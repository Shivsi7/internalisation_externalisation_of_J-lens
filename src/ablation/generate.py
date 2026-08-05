"""Batched generation under J-space ablation, with the paired clean pass.

The exclusion filter needs, at every position, the top-N next-token
predictions of a *clean* forward pass over the same token sequence
(paper §3.5.2). Under incremental decoding that means two KV caches over
identical tokens: at each step the clean model runs first (hooks skipped) to
produce the filter set, then the ablated model runs with that filter. Cost is
2x ablated generation when the filter is on, 1x when it is off (condition B8)
or when there is no ablation.

Left padding + explicit position_ids throughout; sampling defaults are the
Qwen3 recommended settings (T=0.6, top_p=0.95, top_k=20); temperature=0 means
greedy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ablation.hooks import AblationSession
from ablation.jspace import JSpaceAblator


@dataclass
class GenResult:
    prompt: str
    text: str            # generated continuation (special tokens kept)
    n_new_tokens: int
    stop_reason: str     # "eos" | "length"


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    gen: torch.Generator | None,
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits.float() / temperature
    if top_k and top_k > 0:
        kth = logits.topk(top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(-1)
        cum = probs.cumsum(-1)
        remove = (cum - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits
        )
    probs = logits.softmax(-1)
    return torch.multinomial(probs, 1, generator=gen).squeeze(-1)


@torch.no_grad()
def generate_with_ablation(
    model,
    tokenizer,
    prompts: list[str],
    *,
    ablator: JSpaceAblator | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_k: int = 20,
    top_p: float = 0.95,
    seed: int = 0,
    batch_size: int = 16,
    progress=None,
) -> list[GenResult]:
    """Generate continuations for `prompts`, optionally under ablation.

    `prompts` are fully templated strings (chat template already applied,
    including any assistant prefill). Returns results in input order.
    """
    device = model.lm_head.weight.device
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results: list[GenResult] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        gen = None
        if temperature > 0:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed + start)
        results.extend(
            _generate_batch(
                model, tokenizer, chunk, ablator,
                max_new_tokens, temperature, top_k, top_p, gen,
                device, eos_id, pad_id,
            )
        )
        if progress is not None:
            progress(min(start + batch_size, len(prompts)), len(prompts))
    return results


def _generate_batch(
    model, tokenizer, prompts, ablator,
    max_new_tokens, temperature, top_k, top_p, gen,
    device, eos_id, pad_id,
):
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = enc.input_ids.to(device)
    attn = enc.attention_mask.to(device)
    B, S0 = input_ids.shape

    use_filter = ablator is not None and ablator.cfg.exclusion_filter
    filter_n = ablator.cfg.filter_top_n if use_filter else 0
    session = (
        AblationSession(ablator, model.model.layers) if ablator is not None else None
    )

    pos_prefill = (attn.cumsum(-1) - 1).clamp(min=0)
    cur_pos = attn.sum(-1, keepdim=True)  # [B, 1] next position index per seq

    def forward(ids, mask, pos, cache, *, clean, filter_ids=None, active=None):
        if session is not None:
            if clean:
                session.skip = True
            else:
                session.set_state(filter_ids, active)
        return model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        )

    ctx = session if session is not None else _NullCtx()
    with ctx:
        # ---- prefill ----------------------------------------------------
        clean_cache = None
        filter_ids = None
        if use_filter:
            out_c = forward(input_ids, attn, pos_prefill, None, clean=True)
            clean_cache = out_c.past_key_values
            filter_ids = out_c.logits.topk(filter_n, dim=-1).indices  # [B,S0,F]

        if ablator is not None:
            active = attn.bool() if ablator.cfg.ablate_prompt else None
            if ablator.cfg.ablate_prompt:
                out_a = forward(
                    input_ids, attn, pos_prefill, None,
                    clean=False, filter_ids=filter_ids, active=active,
                )
            else:
                out_a = forward(input_ids, attn, pos_prefill, None, clean=True)
        else:
            out_a = (
                out_c if use_filter
                else forward(input_ids, attn, pos_prefill, None, clean=True)
            )
        abl_cache = out_a.past_key_values
        if not use_filter and ablator is None:
            clean_cache = None  # single-cache mode

        next_tok = _sample_token(
            out_a.logits[:, -1, :], temperature, top_k, top_p, gen
        )

        # ---- decode loop ------------------------------------------------
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        generated: list[torch.Tensor] = []
        for _ in range(max_new_tokens):
            next_tok = torch.where(finished, torch.full_like(next_tok, pad_id), next_tok)
            generated.append(next_tok)
            finished = finished | (next_tok == eos_id)
            if bool(finished.all()) or len(generated) == max_new_tokens:
                break

            step_ids = next_tok.unsqueeze(-1)  # [B, 1]
            attn = torch.cat([attn, torch.ones_like(step_ids)], dim=-1)

            step_filter = None
            if use_filter:
                out_c = forward(step_ids, attn, cur_pos, clean_cache, clean=True)
                clean_cache = out_c.past_key_values
                step_filter = out_c.logits.topk(filter_n, dim=-1).indices  # [B,1,F]

            if ablator is not None:
                active = (~finished).unsqueeze(-1)  # [B, 1]
                out_a = forward(
                    step_ids, attn, cur_pos, abl_cache,
                    clean=False, filter_ids=step_filter, active=active,
                )
            else:
                out_a = forward(step_ids, attn, cur_pos, abl_cache, clean=True)
            abl_cache = out_a.past_key_values

            cur_pos = cur_pos + 1
            next_tok = _sample_token(
                out_a.logits[:, -1, :], temperature, top_k, top_p, gen
            )

    gen_ids = torch.stack(generated, dim=1) if generated else input_ids.new_zeros(B, 0)
    results = []
    for i in range(B):
        ids = gen_ids[i].tolist()
        if eos_id in ids:
            stop, ids = "eos", ids[: ids.index(eos_id)]
        else:
            stop = "length"
            ids = [t for t in ids if t != pad_id]
        results.append(
            GenResult(
                prompt=prompts[i],
                text=tokenizer.decode(ids, skip_special_tokens=False),
                n_new_tokens=len(ids),
                stop_reason=stop,
            )
        )
    return results


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None
