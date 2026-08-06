#!/usr/bin/env python3
"""Degradation control C5: clean-vs-ablated top-1 next-token agreement on
WikiText-103, the paper's own selectivity metric (Figure 22: 87/75/65% for
light/medium/heavy vs 96% for the random control on Sonnet 4.5).

Single forward pass per sequence per condition; the clean pass's top-10 also
supplies the exclusion filter, exactly as in generation. Cheap (~minutes).

Writes results/<host>/<ts>_fluency/summary.json.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import socket
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

N_SEQS = 64
SEQ_LEN = 512


def main() -> None:
    import torch
    import jlens
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ablation import AblationConfig, JSpaceAblator, AblationSession, band_layers

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    lens = jlens.JacobianLens.load(str(
        REPO / "lenses/qwen3-4b/qwen3-4b/jlens/Salesforce-wikitext/"
        "Qwen3-4B_jacobian_lens.pt"))
    device = model.lm_head.weight.device

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="validation")
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer.encode(text, add_special_tokens=False)[: N_SEQS * SEQ_LEN]
    seqs = torch.tensor(
        [ids[i * SEQ_LEN:(i + 1) * SEQ_LEN] for i in range(len(ids) // SEQ_LEN)],
        device=device,
    )
    print(f"[fluency] {seqs.shape[0]} seqs x {SEQ_LEN} tokens")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / socket.gethostname() / f"{stamp}_fluency"
    out_dir.mkdir(parents=True, exist_ok=False)

    n_layers = model.config.num_hidden_layers
    conditions = {
        "light": AblationConfig(band=band_layers(n_layers, "light")),
        "medium": AblationConfig(band=band_layers(n_layers, "medium")),
        "heavy": AblationConfig(band=band_layers(n_layers, "heavy")),
        "random_medium": AblationConfig(
            band=band_layers(n_layers, "medium"),
            direction_source="random", norm_match=True,
        ),
    }

    t0 = time.time()
    summary = {"n_sequences": int(seqs.shape[0]), "seq_len": SEQ_LEN}
    BATCH = 8
    with torch.no_grad():
        for name, cfg in conditions.items():
            ablator = JSpaceAblator.from_hf(model, lens, cfg)
            agree = total = 0
            for i in range(0, seqs.shape[0], BATCH):
                batch = seqs[i:i + BATCH]
                out_c = model(input_ids=batch, use_cache=False)
                top_c = out_c.logits.argmax(-1)
                filt = out_c.logits.topk(cfg.filter_top_n, dim=-1).indices
                with AblationSession(ablator, model.model.layers) as sess:
                    sess.set_state(filt, None)
                    out_a = model(input_ids=batch, use_cache=False)
                top_a = out_a.logits.argmax(-1)
                agree += int((top_c == top_a).sum())
                total += top_c.numel()
            summary[name] = {"top1_agreement": agree / total}
            print(f"[fluency] {name:>14}: {agree/total:.4f}", flush=True)

    summary["wall_seconds"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[fluency] -> {out_dir}")


if __name__ == "__main__":
    main()
