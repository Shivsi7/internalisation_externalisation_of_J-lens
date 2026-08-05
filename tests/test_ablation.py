"""CPU tests for src/ablation/ against a tiny random-init Qwen3.

The load-bearing test is test_generation_matches_hf_generate: our manual
paired-cache generation loop must reproduce transformers' own generate()
token-for-token (greedy, left-padded batch) when no ablation is applied.
Everything else checks the ablation math invariants.
"""

from __future__ import annotations

import torch
import pytest

from ablation import AblationConfig, JSpaceAblator, AblationSession, band_layers
from ablation.generate import generate_with_ablation
from ablation.jspace import JSpaceAblator as _A

torch.manual_seed(0)

D, V, LAYERS = 32, 96, 4


# --------------------------------------------------------------------- helpers


class FakeTok:
    """Byte-level tokenizer with just the surface generate_with_ablation uses."""

    eos_token_id = 1
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"
    padding_side = "left"

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=False):
        seqs = [[2 + (b % (V - 2)) for b in t.encode()] for t in texts]
        n = max(len(s) for s in seqs)
        ids = [[self.pad_token_id] * (n - len(s)) + s for s in seqs]
        mask = [[0] * (n - len(s)) + [1] * len(s) for s in seqs]
        from types import SimpleNamespace

        return SimpleNamespace(
            input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask)
        )

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(i)) for i in ids)


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        hidden_size=D,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=V,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    torch.manual_seed(7)
    model = Qwen3ForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_lens_jacobians(n_layers=LAYERS, d=D, seed=3):
    torch.manual_seed(seed)
    return {l: torch.eye(d) + 0.05 * torch.randn(d, d) for l in range(n_layers - 1)}


def make_ablator(model, band, **overrides):
    kw = dict(band=band, k=4, filter_top_n=5)
    kw.update(overrides)
    cfg = AblationConfig(**kw)
    return JSpaceAblator.from_hf(model, _Lens(make_lens_jacobians()), cfg)


class _Lens:
    def __init__(self, jacobians):
        self.jacobians = jacobians


# ------------------------------------------------------------- projection math


def test_ls_projection_zeroes_selected_coefficients():
    torch.manual_seed(1)
    h = torch.randn(6, D)
    A = torch.randn(6, 4, D)
    delta = _A._ls_delta(h, A, ridge=0.0)
    # After removal, the coefficient of h' along every selected direction is 0.
    resid = torch.einsum("nkd,nd->nk", A, h - delta)
    assert resid.abs().max() < 1e-4
    # delta lies in span(A): projecting delta onto span(A) returns delta.
    delta2 = _A._ls_delta(delta, A, ridge=0.0)
    assert torch.allclose(delta, delta2, atol=1e-4)


def test_sequential_equals_ls_for_orthogonal_directions():
    torch.manual_seed(2)
    h = torch.randn(5, D)
    q, _ = torch.linalg.qr(torch.randn(D, 4))
    A = q.T.unsqueeze(0).expand(5, -1, -1)
    d_ls = _A._ls_delta(h, A, ridge=0.0)
    d_seq = _A._sequential_delta(h, A)
    assert torch.allclose(d_ls, d_seq, atol=1e-4)


def test_sequential_differs_from_ls_for_oblique_directions():
    torch.manual_seed(3)
    h = torch.randn(5, D)
    base = torch.randn(4, D)
    A = (base + 0.9 * base.roll(1, dims=0)).unsqueeze(0).expand(5, -1, -1)
    d_ls = _A._ls_delta(h, A, ridge=0.0)
    d_seq = _A._sequential_delta(h, A)
    assert not torch.allclose(d_ls, d_seq, atol=1e-3)


# ---------------------------------------------------- selection and occupancy


def test_exclusion_filter_respected(tiny):
    ab = make_ablator(tiny, band=[1, 2])
    h = torch.randn(8, D)
    # Ban the unfiltered top-5 and check none of them get selected.
    scores = ab._lens_scores(h, 1, use_jacobian=True)
    banned = scores.topk(5, dim=-1).indices
    _, ids = ab.select_directions(h, 1, banned)
    assert not (ids.unsqueeze(-1) == banned.unsqueeze(1)).any()


def test_rankshift_disjoint_from_topk(tiny):
    ab_top = make_ablator(tiny, band=[1])
    ab_shift = make_ablator(
        tiny, band=[1], direction_source="jlens_rankshift", rank_shift=10
    )
    h = torch.randn(8, D)
    _, top_ids = ab_top.select_directions(h, 1, None)
    _, shift_ids = ab_shift.select_directions(h, 1, None)
    assert not (top_ids.unsqueeze(-1) == shift_ids.unsqueeze(1)).any()


def test_occupancy_zero_after_ablation(tiny):
    ab = make_ablator(tiny, band=[1], exclusion_filter=False)
    h = torch.randn(2, 6, D)
    _, ids = ab.select_directions(h.reshape(-1, D).float(), 1, None)
    out = ab.ablate(h.clone(), 1, None, None)
    occ = ab.occupancy(out.reshape(-1, D), 1, ids)
    assert occ.abs().max() < 1e-3
    # ...and it was not zero before.
    occ_before = ab.occupancy(h.reshape(-1, D), 1, ids)
    assert occ_before.abs().max() > 1e-2


def test_active_mask_limits_ablation(tiny):
    ab = make_ablator(tiny, band=[1], exclusion_filter=False)
    h = torch.randn(2, 5, D)
    active = torch.zeros(2, 5, dtype=torch.bool)
    active[:, -1] = True
    out = ab.ablate(h.clone(), 1, None, active)
    assert torch.equal(out[:, :-1], h[:, :-1])
    assert not torch.equal(out[:, -1], h[:, -1])


def test_norm_matched_random_control(tiny):
    ab_j = make_ablator(tiny, band=[1], exclusion_filter=False)
    ab_r = make_ablator(
        tiny, band=[1], exclusion_filter=False,
        direction_source="random", norm_match=True,
    )
    h = torch.randn(1, 8, D)
    out_j = ab_j.ablate(h.clone(), 1, None, None)
    out_r = ab_r.ablate(h.clone(), 1, None, None)
    dj = (h - out_j).norm(dim=-1)
    dr = (h - out_r).norm(dim=-1)
    assert torch.allclose(dj, dr, rtol=0.05)
    assert not torch.allclose(out_j, out_r, atol=1e-3)


# ----------------------------------------------------------- hooks/generation


def test_hooks_skip_is_noop(tiny):
    tok = FakeTok()
    enc = tok(["hello world", "abc"])
    ab = make_ablator(tiny, band=[1, 2], exclusion_filter=False)
    ref = tiny(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits
    with AblationSession(ab, tiny.model.layers) as sess:
        sess.skip = True
        out = tiny(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits
    assert torch.equal(ref, out)


def test_hooks_change_logits(tiny):
    tok = FakeTok()
    enc = tok(["hello world"])
    ab = make_ablator(tiny, band=[1, 2], exclusion_filter=False)
    ref = tiny(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits
    with AblationSession(ab, tiny.model.layers) as sess:
        sess.set_state(None, enc.attention_mask.bool())
        out = tiny(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits
    assert not torch.allclose(ref, out, atol=1e-4)
    assert len(tiny.model.layers[1]._forward_hooks) == 0  # hooks removed


def test_generation_matches_hf_generate(tiny):
    """No-ablation path must reproduce transformers' generate() exactly."""
    tok = FakeTok()
    prompts = ["the quick brown fox", "ab", "some longer prompt here!"]
    ours = generate_with_ablation(
        tiny, tok, prompts, ablator=None, max_new_tokens=12, temperature=0.0
    )
    enc = tok(prompts)
    ref = tiny.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_new_tokens=12,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    ref_new = ref[:, enc.input_ids.shape[1]:]
    for i, r in enumerate(ours):
        ids = ref_new[i].tolist()
        if tok.eos_token_id in ids:
            ids = ids[: ids.index(tok.eos_token_id)]
        ids = [t for t in ids if t != tok.pad_token_id]
        assert r.text == tok.decode(ids), f"seq {i} diverged"


def test_ablated_generation_runs_and_differs(tiny):
    tok = FakeTok()
    prompts = ["the quick brown fox", "ab"]
    band = [1, 2]
    clean = generate_with_ablation(
        tiny, tok, prompts, ablator=None, max_new_tokens=10, temperature=0.0
    )
    ab = make_ablator(tiny, band=band)  # filter on -> paired clean pass path
    ablated = generate_with_ablation(
        tiny, tok, prompts, ablator=ab, max_new_tokens=10, temperature=0.0
    )
    assert all(r.n_new_tokens > 0 for r in ablated)
    stats = ab.stats_summary()
    assert set(stats) == set(band)
    assert all(s["rel"] > 0 for s in stats.values())
    assert any(c.text != a.text for c, a in zip(clean, ablated))


def test_filter_off_is_single_pass(tiny):
    tok = FakeTok()
    ab = make_ablator(tiny, band=[1], exclusion_filter=False)
    out = generate_with_ablation(
        tiny, tok, ["hello"], ablator=ab, max_new_tokens=5, temperature=0.0
    )
    assert out[0].n_new_tokens > 0
