#!/usr/bin/env python3
"""Verify: our occupancy() ranks digit tokens identically to the reference
JacobianLens.apply() J-lens readout at every band layer.

occupancy(v) = <h, J_l^T (g ⊙ W_U[v])>; apply() = W_U · RMSNorm_g(J_l h).
These differ only by the position's RMS scalar, so within a position the
token ORDERING must be identical. This script checks that, and also checks
the logit-lens (use_jacobian=False) ordering DIFFERS mid-band — i.e. the
Jacobian transport is actually doing something.
"""

from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PROMPT = "calc: (4+17)*2+7 = "


def main() -> None:
    import torch
    import jlens
    from jlens.hf import from_hf
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ablation import AblationConfig, JSpaceAblator, band_layers

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    lens = jlens.JacobianLens.load(str(
        REPO / "lenses/qwen3-4b/qwen3-4b/jlens/Salesforce-wikitext/"
        "Qwen3-4B_jacobian_lens.pt"))
    lm = from_hf(model, tokenizer)
    band = band_layers(model.config.num_hidden_layers, "heavy")
    ablator = JSpaceAblator.from_hf(
        model, lens, AblationConfig(band=band, exclusion_filter=False))

    digit_ids = []
    for d in range(10):
        for s in (str(d), " " + str(d)):
            enc = tokenizer.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                digit_ids.append(enc[0])
    ids_t = torch.tensor(digit_ids, dtype=torch.long, device=model.device)

    # Reference readout (the repo's own code path), all positions
    lens_logits, _, input_ids = lens.apply(model=lm, prompt=PROMPT, layers=band)
    # Our residuals for the same forward
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)

    pos = -1  # the "=" position
    agree = disagree_ll = 0
    for layer in band:
        ref_order = torch.tensor(
            [lens_logits[layer][pos, i].item() for i in digit_ids]
        ).argsort(descending=True)
        h = out.hidden_states[layer + 1][0][pos:].float()
        ours = ablator.occupancy(h, layer, ids_t)[0]
        # occupancy uses UNIT-normalised directions; for an ordering
        # comparison against apply() re-scale by ||j_v||.
        A = ablator._dirs_for_ids(ids_t.unsqueeze(0), layer, use_jacobian=True)[0]
        ours_scaled = ours.cpu() * A.norm(dim=-1).cpu()
        our_order = ours_scaled.argsort(descending=True)
        same = torch.equal(ref_order, our_order)
        agree += same

        from scipy.stats import spearmanr
        ref_scores = torch.tensor(
            [lens_logits[layer][pos, i].item() for i in digit_ids])
        rho, _ = spearmanr(ref_scores.numpy(), ours_scaled.numpy())
        ref_rank = ref_order.argsort()
        our_rank = our_order.argsort()
        max_disp = int((ref_rank - our_rank).abs().max())

        ll = ablator._lens_scores(h, layer, use_jacobian=False)[0, digit_ids]
        ll_order = ll.cpu().argsort(descending=True)
        disagree_ll += not torch.equal(ref_order, ll_order)
        print(f"L{layer:>2}: exact={bool(same)}  spearman={rho:.4f}  "
              f"max_rank_shift={max_disp}   "
              f"logit-lens differs: {not torch.equal(ref_order, ll_order)}")

    print(f"\nJ-lens ordering agreement: {agree}/{len(band)} layers")
    print(f"logit-lens (no Jacobian) differs at {disagree_ll}/{len(band)} layers")


if __name__ == "__main__":
    main()
