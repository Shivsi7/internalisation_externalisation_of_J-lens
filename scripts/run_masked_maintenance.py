#!/usr/bin/env python3
"""Masked-maintenance test: the deconfounded version of rung 1.5's maintenance
experiment.

The original maintenance test was confounded because the value's defining
expression stays readable in the frozen context, so a null (no gap occupancy)
cannot separate internal holding from re-reading at the point of use. Here we
remove the confound causally: a 4D attention mask makes the definition span
UNREADABLE from the question span onward, while the gap (filler) positions keep
normal access to it. Internal holding (value transported into gap-position
residuals) remains possible; re-reading at the use site becomes impossible.

Prompt (single forward pass, no generation; the answer is one digit read from
the final-position logits):

  [demo]  w = 3. <demo filler>. Question: ... Answer: w = 3.
  [seg1]  {name} = {value}.
  [seg2]  <filler, no digits>                       (the gap)
  [seg3]  Question: what is the value of {name}? Answer with just the
          number. Answer: {name} =

Conditions:
  baseline   causal mask only. Expect ceiling; validates the prompt.
  masked_use seg3 queries cannot attend seg1 keys. The discovery cell: if
             accuracy survives, the value was carried through gap positions
             (holding is real); if it collapses to the severed floor, the
             model was re-reading and never held anything.
  severed    seg2 AND seg3 queries cannot attend seg1 keys. Negative control;
             the value cannot cross the mask at all, so this is the floor.
  page       masked_use mask, but the filler ends with a restatement
             "Remember that {name} = {value}." inside seg2. The written-page
             analog: a later visible copy should rescue.

Also records, from the baseline forward pass, the J-lens occupancy z-score of
the correct digit at the gap positions (mid-band layers), so holding-as-
measured (occupancy) can be compared with holding-as-causally-tested
(masked_use accuracy).

Writes results/<host>/<ts>_masked_maintenance/{config.json,detail.jsonl,
summary.json}.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import socket
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DEMO = (
    "w = 3. The chairs in the waiting room had been rearranged twice before "
    "anyone arrived. Question: what is the value of w? Answer with just the "
    "number. Answer: w = 3.\n"
)
FILLERS = [
    "The library closes early on Fridays, and the corridor lights were "
    "already dimmed when the last students packed up their notes and left "
    "the reading room.",
    "A delivery van idled outside the bakery while the driver checked the "
    "address again, then carried the boxes of pastries across the wet "
    "street before the rain returned.",
]
NAMES = ["x", "y", "z", "k", "m", "p"]
VALUES = [1, 2, 4, 5, 6, 7, 8, 9]  # 3 excluded: it is the demo's value
CONDITIONS = ["baseline", "masked_use", "severed", "page"]


def main() -> None:
    import torch
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ablation import AblationConfig, JSpaceAblator, band_layers

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = model.lm_head.weight.device
    dtype = model.dtype

    lens = jlens.JacobianLens.load(str(
        REPO / "lenses/qwen3-4b/qwen3-4b/jlens/Salesforce-wikitext/"
        "Qwen3-4B_jacobian_lens.pt"))
    band = band_layers(model.config.num_hidden_layers, "medium")
    ablator = JSpaceAblator.from_hf(
        model, lens, AblationConfig(band=band, exclusion_filter=False),
        device=device)

    # candidate digit token ids, both with and without leading space
    digit_ids = {}
    for d in range(1, 10):
        forms = []
        for s in (str(d), " " + str(d)):
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                forms.append(enc[0])
        digit_ids[d] = forms
    gap_digit_ids = torch.tensor(
        [digit_ids[d][0] for d in range(1, 10)])  # no-space form for occupancy

    def encode(s):
        return tok.encode(s, add_special_tokens=False)

    def build(name, value, filler, restate):
        seg0 = encode(DEMO)
        seg1 = encode(f"{name} = {value}.")
        gap_text = " " + filler
        if restate:
            gap_text += f" Remember that {name} = {value}."
        seg2 = encode(gap_text)
        seg3 = encode(
            f" Question: what is the value of {name}? Answer with just the "
            f"number. Answer: {name} =")
        ids = seg0 + seg1 + seg2 + seg3
        a, b = len(seg0), len(seg0) + len(seg1)          # seg1 = [a, b)
        c = b + len(seg2)                                # seg2 = [b, c)
        return ids, a, b, c

    neg = torch.finfo(dtype).min

    def mask4d(S, a, b, block_from):
        """causal 4D additive mask; queries >= block_from can't see [a, b)."""
        m = torch.full((S, S), neg, dtype=dtype)
        m = torch.triu(m, diagonal=1)  # causal: 0 on/below diagonal
        if block_from is not None:
            m[block_from:, a:b] = neg
        return m.view(1, 1, S, S).to(device)

    results = {c: [] for c in CONDITIONS}
    detail = []
    items = [(n, v, fi) for n in NAMES for v in VALUES
             for fi in range(len(FILLERS))]

    with torch.no_grad():
        for name, value, fi in items:
            row = {"name": name, "value": value, "filler": fi}
            for cond in CONDITIONS:
                restate = cond == "page"
                ids, a, b, c = build(name, value, FILLERS[fi], restate)
                S = len(ids)
                if cond in ("masked_use", "page"):
                    block_from = c
                elif cond == "severed":
                    block_from = b
                else:
                    block_from = None
                am = mask4d(S, a, b, block_from)
                x = torch.tensor([ids], device=device)
                out = model(x, attention_mask=am,
                            output_hidden_states=(cond == "baseline"))
                logits = out.logits[0, -1].float()
                scores = {d: max(logits[t].item() for t in digit_ids[d])
                          for d in range(1, 10)}
                pred = max(scores, key=scores.get)
                row[cond] = {"pred": pred, "correct": pred == value}
                results[cond].append(pred == value)

                if cond == "baseline":
                    # occupancy z of the correct digit at gap positions,
                    # averaged over the medium band
                    zs = []
                    for layer in band:
                        h = out.hidden_states[layer + 1][0, b:c].float()
                        occ = ablator.occupancy(
                            h, layer, gap_digit_ids)          # [gap, 9]
                        col = occ.mean(dim=0)                 # [9]
                        z = ((col[value - 1] - col.mean())
                             / col.std().clamp_min(1e-6)).item()
                        zs.append(z)
                    row["gap_occ_z"] = sum(zs) / len(zs)
            detail.append(row)

    n = len(items)

    def wilson(k, nn, z=1.96):
        if nn == 0:
            return (0.0, 1.0)
        p = k / nn
        d0 = 1 + z * z / nn
        cc = (p + z * z / (2 * nn)) / d0
        h = z * ((p * (1 - p) / nn + z * z / (4 * nn * nn)) ** 0.5) / d0
        return (round(cc - h, 4), round(cc + h, 4))

    summary = {"n": n}
    for cond in CONDITIONS:
        k = sum(results[cond])
        summary[cond] = {
            "acc": round(k / n, 4), "k": k, "wilson95": wilson(k, n)}
    held = [r["gap_occ_z"] for r in detail]
    summary["gap_occ_z_mean"] = round(sum(held) / len(held), 4)
    ok = [r["gap_occ_z"] for r in detail if r["masked_use"]["correct"]]
    bad = [r["gap_occ_z"] for r in detail if not r["masked_use"]["correct"]]
    summary["gap_occ_z_masked_correct"] = (
        round(sum(ok) / len(ok), 4) if ok else None)
    summary["gap_occ_z_masked_wrong"] = (
        round(sum(bad) / len(bad), 4) if bad else None)

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    host = socket.gethostname().split(".")[0]
    outdir = REPO / "results" / host / f"{ts}_masked_maintenance"
    outdir.mkdir(parents=True)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    config = {
        "script": "run_masked_maintenance.py", "argv": sys.argv,
        "model": "Qwen/Qwen3-4B", "band": "medium", "band_layers": band,
        "conditions": CONDITIONS, "names": NAMES, "values": VALUES,
        "n_fillers": len(FILLERS), "n_items": n, "demo_value": 3,
        "decoding": "argmax over digit tokens at final position, no sampling",
        "git_commit": commit, "host": host,
        "gpu": torch.cuda.get_device_name(0),
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2))
    with open(outdir / "detail.jsonl", "w") as f:
        for r in detail:
            f.write(json.dumps(r) + "\n")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote", outdir)


if __name__ == "__main__":
    main()
