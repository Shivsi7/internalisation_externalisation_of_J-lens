#!/usr/bin/env python3
"""Maintenance robustness sweep (hypotheses.md A6.4): the masked-maintenance
cells across gap lengths (~10/30/60/120 tokens) and content type (digits vs
single-token colour words). Same conditions as run_masked_maintenance.py
(baseline / masked_use / severed / page). The no-holding result is robust if
masked_use stays at the severed floor in every cell.

Writes results/<host>/<ts>_maintenance_sweep/{config.json,summary.json}.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import socket
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
import os as _os
HW0_MODEL = _os.environ.get("HW0_MODEL", "Qwen/Qwen3-4B")
HW0_LENS = _os.environ.get(
    "HW0_LENS", str(REPO / "lenses/qwen3-4b/qwen3-4b/jlens/"
                    "Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"))
sys.path.insert(0, str(REPO / "src"))

BASE_FILLER = ("The library closes early on Fridays, and the corridor "
               "lights were already dimmed when the last students packed up "
               "their notes and left the reading room. A delivery van idled "
               "outside the bakery while the driver checked the address "
               "again, then carried the boxes of pastries across the wet "
               "street before the rain returned. ")
DEMO_D = ("w = 3. The chairs had been rearranged twice before anyone "
          "arrived. Question: what is the value of w? Answer with just the "
          "number. Answer: w = 3.\n")
DEMO_W = ("w = lime. The chairs had been rearranged twice before anyone "
          "arrived. Question: what is the value of w? Answer with just the "
          "word. Answer: w = lime.\n")
NAMES = ["x", "y", "z", "k", "m", "p"]
DIGITS = [1, 2, 4, 5, 6, 7, 8, 9]
WORDS = ["red", "blue", "green", "black", "white", "brown", "pink", "gold"]
GAPS = {"g10": 10, "g30": 30, "g60": 60, "g120": 120}
CONDITIONS = ["baseline", "masked_use", "severed", "page"]


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HW0_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        HW0_MODEL, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = model.lm_head.weight.device
    mdtype = model.dtype
    neg = torch.finfo(mdtype).min

    def enc(s):
        return tok.encode(s, add_special_tokens=False)

    def cand_tokens(vals):
        d = {}
        for v in vals:
            forms = [enc(s) for s in (str(v), " " + str(v))]
            d[v] = [f[0] for f in forms if len(f) == 1]
            assert d[v], f"not single-token: {v}"
        return d

    filler_ids_all = enc(" " + BASE_FILLER * 3)

    def build(content, name, value, gap_len, restate):
        demo = DEMO_D if content == "digit" else DEMO_W
        seg0 = enc(demo)
        seg1 = enc(f"{name} = {value}.")
        gap = list(filler_ids_all[:gap_len])
        if restate:
            gap += enc(f" Remember that {name} = {value}.")
        unit = "number" if content == "digit" else "word"
        seg3 = enc(f" Question: what is the value of {name}? Answer with "
                   f"just the {unit}. Answer: {name} =")
        ids = seg0 + seg1 + gap + seg3
        a, b = len(seg0), len(seg0) + len(seg1)
        c = b + len(gap)
        return ids, a, b, c

    def mask4d(S, a, b, block_from):
        m = torch.full((S, S), neg, dtype=mdtype)
        m = torch.triu(m, diagonal=1)
        if block_from is not None:
            m[block_from:, a:b] = neg
        return m.view(1, 1, S, S).to(device)

    summary = {}
    with torch.no_grad():
        for content, values in (("digit", DIGITS), ("word", WORDS)):
            ctoks = cand_tokens(values)
            for gname, glen in GAPS.items():
                accs = {c: 0 for c in CONDITIONS}
                items = [(n, v) for n in NAMES for v in values]
                for name, value in items:
                    for cond in CONDITIONS:
                        ids, a, b, c = build(
                            content, name, value, glen, cond == "page")
                        S = len(ids)
                        bf = (c if cond in ("masked_use", "page")
                              else b if cond == "severed" else None)
                        am = mask4d(S, a, b, bf)
                        x = torch.tensor([ids], device=device)
                        lg = model(x, attention_mask=am).logits[0, -1].float()
                        scores = {v: max(lg[t].item() for t in ctoks[v])
                                  for v in values}
                        accs[cond] += max(scores, key=scores.get) == value
                n = len(items)
                key = f"{content}_{gname}"
                summary[key] = {c: round(accs[c] / n, 4) for c in CONDITIONS}
                summary[key]["n"] = n
                print(f"[{key}] " + " ".join(
                    f"{c}={accs[c]}/{n}" for c in CONDITIONS), flush=True)

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    host = socket.gethostname().split(".")[0]
    outdir = REPO / "results" / host / f"{ts}_maintenance_sweep"
    outdir.mkdir(parents=True)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    (outdir / "config.json").write_text(json.dumps({
        "script": "run_maintenance_sweep.py", "argv": sys.argv,
        "model": HW0_MODEL, "gaps": GAPS,
        "contents": ["digit", "word"], "git_commit": commit, "host": host,
        "gpu": torch.cuda.get_device_name(0)}, indent=2))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", outdir)


if __name__ == "__main__":
    main()
