#!/usr/bin/env python3
"""Injection sufficiency test for duplication (hypotheses.md A8.1).

Masked-maintenance setup, but during the gap we write the value into the
residual stream ourselves: the value digit's J-lens direction, added at
every band layer at gap positions, scaled by 0.1/0.3/1.0 of each position's
residual norm. The definition is masked from the question as before, so the
only route to the answer is the planted internal copy. Wrong-value injection
is the specificity control: if the channel works, answers should track the
injected digit.

Writes results/<host>/<ts>_injection_sufficiency/{config.json,summary.json}.
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
sys.path.insert(0, str(REPO / "scripts"))
from run_masked_maintenance import DEMO, FILLERS, NAMES, VALUES

SCALES = (0.1, 0.3, 1.0)


def main() -> None:
    import torch
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ablation import AblationConfig, JSpaceAblator, band_layers

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

    lens = jlens.JacobianLens.load(HW0_LENS)
    band = band_layers(model.config.num_hidden_layers, "medium")
    ablator = JSpaceAblator.from_hf(
        model, lens, AblationConfig(band=band, exclusion_filter=False),
        device=device)
    blocks = model.model.layers

    digit_ids = {}
    for d in range(1, 10):
        forms = [tok.encode(s, add_special_tokens=False)
                 for s in (str(d), " " + str(d))]
        digit_ids[d] = [f[0] for f in forms if len(f) == 1]

    def unit_dir(value, layer):
        ids = torch.tensor([[digit_ids[value][0]]])
        A = ablator._dirs_for_ids(ids.to(ablator.W_U.device), layer,
                                  use_jacobian=True)
        j = A.reshape(A.shape[0], A.shape[1], -1)[0, 0].float()
        return (j / j.norm().clamp_min(1e-12)).to(device)

    dirs = {v: {l: unit_dir(v, l) for l in band} for v in range(1, 10)}

    def encode(s):
        return tok.encode(s, add_special_tokens=False)

    def build(name, value, filler):
        seg0 = encode(DEMO)
        seg1 = encode(f"{name} = {value}.")
        seg2 = encode(" " + filler)
        seg3 = encode(f" Question: what is the value of {name}? Answer with "
                      f"just the number. Answer: {name} =")
        ids = seg0 + seg1 + seg2 + seg3
        a, b = len(seg0), len(seg0) + len(seg1)
        return ids, a, b, b + len(seg2)

    def mask4d(S, a, b, block_from):
        m = torch.full((S, S), neg, dtype=mdtype)
        m = torch.triu(m, diagonal=1)
        if block_from is not None:
            m[block_from:, a:b] = neg
        return m.view(1, 1, S, S).to(device)

    state = {"on": False, "value": None, "scale": 0.0, "lo": 0, "hi": 0}

    def make_hook(layer):
        def hook(module, inputs, output):
            if not state["on"]:
                return None
            h = output[0] if not torch.is_tensor(output) else output
            j = dirs[state["value"]][layer].to(h.dtype)
            seg = h[0, state["lo"]:state["hi"]]
            norms = seg.float().norm(dim=-1, keepdim=True).to(h.dtype)
            seg = seg + state["scale"] * norms * j
            h = h.clone()
            h[0, state["lo"]:state["hi"]] = seg
            if torch.is_tensor(output):
                return h
            return (h,) + tuple(output[1:])
        return hook

    handles = [blocks[l].register_forward_hook(make_hook(l)) for l in band]

    conds = ["baseline", "masked", "severed"] + \
            [f"inj_{w}_{a}" for w in ("correct", "wrong") for a in SCALES]
    acc = {c: 0 for c in conds}
    track = {c: 0 for c in conds}
    items = [(n, v, fi) for n in NAMES for v in VALUES
             for fi in range(len(FILLERS))]

    with torch.no_grad():
        for name, value, fi in items:
            wrong = VALUES[(VALUES.index(value) + 3) % len(VALUES)]
            ids, a, b, c = build(name, value, FILLERS[fi])
            S = len(ids)
            x = torch.tensor([ids], device=device)
            for cond in conds:
                if cond == "baseline":
                    bf, inj = None, None
                elif cond == "masked":
                    bf, inj = c, None
                elif cond == "severed":
                    bf, inj = b, None
                else:
                    _, w, s = cond.split("_")
                    bf = c
                    inj = (value if w == "correct" else wrong, float(s))
                state["on"] = inj is not None
                if inj is not None:
                    state.update(value=inj[0], scale=inj[1], lo=b, hi=c)
                am = mask4d(S, a, b, bf)
                lg = model(x, attention_mask=am).logits[0, -1].float()
                scores = {d: max(lg[t].item() for t in digit_ids[d])
                          for d in range(1, 10)}
                pred = max(scores, key=scores.get)
                acc[cond] += pred == value
                if cond.startswith("inj_wrong"):
                    track[cond] += pred == wrong
            state["on"] = False

    for h in handles:
        h.remove()

    n = len(items)
    summary = {"n": n}
    for cond in conds:
        summary[cond] = {"acc": round(acc[cond] / n, 4)}
        if cond.startswith("inj_wrong"):
            summary[cond]["tracks_injected"] = round(track[cond] / n, 4)
        print(f"[{cond}] acc={acc[cond]}/{n}"
              + (f" tracks={track[cond]}/{n}" if cond.startswith("inj_wrong")
                 else ""), flush=True)

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    host = socket.gethostname().split(".")[0]
    outdir = REPO / "results" / host / f"{ts}_injection_sufficiency"
    outdir.mkdir(parents=True)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    (outdir / "config.json").write_text(json.dumps({
        "script": "run_injection_sufficiency.py", "argv": sys.argv,
        "model": HW0_MODEL, "band": band, "scales": SCALES,
        "wrong_rule": "VALUES[(i+3) % len]", "git_commit": commit,
        "host": host, "gpu": torch.cuda.get_device_name(0)}, indent=2))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", outdir)


if __name__ == "__main__":
    main()
