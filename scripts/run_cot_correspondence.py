#!/usr/bin/env python3
"""Correspondence test, CoT arm: what does the workspace hold while the model
writes its OWN chain of thought?

The direct-mode correspondence test showed the workspace can represent
page-type values when nothing is externalised. But the complementarity claim
is about the CoT scenario itself, so the measurement must run there, and on
the model's own reasoning rather than a teacher-forced transcript. Here the model solves each program with its own chain of thought
(greedy, chat template), and we measure the first value's occupancy at the
moment the model writes it, during the later lines, and at the line that uses
it, all inside the model's own generated text.

Programs are the persistence-probe generator's (three lines + d = a op c),
with the same constraint that v1 differs from v2, v3 and every literal digit
in lines 2-3, so post-window occupancy of v1 cannot be explained by v1's
digit sitting in the prompt there. Windows are located by the first
occurrence of each value's digit token in the generated ids (v1 cannot
appear earlier by construction). Items where the digits do not appear in
computation order are dropped and counted.

Writes results/<host>/<ts>_cot_correspondence/{config.json,detail.jsonl,
summary.json}.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import random
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

N_PROGRAMS = 80
SEED = int(_os.environ.get("HW0_SEED", "0"))
MAX_NEW = 256
# "var": d = a op c (original). "lit": d = a op <literal> (Addendum 9b),
# matched to the direct arm's serial budget.
FINAL = _os.environ.get("HW0_FINAL", "var")


def gen_programs(n, rng):
    progs, tries = [], 0
    while len(progs) < n and tries < 100000:
        tries += 1
        lines, vals, ok = [], [], True
        for _ in range(3):
            x, y = rng.randint(1, 9), rng.randint(1, 9)
            op = rng.choice(["+", "-"])
            v = x + y if op == "+" else x - y
            if not (1 <= v <= 9):
                ok = False
                break
            lines.append((x, op, y, v))
            vals.append(v)
        if not ok:
            continue
        v1, v2, v3 = vals
        op = rng.choice(["+", "-"])
        if FINAL == "lit":
            k = rng.randint(1, 9)
            d = v1 + k if op == "+" else v1 - k
            if not (1 <= d <= 9) or k == v1 or d == v1:
                continue
        else:
            k = None
            d = v1 + v3 if op == "+" else v1 - v3
            if not (1 <= d <= 9):
                continue
        lits23 = {lines[1][0], lines[1][2], vals[1], lines[2][0],
                  lines[2][2], vals[2]}
        if v1 in lits23 or v1 == v2 or v1 == v3:
            continue
        progs.append({"lines": lines, "final_op": op, "final_lit": k,
                      "d": d})
    return progs


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

    lens = jlens.JacobianLens.load(HW0_LENS)
    band = band_layers(model.config.num_hidden_layers, "medium")
    ablator = JSpaceAblator.from_hf(
        model, lens, AblationConfig(band=band, exclusion_filter=False),
        device=device)

    digit_tok = {}
    for d in range(1, 10):
        forms = [tok.encode(s, add_special_tokens=False)
                 for s in (str(d), " " + str(d))]
        digit_tok[d] = [f[0] for f in forms if len(f) == 1]
    occ_ids = torch.tensor([digit_tok[d][0] for d in range(1, 10)])

    def window_z(hs, lo, hi, value, decoy=None):
        """(mean_z, max_z, decoy_max): sustained presence, best single
        position-layer spike, and the same max for an absent digit (the
        selection-bias null)."""
        if hi <= lo:
            return None, None, None
        zs, zmax, dmax = [], [], []
        for layer in band:
            h = hs[layer + 1][0, lo:hi].float()
            occ = ablator.occupancy(h, layer, occ_ids)
            col = occ.mean(dim=0)
            z = ((col[value - 1] - col.mean())
                 / col.std().clamp_min(1e-6)).item()
            zs.append(z)
            zall = ((occ - occ.mean(dim=1, keepdim=True))
                    / occ.std(dim=1, keepdim=True).clamp_min(1e-6))
            zmax.append(zall[:, value - 1].max().item())
            if decoy is not None:
                dmax.append(zall[:, decoy - 1].max().item())
        return (sum(zs) / len(zs), max(zmax),
                max(dmax) if dmax else None)

    rng = random.Random(SEED)
    progs = gen_programs(N_PROGRAMS, rng)
    detail, dropped = [], 0
    with torch.no_grad():
        for prog in progs:
            (x1, o1, y1, v1), (x2, o2, y2, v2), (x3, o3, y3, v3) = prog["lines"]
            fo, dval = prog["final_op"], prog["d"]
            lit = prog.get("final_lit")
            dline = (f"d = a {fo} {lit}." if lit is not None
                     else f"d = a {fo} c.")
            question = (f"a = {x1} {o1} {y1}. b = {x2} {o2} {y2}. "
                        f"c = {x3} {o3} {y3}. {dline} "
                        "Work through the program line by line, then state "
                        "the value of d.")
            msgs = [{"role": "user", "content": question}]
            enc = tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt")
            ids = (enc["input_ids"] if not hasattr(enc, "shape")
                   else enc).to(device)
            plen = ids.shape[1]
            out = model.generate(
                ids, max_new_tokens=MAX_NEW, do_sample=False,
                pad_token_id=tok.eos_token_id)
            gen = out[0, plen:].tolist()

            def first_pos(val, after=0):
                cands = [i for i, t in enumerate(gen)
                         if t in digit_tok[val] and i >= after]
                return cands[0] if cands else None

            i1 = first_pos(v1)
            i2 = first_pos(v2, (i1 or 0) + 1) if i1 is not None else None
            i3 = first_pos(v3, (i2 or 0) + 1) if i2 is not None else None
            if None in (i1, i2, i3) or not (i1 < i2 < i3):
                dropped += 1
                continue

            hs = model(out, output_hidden_states=True).hidden_states
            base = plen
            used = {v1, v2, v3, dval, x1, y1, x2, y2, x3, y3}
            if lit is not None:
                used.add(lit)
            free = [d9 for d9 in range(1, 10) if d9 not in used]
            decoy = free[0] if free else None
            row = {"prog": prog, "decoy": decoy}
            for key, lo, hi in (
                    ("z_write", base + max(i1 - 1, 0), base + i1 + 2),
                    ("z_post", base + i1 + 4, base + i3),
                    ("z_use", base + i3 + 3, base + len(gen))):
                m, mx, dm = window_z(hs, lo, hi, v1, decoy)
                row[key] = m
                row[key + "_max"] = mx
                row[key + "_decoy_max"] = dm
            last_digit = None
            for t in reversed(gen):
                hit = [d for d in range(1, 10) if t in digit_tok[d]]
                if hit:
                    last_digit = hit[0]
                    break
            row["pred"] = last_digit
            row["correct"] = last_digit == dval
            detail.append(row)

    def agg(key):
        v = [r[key] for r in detail if r[key] is not None]
        if not v:
            return None
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1)) ** 0.5
        return {"mean": round(m, 3), "sd": round(sd, 3),
                "se": round(sd / len(v) ** 0.5, 3), "n": len(v)}

    k = sum(r["correct"] for r in detail)
    summary = {
        "n_scored": len(detail), "n_dropped": dropped,
        "acc": round(k / len(detail), 4) if detail else None,
        "z_write": agg("z_write"), "z_post": agg("z_post"),
        "z_use": agg("z_use"),
        "z_write_max": agg("z_write_max"), "z_post_max": agg("z_post_max"),
        "z_use_max": agg("z_use_max"),
        "z_write_decoy_max": agg("z_write_decoy_max"),
        "z_post_decoy_max": agg("z_post_decoy_max"),
        "z_use_decoy_max": agg("z_use_decoy_max"),
    }

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    host = socket.gethostname().split(".")[0]
    outdir = REPO / "results" / host / f"{ts}_cot_correspondence"
    outdir.mkdir(parents=True)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    config = {
        "script": "run_cot_correspondence.py", "argv": sys.argv,
        "model": HW0_MODEL, "band": "medium", "band_layers": band,
        "n_programs": len(progs), "seed": SEED, "max_new": MAX_NEW,
        "final_variant": FINAL,
        "decoding": "greedy via chat template, model's own CoT",
        "windows": ["write (v1 first written)", "post (between v1 and v3)",
                    "use (after v3 written)"],
        "git_commit": commit, "host": host,
        "gpu": __import__("torch").cuda.get_device_name(0),
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
