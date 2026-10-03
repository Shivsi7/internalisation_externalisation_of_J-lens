#!/usr/bin/env python3
"""Correspondence test, direct arm: the contrast panel for the CoT-arm
correspondence test.

The CoT arm measured the first value's occupancy in three windows of the
model's own transcript (while written, during later lines, at use) and found
in-use-only presence. That alone cannot separate two models of the workspace.
Under the I/O-buffer model the value fades between uses in every mode because
the workspace stores nothing. Under an exchange-style division of labour the
fade is CAUSED by externalisation, so with no page to write, the workspace
should hold the value across the gap.

Here the same programs are answered directly (no chain of thought, chat
template with thinking disabled, the depth ladder's proven format) and the
same three windows are read off the prompt positions instead of generated
positions:

  write  the span of v1's defining line plus two following tokens, where the
         ordering probe shows the value first surfaces
  post   the spans of lines 2-3, which do not involve v1
  use    the span of the final line and answer cue, where v1 is needed

The windows are prompt spans by necessity, in direct mode that is where the
computation lives. The contrast statistic is z_post, direct vs CoT. Note the
interpretive limit, the defining expression stays readable in the prompt, so a
low direct z_post means no holding while re-reading is possible, the same
regime as the maintenance tests.

Writes results/<host>/<ts>_direct_correspondence/{config.json,detail.jsonl,
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
# "var": d = a op c (original, three dependent ops, over the direct budget).
# "lit": d = a op <literal>, one retrieval plus one op (Addendum 9b).
FINAL = _os.environ.get("HW0_FINAL", "var")


def gen_programs(n, rng):
    """Byte-identical generator to run_cot_correspondence.py, so the two arms
    see the same program distribution at the same seed."""
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
        """(mean_z, max_z, decoy_max): mean_z averages positions then layers
        (sustained presence); max_z is the best position-layer cell (sparse-
        spike detector). decoy_max is the same max for a digit absent from
        the program, the selection-bias null for max_z."""
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

    def find_span(ids, text):
        """Locate text's LAST token-subsequence occurrence in ids (the real
        program, not the demo); tries leading-space and leading-newline
        variants for boundary merges."""
        for cand in (text, " " + text, "\n" + text):
            sub = tok.encode(cand, add_special_tokens=False)
            for s in range(len(ids) - len(sub), -1, -1):
                if ids[s:s + len(sub)] == sub:
                    return s, s + len(sub)
        return None

    DEMO_Q = ("a = 2 + 2. b = 1 + 1. c = 3 + 3. d = a + 1. Compute the "
              "final value of d in this program. Answer with only the "
              "number.")
    DEMO_A = "d = 5"

    rng = random.Random(SEED)
    progs = gen_programs(N_PROGRAMS, rng)
    detail, dropped = [], 0
    with torch.no_grad():
        for prog in progs:
            (x1, o1, y1, v1), (x2, o2, y2, v2), (x3, o3, y3, v3) = prog["lines"]
            fo, dval = prog["final_op"], prog["d"]
            line1 = f"a = {x1} {o1} {y1}."
            line23 = f"b = {x2} {o2} {y2}. c = {x3} {o3} {y3}."
            lit = prog.get("final_lit")
            lined = (f"d = a {fo} {lit}." if lit is not None
                     else f"d = a {fo} c.")
            content = (f"{line1} {line23} {lined} Compute the final value of "
                       "d in this program. Answer with only the number.")
            msgs = [{"role": "user", "content": DEMO_Q},
                    {"role": "assistant", "content": DEMO_A},
                    {"role": "user", "content": content}]
            enc = tok.apply_chat_template(
                msgs, add_generation_prompt=True, enable_thinking=False,
                return_tensors="pt")
            x = (enc["input_ids"] if not hasattr(enc, "shape")
                 else enc)
            # prefill the answer frame so the next token is the digit
            pre = tok.encode("d =", add_special_tokens=False)
            x = torch.cat(
                [x, torch.tensor([pre], dtype=x.dtype)], dim=1).to(device)
            ids = x[0].tolist()
            S = len(ids)
            sp1 = find_span(ids, line1)
            sp23 = find_span(ids, line23)
            spd = find_span(ids, lined)
            if None in (sp1, sp23, spd):
                dropped += 1
                continue
            out = model(x, output_hidden_states=True)
            hs = out.hidden_states
            used = {v1, v2, v3, dval, x1, y1, x2, y2, x3, y3}
            if lit is not None:
                used.add(lit)
            free = [d9 for d9 in range(1, 10) if d9 not in used]
            decoy = free[0] if free else None
            row = {"prog": prog, "decoy": decoy}
            for key, lo, hi in (("z_write", sp1[0], min(sp1[1] + 2, S)),
                                ("z_post", sp23[0], sp23[1]),
                                ("z_use", spd[0], S)):
                m, mx, dm = window_z(hs, lo, hi, v1, decoy)
                row[key] = m
                row[key + "_max"] = mx
                row[key + "_decoy_max"] = dm
            logits = out.logits[0, -1].float()
            scores = {d: max(logits[t].item() for t in digit_tok[d])
                      for d in range(1, 10)}
            pred = max(scores, key=scores.get)
            row["pred"] = pred
            row["correct"] = pred == dval
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
    outdir = REPO / "results" / host / f"{ts}_direct_correspondence"
    outdir.mkdir(parents=True)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    config = {
        "script": "run_direct_correspondence.py", "argv": sys.argv,
        "model": HW0_MODEL, "band": "medium", "band_layers": band,
        "n_programs": len(progs), "seed": SEED, "final_variant": FINAL,
        "decoding": ("chat template, enable_thinking=False, one-shot demo, "
                     "assistant prefilled with 'd =', argmax over digit "
                     "tokens at the final position"),
        "windows": ["write (line-1 span + 2)", "post (lines 2-3 spans)",
                    "use (d-line span to end of prompt incl. prefill)"],
        "prompt_note": "v4, demo + prefill readout; v1 bare completion and "
                       "v2/v3 free generation sat at the digit floor "
                       "(format artifact, 26/80 answers with no digit)",
        "demo": {"q": DEMO_Q, "a": DEMO_A},
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
