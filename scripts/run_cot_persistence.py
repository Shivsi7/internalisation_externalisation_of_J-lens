#!/usr/bin/env python3
"""During-CoT persistence probe: what does the workspace hold while values
sit on the page?

The correspondence test measured occupancy in DIRECT mode (nothing
externalised) and answered the by-content-kind question. The duplication and
exchange hypotheses are about the other arm: while a chain of thought has
written a value onto the page, is that value still represented in the
workspace? Duplication says yes (the page copies what the workspace keeps);
exchange says no (written, then dropped).

Design: teacher-forced worked transcripts (the page), single forward pass,
no generation. Programs of four lines with single-digit intermediates,

  cot arm:    a = 9 - 5 = 4. b = 2 + 1 = 3. c = 8 - 6 = 2. d = a - c = 4 - 2 =
  direct arm: a = 9 - 5. b = 2 + 1. c = 8 - 6. d = a - c =

We measure the J-lens occupancy z of v1 (the value of `a`, z-scored against
all nine digits, averaged over the medium band) in three windows:

  write   line 1's own tokens (sanity: computation should surface it)
  post    lines 2-3 (the discriminator: v1 is on the page in the cot arm,
          nowhere in the direct arm; is it in the workspace?)
  use     line 4 up to the final `=`

Constraints on programs: v1 differs from v2, v3, and from every literal
digit in lines 2-3, so post-window occupancy of v1 cannot be driven by v1's
digit appearing in the text there. Also records answer accuracy at the final
position (argmax over digit tokens) for both arms.

Writes results/<host>/<ts>_cot_persistence/{config.json,detail.jsonl,
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
sys.path.insert(0, str(REPO / "src"))

N_PROGRAMS = 80
SEED = 0


def gen_programs(n, rng):
    progs = []
    tries = 0
    while len(progs) < n and tries < 100000:
        tries += 1
        lines = []
        vals = []
        ok = True
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
        # final line: d = a <op> c, needs v1 and v3
        op = rng.choice(["+", "-"])
        d = v1 + v3 if op == "+" else v1 - v3
        if not (1 <= d <= 9):
            continue
        # v1 must not equal v2/v3 nor any literal digit in lines 2-3
        lits23 = {lines[1][0], lines[1][2], vals[1], lines[2][0],
                  lines[2][2], vals[2]}
        if v1 in lits23:
            continue
        progs.append({"lines": lines, "final_op": op, "d": d})
    return progs


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

    lens = jlens.JacobianLens.load(str(
        REPO / "lenses/qwen3-4b/qwen3-4b/jlens/Salesforce-wikitext/"
        "Qwen3-4B_jacobian_lens.pt"))
    band = band_layers(model.config.num_hidden_layers, "medium")
    ablator = JSpaceAblator.from_hf(
        model, lens, AblationConfig(band=band, exclusion_filter=False),
        device=device)

    digit_ids = {}
    for d in range(1, 10):
        forms = [tok.encode(s, add_special_tokens=False)
                 for s in (str(d), " " + str(d))]
        digit_ids[d] = [f[0] for f in forms if len(f) == 1]
    all_digit_t = torch.tensor(
        [digit_ids[d][0] for d in range(1, 10)])

    def enc(s):
        return tok.encode(s, add_special_tokens=False)

    def build(prog, worked):
        (x1, o1, y1, v1), (x2, o2, y2, v2), (x3, o3, y3, v3) = prog["lines"]
        fo, d = prog["final_op"], prog["d"]
        if worked:
            segs = [f"a = {x1} {o1} {y1} = {v1}.",
                    f" b = {x2} {o2} {y2} = {v2}.",
                    f" c = {x3} {o3} {y3} = {v3}.",
                    f" d = a {fo} c = {v1} {fo} {v3} ="]
        else:
            segs = [f"a = {x1} {o1} {y1}.",
                    f" b = {x2} {o2} {y2}.",
                    f" c = {x3} {o3} {y3}.",
                    f" d = a {fo} c ="]
        ids, bounds, at = [], [], 0
        for s in segs:
            t = enc(s)
            ids += t
            bounds.append((at, at + len(t)))
            at += len(t)
        return ids, bounds

    def window_z(hidden_states, span, value):
        zs = []
        for layer in band:
            h = hidden_states[layer + 1][0, span[0]:span[1]].float()
            occ = ablator.occupancy(h, layer, all_digit_t)   # [S, 9]
            col = occ.mean(dim=0)
            z = ((col[value - 1] - col.mean())
                 / col.std().clamp_min(1e-6)).item()
            zs.append(z)
        return sum(zs) / len(zs)

    rng = random.Random(SEED)
    progs = gen_programs(N_PROGRAMS, rng)
    detail = []
    import torch as _t
    with _t.no_grad():
        for prog in progs:
            row = {"prog": prog}
            for arm, worked in (("cot", True), ("direct", False)):
                ids, bounds = build(prog, worked)
                x = _t.tensor([ids], device=device)
                out = model(x, output_hidden_states=True)
                v1 = prog["lines"][0][3]
                row[arm] = {
                    "z_write": window_z(out.hidden_states, bounds[0], v1),
                    "z_post": window_z(
                        out.hidden_states, (bounds[1][0], bounds[2][1]), v1),
                    "z_use": window_z(out.hidden_states, bounds[3], v1),
                }
                logits = out.logits[0, -1].float()
                scores = {d: max(logits[t].item() for t in digit_ids[d])
                          for d in range(1, 10)}
                pred = max(scores, key=scores.get)
                row[arm]["pred"] = pred
                row[arm]["correct"] = pred == prog["d"]
            detail.append(row)

    def agg(key, arm):
        v = [r[arm][key] for r in detail]
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1)) ** 0.5
        return {"mean": round(m, 3), "sd": round(sd, 3),
                "se": round(sd / len(v) ** 0.5, 3)}

    summary = {"n": len(detail)}
    for arm in ("cot", "direct"):
        summary[arm] = {k: agg(k, arm)
                        for k in ("z_write", "z_post", "z_use")}
        k = sum(r[arm]["correct"] for r in detail)
        summary[arm]["acc"] = round(k / len(detail), 4)
        summary[arm]["acc_k"] = k

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    host = socket.gethostname().split(".")[0]
    outdir = REPO / "results" / host / f"{ts}_cot_persistence"
    outdir.mkdir(parents=True)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    config = {
        "script": "run_cot_persistence.py", "argv": sys.argv,
        "model": "Qwen/Qwen3-4B", "band": "medium", "band_layers": band,
        "n_programs": len(detail), "seed": SEED,
        "windows": ["write (line1)", "post (lines2-3)", "use (line4)"],
        "constraint": "v1 distinct from v2, v3 and all literals in lines 2-3",
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
