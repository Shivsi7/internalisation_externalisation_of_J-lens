#!/usr/bin/env python3
"""Context-knockout curve (hypotheses.md A7.1).

Widening attention-mask ladder on the model's own correct transcripts,
applied from the final line onward: L0 nothing, L1 literal value tokens +1
(replicates the page-masking test), L2 values +-3, L3 values +-6, L4 every
generated position before the final line (the prompt stays visible). Random
masks of matched size control L1-L3. Falling accuracy across the ladder
locates where the value information lives.

Writes results/<host>/<ts>_knockout_curve/{config.json,detail.jsonl,
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
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from run_cot_correspondence import gen_programs

N_PROGRAMS = 120
MAX_NEW = 512


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

    digit_tok = {}
    for d in range(1, 10):
        forms = [tok.encode(s, add_special_tokens=False)
                 for s in (str(d), " " + str(d))]
        digit_tok[d] = [f[0] for f in forms if len(f) == 1]
    all_digit_ids = {t for v in digit_tok.values() for t in v}

    rng = random.Random(0)
    progs = gen_programs(N_PROGRAMS, rng)
    levels = ["L0", "L1", "L2", "L3", "L4", "R1", "R2", "R3"]
    acc = {l: [] for l in levels}
    detail = []

    with torch.no_grad():
        for prog in progs:
            (x1, o1, y1, v1), (x2, o2, y2, v2), (x3, o3, y3, v3) = prog["lines"]
            fo, dval = prog["final_op"], prog["d"]
            question = (f"a = {x1} {o1} {y1}. b = {x2} {o2} {y2}. "
                        f"c = {x3} {o3} {y3}. d = a {fo} c. "
                        "Work through the program line by line, then state "
                        "the value of d.")
            enc = tok.apply_chat_template(
                [{"role": "user", "content": question}],
                add_generation_prompt=True, return_tensors="pt")
            ids = (enc["input_ids"] if not hasattr(enc, "shape")
                   else enc).to(device)
            plen = ids.shape[1]
            out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            gen = out[0, plen:].tolist()

            def first_pos(val, after=0):
                c = [i for i, t in enumerate(gen)
                     if t in digit_tok[val] and i >= after]
                return c[0] if c else None

            i1 = first_pos(v1)
            i2 = first_pos(v2, (i1 or 0) + 1) if i1 is not None else None
            i3 = first_pos(v3, (i2 or 0) + 1) if i2 is not None else None
            i_ans = None
            for i in range(len(gen) - 1, -1, -1):
                if gen[i] in all_digit_ids:
                    i_ans = i
                    break
            correct = (i_ans is not None
                       and gen[i_ans] in digit_tok.get(dval, []))
            if None in (i1, i2, i3) or not (i1 < i2 < i3) or not correct \
                    or i_ans <= i3:
                continue

            S = out.shape[1]
            val_ids = set().union(*[digit_tok[v] for v in (v1, v2, v3)])
            base_v = [i for i, t in enumerate(gen[:i_ans]) if t in val_ids]
            q0 = plen + i3 + 2
            arow = plen + i_ans - 1

            def widen(w):
                s = set()
                for p0 in base_v:
                    for dd in range(-w, w + 1):
                        q = p0 + dd
                        if 0 <= q < i_ans:
                            s.add(plen + q)
                return sorted(s)

            L = {"L0": [], "L1": widen(1), "L2": widen(3), "L3": widen(6),
                 "L4": [plen + i for i in range(i_ans)]}
            rng2 = random.Random(i1 * 31337 + i2)
            for lvl, src_lvl in (("R1", "L1"), ("R2", "L2"), ("R3", "L3")):
                pool = [plen + i for i in range(i3)
                        if plen + i not in set(L[src_lvl])]
                L[lvl] = rng2.sample(pool, min(len(L[src_lvl]), len(pool)))

            for lvl in levels:
                m = torch.full((S, S), neg, dtype=mdtype)
                m = torch.triu(m, diagonal=1)
                for p0 in L[lvl]:
                    m[q0:, p0] = neg
                m = m.view(1, 1, S, S).to(device)
                lg = model(out, attention_mask=m).logits[0, arow].float()
                ok = int(lg.argmax().item()) == gen[i_ans]
                acc[lvl].append(ok)
            detail.append({"prog": prog, "n_masked": {k: len(v)
                                                     for k, v in L.items()}})

    def wilson(k, nn, z=1.96):
        p = k / nn
        d0 = 1 + z * z / nn
        c = (p + z * z / (2 * nn)) / d0
        h = z * ((p * (1 - p) / nn + z * z / (4 * nn * nn)) ** 0.5) / d0
        return (round(c - h, 4), round(c + h, 4))

    summary = {"n": len(detail)}
    for lvl in levels:
        k = sum(acc[lvl])
        n = len(acc[lvl])
        summary[lvl] = {"acc": round(k / n, 4), "k": k, "n": n,
                        "wilson95": wilson(k, n)}
        print(f"[{lvl}] {k}/{n} = {k/n:.3f}", flush=True)

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    host = socket.gethostname().split(".")[0]
    outdir = REPO / "results" / host / f"{ts}_knockout_curve"
    outdir.mkdir(parents=True)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    (outdir / "config.json").write_text(json.dumps({
        "script": "run_knockout_curve.py", "argv": sys.argv,
        "model": HW0_MODEL, "n_programs": N_PROGRAMS, "max_new": MAX_NEW,
        "levels": {"L1": "values+-1", "L2": "values+-3", "L3": "values+-6",
                   "L4": "all generated before final line",
                   "R1-3": "random matched-size controls"},
        "git_commit": commit, "host": host,
        "gpu": torch.cuda.get_device_name(0)}, indent=2))
    with open(outdir / "detail.jsonl", "w") as f:
        for r in detail:
            f.write(json.dumps(r) + "\n")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", outdir)


if __name__ == "__main__":
    main()
