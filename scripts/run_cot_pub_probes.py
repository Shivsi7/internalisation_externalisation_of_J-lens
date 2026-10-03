#!/usr/bin/env python3
"""Publication-rigour probes on the model's own CoT (hypotheses.md A6.1-A6.3).

One generation pass over three-line programs (own chain of thought, greedy),
then three measurements per transcript:

  A (page_masking)  can the model still produce its final answer when the
                    final line's positions cannot attend any written
                    occurrence of the intermediate values? Conditions:
                    unmasked re-forward (sanity), value-masked, and a
                    random-position mask of equal size. Scored on items the
                    model answered correctly; metric = final-answer token
                    reproduced at the answer position.
  B (decay)         occupancy z of v1 at token offsets 1..24 after its
                    write (cot arm) and after line 1 ends (unworked direct
                    arm), positions where v1's digit literally appears
                    excluded. Curves written vs unwritten at matched delay.
  C (inventory)     top-5 readout tokens per storage-window position,
                    mid-band, categorized: digit / operator / variable name
                    / token of the line currently being written / other.

Writes results/<host>/<ts>_cot_pub_probes/{config.json,detail.jsonl,
summary.json}.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import random
import socket
import subprocess
import sys
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parents[1]
import os as _os
HW0_MODEL = _os.environ.get("HW0_MODEL", "Qwen/Qwen3-4B")
HW0_LENS = _os.environ.get(
    "HW0_LENS", str(REPO / "lenses/qwen3-4b/qwen3-4b/jlens/"
                    "Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"))
sys.path.insert(0, str(REPO / "src"))

sys.path.insert(0, str(REPO / "scripts"))
from run_cot_correspondence import gen_programs  # same generator + constraints


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--decay-max", type=int, default=24)
    args = ap.parse_args()

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

    lens = jlens.JacobianLens.load(HW0_LENS)
    band = band_layers(model.config.num_hidden_layers, "medium")
    ablator = JSpaceAblator.from_hf(
        model, lens, AblationConfig(band=band, exclusion_filter=False),
        device=device)
    mid_layer = band[len(band) // 2]

    digit_tok = {}
    for d in range(1, 10):
        forms = [tok.encode(s, add_special_tokens=False)
                 for s in (str(d), " " + str(d))]
        digit_tok[d] = [f[0] for f in forms if len(f) == 1]
    occ_ids = torch.tensor([digit_tok[d][0] for d in range(1, 10)])
    all_digit_ids = {t for v in digit_tok.values() for t in v}

    OP_STRINGS = {"+", "-", "=", " +", " -", " =", "*", " *"}
    VAR_STRINGS = {"a", "b", "c", "d", " a", " b", " c", " d"}
    op_ids = {tok.encode(s, add_special_tokens=False)[0]
              for s in OP_STRINGS
              if len(tok.encode(s, add_special_tokens=False)) == 1}
    var_ids = {tok.encode(s, add_special_tokens=False)[0]
               for s in VAR_STRINGS
               if len(tok.encode(s, add_special_tokens=False)) == 1}

    neg = torch.finfo(mdtype).min

    def zscore_at(hs, pos, value):
        zs = []
        for layer in band:
            h = hs[layer + 1][0, pos:pos + 1].float()
            occ = ablator.occupancy(h, layer, occ_ids)
            col = occ[0]
            z = ((col[value - 1] - col.mean())
                 / col.std().clamp_min(1e-6)).item()
            zs.append(z)
        return sum(zs) / len(zs)

    rng = random.Random(0)
    progs = gen_programs(args.n, rng)
    detail = []
    n_dropped = 0
    decay_cot = {d: [] for d in range(1, args.decay_max + 1)}
    decay_dir = {d: [] for d in range(1, args.decay_max + 1)}
    inv_counts = Counter()
    inv_total = 0
    pm = {"unmasked": [], "value_masked": [], "random_masked": []}

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
            out = model.generate(ids, max_new_tokens=args.max_new,
                                 do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            gen = out[0, plen:].tolist()

            def first_pos(val, after=0):
                cands = [i for i, t in enumerate(gen)
                         if t in digit_tok[val] and i >= after]
                return cands[0] if cands else None

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
            if None in (i1, i2, i3) or not (i1 < i2 < i3) or i_ans is None \
                    or i_ans <= i3:
                n_dropped += 1
                continue

            hs = model(out, output_hidden_states=True).hidden_states

            # ---- B: decay curves (cot arm) --------------------------------
            v1_ids = set(digit_tok[v1])
            for delta in range(1, args.decay_max + 1):
                pos = i1 + delta
                if pos >= i3 or gen[pos] in v1_ids:
                    continue
                decay_cot[delta].append(zscore_at(hs, plen + pos, v1))

            # ---- C: inventory over storage window -------------------------
            for pos in range(i1 + 3, i3):
                h = hs[mid_layer + 1][0, plen + pos].float()
                scores = ablator._lens_scores(
                    h.unsqueeze(0).to(device), mid_layer, use_jacobian=True)
                top = scores[0].topk(5).indices.tolist()
                for t in top:
                    inv_total += 1
                    if t in all_digit_ids:
                        inv_counts["digit"] += 1
                    elif t in op_ids:
                        inv_counts["operator"] += 1
                    elif t in var_ids:
                        inv_counts["variable"] += 1
                    elif t in set(gen[max(0, pos - 8):pos + 1]):
                        inv_counts["current_line"] += 1
                    else:
                        inv_counts["other"] += 1

            # ---- A: page masking (correct items only) ---------------------
            if correct:
                S = out.shape[1]
                val_ids = set().union(*[digit_tok[v] for v in (v1, v2, v3)])
                vpos = [plen + i for i, t in enumerate(gen[:i_ans])
                        if t in val_ids]
                vpos = sorted({p for p in vpos} | {p + 1 for p in vpos})
                q0 = plen + i3 + 2          # final line onward
                arow = plen + i_ans - 1     # row that predicts the answer

                def masked_logits(blocked):
                    m = torch.full((S, S), neg, dtype=mdtype)
                    m = torch.triu(m, diagonal=1)
                    for p in blocked:
                        if p < S:
                            m[q0:, p] = neg
                    m = m.view(1, 1, S, S).to(device)
                    o = model(out, attention_mask=m)
                    return o.logits[0, arow].float()

                rng2 = random.Random(i1 * 7919 + i2)
                nonval = [plen + i for i in range(i3)
                          if plen + i not in set(vpos)]
                rand_pos = rng2.sample(nonval, min(len(vpos), len(nonval)))

                for cond, blocked in (("unmasked", []),
                                      ("value_masked", vpos),
                                      ("random_masked", rand_pos)):
                    lg = masked_logits(blocked)
                    pred = int(lg.argmax().item())
                    pm[cond].append(pred == gen[i_ans])

            detail.append({"prog": prog, "correct": correct,
                           "i1": i1, "i3": i3, "i_ans": i_ans})

        # ---- B: direct-arm decay (unworked program) -----------------------
        for prog in progs[:len(detail)]:
            (x1, o1, y1, v1), (x2, o2, y2, v2), (x3, o3, y3, v3) = prog["lines"]
            fo = prog["final_op"]
            text = (f"a = {x1} {o1} {y1}. b = {x2} {o2} {y2}. "
                    f"c = {x3} {o3} {y3}. d = a {fo} c =")
            tids = tok.encode(text, add_special_tokens=False)
            line1 = tok.encode(f"a = {x1} {o1} {y1}.",
                               add_special_tokens=False)
            origin = len(line1) - 1
            x = torch.tensor([tids], device=device)
            hs = model(x, output_hidden_states=True).hidden_states
            for delta in range(1, args.decay_max + 1):
                pos = origin + delta
                if pos >= len(tids):
                    break
                decay_dir[delta].append(zscore_at(hs, pos, v1))

    def curve(dd):
        return {str(k): {"mean": round(sum(v) / len(v), 3), "n": len(v)}
                for k, v in dd.items() if v}

    def rate(xs):
        return {"acc": round(sum(xs) / len(xs), 4), "n": len(xs)} if xs else None

    summary = {
        "n_scored": len(detail), "n_dropped": n_dropped,
        "page_masking": {k: rate(v) for k, v in pm.items()},
        "decay_cot": curve(decay_cot),
        "decay_direct": curve(decay_dir),
        "inventory_top5_shares": {
            k: round(v / inv_total, 4) for k, v in inv_counts.most_common()},
        "inventory_positions_total": inv_total,
    }

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    host = socket.gethostname().split(".")[0]
    outdir = REPO / "results" / host / f"{ts}_cot_pub_probes"
    outdir.mkdir(parents=True)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    (outdir / "config.json").write_text(json.dumps({
        "script": "run_cot_pub_probes.py", "argv": sys.argv,
        "model": HW0_MODEL, "band": "medium", "band_layers": band,
        "mid_layer": mid_layer, "n_programs": args.n,
        "max_new": args.max_new, "decay_max": args.decay_max,
        "git_commit": commit, "host": host,
        "gpu": torch.cuda.get_device_name(0)}, indent=2))
    with open(outdir / "detail.jsonl", "w") as f:
        for r in detail:
            f.write(json.dumps(r) + "\n")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:3000])
    print("wrote", outdir)


if __name__ == "__main__":
    main()
