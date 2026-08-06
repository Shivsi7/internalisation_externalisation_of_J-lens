#!/usr/bin/env python3
"""Rung 1.5 maintenance signature — exchange vs duplication, full contrast.

Synthetic single-digit variable programs with a controlled reuse gap:

    a = 9 - 5        <- carried value defined (computed, single digit)
    b = 2 + 1        |
    c = 8 - 6        |  gap lines: a is not used
    d = a - c        <- a reused

Two presentations of the SAME program:
  Mode W (written):  the model evaluates line by line, rewriting each line
                     with its value ("a = 9 - 5 = 4") — the page holds a.
  Mode U (unwritten): the model answers with only the final value — a is
                     never written; it must be carried internally.

Measurement: workspace occupancy of a's *value digit* during the gap-line
tokens, z-scored across the ten digit classes at each (layer, position)
(same statistic as probe_numeric), meaned over the medium band x gap region.
Hidden states come from a single teacher-forced re-forward over the finished
text (identical to generation-time states for a causal model in eval mode).

Predictions (registered in hypotheses.md A1.3, adapted to this design):
  exchange    -> z_gap(U) > z_gap(W); spike at the reuse line in W
  duplication -> z_gap(U) ~ z_gap(W), both above decoy level
Interpretability gate: z_gap(U) must exceed decoy level (the model must
visibly carry the value at all in U), else the measure is too weak and the
result is 'uninformative', not 'duplication'.

Stages: --n 8 is the parse-rate smoke; scale --n only if it passes.
Writes results/<host>/<ts>_rung15/{summary.json,detail.jsonl}.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import random
import re
import socket
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OPS = {"+": lambda x, y: x + y, "-": lambda x, y: x - y}
LINE_RE = re.compile(r"^\s*([a-z])\s*=\s*(.+?)\s*=\s*(\d)\s*$")


def gen_program(rng: random.Random, gap: int):
    """A program whose first variable is reused after `gap` filler lines.
    All operands and values are distinct digits 1-9 (single-token, and
    role-unambiguous for the z-statistic)."""
    for _ in range(400):
        used: set[int] = set()

        def fresh(lo=1, hi=9):
            v = rng.randint(lo, hi)
            return v if v not in used else None

        lines = []  # (var, expr, value)
        # carried variable a = x op y
        x, y = rng.randint(1, 9), rng.randint(1, 9)
        for op in rng.sample(list(OPS), 2):
            a_val = OPS[op](x, y)
            if 1 <= a_val <= 9 and len({x, y, a_val}) == 3:
                break
        else:
            continue
        used |= {x, y, a_val}
        lines.append(("a", f"{x} {op} {y}", a_val))
        ok = True
        for i in range(gap):  # filler lines, must not involve a
            var = chr(ord("b") + i)
            got = None
            for _ in range(40):
                p, q = rng.randint(1, 9), rng.randint(1, 9)
                o = rng.choice(list(OPS))
                v = OPS[o](p, q)
                if 1 <= v <= 9 and v not in used and p != q:
                    got = (var, f"{p} {o} {q}", v)
                    break
            if got is None:
                ok = False
                break
            used.add(got[2])
            lines.append(got)
        if not ok:
            continue
        # reuse line: target = a op last_filler (or a op const if gap=0)
        other = lines[-1][0] if gap else str(rng.randint(1, 9))
        other_val = lines[-1][2] if gap else int(other)
        tvar = chr(ord("b") + gap)
        got = None
        for o in rng.sample(list(OPS), 2):
            v = OPS[o](a_val, other_val)
            if 1 <= v <= 9 and v not in used:
                got = (tvar, f"a {o} {other}", v)
                break
        if got is None:
            continue
        lines.append(got)
        return {"lines": lines, "carried_value": a_val, "answer": lines[-1][2],
                "gap": gap}
    return None


def program_text(prog) -> str:
    return "\n".join(f"{v} = {e}" for v, e, _ in prog["lines"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8, help="programs per gap value")
    ap.add_argument("--gaps", type=int, nargs="+", default=[2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    import torch
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ablation import AblationConfig, JSpaceAblator, band_layers
    from ablation.generate import generate_with_ablation

    rng = random.Random(args.seed)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / socket.gethostname() / f"{stamp}_rung15"
    out_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    lens = jlens.JacobianLens.load(str(
        REPO / "lenses/qwen3-4b/qwen3-4b/jlens/Salesforce-wikitext/"
        "Qwen3-4B_jacobian_lens.pt"))
    band = band_layers(model.config.num_hidden_layers, "medium")
    ablator = JSpaceAblator.from_hf(
        model, lens, AblationConfig(band=band, exclusion_filter=False))

    dig_ids = {}
    for d in range(10):
        ids = []
        for s in (str(d), " " + str(d)):
            enc = tokenizer.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
        dig_ids[d] = ids
    flat_ids = torch.tensor([i for ids in dig_ids.values() for i in ids],
                            dtype=torch.long, device=model.device)
    widths = [len(dig_ids[d]) for d in range(10)]

    programs = []
    for gap in args.gaps:
        for _ in range(args.n):
            p = gen_program(rng, gap)
            if p:
                programs.append(p)
    print(f"[rung15] {len(programs)} programs, gaps {args.gaps}")

    def w_prompt(prog):
        listing = program_text(prog)
        user = (
            "Evaluate this program line by line. Rewrite each line with its "
            "value appended, exactly in the format `a = 9 - 5 = 4`, one line "
            "per variable. Write nothing else: no explanation, no blank "
            "lines, no final answer sentence.\n\n" + listing
        )
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )

    def u_prompt(prog):
        listing = program_text(prog)
        tvar = prog["lines"][-1][0]
        user = (
            "Compute the final value of "
            f"`{tvar}` in this program. Answer with only the number.\n\n"
            + listing
        )
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        ) + "The answer is"

    t0 = time.time()
    w_out = generate_with_ablation(
        model, tokenizer, [w_prompt(p) for p in programs], ablator=None,
        max_new_tokens=24 * (max(args.gaps) + 3), temperature=0.6,
        seed=args.seed, batch_size=args.batch_size)
    u_out = generate_with_ablation(
        model, tokenizer, [u_prompt(p) for p in programs], ablator=None,
        max_new_tokens=8, temperature=0.6, seed=args.seed,
        batch_size=args.batch_size)

    def digit_z(full_text: str):
        """z of each digit class across the ten classes, per (layer, pos)."""
        enc = tokenizer(full_text, return_tensors="pt",
                        return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            out = model(input_ids=enc.input_ids.to(model.device),
                        output_hidden_states=True)
        per_layer = {}
        for layer in band:
            h = out.hidden_states[layer + 1][0].float()
            occ = ablator.occupancy(h, layer, flat_ids)
            cols, col = [], 0
            for d in range(10):
                w = widths[d]
                cols.append(occ[:, col:col + w].max(-1).values)
                col += w
            occ_d = torch.stack(cols, dim=-1)
            mu = occ_d.mean(-1, keepdim=True)
            sd = occ_d.std(-1, keepdim=True).clamp_min(1e-6)
            per_layer[layer] = ((occ_d - mu) / sd).cpu()
        return per_layer, offsets

    def region_tokens(offsets, lo: int, hi: int):
        return [i for i, (a, b) in enumerate(offsets) if a < hi and b > lo]

    n_parsed = n_correct_w = n_correct_u = 0
    acc_by_gap: dict[int, dict[str, list]] = {}
    z_by_gap: dict[int, dict[str, list]] = {}
    prof_acc: dict[str, dict[int, list]] = {}
    detail = []
    for prog, wo, uo in zip(programs, w_out, u_out):
        item = {"program": program_text(prog), "gap": prog["gap"],
                "carried": prog["carried_value"], "answer": prog["answer"],
                "w_text": wo.text, "u_text": uo.text}
        detail.append(item)

        # ---- mode U: carry region = filler lines inside the prompt --------
        m = re.search(r"\d", uo.text)
        u_ok = bool(m) and int(m.group()) == prog["answer"]
        n_correct_u += u_ok
        item["u_correct"] = u_ok

        # ---- mode W: parse the rewritten lines ----------------------------
        parsed = {}
        for line in wo.text.splitlines():
            mm = LINE_RE.match(line.strip())
            if mm:
                parsed[mm.group(1)] = (mm.group(2), int(mm.group(3)))
        want = {v: val for v, _, val in prog["lines"]}
        w_parse = set(parsed) == set(want)
        w_ok = w_parse and all(parsed[v][1] == want[v] for v in want)
        item["w_parsed"] = w_parse
        item["w_correct"] = w_ok
        n_parsed += w_parse
        n_correct_w += w_ok
        g = prog["gap"]
        acc_by_gap.setdefault(g, {"w": [], "u": []})
        acc_by_gap[g]["w"].append(w_ok)
        acc_by_gap[g]["u"].append(u_ok)

        carried = prog["carried_value"]
        gap_vars = [v for v, _, _ in prog["lines"][1:-1]]
        use_var = prog["lines"][-1][0]
        decoys = [d for d in range(1, 10)
                  if d != carried and d not in
                  {val for _, _, val in prog["lines"]}]

        def mean_z(zmap, toks, digit):
            vals = [float(zmap[l][t, digit]) for l in band for t in toks]
            return sum(vals) / len(vals) if vals else None

        def max_z(zmap, toks, digit):
            vals = [float(zmap[l][t, digit]) for l in band for t in toks]
            return max(vals) if vals else None

        def layer_profile(zmap, toks, digit):
            return {l: (sum(float(zmap[l][t, digit]) for t in toks)
                        / len(toks)) for l in band} if toks else {}

        rec = {}
        # ---- W side: regions live in the GENERATED text (needs w_ok) ------
        if w_ok:
            w_full = wo.prompt + wo.text
            zW, offW = digit_z(w_full)
            base = len(wo.prompt)
            spans = {}
            for line in wo.text.splitlines():
                mm = LINE_RE.match(line.strip())
                if mm:
                    lo = w_full.find(line, base)
                    spans[mm.group(1)] = (lo, lo + len(line))
            if all(v in spans for v in gap_vars + [use_var]):
                gap_toks = []
                for v in gap_vars:
                    gap_toks += region_tokens(offW, *spans[v])
                use_toks = region_tokens(offW, *spans[use_var])
                zw = mean_z(zW, gap_toks, carried)
                if zw is not None:
                    rec.update({
                        "z_gap_w": zw,
                        "z_use_w": mean_z(zW, use_toks, carried),
                        "z_decoy_w": sum(mean_z(zW, gap_toks, d)
                                         for d in decoys) / len(decoys),
                        "z_gap_w_max": max_z(zW, gap_toks, carried),
                        "z_gap_w_decoy_max": sum(max_z(zW, gap_toks, d)
                                                 for d in decoys) / len(decoys),
                        "profile_w": layer_profile(zW, gap_toks, carried),
                    })
        # ---- U side: gap lines live in the PROMPT (needs u_ok) ------------
        if u_ok:
            u_full = uo.prompt + uo.text
            zU, offU = digit_z(u_full)
            gap_toks_u = []
            for v in gap_vars:
                for line in program_text(prog).splitlines():
                    if line.startswith(v + " ="):
                        lo = u_full.find(line)
                        gap_toks_u += region_tokens(offU, lo, lo + len(line))
            zu = mean_z(zU, gap_toks_u, carried)
            if zu is not None:
                rec.update({
                    "z_gap_u": zu,
                    "z_decoy_u": sum(mean_z(zU, gap_toks_u, d)
                                     for d in decoys) / len(decoys),
                    "z_gap_u_max": max_z(zU, gap_toks_u, carried),
                    "z_gap_u_decoy_max": sum(max_z(zU, gap_toks_u, d)
                                             for d in decoys) / len(decoys),
                    "profile_u": layer_profile(zU, gap_toks_u, carried),
                })
        for k, v in rec.items():
            if k.startswith("profile_"):
                for l, lz in v.items():
                    prof_acc.setdefault(k, {}).setdefault(l, []).append(lz)
            else:
                z_by_gap.setdefault(g, {}).setdefault(k, []).append(v)
        item.update({k: v for k, v in rec.items()
                     if not k.startswith("profile_")})

    def mean(x):
        x = [v for v in x if v is not None]
        return sum(x) / len(x) if x else None

    n = len(programs)
    summary = {
        "n_programs": n, "gaps": args.gaps,
        "w_parse_rate": n_parsed / n,
        "w_correct_rate": n_correct_w / n,
        "u_correct_rate": n_correct_u / n,
        "accuracy_by_gap": {
            g: {m: {"acc": mean([float(b) for b in v]), "n": len(v)}
                for m, v in d.items()}
            for g, d in sorted(acc_by_gap.items())
        },
        "z_by_gap": {
            g: {k: {"mean": mean(v), "n": len(v)} for k, v in sorted(d.items())}
            for g, d in sorted(z_by_gap.items())
        },
        "layer_profile_mean": {
            k: {l: mean(v) for l, v in sorted(d.items())}
            for k, d in prof_acc.items()
        },
        "wall_seconds": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "detail.jsonl").open("w") as f:
        for d in detail:
            f.write(json.dumps(d) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"[rung15] -> {out_dir}")


if __name__ == "__main__":
    main()
