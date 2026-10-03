#!/usr/bin/env python3
"""Numeric-readout gate (hypotheses.md A1.4): are arithmetic intermediates
J-lens-readable on Qwen3-4B, once the readout is scored properly?

Tokenization fact that shapes this probe: Qwen3 tokenizes numbers
digit-by-digit ("42" -> "4","2"; there are no multi-digit tokens). So:

  Battery A (the gate): chains whose intermediates are SINGLE DIGITS
    (2..9), e.g. calc: ((7-3)*2)-6 with intermediates 4, 8, 2. Occupancy of
    each digit token, z-scored across the nine digit tokens at the same
    (layer, position); statistic = best z over band x positions. AUC of
    digit-in-target-role vs digit-in-no-role (not operand, not target),
    pooled over prompts. Operand digits reported separately as the positive
    control (they are literally in context).

  Battery B (diagnostic): the paper-style 2-digit form (a+b)*c+d, scored by
    the LEADING digit token of each intermediate. Multi-token readout is
    where the paper itself reports sharp degradation, so B failing while A
    passes localises the problem to tokenization, not to the workspace.

Gate: AUC_A >= 0.70 with bootstrap CI excluding 0.5 (occupancy-z or
digit-rank statistic). Ordering check: Spearman(computation index, first
layer at which z >= 2).

Writes results/<host>/<ts>_numeric_probe/{summary.json,detail.jsonl}.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import pathlib
import random
import socket
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

LENS_PATH = (REPO / "lenses/qwen3-4b/qwen3-4b/jlens/Salesforce-wikitext/"
             "Qwen3-4B_jacobian_lens.pt")
N_PROMPTS_A = 40
N_PROMPTS_B = 20
import os as _os
SEED = int(_os.environ.get("HW0_SEED", "0"))
OPS = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y}


def make_battery_a(rng: random.Random):
    """3-step chains, all operands and intermediates distinct digits 1..9."""
    battery, seen = [], set()
    while len(battery) < N_PROMPTS_A:
        a, b, c, d = (rng.randint(1, 9) for _ in range(4))
        o1, o2, o3 = (rng.choice(list(OPS)) for _ in range(3))
        m1 = OPS[o1](a, b)
        if not 2 <= m1 <= 9:
            continue
        m2 = OPS[o2](m1, c)
        if not 2 <= m2 <= 9:
            continue
        ans = OPS[o3](m2, d)
        if not 2 <= ans <= 9:
            continue
        vals = (a, b, c, d, m1, m2, ans)
        if len(set(vals)) != 7:
            continue
        prompt = f"calc: (({a}{o1}{b}){o2}{c}){o3}{d} = "
        if prompt in seen:
            continue
        seen.add(prompt)
        battery.append({
            "prompt": prompt,
            "operands": [a, b, c, d],
            "targets": [m1, m2, ans],
        })
    return battery


def make_battery_b(rng: random.Random):
    """Paper-style (a+b)*c+d with 2-digit intermediates."""
    battery = []
    while len(battery) < N_PROMPTS_B:
        a, b = rng.randint(3, 19), rng.randint(3, 19)
        c, d = rng.randint(2, 4), rng.randint(3, 9)
        m1, m2 = a + b, (a + b) * c
        ans = m2 + d
        if not all(10 <= v <= 99 for v in (m1, m2, ans)):
            continue
        if len({m1, m2, ans}) != 3:
            continue
        battery.append({
            "prompt": f"calc: ({a}+{b})*{c}+{d} = ",
            "operands": [a, b, c, d],
            "targets": [m1, m2, ans],
        })
    return battery


def digit_token_ids(tokenizer) -> dict[int, list[int]]:
    """digit -> its single-token ids (bare and leading-space variants)."""
    out = {}
    for digit in range(10):
        ids = []
        for s in (str(digit), " " + str(digit)):
            enc = tokenizer.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
        out[digit] = ids
    return out


def auc_with_ci(pos: list[float], neg: list[float], rng, n_boot=2000):
    def auc(p, n):
        wins = sum((x > y) + 0.5 * (x == y) for x in p for y in n)
        return wins / (len(p) * len(n))
    point = auc(pos, neg)
    boots = []
    for _ in range(n_boot):
        bp = [pos[rng.randrange(len(pos))] for _ in pos]
        bn = [neg[rng.randrange(len(neg))] for _ in neg]
        boots.append(auc(bp, bn))
    boots.sort()
    return {"point": point, "ci95": [boots[int(0.025 * n_boot)],
                                     boots[int(0.975 * n_boot)]]}


def main() -> None:
    import torch
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ablation import AblationConfig, JSpaceAblator, band_layers

    rng = random.Random(SEED)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / socket.gethostname() / f"{stamp}_numeric_probe"
    out_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    lens = jlens.JacobianLens.load(str(LENS_PATH))
    band = band_layers(model.config.num_hidden_layers, "heavy")
    ablator = JSpaceAblator.from_hf(
        model, lens, AblationConfig(band=band, exclusion_filter=False)
    )
    dig_ids = digit_token_ids(tokenizer)
    flat_ids = [i for ids in dig_ids.values() for i in ids]
    flat_ids_t = torch.tensor(flat_ids, dtype=torch.long, device=model.device)
    widths = [len(dig_ids[d]) for d in range(10)]
    print(f"band L{band[0]}..L{band[-1]}; digit token variants: {widths}")

    def digit_occ_z(prompt: str):
        """Per (layer, position): z of each digit's occupancy across the ten
        digit classes. Returns {layer: z [S, 10]} plus per-digit best-z and
        the first layer at which z >= 2."""
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        per_layer = {}
        for layer in band:
            h = out.hidden_states[layer + 1][0].float()
            occ = ablator.occupancy(h, layer, flat_ids_t)  # [S, n_flat]
            cols, col = [], 0
            for d in range(10):
                w = len(dig_ids[d])
                cols.append(occ[:, col:col + w].max(-1).values)
                col += w
            occ_d = torch.stack(cols, dim=-1)  # [S, 10]
            mu = occ_d.mean(-1, keepdim=True)
            sd = occ_d.std(-1, keepdim=True).clamp_min(1e-6)
            per_layer[layer] = (occ_d - mu) / sd
        return per_layer

    # ---------------- Battery A -----------------------------------------
    battery_a = make_battery_a(rng)
    a_stats = {"target": [], "operand": [], "none": []}
    a_final = {"target": [], "operand": [], "none": []}  # final position only
    ordering = []
    detail = []
    for item in battery_a:
        z_by_layer = digit_occ_z(item["prompt"])
        best = [max(float(z[:, d].max()) for z in z_by_layer.values())
                for d in range(10)]
        # The paper reads the chain at the answer-adjacent position ("=");
        # maxing over all positions inflates the null, so score the final
        # position separately (max over band layers only).
        best_final = [max(float(z[-1, d]) for z in z_by_layer.values())
                      for d in range(10)]
        first_layer = [None] * 10       # final position only
        first_layer_all = [None] * 10   # any position
        for layer in band:  # ascending
            z_fin = z_by_layer[layer][-1]
            z_any = z_by_layer[layer].max(0).values
            for d in range(10):
                if first_layer[d] is None and float(z_fin[d]) >= 2.0:
                    first_layer[d] = layer
                if first_layer_all[d] is None and float(z_any[d]) >= 2.0:
                    first_layer_all[d] = layer
        tset, oset = set(item["targets"]), set(item["operands"])
        for d in range(1, 10):
            role = ("target" if d in tset
                    else "operand" if d in oset else "none")
            a_stats[role].append(best[d])
            a_final[role].append(best_final[d])
        for idx, t in enumerate(item["targets"]):
            if first_layer[t] is not None:
                ordering.append((idx, first_layer[t]))
        detail.append({**item, "battery": "A", "best_z_by_digit": best,
                       "best_z_final_pos": best_final,
                       "first_layer_z2_final_pos": first_layer,
                       "first_layer_z2_allpos": first_layer_all})

    # ---------------- Battery B (leading-digit diagnostic) ---------------
    battery_b = make_battery_b(rng)
    b_stats = {"target_lead": [], "none": []}
    for item in battery_b:
        z_by_layer = digit_occ_z(item["prompt"])
        best = [max(float(z[:, d].max()) for z in z_by_layer.values())
                for d in range(10)]
        lead = {int(str(v)[0]) for v in item["targets"]}
        used = (lead
                | {int(ch) for v in item["targets"] for ch in str(v)}
                | {int(ch) for v in item["operands"] for ch in str(v)})
        for d in range(10):
            if d in lead:
                b_stats["target_lead"].append(best[d])
            elif d not in used:
                b_stats["none"].append(best[d])
        detail.append({**item, "battery": "B", "best_z_by_digit": best})

    boot = random.Random(SEED + 1)
    auc_a = auc_with_ci(a_stats["target"], a_stats["none"], boot)
    auc_a_operand = auc_with_ci(a_stats["operand"], a_stats["none"], boot)
    auc_a_fin = auc_with_ci(a_final["target"], a_final["none"], boot)
    auc_a_fin_op = auc_with_ci(a_final["operand"], a_final["none"], boot)
    auc_b = (auc_with_ci(b_stats["target_lead"], b_stats["none"], boot)
             if b_stats["target_lead"] and b_stats["none"] else None)

    spearman = None
    if len(ordering) >= 8:
        from scipy.stats import spearmanr
        rho, p = spearmanr([a for a, _ in ordering], [b for _, b in ordering])
        spearman = {"rho": float(rho), "p": float(p), "n": len(ordering)}

    def med(x):
        return sorted(x)[len(x) // 2] if x else None

    summary = {
        "batteryA": {
            "n_prompts": len(battery_a),
            "auc_target_vs_none": auc_a,
            "auc_operand_vs_none (positive control)": auc_a_operand,
            "final_pos_auc_target_vs_none": auc_a_fin,
            "final_pos_auc_operand_vs_none (positive control)": auc_a_fin_op,
            "median_z": {k: med(v) for k, v in a_stats.items()},
            "median_z_final_pos": {k: med(v) for k, v in a_final.items()},
            "ordering_spearman (final pos)": spearman,
        },
        "batteryB_leading_digit": {
            "n_prompts": len(battery_b),
            "auc_targetlead_vs_none": auc_b,
            "median_z": {k: med(v) for k, v in b_stats.items()},
        },
        "gate_pass": bool(any(
            a["point"] >= 0.70 and a["ci95"][0] > 0.5 for a in (auc_a, auc_a_fin)
        )),
        "band": [band[0], band[-1]],
        "seed": SEED,
        "note": "Qwen3 tokenizes numbers digit-by-digit; battery A uses "
                "single-digit intermediates (clean single-token readout), "
                "battery B scores only the leading digit of 2-digit values.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "detail.jsonl").open("w") as f:
        for d in detail:
            f.write(json.dumps(d) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"[numeric probe] -> {out_dir}")


if __name__ == "__main__":
    main()
