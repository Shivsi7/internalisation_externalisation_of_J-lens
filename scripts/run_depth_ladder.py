#!/usr/bin/env python3
"""Depth ladder D-measurement (A2.4): the serial depth budget of Qwen3-4B.

Fully sequential single-digit chains — each line consumes the previous
line's value:

    a = 7 - 3
    b = a + 5
    c = b - 6
    ...

Direct mode ("What is <last>? Answer with only the number."). For each
chain length L: behavioural accuracy, and whether the EMITTED answer digit
surfaces in the J-lens readout at the answer-adjacent position (z >= 2
across the ten digit classes, max over band layers) — the paper-§3.3-style
presence test. D := largest L where the surfacing fraction >= 0.5.

Registered prediction (hypotheses.md §3): D in [3, 7].

Writes results/<host>/<ts>_depth_ladder/summary.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import random
import socket
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OPS = {"+": lambda x, y: x + y, "-": lambda x, y: x - y}
VARS = "abcdefghij"


def gen_chain(rng: random.Random, L: int):
    for _ in range(300):
        lines = []
        x, y = rng.randint(1, 9), rng.randint(1, 9)
        op = rng.choice(list(OPS))
        v = OPS[op](x, y)
        if not 1 <= v <= 9:
            continue
        lines.append((VARS[0], f"{x} {op} {y}", v))
        ok = True
        for i in range(1, L):
            got = None
            for _ in range(40):
                c = rng.randint(1, 9)
                op = rng.choice(list(OPS))
                nv = OPS[op](v, c)
                if 1 <= nv <= 9 and nv != v:
                    got = (VARS[i], f"{VARS[i-1]} {op} {c}", nv)
                    break
            if got is None:
                ok = False
                break
            lines.append(got)
            v = got[2]
        if ok:
            return {"lines": lines, "answer": v, "L": L}
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30, help="chains per length")
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[2, 3, 4, 5, 6, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    import torch
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ablation import AblationConfig, JSpaceAblator, band_layers
    from ablation.generate import generate_with_ablation

    rng = random.Random(args.seed)
    chains = []
    for L in args.lengths:
        for _ in range(args.n):
            c = gen_chain(rng, L)
            if c:
                chains.append(c)
    print(f"[depth] {len(chains)} chains, lengths {args.lengths}")

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
    band = band_layers(model.config.num_hidden_layers, "heavy")
    ablator = JSpaceAblator.from_hf(model, lens, AblationConfig(
        band=band, exclusion_filter=False))
    mid_band = [l for l in band if l <= 28]

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

    def prompt_for(chain) -> str:
        listing = "\n".join(f"{v} = {e}" for v, e, _ in chain["lines"])
        last = chain["lines"][-1][0]
        user = (f"Compute the final value of `{last}` in this program. "
                f"Answer with only the number.\n\n{listing}")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        ) + "The answer is"

    prompts = [prompt_for(c) for c in chains]
    t0 = time.time()
    res = generate_with_ablation(
        model, tokenizer, prompts, ablator=None, max_new_tokens=8,
        temperature=0.6, top_k=20, top_p=0.95, seed=args.seed,
        batch_size=args.batch_size)

    per_L: dict[int, dict[str, list]] = {}
    for chain, prompt, r in zip(chains, prompts, res):
        L = chain["L"]
        st = per_L.setdefault(L, {"acc": [], "surf_mid": [], "surf_any": [],
                                  "z_mid": []})
        m = None
        for i, ch in enumerate(r.text):
            if ch.isdigit():
                m = i
                break
        if m is None:
            st["acc"].append(False)
            continue
        emitted = int(r.text[m])
        st["acc"].append(emitted == chain["answer"])

        full = prompt + r.text
        enc = tokenizer(full, return_tensors="pt",
                        return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc.pop("offset_mapping")[0].tolist()
        digit_char = len(prompt) + m
        tok = next((i for i, (a, b) in enumerate(offsets)
                    if a <= digit_char < b), None)
        if tok is None or tok < 2:
            continue
        pos = [tok - 1]  # answer-adjacent position
        with torch.no_grad():
            out = model(input_ids=enc.input_ids.to(model.device),
                        output_hidden_states=True)
        z_mid, z_all = [], []
        for layer in band:
            h = out.hidden_states[layer + 1][0][pos].float()
            occ = ablator.occupancy(h, layer, flat_ids)
            cols, col = [], 0
            for d in range(10):
                w = widths[d]
                cols.append(occ[:, col:col + w].max(-1).values)
                col += w
            occ_d = torch.stack(cols, dim=-1)
            mu = occ_d.mean(-1, keepdim=True)
            sd = occ_d.std(-1, keepdim=True).clamp_min(1e-6)
            zval = float(((occ_d - mu) / sd)[0, emitted])
            z_all.append(zval)
            if layer in mid_band:
                z_mid.append(zval)
        st["z_mid"].append(max(z_mid))
        st["surf_mid"].append(max(z_mid) >= 2.0)
        st["surf_any"].append(max(z_all) >= 2.0)

    def frac(x):
        return sum(x) / len(x) if x else None

    table = {}
    for L in sorted(per_L):
        st = per_L[L]
        table[L] = {
            "n": len(st["acc"]),
            "direct_acc": frac(st["acc"]),
            "surfacing_frac_midband": frac(st["surf_mid"]),
            "surfacing_frac_anyband": frac(st["surf_any"]),
            "mean_max_z_midband": frac(st["z_mid"]),
        }
    D_mid = max((L for L, v in table.items()
                 if v["surfacing_frac_midband"] is not None
                 and v["surfacing_frac_midband"] >= 0.5), default=None)
    D_any = max((L for L, v in table.items()
                 if v["surfacing_frac_anyband"] is not None
                 and v["surfacing_frac_anyband"] >= 0.5), default=None)

    summary = {
        "per_length": table,
        "D_midband": D_mid,
        "D_anyband": D_any,
        "registered_prediction": "D in [3, 7] (hypotheses.md §3)",
        "wall_seconds": time.time() - t0,
    }
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / socket.gethostname() / f"{stamp}_depth_ladder"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[depth] -> {out_dir}")


if __name__ == "__main__":
    main()
