#!/usr/bin/env python3
"""Does the J-lens workspace show the computed GSM8K answer before it is
spoken? (Paper §3.3 signature, on real GSM8K, tokenization-compatible.)

Uses the existing direct-mode clean rollouts (no new generation): for each,
one teacher-forced forward pass over prompt + "The answer is <digits>",
reading digit occupancy at positions BEFORE the first answer digit.

Statistics, all z-scored across the ten digit classes per (layer, position):
  emitted  - leading digit of the answer the model actually gave
  gold     - leading digit of the correct answer (differs when wrong)
  decoys   - digits absent from question, gold, and emitted answer

Position classes:
  imminent - the token right before the first digit (next token IS the
             digit; late layers are trivially "motor" here, mid-band is not)
  early    - 2..6 tokens before that (digit is not the next token anywhere)

Expected if the workspace is real on GSM8K: emitted > decoys in the
mid-band, already at 'early' positions; on incorrect rollouts the workspace
tracks the emitted (wrong) answer, not the gold one.

Writes results/<host>/<ts>_presurface/summary.json.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import pathlib
import socket
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def newest_direct_clean() -> pathlib.Path:
    cands = []
    for d in glob.glob(str(REPO / "results/*/*gsm8k_direct_none*")):
        d = pathlib.Path(d)
        try:
            cfg = json.loads((d / "config.json").read_text())
        except Exception:
            continue
        if cfg.get("n", 0) >= 100 and cfg.get("model", "Qwen/Qwen3-4B") == "Qwen/Qwen3-4B":
            cands.append(d)
    if not cands:
        raise SystemExit("no direct clean run found")
    return sorted(cands)[-1]


def main() -> None:
    import argparse

    import torch
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ablation import AblationConfig, JSpaceAblator, band_layers
    from eval.gsm8k import PROMPTS_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default=None,
                    help="specific results/<host>/<run> dir; default = newest "
                         "4B gsm8k direct clean run")
    args = ap.parse_args()

    src_dir = (pathlib.Path(args.source_dir) if args.source_dir
               else newest_direct_clean())
    print(f"[presurface] rollouts from {src_dir.name}")
    rollouts = [json.loads(l) for l in
                (src_dir / "generations.jsonl").read_text().splitlines()]

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
    n_layers = model.config.num_hidden_layers
    band = band_layers(n_layers, "heavy")  # L15..32: probe widely, slice later
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

    user_tpl = (PROMPTS_DIR / "gsm8k_direct_user.txt").read_text()
    prefill = (PROMPTS_DIR / "gsm8k_direct_prefill.txt").read_text()

    mid_band = [l for l in band if l <= 28]
    late = [l for l in band if l > 28]

    stats: dict[str, dict[str, list]] = {}

    def add(bucket: str, key: str, val: float) -> None:
        stats.setdefault(bucket, {}).setdefault(key, []).append(val)

    t0 = time.time()
    n_used = 0
    for r in rollouts:
        # reconstruct the exact sequence: templated prompt + generation
        prompt = tokenizer.apply_chat_template(
            [{"role": "user",
              "content": user_tpl.format(question=r["question"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ) + prefill
        full = prompt + r["text"]
        m = None
        for i, ch in enumerate(r["text"]):
            if ch.isdigit():
                m = i
                break
        if m is None or r["pred"] is None:
            continue
        emitted_digit = int(r["text"][m])
        gold_digit = int(str(r["gold"]).lstrip("-$")[0])
        qdigits = {int(c) for c in r["question"] if c.isdigit()}
        used_digits = (qdigits | {emitted_digit, gold_digit}
                       | {int(c) for c in str(r["pred"]) if c.isdigit()}
                       | {int(c) for c in str(r["gold"]) if c.isdigit()})
        decoys = [d for d in range(10) if d not in used_digits]
        if not decoys:
            continue

        enc = tokenizer(full, return_tensors="pt",
                        return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc.pop("offset_mapping")[0].tolist()
        digit_char = len(prompt) + m
        first_digit_tok = next(
            (i for i, (a, b) in enumerate(offsets) if a <= digit_char < b),
            None)
        if first_digit_tok is None or first_digit_tok < 7:
            continue
        pos_imminent = [first_digit_tok - 1]
        pos_early = list(range(first_digit_tok - 6, first_digit_tok - 1))

        with torch.no_grad():
            out = model(input_ids=enc.input_ids.to(model.device),
                        output_hidden_states=True)

        for layer_set, lname in ((mid_band, "mid"), (late, "late")):
            for pos_set, pname in ((pos_imminent, "imminent"),
                                   (pos_early, "early")):
                zs = {}
                for layer in layer_set:
                    h = out.hidden_states[layer + 1][0][pos_set].float()
                    occ = ablator.occupancy(h, layer, flat_ids)
                    cols, col = [], 0
                    for d in range(10):
                        w = widths[d]
                        cols.append(occ[:, col:col + w].max(-1).values)
                        col += w
                    occ_d = torch.stack(cols, dim=-1)
                    mu = occ_d.mean(-1, keepdim=True)
                    sd = occ_d.std(-1, keepdim=True).clamp_min(1e-6)
                    z = (occ_d - mu) / sd
                    for d in range(10):
                        zs.setdefault(d, []).append(float(z[:, d].mean()))
                zmean = {d: sum(v) / len(v) for d, v in zs.items()}
                bucket = f"{lname}_{pname}_" + (
                    "correct" if r["correct"] else "incorrect")
                add(bucket, "emitted", zmean[emitted_digit])
                add(bucket, "gold", zmean[gold_digit])
                add(bucket, "decoy",
                    sum(zmean[d] for d in decoys) / len(decoys))
        n_used += 1

    def mean(x):
        return sum(x) / len(x) if x else None

    summary = {
        "source_run": src_dir.name,
        "n_rollouts_used": n_used,
        "band": [band[0], band[-1]],
        "mid_band_max_layer": 28,
        "buckets": {
            b: {k: {"mean": mean(v), "n": len(v)} for k, v in kv.items()}
            for b, kv in sorted(stats.items())
        },
        "wall_seconds": time.time() - t0,
    }
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / socket.gethostname() / f"{stamp}_presurface"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[presurface] -> {out_dir}")


if __name__ == "__main__":
    main()
