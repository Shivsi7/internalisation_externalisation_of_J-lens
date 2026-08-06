#!/usr/bin/env python3
"""Prefix rescue + filler control (rung 4 lite + rung 2's null), GSM8K.

Question: when the workspace is ablated, does a CoT *prefix* rescue the
answer because of its CONTENT, or would any token-matched page do (the
epiphenomenal reading: CoT = more positions/compute, not storage)?

Conditions (crossed with {clean, medium ablation}); the model only generates
the short answer ("The answer is ..."), the prefix is prefilled:
  own_cot       - the clean model's own think block for this problem
                  (only problems the clean CoT solved correctly)
  filler        - content-free filler, token-matched to own_cot
  wrong_problem - another problem's think block, token-matched cohort
  none          - no prefix (replicates the direct condition's shape)

Readings: own_cot >> filler under ablation = content matters (scratchpad
reading supported; epiphenomenal killed). own_cot ~ filler = protection is
positions/compute. wrong_problem isolates "any coherent math text".
Note: the think block may state the final answer; rescue here includes
"reading the answer off the page through a damaged band" — that is the
point of rung 4's sufficiency test, and the filler/wrong-problem nulls are
what make the comparison meaningful.

Writes results/<host>/<ts>_prefix_rescue/ with per-condition jsonl + one
summary.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import pathlib
import socket
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FILLER_SENTENCE = "Let me think about this problem carefully and step by step. "


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (c - h, c + h)


def newest_cot_clean_generations() -> pathlib.Path:
    """Latest v2 (max_new>=2048) gsm8k cot clean generations.jsonl."""
    cands = []
    for d in glob.glob(str(REPO / "results/*/*gsm8k_cot_none*")):
        d = pathlib.Path(d)
        try:
            cfg = json.loads((d / "config.json").read_text())
        except Exception:
            continue
        if cfg["sampling"]["max_new_tokens"] >= 2048:
            cands.append(d)
    if not cands:
        raise SystemExit("no v2 cot clean run found")
    return sorted(cands)[-1] / "generations.jsonl"


def think_block(text: str) -> str | None:
    if "<think>" in text and "</think>" in text:
        return text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    import torch
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ablation import AblationConfig, JSpaceAblator, band_layers
    from ablation.generate import generate_with_ablation
    from eval.gsm8k import PROMPTS_DIR, extract_answer, is_correct

    src = newest_cot_clean_generations()
    print(f"[rescue] prefix source: {src}")
    items = []
    for line in src.read_text().splitlines():
        d = json.loads(line)
        tb = think_block(d["text"])
        if d["correct"] and tb:
            items.append({"question": d["question"], "gold": d["gold"], "think": tb})
    items = items[: args.n]
    print(f"[rescue] {len(items)} correct-CoT problems")

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

    user_tpl = (PROMPTS_DIR / "gsm8k_cot_user.txt").read_text()

    def n_tokens(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    def match_tokens(base: str, target_n: int) -> str:
        """Repeat filler to ~target_n tokens (never over by more than one
        sentence), so position count is matched."""
        out = base
        while n_tokens(out) < target_n:
            out += base
        ids = tokenizer.encode(out, add_special_tokens=False)[:target_n]
        return tokenizer.decode(ids)

    def build_prompt(question: str, prefix_think: str | None) -> str:
        head = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_tpl.format(question=question)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        think = prefix_think if prefix_think is not None else ""
        return head + "<think>\n" + think + "\n</think>\n\nThe answer is"

    wrong = items[1:] + items[:1]  # derangement for wrong_problem
    conditions: dict[str, list[str]] = {
        "own_cot": [build_prompt(it["question"], it["think"]) for it in items],
        "filler": [
            build_prompt(it["question"],
                         match_tokens(FILLER_SENTENCE, n_tokens(it["think"])))
            for it in items
        ],
        "wrong_problem": [
            build_prompt(it["question"], w["think"])
            for it, w in zip(items, wrong)
        ],
        "none": [build_prompt(it["question"], None) for it in items],
    }

    band = band_layers(model.config.num_hidden_layers, "medium")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / socket.gethostname() / f"{stamp}_prefix_rescue"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "config.json").write_text(json.dumps({
        "argv": sys.argv, "n": len(items), "seed": args.seed,
        "prefix_source": str(src), "band": [band[0], band[-1]],
        "filler_sentence": FILLER_SENTENCE,
    }, indent=2))

    summary = {}
    t0 = time.time()
    for abl_name in ("medium", "none"):
        ablator = None
        if abl_name == "medium":
            cfg = AblationConfig(band=band, k=10, exclusion_filter=True,
                                 random_seed=args.seed)
            ablator = JSpaceAblator.from_hf(model, lens, cfg)
        for cond, prompts in conditions.items():
            results = generate_with_ablation(
                model, tokenizer, prompts, ablator=ablator,
                max_new_tokens=24, temperature=0.6, top_k=20, top_p=0.95,
                seed=args.seed, batch_size=args.batch_size,
            )
            n_ok = 0
            with (out_dir / f"{cond}_{abl_name}.jsonl").open("w") as f:
                for it, r in zip(items, results):
                    pred = extract_answer("The answer is" + r.text)
                    ok = is_correct(pred, it["gold"])
                    n_ok += ok
                    f.write(json.dumps({
                        "question": it["question"], "gold": it["gold"],
                        "pred": pred, "correct": ok, "text": r.text,
                    }) + "\n")
            lo, hi = wilson_ci(n_ok, len(items))
            summary[f"{cond}_{abl_name}"] = {
                "acc": n_ok / len(items), "n": len(items), "wilson95": [lo, hi],
            }
            print(f"[rescue] {cond:>13} / {abl_name:<6} "
                  f"acc {n_ok}/{len(items)} = {n_ok/len(items):.3f}", flush=True)

    summary["wall_seconds"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[rescue] -> {out_dir}")


if __name__ == "__main__":
    main()
