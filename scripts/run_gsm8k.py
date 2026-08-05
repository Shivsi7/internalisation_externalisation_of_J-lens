#!/usr/bin/env python3
"""Run one GSM8K condition: (mode) x (ablation) -> results/<host>/<ts>_<tag>/.

Every run writes its complete parameter set (config.json), raw generations
(generations.jsonl), and scored summary (summary.json). Nothing is ever
overwritten; the timestamp in the directory name is the run identity.

Example:
    python scripts/run_gsm8k.py --mode cot --ablation medium --n 200
    python scripts/run_gsm8k.py --mode direct --ablation medium \
        --direction-source random --norm-match --n 200
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import socket
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (center - half, center + half)


def git_head(path: pathlib.Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["cot", "direct"], required=True)
    ap.add_argument("--ablation", default="none",
                    choices=["none", "light", "medium", "heavy"])
    ap.add_argument("--direction-source", default="jlens",
                    choices=["jlens", "jlens_rankshift", "logitlens", "random"])
    ap.add_argument("--norm-match", action="store_true")
    ap.add_argument("--no-filter", action="store_true",
                    help="disable the clean-pass exclusion filter (condition B8)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=None,
                    help="default: 1024 for cot, 16 for direct")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--lens-path",
                    default=str(REPO / "lenses/qwen3-4b/qwen3-4b/jlens/"
                                "Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    max_new = args.max_new or (1024 if args.mode == "cot" else 16)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ablation import AblationConfig, JSpaceAblator, band_layers
    from ablation.generate import generate_with_ablation
    from eval.gsm8k import build_prompts, extract_answer, is_correct, load_gsm8k

    tag = args.tag or (
        f"gsm8k_{args.mode}_{args.ablation}"
        + (f"_{args.direction_source}" if args.direction_source != "jlens" else "")
        + ("_normmatch" if args.norm_match else "")
        + ("_nofilter" if args.no_filter else "")
        + f"_n{args.n}_s{args.seed}"
    )
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / socket.gethostname() / f"{stamp}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"[run] output -> {out_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ablator = None
    abl_cfg = None
    if args.ablation != "none":
        import jlens

        band = band_layers(model.config.num_hidden_layers, args.ablation)
        abl_cfg = AblationConfig(
            band=band,
            k=args.k,
            direction_source=args.direction_source,
            norm_match=args.norm_match,
            exclusion_filter=not args.no_filter,
            random_seed=args.seed,
        )
        needs_lens = args.direction_source in ("jlens", "jlens_rankshift") or args.norm_match
        lens = jlens.JacobianLens.load(args.lens_path) if needs_lens else None
        ablator = JSpaceAblator.from_hf(model, lens, abl_cfg)
        print(f"[run] ablation band {band[0]}..{band[-1]} "
              f"({len(band)} layers), source={args.direction_source}, "
              f"filter={'off' if args.no_filter else 'on'}", flush=True)

    items = load_gsm8k(n=args.n, seed=args.seed)
    prompts = build_prompts(items, tokenizer, args.mode)

    config = {
        "argv": sys.argv,
        "task": "gsm8k", "mode": args.mode, "n": args.n, "seed": args.seed,
        "sampling": {"temperature": args.temperature, "top_k": args.top_k,
                     "top_p": args.top_p, "max_new_tokens": max_new,
                     "batch_size": args.batch_size},
        "model": args.model,
        "lens_path": args.lens_path if ablator is not None else None,
        "ablation": abl_cfg.asdict() if abl_cfg else None,
        "env": {
            "hostname": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "repo_head": git_head(REPO),
            "started_utc": stamp,
        },
    }
    try:
        import transformers
        config["env"]["transformers"] = transformers.__version__
    except Exception:
        pass
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    t0 = time.time()

    def progress(done, total):
        rate = done / max(time.time() - t0, 1e-9)
        print(f"[run] {done}/{total} prompts ({rate:.2f}/s)", flush=True)

    results = generate_with_ablation(
        model, tokenizer, prompts,
        ablator=ablator,
        max_new_tokens=max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        batch_size=args.batch_size,
        progress=progress,
    )

    n_correct = 0
    with (out_dir / "generations.jsonl").open("w") as f:
        for it, r in zip(items, results):
            pred = extract_answer(r.text)
            ok = is_correct(pred, it["gold"])
            n_correct += ok
            f.write(json.dumps({
                "question": it["question"], "gold": it["gold"],
                "text": r.text, "pred": pred, "correct": ok,
                "n_new_tokens": r.n_new_tokens, "stop_reason": r.stop_reason,
            }) + "\n")

    n = len(results)
    lo, hi = wilson_ci(n_correct, n)
    summary = {
        "tag": tag,
        "accuracy": n_correct / n,
        "n_correct": n_correct,
        "n": n,
        "wilson95": [lo, hi],
        "mean_new_tokens": sum(r.n_new_tokens for r in results) / n,
        "stop_reasons": {
            s: sum(r.stop_reason == s for r in results) for s in ("eos", "length")
        },
        "wall_seconds": time.time() - t0,
        "ablation_stats": ablator.stats_summary() if ablator else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[run] DONE {tag}: acc {n_correct}/{n} = {n_correct/n:.3f} "
          f"[{lo:.3f}, {hi:.3f}] in {summary['wall_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
