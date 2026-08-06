#!/usr/bin/env python3
"""Run one (task, mode, ablation) condition -> results/<host>/<ts>_<tag>/.

Generalizes run_gsm8k.py to MATH-500 and AIME (task adapters in
src/eval/tasks.py, dataset revisions pinned there). Same manifest discipline:
config.json (full params), generations.jsonl (raw), summary.json (scored).

    python scripts/run_eval.py --task math500 --mode cot --ablation medium --n 250
    python scripts/run_eval.py --task aime --mode direct --ablation none
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
    ap.add_argument("--task", choices=["gsm8k", "math500", "aime"], required=True)
    ap.add_argument("--mode", choices=["cot", "direct"], required=True)
    ap.add_argument("--ablation", default="none",
                    choices=["none", "light", "medium", "heavy"])
    ap.add_argument("--direction-source", default="jlens",
                    choices=["jlens", "jlens_rankshift", "logitlens", "random"])
    ap.add_argument("--norm-match", action="store_true")
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=None,
                    help="subsample (stratified by level for math500); "
                         "aime always runs all 60")
    ap.add_argument("--single-digit-answers", action="store_true",
                    help="gsm8k only: restrict to single-digit gold answers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=None,
                    help="default: task cot_max_new for cot, 32 for direct")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--lens-path",
                    default=str(REPO / "lenses/qwen3-4b/qwen3-4b/jlens/"
                                "Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ablation import AblationConfig, JSpaceAblator, band_layers
    from ablation.generate import generate_with_ablation
    from eval.tasks import TASKS, SOURCES, build_prompts

    spec = TASKS[args.task]
    max_new = args.max_new or (spec["cot_max_new"] if args.mode == "cot" else 32)

    model_slug = ""
    if args.model != "Qwen/Qwen3-4B":
        model_slug = "_" + args.model.split("/")[-1].lower().replace("qwen3-", "m")
    sd = "_sd" if args.single_digit_answers else ""
    tag = args.tag or (
        f"{args.task}{model_slug}{sd}_{args.mode}_{args.ablation}"
        + (f"_{args.direction_source}" if args.direction_source != "jlens" else "")
        + ("_normmatch" if args.norm_match else "")
        + ("_nofilter" if args.no_filter else "")
        + (f"_n{args.n}" if args.n else "_all")
        + f"_s{args.seed}"
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
            band=band, k=args.k,
            direction_source=args.direction_source,
            norm_match=args.norm_match,
            exclusion_filter=not args.no_filter,
            random_seed=args.seed,
        )
        needs_lens = (args.direction_source in ("jlens", "jlens_rankshift")
                      or args.norm_match)
        lens = jlens.JacobianLens.load(args.lens_path) if needs_lens else None
        ablator = JSpaceAblator.from_hf(model, lens, abl_cfg)
        print(f"[run] band {band[0]}..{band[-1]} source={args.direction_source} "
              f"filter={'off' if args.no_filter else 'on'}", flush=True)

    load_kw = {"n": args.n, "seed": args.seed}
    if args.single_digit_answers:
        if args.task != "gsm8k":
            raise SystemExit("--single-digit-answers is gsm8k-only")
        load_kw["single_digit"] = True
    items = spec["load"](**load_kw)
    prompts = build_prompts(items, tokenizer, args.task, args.mode)
    print(f"[run] {len(items)} problems", flush=True)

    config = {
        "argv": sys.argv,
        "task": args.task, "mode": args.mode, "n": len(items), "seed": args.seed,
        "dataset_sources": SOURCES,
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
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    t0 = time.time()

    def progress(done, total):
        print(f"[run] {done}/{total} ({done/max(time.time()-t0,1e-9):.2f}/s)",
              flush=True)

    results = generate_with_ablation(
        model, tokenizer, prompts, ablator=ablator,
        max_new_tokens=max_new, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, seed=args.seed,
        batch_size=args.batch_size, progress=progress,
    )

    n_correct = 0
    per_group: dict[str, list[bool]] = {}
    with (out_dir / "generations.jsonl").open("w") as f:
        for it, r in zip(items, results):
            pred = spec["extract"](r.text)
            ok = bool(spec["score"](pred, it["gold"]))
            n_correct += ok
            group = str(it.get("level", it.get("year", "all")))
            per_group.setdefault(group, []).append(ok)
            f.write(json.dumps({
                **{k: v for k, v in it.items()},
                "text": r.text, "pred": pred, "correct": ok,
                "n_new_tokens": r.n_new_tokens, "stop_reason": r.stop_reason,
            }) + "\n")

    n = len(results)
    lo, hi = wilson_ci(n_correct, n)
    summary = {
        "tag": tag, "task": args.task,
        "accuracy": n_correct / n, "n_correct": n_correct, "n": n,
        "wilson95": [lo, hi],
        "by_group": {
            g: {"acc": sum(v) / len(v), "n": len(v)}
            for g, v in sorted(per_group.items())
        },
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
