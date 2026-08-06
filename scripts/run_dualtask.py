#!/usr/bin/env python3
"""Dual-task word probe — rung 1.5's corrected primary instrument (A2.3).

The model holds a single-token word while solving a single-digit-answer
GSM8K problem, in both arms (CoT / direct), clean and medium-ablated:

  behavioural: answer accuracy + word recall accuracy per condition
  lens: J-lens occupancy of the held word during the SOLVING span of the
        clean rollouts (direct arm: the question span; CoT arm: the think
        block), z-scored across the candidate word set at each (layer,
        position). Motor check: occupancy at the "The word was" position.

Registered outcomes in hypotheses.md A2.3 (persistence / token-exclusion /
arm asymmetry); recall predicted near ceiling everywhere (word stays in
context), with a direct-arm-specific drop under ablation reading instead as
workspace-dependent retrieval.

Writes results/<host>/<ts>_dualtask/{summary.json,<cond>.jsonl}.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import socket
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

WORDS = [
    "banana", "violin", "castle", "pepper", "rocket", "candle", "tiger",
    "marble", "helmet", "garden", "bridge", "winter", "copper", "velvet",
    "planet", "hammer", "island", "jungle", "ladder", "mirror", "needle",
    "orange", "pencil", "ribbon", "saddle", "temple", "turtle", "wallet",
]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (c - h, c + h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cot-max-new", type=int, default=2048)
    ap.add_argument("--skip-occupancy", action="store_true")
    args = ap.parse_args()

    import torch
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ablation import AblationConfig, JSpaceAblator, band_layers
    from ablation.generate import generate_with_ablation
    from eval.gsm8k import extract_answer, is_correct, load_gsm8k

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

    words = []
    word_tok = {}
    for w in WORDS:
        enc = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(enc) == 1:
            words.append(w)
            word_tok[w] = enc[0]
    if len(words) < 10:
        raise SystemExit(f"only {len(words)} single-token words: {words}")
    print(f"[dualtask] {len(words)} single-token words usable")
    wid_t = torch.tensor([word_tok[w] for w in words], dtype=torch.long,
                         device=model.device)

    items = load_gsm8k(n=args.n, seed=args.seed, single_digit=True)
    for i, it in enumerate(items):
        it["word"] = words[i % len(words)]

    def build_prompt(it, arm: str) -> str:
        if arm == "direct":
            user = (f"Remember this word: {it['word']}.\n\n{it['question']}\n\n"
                    "Give only the final numeric answer, then on a new line "
                    "write: The word was ...")
            head = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            return head + "The answer is"
        user = (f"Remember this word: {it['word']}.\n\n{it['question']}\n\n"
                "Please reason step by step, put your final answer after "
                '"The answer is", then on a new line write "The word was" '
                "followed by the remembered word.")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / socket.gethostname() / f"{stamp}_dualtask"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "config.json").write_text(json.dumps({
        "argv": sys.argv, "n": len(items), "seed": args.seed,
        "band": [band[0], band[-1]], "n_words": len(words),
    }, indent=2))

    summary = {}
    clean_rollouts = {}
    t0 = time.time()
    for arm in ("direct", "cot"):
        prompts = [build_prompt(it, arm) for it in items]
        for abl in ("none", "medium"):
            ablator = None
            if abl == "medium":
                ablator = JSpaceAblator.from_hf(model, lens, AblationConfig(
                    band=band, k=10, exclusion_filter=True,
                    random_seed=args.seed))
            max_new = args.cot_max_new if arm == "cot" else 24
            res = generate_with_ablation(
                model, tokenizer, prompts, ablator=ablator,
                max_new_tokens=max_new, temperature=0.6, top_k=20, top_p=0.95,
                seed=args.seed, batch_size=args.batch_size)
            n_ans = n_rec = 0
            rows = []
            for it, r in zip(items, res):
                text = ("The answer is" + r.text) if arm == "direct" else r.text
                pred = extract_answer(text)
                a_ok = is_correct(pred, it["gold"])
                m = re.findall(r"word was\W*([A-Za-z]+)", text,
                               flags=re.IGNORECASE)
                rec_ok = bool(m) and m[-1].lower() == it["word"]
                n_ans += a_ok
                n_rec += rec_ok
                rows.append({**it, "text": r.text, "pred": pred,
                             "answer_ok": a_ok, "recall_ok": rec_ok,
                             "stop": r.stop_reason})
            cond = f"{arm}_{abl}"
            with (out_dir / f"{cond}.jsonl").open("w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            n = len(items)
            summary[cond] = {
                "answer_acc": n_ans / n, "answer_ci": wilson_ci(n_ans, n),
                "recall_acc": n_rec / n, "recall_ci": wilson_ci(n_rec, n),
                "n": n,
            }
            print(f"[dualtask] {cond:>13}: answer {n_ans}/{n}  "
                  f"recall {n_rec}/{n}", flush=True)
            if abl == "none":
                clean_rollouts[arm] = [(build_prompt(it, arm), r.text, it)
                                       for it, r in zip(items, res)]

    # ---- occupancy of the held word during solving (clean rollouts) -------
    if not args.skip_occupancy:
        ablator = JSpaceAblator.from_hf(model, lens, AblationConfig(
            band=band, exclusion_filter=False))
        occ_stats = {}
        for arm, rolls in clean_rollouts.items():
            held_mean, held_max, decoy_mean, motor = [], [], [], []
            for prompt, gen, it in rolls:
                full = prompt + gen
                if arm == "cot":
                    if "<think>" not in full or "</think>" not in full:
                        continue
                    lo = full.index("<think>") + len("<think>")
                    hi = full.index("</think>")
                else:
                    lo = full.index(it["question"][:40])
                    hi = lo + len(it["question"])
                enc = tokenizer(full, return_tensors="pt",
                                return_offsets_mapping=True,
                                add_special_tokens=False)
                offsets = enc.pop("offset_mapping")[0].tolist()
                span = [i for i, (a, b) in enumerate(offsets)
                        if a < hi and b > lo]
                if len(span) < 3:
                    continue
                wm = re.search(r"word was\W*[A-Za-z]", full[hi:],
                               flags=re.IGNORECASE)
                with torch.no_grad():
                    out = model(input_ids=enc.input_ids.to(model.device),
                                output_hidden_states=True)
                widx = words.index(it["word"])
                zs_mean, zs_max, zd = [], [], []
                for layer in band:
                    h = out.hidden_states[layer + 1][0][span].float()
                    occ = ablator.occupancy(h, layer, wid_t)  # [S, n_words]
                    mu = occ.mean(-1, keepdim=True)
                    sd = occ.std(-1, keepdim=True).clamp_min(1e-6)
                    z = (occ - mu) / sd
                    zs_mean.append(float(z[:, widx].mean()))
                    zs_max.append(float(z[:, widx].max()))
                    others = [i for i in range(len(words)) if i != widx]
                    zd.append(float(z[:, others].mean()))
                held_mean.append(sum(zs_mean) / len(zs_mean))
                held_max.append(max(zs_max))
                decoy_mean.append(sum(zd) / len(zd))
                if wm:
                    mpos = hi + wm.start()
                    mtoks = [i for i, (a, b) in enumerate(offsets)
                             if a < mpos + 8 and b > mpos]
                    if mtoks:
                        vals = []
                        for layer in band:
                            h = out.hidden_states[layer + 1][0][mtoks].float()
                            occ = ablator.occupancy(h, layer, wid_t)
                            mu = occ.mean(-1, keepdim=True)
                            sd = occ.std(-1, keepdim=True).clamp_min(1e-6)
                            vals.append(float(((occ - mu) / sd)[:, widx].max()))
                        motor.append(max(vals))

            def mean(x):
                return sum(x) / len(x) if x else None

            occ_stats[arm] = {
                "n": len(held_mean),
                "held_word_mean_z": mean(held_mean),
                "held_word_max_z": mean(held_max),
                "decoy_mean_z": mean(decoy_mean),
                "recall_site_max_z (motor check)": mean(motor),
            }
            print(f"[dualtask] occupancy {arm}: {occ_stats[arm]}", flush=True)
        summary["occupancy"] = occ_stats

    summary["wall_seconds"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[dualtask] -> {out_dir}")


if __name__ == "__main__":
    main()
