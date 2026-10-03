#!/usr/bin/env python3
"""Task-type balance (extension 3, hypotheses.md A5.2).

Does reliance on the page depend on the kind of thinking? Four synthetic
suites, each run through {clean, medium ablation} x {CoT, direct}:

  arith      three-line single-digit programs plus d = a op c (chained
             computation; the depth ladder says this exceeds the internal
             budget)
  recall     five colour facts about named people, query one (single lookup,
             the context already holds the answer)
  transitive four-name taller-than chain, query tallest/shortest (ordering)
  verbal     odd-one-out from four words (lexical/semantic lookup)

Metric per suite: accuracy per cell, retention (ablated/clean) per mode.
The report's picture predicts the ablation gap concentrates on the suites
that need chained computation. n=80 per suite, one seed; leads, not
conclusions.

Writes results/<host>/<ts>_task_balance/{config.json,detail.jsonl,
summary.json}.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import random
import re
import socket
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
import os as _os
HW0_MODEL = _os.environ.get("HW0_MODEL", "Qwen/Qwen3-4B")
HW0_LENS = _os.environ.get(
    "HW0_LENS", str(REPO / "lenses/qwen3-4b/qwen3-4b/jlens/"
                    "Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"))
sys.path.insert(0, str(REPO / "src"))

NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry"]
OBJECTS = ["hat", "coat", "scarf", "cup", "bike", "chair"]
COLOURS = ["red", "blue", "green", "black", "white", "purple", "orange",
           "brown"]
CATEGORIES = {
    "fruit": ["apple", "banana", "cherry", "grape", "mango", "peach"],
    "animal": ["dog", "horse", "rabbit", "tiger", "sheep", "goat"],
    "tool": ["hammer", "wrench", "saw", "drill", "chisel", "pliers"],
    "vehicle": ["car", "truck", "bus", "train", "boat", "tram"],
    "clothing": ["shirt", "jacket", "sock", "glove", "skirt", "belt"],
}


def gen_arith(rng):
    while True:
        lines, vals, ok = [], [], True
        for _ in range(3):
            x, y = rng.randint(1, 9), rng.randint(1, 9)
            op = rng.choice(["+", "-"])
            v = x + y if op == "+" else x - y
            if not (1 <= v <= 9):
                ok = False
                break
            lines.append(f"{'abc'[len(vals)]} = {x} {op} {y}.")
            vals.append(v)
        if not ok:
            continue
        op = rng.choice(["+", "-"])
        d = vals[0] + vals[2] if op == "+" else vals[0] - vals[2]
        if not (1 <= d <= 9):
            continue
        q = " ".join(lines) + f" d = a {op} c. What is the value of d?"
        return {"question": q, "gold": str(d),
                "candidates": [str(i) for i in range(1, 10)]}


def gen_recall(rng):
    people = rng.sample(NAMES, 5)
    cols = rng.sample(COLOURS, 5)
    objs = [rng.choice(OBJECTS) for _ in range(5)]
    facts = [f"{p}'s {o} is {c}." for p, o, c in zip(people, objs, cols)]
    i = rng.randrange(5)
    q = " ".join(facts) + f" What colour is {people[i]}'s {objs[i]}?"
    return {"question": q, "gold": cols[i], "candidates": cols}


def gen_transitive(rng):
    people = rng.sample(NAMES, 4)
    facts = [f"{people[i]} is taller than {people[i+1]}." for i in range(3)]
    if rng.random() < 0.5:
        q, gold = "Who is the tallest?", people[0]
    else:
        q, gold = "Who is the shortest?", people[-1]
    return {"question": " ".join(facts) + " " + q, "gold": gold,
            "candidates": people}


def gen_verbal(rng):
    cat, other = rng.sample(list(CATEGORIES), 2)
    words = rng.sample(CATEGORIES[cat], 3)
    intruder = rng.choice(CATEGORIES[other])
    options = words + [intruder]
    rng.shuffle(options)
    q = ("Which word does not belong with the others: "
         + ", ".join(options) + "?")
    return {"question": q, "gold": intruder, "candidates": options}


SUITES = {"arith": gen_arith, "recall": gen_recall,
          "transitive": gen_transitive, "verbal": gen_verbal}


def extract(text, candidates):
    matches = list(re.finditer(r"[Tt]he answer is", text))
    tail = text[matches[-1].end():] if matches else text
    low = tail.lower()
    found = None
    for c in candidates:
        idx = low.find(c.lower())
        if idx != -1 and (found is None or idx < found[1]):
            found = (c, idx)
    return found[0] if found else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cot-max-new", type=int, default=1024)
    args = ap.parse_args()

    import torch
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ablation import AblationConfig, JSpaceAblator, band_layers
    from ablation.generate import generate_with_ablation

    tokenizer = AutoTokenizer.from_pretrained(HW0_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        HW0_MODEL, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    lens = jlens.JacobianLens.load(HW0_LENS)
    band = band_layers(model.config.num_hidden_layers, "medium")

    def build_prompt(question, arm):
        if arm == "direct":
            user = (f"{question}\n\nGive only the final answer, nothing "
                    "else.")
            head = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            return head + "The answer is"
        user = (f"{question}\n\nPlease reason step by step, and put your "
                'final answer after "The answer is".')
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True)

    rng = random.Random(args.seed)
    items = {s: [g(rng) for _ in range(args.n)] for s, g in SUITES.items()}

    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    host = socket.gethostname().split(".")[0]
    out_dir = REPO / "results" / host / f"{stamp}_task_balance"
    out_dir.mkdir(parents=True)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    (out_dir / "config.json").write_text(json.dumps({
        "script": "run_task_balance.py", "argv": sys.argv,
        "model": HW0_MODEL, "band": "medium", "band_layers": band,
        "n_per_suite": args.n, "seed": args.seed,
        "cot_max_new": args.cot_max_new, "temperature": 0.6,
        "ablation": "medium k=10 exclusion_filter=on (matches main grid)",
        "git_commit": commit, "host": host,
        "gpu": torch.cuda.get_device_name(0),
    }, indent=2))

    def wilson(k, nn, z=1.96):
        if nn == 0:
            return (0.0, 1.0)
        p = k / nn
        d0 = 1 + z * z / nn
        c = (p + z * z / (2 * nn)) / d0
        h = z * ((p * (1 - p) / nn + z * z / (4 * nn * nn)) ** 0.5) / d0
        return (round(c - h, 4), round(c + h, 4))

    summary = {}
    detail_f = open(out_dir / "detail.jsonl", "w")
    for suite, its in items.items():
        for arm in ("direct", "cot"):
            prompts = [build_prompt(it["question"], arm) for it in its]
            for abl in ("none", "medium"):
                ablator = None
                if abl == "medium":
                    ablator = JSpaceAblator.from_hf(
                        model, lens, AblationConfig(
                            band=band, k=10, exclusion_filter=True,
                            random_seed=args.seed))
                max_new = args.cot_max_new if arm == "cot" else 16
                res = generate_with_ablation(
                    model, tokenizer, prompts, ablator=ablator,
                    max_new_tokens=max_new, temperature=0.6, top_k=20,
                    top_p=0.95, seed=args.seed,
                    batch_size=args.batch_size)
                k = 0
                n_len = 0
                for it, r in zip(its, res):
                    text = ("The answer is" + r.text) if arm == "direct" \
                        else r.text
                    pred = extract(text, it["candidates"])
                    ok = pred == it["gold"]
                    k += ok
                    n_len += r.stop_reason == "length"
                    detail_f.write(json.dumps({
                        "suite": suite, "arm": arm, "abl": abl,
                        "q": it["question"], "gold": it["gold"],
                        "pred": pred, "correct": ok,
                        "stop": r.stop_reason}) + "\n")
                key = f"{suite}_{arm}_{abl}"
                summary[key] = {"acc": round(k / len(its), 4), "k": k,
                                "n": len(its), "wilson95": wilson(k, len(its)),
                                "length_stopped": n_len}
                print(f"[{key}] acc={k}/{len(its)} len_stop={n_len}",
                      flush=True)
    detail_f.close()

    for suite in SUITES:
        for arm in ("direct", "cot"):
            c = summary[f"{suite}_{arm}_none"]["acc"]
            a = summary[f"{suite}_{arm}_medium"]["acc"]
            summary[f"{suite}_{arm}_retention"] = (
                round(a / c, 3) if c > 0 else None)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items()
                      if k.endswith("retention")}, indent=2))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
