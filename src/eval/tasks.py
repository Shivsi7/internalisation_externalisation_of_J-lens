"""Task adapters: GSM8K, MATH-500, AIME 2024+2025.

Each task provides load(n, seed) -> [{question, gold, **meta}], a prompt
template pair (cot/direct), and a scorer. Sources and revisions are pinned
here and recorded in every run's config.json.

MATH-500 subsampling is stratified by level (equal per level) so the
difficulty gradient survives a compute-limited n.
"""

from __future__ import annotations

import pathlib
import random
import re

from eval import gsm8k as _gsm8k

PROMPTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "prompts"

SOURCES = {
    "gsm8k": ("openai/gsm8k", "main"),
    "math500": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"),
    "aime24": ("HuggingFaceH4/aime_2024", "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"),
    "aime25": ("MathArena/aime_2025", "c94da77eb22bbd6439e62a323bec18493a421302"),
}


def _stratified_sample(items, key, n, seed):
    rng = random.Random(seed)
    by = {}
    for it in items:
        by.setdefault(key(it), []).append(it)
    per = max(1, n // len(by))
    out = []
    for k in sorted(by):
        pool = by[k]
        out.extend(pool if len(pool) <= per else rng.sample(pool, per))
    return out


def load_math500(n: int | None = None, seed: int = 0):
    from datasets import load_dataset

    ds = load_dataset(SOURCES["math500"][0], split="test",
                      revision=SOURCES["math500"][1])
    items = [
        {"question": ex["problem"], "gold": ex["answer"], "level": ex["level"],
         "subject": ex.get("subject", "")}
        for ex in ds
    ]
    if n is not None and n < len(items):
        items = _stratified_sample(items, lambda it: it["level"], n, seed)
    return items


def load_aime(n: int | None = None, seed: int = 0):
    """AIME 2024 I+II + 2025 I+II, 60 problems, years tagged for the
    contamination split. Revisions pinned in SOURCES."""
    from datasets import load_dataset

    items = []
    ds24 = load_dataset(SOURCES["aime24"][0], split="train",
                        revision=SOURCES["aime24"][1])
    for ex in ds24:
        items.append({"question": ex["problem"], "gold": str(ex["answer"]),
                      "year": 2024})
    ds25 = load_dataset(SOURCES["aime25"][0], split="train",
                        revision=SOURCES["aime25"][1])
    for ex in ds25:
        items.append({"question": ex["problem"], "gold": str(ex["answer"]),
                      "year": 2025})
    return items  # n ignored: always the full 60, per the spec


_BOXED = re.compile(r"\\boxed\{")


def _last_boxed(text: str) -> str | None:
    """Contents of the last \\boxed{...}, brace-aware."""
    starts = [m.end() for m in _BOXED.finditer(text)]
    if not starts:
        return None
    start = starts[-1]
    depth, i = 1, start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1] if depth == 0 else text[start:]


def extract_math(text: str) -> str | None:
    """Predicted answer string for MATH-style (latex) answers."""
    visible = text.split("</think>")[-1] if "</think>" in text else text
    for source in (visible, text):
        boxed = _last_boxed(source)
        if boxed is not None:
            return boxed.strip()
        hits = re.findall(r"answer is\s*:?\s*\$?([^\n$.]+)", source,
                          flags=re.IGNORECASE)
        if hits:
            return hits[-1].strip().rstrip(".")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", visible)
    return nums[-1].replace(",", "") if nums else None


def score_math(pred: str | None, gold: str) -> bool:
    """math-verify equivalence with a plain-string fallback."""
    if pred is None:
        return False
    if pred.strip() == gold.strip():
        return True
    try:
        from math_verify import parse, verify

        return bool(verify(parse(gold), parse(pred)))
    except Exception:
        return False


def score_aime(pred: str | None, gold: str) -> bool:
    """AIME answers are integers in [0, 999]."""
    if pred is None:
        return False
    nums = re.findall(r"-?\d+", pred.replace(",", ""))
    return bool(nums) and nums[-1] == str(int(gold))


TASKS = {
    "gsm8k": {
        "load": _gsm8k.load_gsm8k,
        "cot_user": "gsm8k_cot_user.txt",
        "direct_user": "gsm8k_direct_user.txt",
        "extract": _gsm8k.extract_answer,
        "score": _gsm8k.is_correct,
        "cot_max_new": 2048,
    },
    "math500": {
        "load": load_math500,
        "cot_user": "math_cot_user.txt",
        "direct_user": "math_direct_user.txt",
        "extract": extract_math,
        "score": score_math,
        "cot_max_new": 3072,
    },
    "aime": {
        "load": load_aime,
        "cot_user": "math_cot_user.txt",
        "direct_user": "math_direct_user.txt",
        "extract": extract_math,
        "score": score_aime,
        "cot_max_new": 4096,
    },
}


def build_prompts(items, tokenizer, task: str, mode: str) -> list[str]:
    spec = TASKS[task]
    if mode == "cot":
        tpl = (PROMPTS_DIR / spec["cot_user"]).read_text()
        prefill, thinking = "", True
    else:
        tpl = (PROMPTS_DIR / spec["direct_user"]).read_text()
        prefill = (PROMPTS_DIR / "gsm8k_direct_prefill.txt").read_text()
        thinking = False
    prompts = []
    for it in items:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": tpl.format(question=it["question"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
        )
        prompts.append(text + prefill)
    return prompts
