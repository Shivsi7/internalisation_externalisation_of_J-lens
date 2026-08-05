"""GSM8K loading, prompt construction, and numeric answer scoring.

Two externalisation arms (design.md factor A):
  cot    - thinking mode; the model writes its reasoning before answering.
  direct - non-thinking mode, answer-only instruction, assistant prefilled
           with "The answer is" so the very first sampled tokens are the
           answer. The paper never states its direct-answer format; this one
           is ours and is recorded verbatim in prompts/.
"""

from __future__ import annotations

import pathlib
import re

PROMPTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "prompts"

_NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def load_gsm8k(n: int | None = None, seed: int = 0):
    """Test-split problems as [{question, gold}]. Deterministic subsample."""
    import random

    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = [
        {"question": ex["question"], "gold": ex["answer"].split("####")[-1].strip()}
        for ex in ds
    ]
    if n is not None and n < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, n)
    return items


def build_prompts(items, tokenizer, mode: str) -> list[str]:
    """Fully templated prompt strings (chat template + prefill applied)."""
    if mode == "cot":
        user_tpl = (PROMPTS_DIR / "gsm8k_cot_user.txt").read_text()
        prefill = ""
        thinking = True
    elif mode == "direct":
        user_tpl = (PROMPTS_DIR / "gsm8k_direct_user.txt").read_text()
        prefill = (PROMPTS_DIR / "gsm8k_direct_prefill.txt").read_text()
        thinking = False
    else:
        raise ValueError(f"unknown mode {mode!r}")

    prompts = []
    for it in items:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_tpl.format(question=it["question"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        prompts.append(text + prefill)
    return prompts


def _to_float(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_answer(text: str) -> float | None:
    """Predicted number from a generation.

    Preference order: last "answer is <num>" (outside the think block if one
    exists), then \\boxed{<num>}, then the last number in the post-think text.
    """
    if "</think>" in text:
        visible = text.split("</think>")[-1]
    else:
        visible = text

    for source in (visible, text):
        hits = re.findall(r"answer is[^\d\-]*(-?\$?\d[\d,]*(?:\.\d+)?)", source,
                          flags=re.IGNORECASE)
        if hits:
            return _to_float(hits[-1])
        boxed = re.findall(r"\\boxed\{([^}]*)\}", source)
        if boxed:
            nums = _NUM.findall(boxed[-1])
            if nums:
                return _to_float(nums[-1])
    nums = _NUM.findall(visible)
    if nums:
        return _to_float(nums[-1])
    return None


def is_correct(pred: float | None, gold: str) -> bool:
    g = _to_float(gold)
    if pred is None or g is None:
        return False
    return abs(pred - g) < max(1e-4, 1e-4 * abs(g))
