#!/usr/bin/env python3
"""Validation-gate probes (design.md §4): is the lens interpretable on Qwen3-4B?

1. Readout probes: the paper's arithmetic chain (`calc: (4+17)*2+7 =` should
   surface 21 -> 42 -> 49 in order across layers, §3.3) and a multi-hop
   factual probe (planet fourth from the sun -> Mars -> red).
2. Ablation smoke: short generations, clean vs heavy ablation, to eyeball
   coherence before spending GPU-hours.

Writes a human-readable transcript to results/<host>/<ts>_probes/probes.txt.
"""

from __future__ import annotations

import datetime as dt
import pathlib
import socket
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

LENS_PATH = (REPO / "lenses/qwen3-4b/qwen3-4b/jlens/Salesforce-wikitext/"
             "Qwen3-4B_jacobian_lens.pt")

READOUT_PROBES = [
    ("calc chain", "calc: (4+17)*2+7 = ", ["21", "42", "49"]),
    ("multi-hop planets", "The colour of the planet fourth from the sun is",
     ["Mars", "red"]),
    ("multi-hop capital", "The famous tower in the capital city of France is the",
     ["Paris", "Eiffel"]),
]

GEN_PROBES = [
    "The capital of France is",
    "Q: If 3 apples cost 6 dollars, how much do 5 apples cost?\nA:",
]


def main() -> None:
    import torch
    import jlens
    from jlens.hf import from_hf
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ablation import AblationConfig, JSpaceAblator, band_layers
    from ablation.generate import generate_with_ablation

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / socket.gethostname() / f"{stamp}_probes"
    out_dir.mkdir(parents=True, exist_ok=False)
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    lens = jlens.JacobianLens.load(str(LENS_PATH))
    lm = from_hf(model, tokenizer)
    emit(f"model: Qwen3-4B ({lm.n_layers} layers), lens: {lens!r}")

    n_layers = model.config.num_hidden_layers
    band = band_layers(n_layers, "heavy")  # widest band: 15..32
    emit(f"probing layers {band[0]}..{band[-1]}")

    # ---- 1. readout probes ---------------------------------------------
    for name, prompt, expected in READOUT_PROBES:
        emit(f"\n=== readout: {name!r}  prompt={prompt!r}")
        emit(f"    expecting to see: {expected}")
        lens_logits, model_logits, input_ids = lens.apply(
            model=lm, prompt=prompt, layers=band, positions=[-1],
        )
        for layer in band:
            top = lens_logits[layer][0].topk(8).indices.tolist()
            toks = [tokenizer.decode([t]).strip() for t in top]
            emit(f"  L{layer:>2}: {toks}")
        top = model_logits[0].topk(5).indices.tolist()
        emit(f"  model next-token: {[tokenizer.decode([t]) for t in top]}")

    # ---- 2. ablation smoke ---------------------------------------------
    emit("\n=== ablation smoke (clean vs heavy, greedy, 40 tokens)")
    cfg = AblationConfig(band=band, k=10, exclusion_filter=True)
    ablator = JSpaceAblator.from_hf(model, lens, cfg)
    clean = generate_with_ablation(
        model, tokenizer, GEN_PROBES, ablator=None,
        max_new_tokens=40, temperature=0.0,
    )
    ablated = generate_with_ablation(
        model, tokenizer, GEN_PROBES, ablator=ablator,
        max_new_tokens=40, temperature=0.0,
    )
    for c, a in zip(clean, ablated):
        emit(f"\nprompt : {c.prompt!r}")
        emit(f"clean  : {c.text!r}")
        emit(f"ablated: {a.text!r}")
    emit("\nper-layer relative norm removed (heavy):")
    for layer, s in ablator.stats_summary().items():
        emit(f"  L{layer:>2}: {s['rel']:.4f}")

    (out_dir / "probes.txt").write_text("\n".join(lines))
    print(f"\n[probes] written to {out_dir/'probes.txt'}")


if __name__ == "__main__":
    main()
