#!/usr/bin/env python3
"""Download a pre-fitted Jacobian lens from the Neuronpedia Hub repo.

The paper's companion code (github.com/anthropics/jacobian-lens) can fit a lens
itself, but Neuronpedia already hosts fitted lenses for the whole Qwen3 family,
including Qwen3-4B. Fitting is therefore a download, not a GPU job.

    python scripts/fetch_lens.py                # Qwen3-4B (the headline model)
    python scripts/fetch_lens.py --model qwen3-8b
    python scripts/fetch_lens.py --all          # the full scale sweep

Provenance of every lens fetched here, from its own config.yaml: fitted on
Salesforce/wikitext-103-raw-v1, max_seq_len 128, bfloat16, target_layer=null
(i.e. the final layer -- note the paper's default reads to the *penultimate*
layer instead; see design.md 6.1).
"""

from __future__ import annotations

import argparse
import pathlib

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "main"

# np_model_id -> (hf_model_name, filename within the Hub repo)
MODELS = {
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "Qwen3-1.7B_jacobian_lens.pt"),
    "qwen3-4b": ("Qwen/Qwen3-4B", "Qwen3-4B_jacobian_lens.pt"),
    "qwen3-8b": ("Qwen/Qwen3-8B", "Qwen3-8B_jacobian_lens.pt"),
    "qwen3-14b": ("Qwen/Qwen3-14B", "Qwen3-14B_jacobian_lens.pt"),
    "qwen3-32b": ("Qwen/Qwen3-32B", "Qwen3-32B_jacobian_lens.pt"),
    "llama3.1-8b-it": ("unsloth/Meta-Llama-3.1-8B-Instruct",
                       "Llama-3.1-8B-Instruct_jacobian_lens.pt"),
}

DEFAULT_MODEL = "qwen3-4b"


def remote_path(np_id: str, filename: str) -> str:
    return f"{np_id}/jlens/Salesforce-wikitext/{filename}"


def fetch(np_id: str, out_dir: pathlib.Path, *, verify: bool = True) -> pathlib.Path:
    from huggingface_hub import hf_hub_download

    hf_name, filename = MODELS[np_id]
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name in (filename, "config.yaml", filename.replace(
        "_jacobian_lens.pt", "_convergence.csv"
    )):
        paths[name] = hf_hub_download(
            repo_id=LENS_REPO,
            filename=remote_path(np_id, name),
            revision=LENS_REVISION,
            local_dir=out_dir,
        )

    lens_path = pathlib.Path(paths[filename])
    print(f"{np_id}: {lens_path} ({lens_path.stat().st_size / 1e6:.0f} MB)")

    if verify:
        _verify(lens_path, hf_name)
    return lens_path


def _verify(lens_path: pathlib.Path, expected_hf_name: str) -> None:
    """Check the lens loads and covers the layers we intend to ablate.

    Guards the two things that would silently invalidate every downstream
    result: a lens fitted for a different checkpoint, and a lens missing part
    of the workspace band.
    """
    import jlens

    lens = jlens.JacobianLens.from_pretrained(str(lens_path))
    print(f"  {lens!r}")

    # design.md 4: the naive port of the paper's band is 38-92% of depth.
    layers = lens.source_layers
    depth = max(layers) + 2  # source layers run 0..n_layers-2
    band = [l for l in layers if 0.38 * depth <= l <= 0.92 * depth]
    print(f"  d_model={lens.d_model}  n_prompts={lens.n_prompts}")
    print(f"  source layers {min(layers)}..{max(layers)} ({len(layers)} fitted)")
    print(f"  naive workspace band (38-92% of {depth}): {min(band)}..{max(band)}")

    missing = [l for l in band if l not in lens.jacobians]
    if missing:
        raise SystemExit(f"  FAIL: lens is missing band layers {missing}")
    print(f"  OK: all {len(band)} band layers present ({expected_hf_name})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL)
    parser.add_argument("--all", action="store_true", help="fetch the scale sweep")
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("lenses"))
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    targets = sorted(MODELS) if args.all else [args.model]
    for np_id in targets:
        fetch(np_id, args.out_dir / np_id, verify=not args.no_verify)


if __name__ == "__main__":
    main()
