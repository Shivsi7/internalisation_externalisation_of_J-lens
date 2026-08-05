"""Ablation configuration and layer-band arithmetic.

Band fractions come from the paper's Figure 22 (layers reindexed 0-100 across
depth): light 62-71, medium 54-79, heavy 42-88. The naive port to an n-layer
model is round(frac * n) inclusive at both ends (design.md §4.2 gives the
resulting Qwen3-4B bands: light 22-26, medium 19-28, heavy 15-32). The
validation gate re-derives the band empirically; these are the starting point.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

BAND_FRACTIONS: dict[str, tuple[float, float]] = {
    "light": (0.62, 0.71),
    "medium": (0.54, 0.79),
    "heavy": (0.42, 0.88),
}


def band_layers(n_layers: int, name: str) -> list[int]:
    """Layer indices (block outputs, 0-based) for a named ablation strength."""
    lo, hi = BAND_FRACTIONS[name]
    return list(range(round(lo * n_layers), round(hi * n_layers) + 1))


@dataclass
class AblationConfig:
    """One ablation condition. Every field is a documented decision.

    direction_source:
        "jlens"           - top-k J-lens directions (the effect, B1)
        "jlens_rankshift" - ranks rank_shift+1 .. rank_shift+k (control B3)
        "logitlens"       - top-k rows of W_U, no Jacobian transport (B5)
        "random"          - k fixed random orthonormal directions (B2);
                            with norm_match=True the removed component is
                            rescaled per position to the norm the jlens
                            ablation would have removed on the same residual.

    exclusion_filter: the paper's confound guard — tokens in the clean
        forward pass's top `filter_top_n` next-token predictions are exempt
        from selection. Toggleable because it is itself condition B8: it is
        not neutral between CoT and direct arms (design.md §6.2(4)).

    projection: "least_squares" projects onto the span of the k selected
        (non-orthogonal) vectors; "sequential" zeroes projections one at a
        time in rank order (order-dependent, kept as a sensitivity variant).

    include_norm_weight: fold the final RMSNorm weight g into the lens
        vector, j_v = J_l^T (g ⊙ W_U[v]). The readout that *selects* tokens
        goes through the final norm, so the vector removed should match the
        vector scored; set False to measure how much that choice matters.

    ablate_prompt: apply the ablation at prompt positions as well as
        generated ones (the paper ablates "at each token position").
    """

    band: list[int] = field(default_factory=list)
    k: int = 10
    direction_source: str = "jlens"
    rank_shift: int = 50
    exclusion_filter: bool = True
    filter_top_n: int = 10
    norm_match: bool = False
    projection: str = "least_squares"
    include_norm_weight: bool = True
    ablate_prompt: bool = True
    random_seed: int = 0
    ridge: float = 1e-6  # relative Tikhonov term on the Gram matrix

    def __post_init__(self) -> None:
        if self.direction_source not in (
            "jlens", "jlens_rankshift", "logitlens", "random",
        ):
            raise ValueError(f"unknown direction_source {self.direction_source!r}")
        if self.projection not in ("least_squares", "sequential"):
            raise ValueError(f"unknown projection {self.projection!r}")
        if self.norm_match and self.direction_source != "random":
            raise ValueError("norm_match is only meaningful for random directions")

    def asdict(self) -> dict:
        return asdict(self)
