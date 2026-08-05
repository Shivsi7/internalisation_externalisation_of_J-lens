"""J-space ablation for Qwen3-4B — the intervention the reference repo does not ship.

`anthropics/jacobian-lens` provides fitting and readout only. Everything here —
top-k J-lens direction selection, least-squares projection removal, the paired
clean pass for the exclusion filter, and batched ablated generation — is ours,
written against the paper's §3.5.2 description and the decisions documented in
`design.md` §6.2.
"""

from ablation.config import AblationConfig, band_layers, BAND_FRACTIONS
from ablation.jspace import JSpaceAblator
from ablation.hooks import AblationSession
from ablation.generate import generate_with_ablation

__all__ = [
    "AblationConfig",
    "band_layers",
    "BAND_FRACTIONS",
    "JSpaceAblator",
    "AblationSession",
    "generate_with_ablation",
]
