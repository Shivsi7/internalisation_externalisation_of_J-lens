"""Forward-hook session that applies the ablation during model forwards.

Mirrors jlens.hooks.ActivationRecorder structurally (hook the residual blocks,
handle tensor-or-tuple outputs) but *modifies* the output instead of recording
it. The same hooked model serves both the ablated and the paired clean pass:
`session.skip = True` makes every hook a no-op, so the generation loop can
interleave clean and ablated forwards without re-registering hooks.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ablation.jspace import JSpaceAblator


class AblationSession:
    """Context manager owning the hooks and the per-forward state.

    Before each ablated forward, call `set_state(filter_ids, active)`;
    before each clean forward, set `skip = True`.

    filter_ids: [B, S, F] token ids exempt from selection (clean-pass top-F),
        or None when the exclusion filter is off.
    active: [B, S] bool mask of positions to ablate, or None for all.
        S must match the sequence length of the *current* forward (1 during
        incremental decoding).
    """

    def __init__(self, ablator: JSpaceAblator, blocks: Sequence[nn.Module]) -> None:
        self.ablator = ablator
        self._blocks = blocks
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.skip = False
        self.filter_ids: torch.Tensor | None = None
        self.active: torch.Tensor | None = None

    def set_state(
        self,
        filter_ids: torch.Tensor | None,
        active: torch.Tensor | None,
    ) -> None:
        self.skip = False
        self.filter_ids = filter_ids
        self.active = active

    def _make_hook(self, layer: int):
        def hook(module: nn.Module, inputs, output):
            if self.skip:
                return None  # leave output untouched
            tensor = output if torch.is_tensor(output) else output[0]
            new = self.ablator.ablate(tensor, layer, self.filter_ids, self.active)
            if torch.is_tensor(output):
                return new
            return (new,) + tuple(output[1:])

        return hook

    def __enter__(self) -> "AblationSession":
        try:
            for layer in self.ablator.cfg.band:
                self._handles.append(
                    self._blocks[layer].register_forward_hook(self._make_hook(layer))
                )
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
