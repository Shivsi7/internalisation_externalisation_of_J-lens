"""Core J-space ablation math: selection, projection, occupancy.

The quantity manipulated and the quantity measured are the same by
construction (HANDOFF.md): `occupancy()` reports the projection coefficients
that `ablate()` removes.

Conventions
-----------
The lens readout at layer l is `lm_head(final_norm(J_l @ h))` (jlens.lens /
jlens.hf). RMSNorm rescales by a token-independent positive scalar, so token
*ranking* is by `<J_l^T (g ⊙ W_U[v]), h>` where g is the final-norm weight.
The "J-lens vector" of token v at layer l is therefore

    j_v = J_l^T (g ⊙ W_U[v])            (rows of the direction matrix A)

and ablation removes the least-squares projection of h onto span{j_v : v in
top-k}, which is the unique minimal-norm edit after which every selected
token's lens coefficient <h, j_v> is exactly zero. ("Zero the projection onto
each" in the paper is ambiguous for non-orthogonal vectors; design.md §6.2(2)
picks least-squares as the default and keeps sequential zeroing as a variant.)

All projection math runs in fp32; the residual is cast back to its original
dtype (bf16) at the end. Scoring for selection is done through the same
bf16 unembedding matmul the reference readout uses.
"""

from __future__ import annotations

import torch

from ablation.config import AblationConfig


class JSpaceAblator:
    """Holds the per-layer J matrices and applies the ablation to residuals.

    Stateless across forward passes except for accumulated statistics; the
    per-forward state (exclusion-filter token ids, active-position mask) is
    passed in by the hook session.
    """

    def __init__(
        self,
        cfg: AblationConfig,
        *,
        jacobians: dict[int, torch.Tensor] | None,
        unembed_weight: torch.Tensor,
        norm_weight: torch.Tensor | None,
        norm_eps: float = 1e-6,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.W_U = unembed_weight  # [V, d], model dtype, model device
        d = self.W_U.shape[1]
        self.d_model = d
        self.norm_eps = norm_eps

        g = norm_weight if norm_weight is not None else torch.ones(d)
        if not cfg.include_norm_weight:
            g = torch.ones_like(g)
        self.g32 = g.detach().to(self.device, torch.float32)

        needs_J = cfg.direction_source in ("jlens", "jlens_rankshift") or cfg.norm_match
        if needs_J:
            if jacobians is None:
                raise ValueError(f"{cfg.direction_source} needs a fitted lens")
            missing = [l for l in cfg.band if l not in jacobians]
            if missing:
                raise ValueError(f"lens missing band layers {missing}")
            self.J = {
                l: jacobians[l].detach().to(self.device, torch.float32)
                for l in cfg.band
            }
        else:
            self.J = {}

        if cfg.direction_source == "random":
            self.rand_dirs = {}
            for l in cfg.band:
                gen = torch.Generator().manual_seed(cfg.random_seed * 1000 + l)
                raw = torch.randn(d, cfg.k, generator=gen)
                q, _ = torch.linalg.qr(raw)  # [d, k], orthonormal columns
                self.rand_dirs[l] = q.T.contiguous().to(self.device)  # [k, d]

        self.capture_tokens = False
        self.captured: dict[int, torch.Tensor] = {}
        self.reset_stats()

    # ------------------------------------------------------------------ stats

    def reset_stats(self) -> None:
        self._stats: dict[int, dict[str, float]] = {
            l: {"delta_sq": 0.0, "h_sq": 0.0, "n": 0} for l in self.cfg.band
        }

    def stats_summary(self) -> dict[int, dict[str, float]]:
        """Per-layer mean relative norm removed: ||delta|| / ||h|| (RMS over positions)."""
        out = {}
        for l, s in self._stats.items():
            if s["n"]:
                out[l] = {
                    "rms_delta": (s["delta_sq"] / s["n"]) ** 0.5,
                    "rms_h": (s["h_sq"] / s["n"]) ** 0.5,
                    "rel": (s["delta_sq"] / max(s["h_sq"], 1e-30)) ** 0.5,
                    "n_positions": s["n"],
                }
        return out

    # -------------------------------------------------------------- selection

    def _lens_scores(self, h32: torch.Tensor, layer: int, *, use_jacobian: bool) -> torch.Tensor:
        """Readout scores [N, V], matching the reference unembed pipeline
        (transport -> RMSNorm(+g) -> W_U) with the matmul in W_U's dtype."""
        y = h32 @ self.J[layer].T if use_jacobian else h32
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        y = y * self.g32
        return (y.to(self.W_U.dtype) @ self.W_U.T).float()

    def _topk_ids(self, scores: torch.Tensor, filter_ids: torch.Tensor | None) -> torch.Tensor:
        """Top-k token ids per row, skipping exclusion-filter tokens."""
        cfg = self.cfg
        if filter_ids is not None:
            scores = scores.scatter(-1, filter_ids.to(scores.device), float("-inf"))
        if cfg.direction_source == "jlens_rankshift":
            ids = scores.topk(cfg.rank_shift + cfg.k, dim=-1).indices
            return ids[..., cfg.rank_shift:]
        return scores.topk(cfg.k, dim=-1).indices

    def _dirs_for_ids(self, ids: torch.Tensor, layer: int, *, use_jacobian: bool) -> torch.Tensor:
        """Direction rows A [N, k, d] for the given token ids: (g ⊙ W_U[v]) J_l."""
        w = self.W_U[ids].float() * self.g32  # [N, k, d]
        if use_jacobian:
            w = w @ self.J[layer]
        return w

    def select_directions(
        self, h32: torch.Tensor, layer: int, filter_ids: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Directions to remove at this layer: (A [N,k,d], token ids or None)."""
        cfg = self.cfg
        if cfg.direction_source == "random":
            A = self.rand_dirs[layer].unsqueeze(0).expand(h32.shape[0], -1, -1)
            return A, None
        use_j = cfg.direction_source in ("jlens", "jlens_rankshift")
        scores = self._lens_scores(h32, layer, use_jacobian=use_j)
        ids = self._topk_ids(scores, filter_ids)
        return self._dirs_for_ids(ids, layer, use_jacobian=use_j), ids

    # ------------------------------------------------------------- projection

    @staticmethod
    def _ls_delta(h32: torch.Tensor, A: torch.Tensor, ridge: float) -> torch.Tensor:
        """Least-squares projection of h onto span(rows of A): [N, d]."""
        b = torch.einsum("nkd,nd->nk", A, h32)
        G = A @ A.transpose(1, 2)  # [N, k, k]
        k = G.shape[-1]
        eye = torch.eye(k, device=G.device, dtype=G.dtype)
        scale = G.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).unsqueeze(-1)
        c = torch.linalg.solve(G + ridge * scale * eye, b.unsqueeze(-1)).squeeze(-1)
        return torch.einsum("nk,nkd->nd", c, A)

    @staticmethod
    def _sequential_delta(h32: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Zero projections one direction at a time, in selection order."""
        h = h32.clone()
        for i in range(A.shape[1]):
            a = A[:, i, :]
            coef = (h * a).sum(-1) / a.pow(2).sum(-1).clamp_min(1e-30)
            h = h - coef.unsqueeze(-1) * a
        return h32 - h

    def _delta(self, h32: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        if self.cfg.projection == "sequential":
            return self._sequential_delta(h32, A)
        return self._ls_delta(h32, A, self.cfg.ridge)

    # ---------------------------------------------------------------- ablate

    def ablate(
        self,
        hidden: torch.Tensor,
        layer: int,
        filter_ids: torch.Tensor | None,
        active: torch.Tensor | None,
    ) -> torch.Tensor:
        """Ablate a residual tensor [B, S, d] at `layer`.

        filter_ids: [B, S, F] clean-pass top-F token ids, or None (filter off).
        active:     [B, S] bool; positions where the ablation applies
                    (False for padding, and for prompt positions when
                    ablate_prompt is off). None means all positions.
        """
        B, S, d = hidden.shape
        flat = hidden.reshape(B * S, d)
        if active is not None:
            idx = active.reshape(B * S).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                return hidden
        else:
            idx = None

        h32_all = (flat[idx] if idx is not None else flat).float()
        f_all = None
        if filter_ids is not None:
            f_all = filter_ids.reshape(B * S, -1)
            if idx is not None:
                f_all = f_all[idx]

        # Chunk over positions: the [n, vocab] score tensor is ~600 MB per
        # 1024 positions in fp32, and prefill can have thousands of positions.
        CHUNK = 1024
        new_parts, cap_parts = [], []
        for i in range(0, h32_all.shape[0], CHUNK):
            h32 = h32_all[i : i + CHUNK]
            f = f_all[i : i + CHUNK] if f_all is not None else None

            A, ids = self.select_directions(h32, layer, f)
            delta = self._delta(h32, A)

            if self.cfg.direction_source == "random" and self.cfg.norm_match:
                # Rescale so ||delta|| per position equals what the jlens
                # ablation would have removed from this same residual.
                scores = self._lens_scores(h32, layer, use_jacobian=True)
                j_ids = self._topk_ids(scores, f)
                A_j = self._dirs_for_ids(j_ids, layer, use_jacobian=True)
                target = self._delta(h32, A_j).norm(dim=-1)
                scale = target / delta.norm(dim=-1).clamp_min(1e-12)
                delta = delta * scale.unsqueeze(-1)

            s = self._stats[layer]
            s["delta_sq"] += float(delta.pow(2).sum(-1).sum())
            s["h_sq"] += float(h32.pow(2).sum(-1).sum())
            s["n"] += h32.shape[0]
            if self.capture_tokens and ids is not None:
                cap_parts.append(ids.detach().cpu())
            new_parts.append((h32 - delta).to(hidden.dtype))

        if self.capture_tokens and cap_parts:
            self.captured[layer] = torch.cat(cap_parts, dim=0)

        src = torch.cat(new_parts, dim=0)
        out = flat.clone()
        if idx is not None:
            out[idx] = src
        else:
            out = src
        return out.reshape(B, S, d)

    # ----------------------------------------------------------- measurement

    def occupancy(
        self, hidden: torch.Tensor, layer: int, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """occ(v, l, t) = <h, j_v / ||j_v||> for given tokens.

        hidden: [N, d] residuals at `layer`; token_ids: [m] or [N, m].
        Returns [N, m]. This is the measurement side of the same coin as
        `ablate` — rung 1.5 reads it, the intervention zeroes it.
        """
        h32 = hidden.float().to(self.device)
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0).expand(h32.shape[0], -1)
        use_j = self.cfg.direction_source != "logitlens"
        A = self._dirs_for_ids(token_ids.to(self.W_U.device), layer, use_jacobian=use_j)
        A = A / A.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.einsum("nkd,nd->nk", A, h32)

    # ---------------------------------------------------------- construction

    @classmethod
    def from_hf(cls, model, lens, cfg: AblationConfig, device=None) -> "JSpaceAblator":
        """Build from a HF *ForCausalLM (Qwen3 layout) and a JacobianLens."""
        device = device or model.lm_head.weight.device
        return cls(
            cfg,
            jacobians=lens.jacobians if lens is not None else None,
            unembed_weight=model.lm_head.weight,
            norm_weight=model.model.norm.weight,
            norm_eps=getattr(model.config, "rms_norm_eps", 1e-6),
            device=device,
        )
