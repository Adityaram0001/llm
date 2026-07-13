"""Lion (Chen et al. '23) and Muon (Jordan '24) optimizers, phase 5 Wave D.

Both are drop-in `torch.optim.Optimizer` subclasses so `Trainer` can mix-and-match them with
plain `torch.optim.AdamW` via config (`OptimConfig.optimizer`). Muon only makes sense for 2D
"hidden" weight matrices (see `build_muon_param_groups` in `trainer.py`) -- embeddings, the LM
head, and norm gains keep using AdamW regardless of `optimizer="muon"`, per the nanoGPT
speedrun recipe the phase-5 spec names.
"""

from __future__ import annotations

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration (Jordan '24) approximating `G`'s nearest semi-orthogonal
    matrix (i.e. the U V^T of its SVD, with every singular value pushed to ~1) using only
    matmuls -- no actual SVD. The quintic map x -> ax + bx^3 + cx^5, with these specific
    (a, b, c), converges singular values in [0, 1] towards 1 in ~5 iterations (the coefficients
    are chosen so the map's derivative is flat near 1, unlike the naive x -> 1.5x - 0.5x^3
    Newton iteration for the matrix sign function, which converges much slower near 0).
    Runs in bf16 (Muon's update doesn't need fp32 precision here, and this is the hot loop)."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon: SGD-momentum whose update is orthogonalized (via Newton-Schulz) before the step,
    for 2D hidden weight matrices only. Orthogonalizing means every singular direction of the
    weight moves by ~the same amount per step, instead of the update being dominated by
    whichever direction the raw gradient happens to be largest in -- the claimed reason Muon
    trains faster than AdamW on hidden matrices in the nanoGPT speedrun results this ablation
    is checking. The `max(1, fan_out/fan_in)**0.5` scale keeps the update's RMS magnitude
    roughly matched across non-square matrices (attention/FFN projections aren't always
    square), matching Keller Jordan's reference implementation.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2D parameters (hidden weight matrices), got shape {tuple(g.shape)}"
                    )
                state = self.state[p]
                buf = state.setdefault("momentum_buffer", torch.zeros_like(g))
                buf.mul_(group["momentum"]).add_(g)
                update = buf.add(g, alpha=group["momentum"]) if group["nesterov"] else buf
                update = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])
                update = update * max(1.0, g.size(0) / g.size(1)) ** 0.5
                p.add_(update, alpha=-group["lr"])
        return loss


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al. '23, "Symbolic Discovery of Optimization Algorithms"): the update is
    `sign(beta1 * m + (1 - beta1) * g)`, so every parameter moves by the same step *size* each
    update -- only the sign varies, and only one momentum buffer is kept (half of AdamW's
    optimizer-state memory). Because the update magnitude no longer reflects the gradient's
    actual scale, the paper recommends an lr ~3-10x smaller and a weight_decay ~3-10x larger
    than the AdamW recipe it replaces -- pass those directly via `OptimConfig.lr`/
    `weight_decay` for a Lion run rather than reusing the AdamW config verbatim."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                m = state.setdefault("momentum", torch.zeros_like(p))
                if group["weight_decay"] != 0.0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                update = (m * beta1 + g * (1 - beta1)).sign()
                p.add_(update, alpha=-group["lr"])
                m.mul_(beta2).add_(g, alpha=1 - beta2)
        return loss
