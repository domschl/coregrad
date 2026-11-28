"""Optimizers for coregrad: fused AdamW and hybrid Muon + AdamW."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz iteration to orthogonalize a 2D gradient matrix."""
    assert g.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16() if g.is_cuda else g.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-schulz for 2D hidden weight matrices."""

    def __init__(
        self,
        params: Any,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            wd = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1.0 - beta)
                update = g.lerp(buf, beta) if nesterov else buf
                orth = zeropower_via_newtonschulz5(update, steps=ns_steps)
                # Keller Jordan aspect-ratio scaling for semi-orthogonal NS updates (muon_lr ~ 0.02)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(orth, alpha=-lr * scale)
        return loss


class HybridOptimizer:
    """Wraps one or more PyTorch optimizers with a unified step/zero_grad/state_dict API."""

    def __init__(self, optimizers: list[torch.optim.Optimizer], base_lrs: list[float]) -> None:
        self.optimizers = optimizers
        self.base_lrs = base_lrs

    def set_lr_scale(self, scale: float) -> None:
        for opt, base_lr in zip(self.optimizers, self.base_lrs, strict=True):
            for pg in opt.param_groups:
                pg["lr"] = base_lr * scale

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def state_dict(self) -> dict[str, Any]:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for opt, sd in zip(self.optimizers, state["optimizers"], strict=True):
            opt.load_state_dict(sd)


def build_optimizer(
    model: nn.Module,
    *,
    optimizer_type: str = "adamw",
    lr: float = 6e-4,
    muon_lr: float = 0.02,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    use_fused: bool = True,
) -> HybridOptimizer:
    """Create either fused AdamW or hybrid Muon (2D hidden) + AdamW (embed/norms)."""
    seen: set[int] = set()
    muon_params: list[nn.Parameter] = []
    decay_params: list[nn.Parameter] = []
    nodecay_params: list[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))

        is_embed_or_head = (
            "tok_emb" in name or "lm_head" in name or "loop_emb" in name or "router." in name
        )
        if optimizer_type == "muon" and p.ndim == 2 and not is_embed_or_head:
            muon_params.append(p)
        elif p.ndim >= 2 and "loop_emb" not in name:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    fused_ok = use_fused and any(p.is_cuda for p in model.parameters())
    adam_groups: list[dict[str, Any]] = []
    if decay_params:
        adam_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if nodecay_params:
        adam_groups.append({"params": nodecay_params, "weight_decay": 0.0})

    adamw = torch.optim.AdamW(adam_groups, lr=lr, betas=betas, eps=1e-8, fused=fused_ok)

    if optimizer_type == "adamw":
        return HybridOptimizer([adamw], [lr])
    if optimizer_type == "muon":
        muon = Muon(muon_params, lr=muon_lr, momentum=0.95, weight_decay=weight_decay)
        return HybridOptimizer([muon, adamw], [muon_lr, lr])
    raise ValueError(f"Unknown optimizer_type: {optimizer_type!r} (expected 'adamw' or 'muon')")
