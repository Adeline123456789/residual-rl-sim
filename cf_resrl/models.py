"""CPU probes for direct and factorial refresh gains."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


_METHODS = {
    "norm": "Norm",
    "linear": "Linear",
    "linear-full": "Linear",
    "b5": "B5",
    "b6": "B6",
    "p1": "P1",
}
_FOUR_BRANCH_METHODS = ("B5", "P1")


def _matrix(values, name, *, allow_empty=False):
    matrix = torch.as_tensor(values, device="cpu", dtype=torch.float32)
    if matrix.ndim != 2 or not matrix.shape[1] or (not allow_empty and not matrix.shape[0]):
        raise ValueError(f"{name} must be a matrix with at least one feature")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    return matrix


def _target(values, rows, name):
    target = torch.as_tensor(values, device="cpu", dtype=torch.float32)
    if target.shape == (rows, 1):
        target = target.squeeze(1)
    if target.shape != (rows,) or not torch.isfinite(target).all():
        raise ValueError(f"{name} must have one finite gain per source group")
    return target


def _scale(values):
    std = values.std(dim=0, unbiased=False)
    return torch.where(std > 1e-6, std, torch.ones_like(std))


def _mlp(in_features, width):
    return nn.Sequential(
        nn.Linear(in_features, width),
        nn.SiLU(),
        nn.Linear(width, 1),
    )


class Probe(nn.Module):
    """Share the same two-tower capacity across B5, B6 and P1."""

    def __init__(self, method: str, context_dim: int, residual_dim: int, width: int):
        super().__init__()
        self.method = method
        self.context_dim = context_dim
        self.residual_dim = residual_dim
        self.width = width
        input_residual_dim = 1 if method == "Norm" else residual_dim
        self.register_buffer("context_mean", torch.zeros(context_dim))
        self.register_buffer("context_scale", torch.ones(context_dim))
        self.register_buffer("residual_mean", torch.zeros(input_residual_dim))
        self.register_buffer("residual_scale", torch.ones(input_residual_dim))
        self.register_buffer("target_mean", torch.zeros(()))
        self.register_buffer("target_scale", torch.ones(()))
        if method == "Linear":
            self.direct = nn.Linear(context_dim + residual_dim, 1)
        elif method == "Norm":
            self.direct = _mlp(context_dim + input_residual_dim, width)
        else:
            self.base_head = _mlp(context_dim, width)
            self.residual_head = _mlp(context_dim + residual_dim, width)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _residual_inputs(self, residual):
        if self.method == "Norm":
            return torch.linalg.vector_norm(residual, dim=1, keepdim=True)
        return residual

    def _normalized_gains(self, context, residual):
        context = (context - self.context_mean) / self.context_scale
        residual = (self._residual_inputs(residual) - self.residual_mean) / self.residual_scale
        combined = torch.cat((context, residual), dim=1)
        if self.method in ("Norm", "Linear"):
            return {"g_delta": self.direct(combined).squeeze(1)}
        base = self.base_head(context).squeeze(1)
        at_delta = self.residual_head(combined).squeeze(1)
        if self.method == "B6":
            return {"g_delta": base + at_delta}
        normalized_zero = (-self.residual_mean / self.residual_scale).expand_as(residual)
        at_zero = self.residual_head(torch.cat((context, normalized_zero), dim=1)).squeeze(1)
        if self.method == "P1":
            interaction = at_delta - at_zero
            return {"g_delta": base + interaction, "g_zero": base, "interaction": interaction}
        return {"g_delta": base + at_delta, "g_zero": base + at_zero, "interaction": at_delta - at_zero}

    def forward(self, context, residual):
        return {
            name: gain * self.target_scale + (0 if name == "interaction" else self.target_mean)
            for name, gain in self._normalized_gains(context, residual).items()
        }


def fit_probe(
    method, context, residual, target_g0, target_g_delta, *, steps, batch_size,
    learning_rate, seed, width,
) -> Probe:
    """Fit train-only preprocessing and carry out a fixed number of Adam steps."""
    canonical = _METHODS.get(method.casefold())
    if canonical is None:
        raise ValueError(f"Unknown refresh gain probe: {method}")
    if any(type(value) is not int or value <= 0 for value in (steps, batch_size, width)):
        raise ValueError("steps, batch_size and width must be positive integers")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    context = _matrix(context, "context")
    residual = _matrix(residual, "residual")
    rows = len(context)
    if len(residual) != rows:
        raise ValueError("context and residual must align by source group")
    target_delta = _target(target_g_delta, rows, "target_g_delta")
    four_branch = canonical in _FOUR_BRANCH_METHODS
    if four_branch and target_g0 is None:
        raise ValueError(f"{canonical} requires the paired zero-residual gain")
    target_zero = _target(target_g0, rows, "target_g0") if four_branch else None
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        probe = Probe(canonical, context.shape[1], residual.shape[1], width)

    residual_input = probe._residual_inputs(residual)
    all_targets = torch.cat((target_zero, target_delta)) if four_branch else target_delta
    probe.context_mean.copy_(context.mean(dim=0))
    probe.context_scale.copy_(_scale(context))
    probe.residual_mean.copy_(residual_input.mean(dim=0))
    probe.residual_scale.copy_(_scale(residual_input))
    probe.target_mean.copy_(all_targets.mean())
    probe.target_scale.copy_(_scale(all_targets))
    target_delta = (target_delta - probe.target_mean) / probe.target_scale
    if four_branch:
        target_zero = (target_zero - probe.target_mean) / probe.target_scale

    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    sampler = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        sample = torch.randint(rows, (batch_size,), generator=sampler)
        gains = probe._normalized_gains(context[sample], residual[sample])
        loss = F.mse_loss(gains["g_delta"], target_delta[sample])
        if four_branch:
            loss = (loss + F.mse_loss(gains["g_zero"], target_zero[sample])) / 2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in probe.parameters()):
        raise RuntimeError(f"{canonical} produced non-finite parameters")
    return probe.eval()


@torch.inference_mode()
def predict_probe(probe: Probe, context, residual):
    context = _matrix(context, "context", allow_empty=True)
    residual = _matrix(residual, "residual", allow_empty=True)
    if context.shape != (len(residual), probe.context_dim) or residual.shape[1] != probe.residual_dim:
        raise ValueError("prediction inputs differ from the fitted feature dimensions")
    return {name: value.numpy() for name, value in probe(context, residual).items()}


def save_probe(probe: Probe, path: str | Path) -> None:
    """Write fitted probe weights and train-only preprocessing."""
    torch.save(
        {
            "method": probe.method,
            "context_dim": probe.context_dim,
            "residual_dim": probe.residual_dim,
            "width": probe.width,
            "state": probe.state_dict(),
        },
        path,
    )


def load_probe(path: str | Path) -> Probe:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        probe = Probe(payload["method"], payload["context_dim"], payload["residual_dim"], payload["width"])
    probe.load_state_dict(payload["state"], strict=True)
    return probe.eval()
