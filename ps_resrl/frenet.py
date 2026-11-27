"""Frenet tangential-normal residual decomposition and phase-hold rule."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class FrenetDecomposition:
    """Orthogonal decomposition of translational residual relative to base motion."""

    tangent: tuple[float, float, float]
    delta_parallel: tuple[float, float, float]
    delta_perp: tuple[float, float, float]
    parallel_signed: float
    perp_norm: float
    brake_ratio: float


def _vec3(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(values) < 3:
        raise ValueError(f"{name} must have at least 3 translational components")
    result = (float(values[0]), float(values[1]), float(values[2]))
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"{name} translational components must be finite")
    return result


def frenet_decompose(
    base_action: Sequence[float],
    residual_action: Sequence[float],
    *,
    eps: float = 1e-6,
    eps0: float = 1e-3,
) -> FrenetDecomposition:
    """Decompose translational residual along and orthogonal to the base action."""
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    if eps0 <= 0.0 or not math.isfinite(eps0):
        raise ValueError("eps0 must be finite and positive")

    vb = _vec3(base_action, "base_action")
    vd = _vec3(residual_action, "residual_action")

    b_norm_sq = vb[0] * vb[0] + vb[1] * vb[1] + vb[2] * vb[2]
    b_norm = math.sqrt(b_norm_sq)
    denom = b_norm + eps
    tangent = (vb[0] / denom, vb[1] / denom, vb[2] / denom)

    dot_t = vd[0] * tangent[0] + vd[1] * tangent[1] + vd[2] * tangent[2]
    delta_parallel = (dot_t * tangent[0], dot_t * tangent[1], dot_t * tangent[2])
    delta_perp = (
        vd[0] - delta_parallel[0],
        vd[1] - delta_parallel[1],
        vd[2] - delta_parallel[2],
    )
    perp_norm = math.sqrt(
        delta_perp[0] * delta_perp[0]
        + delta_perp[1] * delta_perp[1]
        + delta_perp[2] * delta_perp[2]
    )
    dot_b = vd[0] * vb[0] + vd[1] * vb[1] + vd[2] * vb[2]
    brake_ratio = dot_b / (b_norm_sq + eps0)
    return FrenetDecomposition(
        tangent=tangent,
        delta_parallel=delta_parallel,
        delta_perp=delta_perp,
        parallel_signed=dot_t,
        perp_norm=perp_norm,
        brake_ratio=brake_ratio,
    )


def phase_hold_velocity(
    base_action: Sequence[float],
    residual_action: Sequence[float],
    *,
    kappa: float = 1.0,
    nu_min: float = 0.0,
    nu_max: float = 1.0,
    eps0: float = 1e-3,
) -> float:
    """Compute analytical phase velocity that brakes chunk clock when residual opposes base."""
    if not math.isfinite(kappa) or kappa < 0.0:
        raise ValueError("kappa must be finite and nonnegative")
    if not (math.isfinite(nu_min) and math.isfinite(nu_max)) or nu_min > nu_max:
        raise ValueError("nu_min and nu_max must be finite with nu_min <= nu_max")
    decomp = frenet_decompose(base_action, residual_action, eps0=eps0)
    raw = 1.0 + kappa * min(0.0, decomp.brake_ratio)
    return min(nu_max, max(nu_min, raw))
