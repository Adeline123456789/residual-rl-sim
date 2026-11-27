"""Core algebra and accounting for CF-ResRL experiments.

This module does not implement ACT, ResFiT, simulator snapshots, training, or
robot evaluation. It exists to pin down the 2x2 branch semantics so that the
same quantities are used by collection, model training, and evaluation code.

Notation:
    V00: zero residual, keep cached chunk
    V01: zero residual, refresh chunk
    V10: execute candidate residual, keep cached chunk
    V11: execute candidate residual, refresh chunk

The first action must be identical within each keep/refresh pair. Refresh is a
request made after that action; any inference delay continues to advance the
simulator under the registered fallback / old-chunk execution rule.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable, Iterable


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class FactorialOutcome:
    """Matched 2x2 branch returns for one source snapshot and residual.

    Values must use the same horizon, reward definition, downstream policies,
    random-number protocol, and refresh-delay semantics. A replan cost is not
    baked into the four values; it is supplied when evaluating a decision.
    """

    v00: float
    v01: float
    v10: float
    v11: float

    def __post_init__(self) -> None:
        for name in ("v00", "v01", "v10", "v11"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))

    @property
    def base_refresh_gain(self) -> float:
        """G(0) = V01 - V00."""
        return self.v01 - self.v00

    @property
    def residual_refresh_gain(self) -> float:
        """G(delta) = V11 - V10."""
        return self.v11 - self.v10

    @property
    def interaction(self) -> float:
        """I(delta) = V11 - V10 - V01 + V00."""
        return self.v11 - self.v10 - self.v01 + self.v00

    def verify_identity(self, *, atol: float = 1e-12) -> bool:
        """Check G(delta) = G(0) + I(delta)."""
        if atol < 0 or not math.isfinite(atol):
            raise ValueError("atol must be finite and nonnegative")
        return math.isclose(
            self.residual_refresh_gain,
            self.base_refresh_gain + self.interaction,
            abs_tol=atol,
            rel_tol=0.0,
        )

    def oracle_refresh(self, cost: float, *, residual: bool = True) -> bool:
        """Return the matched-branch optimal KEEP/REFRESH decision.

        Ties choose KEEP to avoid an unnecessary policy call. This oracle is
        only a diagnostic label from already collected branches; it is not a
        deployable policy and is not evidence of global optimality.
        """
        c = _finite("cost", cost)
        keep = self.v10 if residual else self.v00
        refresh = (self.v11 if residual else self.v01) - c
        return refresh > keep

    def decision_regret(
        self, refresh: bool, cost: float, *, residual: bool = True
    ) -> float:
        """Matched value loss relative to the local branch oracle."""
        c = _finite("cost", cost)
        keep = self.v10 if residual else self.v00
        refresh_value = (self.v11 if residual else self.v01) - c
        chosen = refresh_value if bool(refresh) else keep
        return max(keep, refresh_value) - chosen


@dataclass(frozen=True)
class BranchBudget:
    """Interaction accounting for equal-budget model comparisons.

    A four-branch factorial context costs four branch rollouts. A direct
    residual-conditioned target needs two branches (delta+KEEP, delta+REFRESH).
    Environment control steps should also be reported because suffix lengths
    can differ; this object only covers branch-count accounting.
    """

    total_branch_rollouts: int

    def __post_init__(self) -> None:
        if isinstance(self.total_branch_rollouts, bool) or self.total_branch_rollouts < 0:
            raise ValueError("total_branch_rollouts must be a nonnegative integer")

    @property
    def factorial_contexts(self) -> int:
        return self.total_branch_rollouts // 4

    @property
    def direct_two_branch_contexts(self) -> int:
        return self.total_branch_rollouts // 2

    @property
    def factorial_unused(self) -> int:
        return self.total_branch_rollouts % 4

    @property
    def direct_unused(self) -> int:
        return self.total_branch_rollouts % 2


def assert_disjoint_groups(
    train_groups: Iterable[Hashable],
    validation_groups: Iterable[Hashable],
    test_groups: Iterable[Hashable],
) -> None:
    """Fail on episode/snapshot leakage across dataset partitions."""
    train, validation, test = set(train_groups), set(validation_groups), set(test_groups)
    overlap = {
        "train-validation": train & validation,
        "train-test": train & test,
        "validation-test": validation & test,
    }
    bad = {name: values for name, values in overlap.items() if values}
    if bad:
        raise ValueError(f"Group leakage detected: {bad}")
