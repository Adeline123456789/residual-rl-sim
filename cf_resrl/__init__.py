"""Dependency-free contracts for CF-ResRL factorial supervision experiments."""

from .factorial import (
    BranchBudget,
    FactorialOutcome,
    assert_disjoint_groups,
)

__all__ = ["BranchBudget", "FactorialOutcome", "assert_disjoint_groups"]
