"""Summarize matched Square branch outcomes and the interactions used to collect them."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from itertools import accumulate
import math
import random
from typing import Any, Iterable, Mapping, Sequence

from cf_resrl.factorial import FactorialOutcome


_BRANCHES = {
    "factorial": ("V00", "V01", "V10", "V11"),
    "pair": ("V10", "V11"),
}
_QUANTILES = {
    "min": 0.0, "q05": 0.05, "q25": 0.25, "median": 0.5,
    "q75": 0.75, "q95": 0.95, "max": 1.0,
}


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _resources(rows: Iterable[tuple[int, int, float]]) -> dict[str, int | float]:
    rows = list(rows)
    return {
        "environment_steps": sum(row[0] for row in rows),
        "act_calls": sum(row[1] for row in rows),
        "wall_seconds": math.fsum(row[2] for row in rows),
    }


def _source_resources(row: Mapping[str, Any]) -> tuple[int, int, float]:
    steps = _nonnegative_int(row["source_episode_steps"], "source_episode_steps")
    if not steps:
        raise ValueError("A collected source episode must contain a control step")
    return (
        steps,
        _nonnegative_int(row["source_episode_act_calls"], "source_episode_act_calls"),
        _nonnegative_float(row["source_episode_wall_seconds"], "source_episode_wall_seconds"),
    )


def _branch_resources(branch: Mapping[str, Any]) -> tuple[int, int, float]:
    steps = _nonnegative_int(branch["environment_steps"], "branch environment_steps")
    if not steps:
        raise ValueError("A collected branch must contain a control step")
    return (
        steps,
        _nonnegative_int(branch["act_calls"], "branch act_calls"),
        _nonnegative_float(branch["wall_seconds"], "branch wall_seconds"),
    )


def _budget(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    branch_costs = defaultdict(list)
    for row in rows:
        for label, branch in row["branches"].items():
            branch_costs[label].append(_branch_resources(branch))
    return {
        "source_scout": {"episodes": len(rows), **_resources(_source_resources(row) for row in rows)},
        "branches": {
            "rollouts": sum(map(len, branch_costs.values())),
            **_resources(cost for costs in branch_costs.values() for cost in costs),
            "by_label": {
                label: {"rollouts": len(costs), **_resources(costs)}
                for label, costs in sorted(branch_costs.items())
            },
        },
    }


def _quantile(
    sorted_values: Sequence[float], fraction: float, cumulative_weights: list[int] | None = None
) -> float:
    # Interpolate the order statistics of the empirical distribution. Integer
    # weights represent repeated observations without materializing them.
    count = cumulative_weights[-1] if cumulative_weights is not None else len(sorted_values)
    rank = (count - 1) * fraction
    lower, upper = math.floor(rank), math.ceil(rank)
    if cumulative_weights is not None:
        lower_index = bisect_right(cumulative_weights, lower)
        upper_index = bisect_right(cumulative_weights, upper)
    else:
        lower_index, upper_index = lower, upper
    value = sorted_values[lower_index]
    return value + (sorted_values[upper_index] - value) * (rank - lower)


def _weighted_mean(values: tuple[float, ...], weights: list[int]) -> float:
    return math.fsum(value * weight for value, weight in zip(values, weights)) / sum(weights)


def _bootstrap_intervals(
    columns: list[tuple[float, ...]],
    weights: list[int],
    *,
    samples: int,
    seed: str,
) -> list[list[float]]:
    rng = random.Random(seed)
    population = range(len(weights))
    distributions: list[list[float]] = [[] for _ in columns]
    weighted = [tuple(value * weight for value, weight in zip(column, weights)) for column in columns]
    for _ in range(samples):
        indices = rng.choices(population, k=len(weights))
        total_weight = sum(weights[index] for index in indices)
        for saved, column in zip(distributions, weighted):
            saved.append(math.fsum(column[index] for index in indices) / total_weight)
    for values in distributions:
        values.sort()
    return [[_quantile(values, 0.025), _quantile(values, 0.975)] for values in distributions]


def _estimate(
    outcomes: list[FactorialOutcome],
    weights: list[int],
    *,
    costs: list[float],
    bootstrap_samples: int,
    bootstrap_seed: str,
) -> dict[str, Any]:
    gains = [
        tuple(o.base_refresh_gain for o in outcomes),
        tuple(o.residual_refresh_gain for o in outcomes),
        tuple(o.interaction for o in outcomes),
    ]
    flips = []
    decision_flips = []
    for cost in costs:
        base_refresh = tuple(gain > cost for gain in gains[0])
        residual_refresh = tuple(gain > cost for gain in gains[1])
        flip = tuple(base != residual for base, residual in zip(base_refresh, residual_refresh))
        flips.append(flip)
        decision_flips.append({
            "cost": cost,
            "flip_rate": _weighted_mean(flip, weights),
            "base_refresh_rate": _weighted_mean(base_refresh, weights),
            "residual_refresh_rate": _weighted_mean(residual_refresh, weights),
            "keep_to_refresh_rate": _weighted_mean(
                tuple(not base and residual for base, residual in zip(base_refresh, residual_refresh)), weights
            ),
            "refresh_to_keep_rate": _weighted_mean(
                tuple(base and not residual for base, residual in zip(base_refresh, residual_refresh)), weights
            ),
        })
    intervals = _bootstrap_intervals(
        gains + flips, weights, samples=bootstrap_samples, seed=bootstrap_seed
    )
    result = {"source_episodes": len(outcomes), "total_weight": sum(weights)}
    for name, values, interval in zip(("g0", "g_delta", "interaction"), gains, intervals):
        ordered = sorted(zip(values, weights))
        sorted_values, sorted_weights = zip(*ordered)
        cumulative = list(accumulate(sorted_weights))
        result[name] = {
            "mean": _weighted_mean(values, weights),
            "mean_ci95": interval,
            **{key: _quantile(sorted_values, fraction, cumulative) for key, fraction in _QUANTILES.items()},
        }
    for item, interval in zip(decision_flips, intervals[3:]):
        item["flip_rate_ci95"] = interval
    result["decision_flips"] = decision_flips
    return result


def summarize_e0(
    rows: Iterable[Mapping[str, Any]],
    *,
    costs: Iterable[float],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Aggregate one collected group per source episode.

    Each bootstrap draw samples entire source episodes within a distribution.
    The control-tick estimate weights the uniformly drawn natural ticks by their
    source episode lengths. Enriched estimates characterize the enrichment rule.
    """
    costs = [_nonnegative_float(cost, "refresh cost") for cost in costs]
    if not costs or len(set(costs)) != len(costs):
        raise ValueError("Expected distinct registered refresh costs")
    if not _nonnegative_int(bootstrap_samples, "bootstrap_samples"):
        raise ValueError("bootstrap_samples must be positive")
    _nonnegative_int(bootstrap_seed, "bootstrap_seed")

    checked = []
    by_bank = defaultdict(list)
    by_distribution = defaultdict(list)
    by_split = defaultdict(list)
    episode_ids = set()
    for row in rows:
        episode_id = row["source_episode_id"]
        if type(episode_id) not in (str, int) or episode_id == "" or episode_id in episode_ids:
            raise ValueError(f"Invalid or duplicate source episode ID: {episode_id!r}")
        episode_ids.add(episode_id)
        distribution, split, bank = row["distribution"], row["split"], row["bank"]
        if distribution not in ("natural", "enriched") or not isinstance(split, str) or not split:
            raise ValueError("Expected a natural or enriched distribution and a split name")
        if bank not in _BRANCHES or set(row["branches"]) != set(_BRANCHES[bank]):
            raise ValueError(f"Unexpected branch labels for {bank!r} bank")
        for branch in row["branches"].values():
            if not math.isfinite(branch["discounted_return"]):
                raise ValueError("Branch discounted_return must be finite")
        checked.append(row)
        by_bank[bank].append(row)
        by_distribution[distribution].append(row)
        by_split[split].append(row)

    distributions = {}
    for distribution, group in sorted(by_distribution.items()):
        factorial_rows = sorted(
            (row for row in group if row["bank"] == "factorial"),
            key=lambda row: (type(row["source_episode_id"]).__name__, row["source_episode_id"]),
        )
        if not factorial_rows:
            continue
        outcomes = [
            FactorialOutcome(*(row["branches"][label]["discounted_return"] for label in _BRANCHES["factorial"]))
            for row in factorial_rows
        ]
        options = {
            "costs": costs,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": f"{bootstrap_seed}:{distribution}",
        }
        estimates = {"episode_uniform": _estimate(outcomes, [1] * len(outcomes), **options)}
        if distribution == "natural":
            weights = [_source_resources(row)[0] for row in factorial_rows]
            estimates["control_tick_weighted"] = _estimate(outcomes, weights, **options)
        distributions[distribution] = estimates
    return {
        "source_episodes": len(checked),
        "bootstrap": {
            "unit": "source_episode",
            "stratified_by": "distribution",
            "interval": "percentile_95",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "decision_rule": "refresh iff gain > cost",
        "budgets": {
            "overall": _budget(checked),
            "by_bank": {key: _budget(group) for key, group in sorted(by_bank.items())},
            "by_distribution": {key: _budget(group) for key, group in sorted(by_distribution.items())},
            "by_split": {key: _budget(group) for key, group in sorted(by_split.items())},
        },
        "distributions": distributions,
    }
