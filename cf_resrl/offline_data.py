"""Verified deployment-visible features and episode budgets for Square probes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

import numpy as np

from .dataset import sha256_file
from .factorial import BranchBudget, FactorialOutcome, assert_disjoint_groups


CONTEXT_FIELDS = ("visual", "state", "base", "chunk_suffix", "chunk_mask", "tick", "chunk_index")
FEATURE_SHAPES = {
    "visual": (128,),
    "state": (9,),
    "base": (7,),
    "chunk_suffix": (20, 7),
    "chunk_mask": (20,),
    "tick": (1,),
    "chunk_index": (1,),
    "residual": (7,),
}
FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.0)
DISTRIBUTIONS = ("natural", "enriched")
FACTORIAL_BRANCHES = ("V00", "V01", "V10", "V11")
PAIR_BRANCHES = ("V10", "V11")
CONTEXT_DIM = sum(math.prod(FEATURE_SHAPES[field]) for field in CONTEXT_FIELDS)
RESIDUAL_DIM = FEATURE_SHAPES["residual"][0]


@dataclass(frozen=True)
class _Episode:
    row: dict
    context: np.ndarray
    residual: np.ndarray

    @property
    def source_episode_id(self) -> str:
        return self.row["source_episode_id"]


@dataclass(frozen=True)
class E0Dataset:
    _episodes: tuple[_Episode, ...]
    provenance: dict
    feature_schema: dict


@dataclass(frozen=True)
class ProbeSlice:
    source_episode_ids: tuple[str, ...]
    context: np.ndarray
    residual: np.ndarray
    g0: np.ndarray | None
    g_delta: np.ndarray
    v10: np.ndarray
    v11: np.ndarray
    budget: dict


@dataclass(frozen=True)
class BudgetSelection:
    factorial: ProbeSlice
    b6_rollout: ProbeSlice
    b6_joint: ProbeSlice
    validation: ProbeSlice
    test: ProbeSlice
    manifest: dict


def pack_features(mapping: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Copy the pre-action, deployment-visible fields into a frozen layout."""
    if set(mapping) != set(FEATURE_SHAPES):
        raise ValueError("Expected exactly the registered deployment-visible E0 feature fields")
    values = {}
    for field, shape in FEATURE_SHAPES.items():
        array = np.asarray(mapping[field])
        if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Expected finite float32 {field} with shape {shape}")
        values[field] = array
    context = np.concatenate([values[field].reshape(-1) for field in CONTEXT_FIELDS])
    return context, values["residual"].copy()


def _verify_resources(resource: dict, *, source: bool = False) -> None:
    prefix = "source_episode_" if source else ""
    steps = resource[prefix + ("steps" if source else "environment_steps")]
    calls = resource[prefix + "act_calls"]
    seconds = resource[prefix + "wall_seconds"]
    if (
        type(steps) is not int or steps <= 0
        or type(calls) is not int or calls < 0
        or type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds < 0
    ):
        raise ValueError("Invalid measured E0 interaction resources")


def _read_verified_feature(attempt: Path, row: dict) -> tuple[np.ndarray, np.ndarray]:
    episode_id = row["source_episode_id"]
    expected = Path("features") / f"{episode_id}.npz"
    path = (attempt / expected).resolve()
    if row["features"] != expected.as_posix() or path.parent != attempt / "features":
        raise ValueError(f"Unexpected feature path for {episode_id}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != row["features_sha256"]:
        raise ValueError(f"Feature checksum mismatch for {episode_id}")
    with np.load(io.BytesIO(data), allow_pickle=False) as mapping:
        return pack_features(mapping)


def load_e0(attempt: str | Path) -> E0Dataset:
    """Verify a completed E0 run and load only the model-visible feature files."""
    attempt = Path(attempt).resolve()
    status = json.loads((attempt / "status.json").read_bytes())
    index_data = (attempt / "group_index.jsonl").read_bytes()
    index_sha = hashlib.sha256(index_data).hexdigest()
    summary_sha = sha256_file(attempt / "summary.json")
    if status["status"] != "COMPLETED" or (
        status["index_sha256"], status["summary_sha256"]
    ) != (index_sha, summary_sha):
        raise ValueError("E0 attempt is incomplete or index/summary checksums differ")

    manifest_path = attempt / "run_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    schema_path = attempt / "feature_schema.json"
    schema = json.loads(schema_path.read_bytes())
    expected_schema = {
        field: {"dtype": "float32", "shape": list(shape)} for field, shape in FEATURE_SHAPES.items()
    }
    if schema != expected_schema:
        raise ValueError("E0 feature manifest differs from the frozen visible schema")

    rows = [json.loads(line) for line in index_data.splitlines()]
    plan = manifest["source_plan"]
    if not len(rows) == len(plan) == status["planned_groups"] == status["completed_groups"]:
        raise ValueError("E0 episode count differs from the source plan")
    protocol = manifest["protocol"]
    if protocol["task"] != "Square" or protocol["refresh_delay_steps"] != 2:
        raise ValueError("Expected the registered Square protocol and two-tick refresh")
    counts = Counter()
    seen_seeds = set()
    episodes = []
    for row, planned in zip(rows, plan, strict=True):
        distribution, bank, split = row["distribution"], row["bank"], row["split"]
        seed = row["seed"]
        if (
            type(seed) is not int
            or seed in seen_seeds
            or row["source_episode_id"] != f"square_e0_{seed}"
            or any(row[key] != planned[key] for key in (
                "seed", "source_episode_id", "distribution", "bank", "split", "selector_seed",
            ))
            or row["selector_seed"] != protocol["selector_seed"] + seed
            or distribution not in DISTRIBUTIONS
            or bank not in ("factorial", "pair")
            or split not in ("train", "validation", "test")
            or (bank == "pair" and split != "train")
        ):
            raise ValueError(f"Invalid or duplicate E0 episode assignment for seed {seed}")
        seen_seeds.add(seed)
        cohort = protocol["factorial_cohorts" if bank == "factorial" else "pair_training_cohorts"][distribution]
        if not cohort["first_seed"] <= seed < cohort["first_seed"] + cohort["size"]:
            raise ValueError(f"E0 episode seed {seed} leaves its registered cohort")
        counts[distribution, bank, split] += 1
        labels = FACTORIAL_BRANCHES if bank == "factorial" else PAIR_BRANCHES
        if set(row["branches"]) != set(labels):
            raise ValueError(f"E0 episode {seed} has an unexpected branch set")
        _verify_resources(row, source=True)
        if type(row["source_tick"]) is not int or not 0 <= row["source_tick"] < row["source_episode_steps"]:
            raise ValueError(f"E0 episode {seed} has an invalid source tick")
        for label, branch in row["branches"].items():
            _verify_resources(branch)
            if (
                branch["label"] != label
                or branch["environment_steps"] > branch["remaining_horizon"]
                or type(branch["discounted_return"]) not in (float, int)
                or not math.isfinite(branch["discounted_return"])
            ):
                raise ValueError(f"Invalid branch {label} for E0 episode {seed}")
        context, residual = _read_verified_feature(attempt, row)
        episodes.append(_Episode(row, context, residual))

    for distribution in DISTRIBUTIONS:
        for split, count in protocol["factorial_split_counts"].items():
            if counts[distribution, "factorial", split] != count:
                raise ValueError(f"Unexpected {distribution} / {split} factorial cohort size")
        if counts[distribution, "pair", "train"] != protocol["pair_training_cohorts"][distribution]["size"]:
            raise ValueError(f"Unexpected {distribution} pair-only cohort size")
    assert_disjoint_groups(*(
        [episode.source_episode_id for episode in episodes if episode.row["split"] == split]
        for split in ("train", "validation", "test")
    ))
    return E0Dataset(
        _episodes=tuple(episodes),
        provenance={
            "attempt": str(attempt),
            "status_sha256": sha256_file(attempt / "status.json"),
            "index_sha256": index_sha,
            "summary_sha256": summary_sha,
            "run_manifest_sha256": sha256_file(manifest_path),
            "feature_schema_sha256": sha256_file(schema_path),
            "dataset": protocol["dataset"],
            "identities": manifest["identities"],
        },
        feature_schema={
            "context_fields": list(CONTEXT_FIELDS),
            "context_dim": CONTEXT_DIM,
            "residual_field": "residual",
            "residual_dim": RESIDUAL_DIM,
            "fields": expected_schema,
        },
    )


def _budget(episodes: Sequence[_Episode], labels: tuple[str, ...]) -> dict:
    rows = [episode.row for episode in episodes]
    branches = [row["branches"][label] for row in rows for label in labels]
    return {
        "source_episodes": [row["source_episode_id"] for row in rows],
        "source_banks": dict(Counter(row["bank"] for row in rows)),
        "source_scout": {
            "episodes": len(rows),
            "environment_steps": sum(row["source_episode_steps"] for row in rows),
            "act_calls": sum(row["source_episode_act_calls"] for row in rows),
            "wall_seconds": math.fsum(row["source_episode_wall_seconds"] for row in rows),
        },
        "branches": {
            "labels": list(labels),
            "rollouts": len(branches),
            "environment_steps": sum(branch["environment_steps"] for branch in branches),
            "act_calls": sum(branch["act_calls"] for branch in branches),
            "wall_seconds": math.fsum(branch["wall_seconds"] for branch in branches),
        },
    }


def _slice(episodes: Sequence[_Episode], labels: tuple[str, ...]) -> ProbeSlice:
    context = np.array([episode.context for episode in episodes], dtype=np.float32).reshape(-1, CONTEXT_DIM)
    residual = np.array([episode.residual for episode in episodes], dtype=np.float32).reshape(-1, RESIDUAL_DIM)
    returns = {
        label: np.array([episode.row["branches"][label]["discounted_return"] for episode in episodes])
        for label in labels
    }
    g0 = None
    if labels == FACTORIAL_BRANCHES:
        g0 = np.array([
            FactorialOutcome(v00, v01, v10, v11).base_refresh_gain
            for v00, v01, v10, v11 in zip(*(returns[label] for label in labels), strict=True)
        ])
    v10, v11 = (returns[label] for label in PAIR_BRANCHES)
    return ProbeSlice(
        source_episode_ids=tuple(episode.source_episode_id for episode in episodes),
        context=context,
        residual=residual,
        g0=g0,
        g_delta=v11 - v10,
        v10=v10,
        v11=v11,
        budget=_budget(episodes, labels),
    )


def _seeded_order(episodes: Sequence[_Episode], *, seed: int, distribution: str, stream: str) -> list[_Episode]:
    ordered = sorted(episodes, key=lambda episode: episode.source_episode_id)
    random.Random(f"{seed}:{distribution}:{stream}").shuffle(ordered)
    return ordered


def select_budget(dataset: E0Dataset, distribution: str, fraction: float, seed: int) -> BudgetSelection:
    """Choose nested prefixes by episode and charge real rollouts and control steps."""
    if distribution not in DISTRIBUTIONS or fraction not in FRACTIONS or type(seed) is not int:
        raise ValueError("Choose a registered E0 distribution, fraction and integer seed")
    strata = [episode for episode in dataset._episodes if episode.row["distribution"] == distribution]
    factorial_train = [
        episode for episode in strata
        if episode.row["bank"] == "factorial" and episode.row["split"] == "train"
    ]
    pair_train = [
        episode for episode in strata
        if episode.row["bank"] == "pair" and episode.row["split"] == "train"
    ]
    if len(pair_train) != len(factorial_train) or not factorial_train:
        raise ValueError("Each stratum needs disjoint balanced factorial and pair training pools")

    k = int(len(factorial_train) * fraction)
    factorial_order = _seeded_order(factorial_train, seed=seed, distribution=distribution, stream="factorial")
    factorial = _slice(factorial_order[:k], FACTORIAL_BRANCHES)
    branch_budget = BranchBudget(factorial.budget["branches"]["rollouts"])
    step_cap = factorial.budget["branches"]["environment_steps"]

    b6_first = _seeded_order(factorial_train, seed=seed, distribution=distribution, stream="b6_factorial")
    b6_second = _seeded_order(pair_train, seed=seed, distribution=distribution, stream="b6_pair")
    b6_order = [episode for pair in zip(b6_first, b6_second, strict=True) for episode in pair]
    b6_rows = b6_order[:branch_budget.direct_two_branch_contexts]
    remaining = step_cap
    cap_rows = []
    for episode in b6_rows:
        measured_steps = sum(episode.row["branches"][label]["environment_steps"] for label in PAIR_BRANCHES)
        if measured_steps > remaining:
            break
        cap_rows.append(episode)
        remaining -= measured_steps
    b6_rollout = _slice(b6_rows, PAIR_BRANCHES)
    b6_joint = _slice(cap_rows, PAIR_BRANCHES)
    validation, test = (
        _slice(
            sorted((episode for episode in strata if episode.row["split"] == split),
                   key=lambda episode: episode.source_episode_id),
            FACTORIAL_BRANCHES,
        )
        for split in ("validation", "test")
    )
    assert_disjoint_groups(
        (*factorial.source_episode_ids, *b6_rollout.source_episode_ids),
        validation.source_episode_ids,
        test.source_episode_ids,
    )
    return BudgetSelection(
        factorial=factorial,
        b6_rollout=b6_rollout,
        b6_joint=b6_joint,
        validation=validation,
        test=test,
        manifest={
            "distribution": distribution,
            "fraction": fraction,
            "order_seed": seed,
            "budget": {
                "branch_rollouts": branch_budget.total_branch_rollouts,
                "branch_environment_steps": step_cap,
            },
            "factorial": factorial.budget,
            "b6_rollout": b6_rollout.budget,
            "b6_joint": b6_joint.budget,
            "b6_joint_unused": {
                "branch_rollouts": branch_budget.total_branch_rollouts - b6_joint.budget["branches"]["rollouts"],
                "branch_environment_steps": remaining,
            },
            "b6_rollout_environment_step_delta": b6_rollout.budget["branches"]["environment_steps"] - step_cap,
            "validation": validation.budget,
            "test": test.budget,
        },
    )
