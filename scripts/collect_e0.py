"""Collect registered natural and enriched Square counterfactual data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import random
import sys

from collect_factorial import DEFAULT_CONFIG as GATE_CONFIG
from collect_factorial import REPO_ROOT, _json, _sha256, _write, load_square_runner


DEFAULT_CONFIG = REPO_ROOT / "configs/cf_resrl/square_e0.json"
FACTORIAL_LABELS = ("V00", "V01", "V10", "V11")
PAIR_LABELS = ("V10", "V11")


def _plan(config):
    sizes = {"train": 160, "validation": 20, "test": 20}
    if (
        config["task"] != "Square"
        or config["refresh_delay_steps"] != 2
        or config["factorial_split_counts"] != sizes
    ):
        raise ValueError("E0 requires the registered cohorts, two-tick refresh and the 160/20/20 episode split")
    rng = random.Random(config["split_seed"])
    plan = []
    for bank, key, size in (
        ("factorial", "factorial_cohorts", 200), ("pair", "pair_training_cohorts", 160)
    ):
        for distribution in ("natural", "enriched"):
            cohort = config[key][distribution]
            if cohort["size"] != size:
                raise ValueError(f"Unexpected {bank} cohort size for {distribution}")
            seeds = list(range(cohort["first_seed"], cohort["first_seed"] + size))
            order = seeds[:]
            rng.shuffle(order)
            splits = ["train"] * size if bank == "pair" else [
                split for split, count in sizes.items() for _ in range(count)
            ]
            split_by_seed = dict(zip(order, splits, strict=True))
            for seed in seeds:
                plan.append({
                    "source_episode_id": f"square_e0_{seed}",
                    "seed": seed,
                    "distribution": distribution,
                    "bank": bank,
                    "split": split_by_seed[seed],
                    "selector_seed": config["selector_seed"] + seed,
                })
    if len({row["seed"] for row in plan}) != len(plan):
        raise ValueError("Source episodes must be disjoint across all E0 cohorts")
    return plan


def _gate(config):
    gate = REPO_ROOT / config["gate_attempt"]
    status = _json(gate / "status.json")
    summary = _json(gate / "summary.json")
    digest = _sha256(gate / "summary.json")
    if (
        status["status"] != "COMPLETED"
        or status["summary_sha256"] != digest
        or config["gate_summary_sha256"] != digest
        or summary["status"] != "PASS"
        or summary["branch_rollouts"] != 32
        or len(summary["groups"]) != 8
    ):
        raise ValueError("The registered Square branch gate must have passed before E0 collection")
    return gate, summary


def _status(out, status, **fields):
    temporary = out / ".status.tmp"
    _write(temporary, {"status": status, **fields})
    temporary.replace(out / "status.json")


def _collect(config, plan, out):
    gate, gate_summary = _gate(config)
    gate_seeds = {group["seed"] for group in gate_summary["groups"]}
    if any(row["seed"] in gate_seeds for row in plan):
        raise ValueError("E0 source seeds must be distinct from the hard gate")
    gate_config = _json(GATE_CONFIG)
    if config["refresh_delay_steps"] != gate_config["refresh_delay_steps"]:
        raise ValueError("E0 delay differs from the validated gate")
    runner, provenance = load_square_runner(gate_config)
    try:
        _collect_with_runner(config, plan, out, gate, runner, provenance)
    finally:
        runner.wrapper.close()


def _collect_with_runner(config, plan, out, gate, runner, provenance):
    identities = _json(gate / "run_manifest.json")["identities"]
    if provenance["identities"] != identities:
        raise ValueError("E0 policy and normalization identities differ from the passing gate")
    collect_branch_bank(config, plan, out, runner, provenance)


def collect_branch_bank(config, plan, out, runner, provenance):
    import numpy as np
    from cf_resrl.e0_collector import collect_episode
    from cf_resrl.e0_report import summarize_e0

    groups_dir = out / "groups"
    features_dir = out / "features"
    groups_dir.mkdir()
    features_dir.mkdir()
    residual_feature = "scaled and clipped residual" if "x1_cell" in config else "original residual"
    _write(out / "run_manifest.json", {
        **provenance,
        "protocol": config,
        "source_plan": plan,
        "feature_scope": f"frozen visual features, current observed state and active chunk, {residual_feature}",
        "raw_group_scope": "nominal source rollout and real intervention trajectories for evidence only",
        "pair_training_budget": "additional train-only V10/V11 contexts; account true control steps separately",
    })
    expected = len(plan)
    _status(out, "RUNNING", pid=os.getpid(), completed_groups=0, planned_groups=expected)
    feature_schema = None
    rows = []
    with (out / "group_index.jsonl").open("x") as index:
        for number, source in enumerate(plan, start=1):
            labels = FACTORIAL_LABELS if source["bank"] == "factorial" else PAIR_LABELS
            group, features = collect_episode(
                runner, seed=source["seed"], distribution=source["distribution"],
                labels=labels, selector_seed=source["selector_seed"],
            )
            group.update(source)
            observed_schema = {
                name: {"shape": list(values.shape), "dtype": str(values.dtype)}
                for name, values in sorted(features.items())
            }
            if feature_schema is None:
                feature_schema = observed_schema
                _write(out / "feature_schema.json", feature_schema)
            if observed_schema != feature_schema or any(
                array.dtype.kind != "f" or not np.isfinite(array).all() for array in features.values()
            ):
                raise RuntimeError("Collected features have an invalid schema or non-finite value")
            stem = source["source_episode_id"]
            raw_file = groups_dir / (stem + ".json.gz")
            with gzip.open(raw_file, "xt", compresslevel=1) as compressed:
                json.dump(group, compressed, allow_nan=False, separators=(",", ":"))
            feature_file = features_dir / (stem + ".npz")
            with feature_file.open("xb") as stream:
                np.savez_compressed(stream, **features)
            scout = group["nominal_scout"]
            row = {
                **source,
                "source_tick": group["source_tick"],
                "source_episode_steps": scout["environment_steps"],
                "source_episode_act_calls": scout["act_calls"],
                "source_episode_wall_seconds": scout["wall_seconds"],
                "branches": {
                    label: {key: value for key, value in branch.items() if key != "steps"}
                    for label, branch in group["branches"].items()
                },
                "raw_group": str(raw_file.relative_to(out)),
                "raw_group_sha256": _sha256(raw_file),
                "features": str(feature_file.relative_to(out)),
                "features_sha256": _sha256(feature_file),
            }
            index.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            index.flush()
            os.fsync(index.fileno())
            rows.append(row)
            _status(out, "RUNNING", pid=os.getpid(), completed_groups=number, planned_groups=expected)
            print(f"E0 episode {number}/{expected}: {stem}, {source['distribution']} {source['bank']}, tick {group['source_tick']}", flush=True)
    summary = summarize_e0(
        rows,
        costs=config["decision_costs"],
        bootstrap_samples=config["bootstrap_episodes"],
        bootstrap_seed=config["bootstrap_seed"],
    )
    _write(out / "summary.json", summary)
    manifest_checksum = {"run_manifest_sha256": _sha256(out / "run_manifest.json")} if "x1_cell" in config else {}
    _status(out, "COMPLETED", completed_groups=expected, planned_groups=expected,
            summary_sha256=_sha256(out / "summary.json"), index_sha256=_sha256(out / "group_index.jsonl"),
            **manifest_checksum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = _json(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.out or REPO_ROOT / config["output_root"] / "attempts" / stamp
    out.mkdir(parents=True, exist_ok=False)
    print(f"Square E0 evidence: {out}", flush=True)
    try:
        _collect(config, _plan(config), out)
    except Exception as error:
        previous = out / "status.json"
        progress = _json(previous).get("completed_groups", 0) if previous.exists() else 0
        _status(out, "FAILED", completed_groups=progress, error=str(error), type=type(error).__name__)
        raise


if __name__ == "__main__":
    main()
