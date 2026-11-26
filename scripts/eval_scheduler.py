"""Evaluate frozen Square schedulers with validation-only tuning and hard ACT caps."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_factorial import DEFAULT_CONFIG as GATE_CONFIG
from scripts.collect_factorial import _json, _sha256, _write, load_square_runner


DEFAULT_CONFIG = ROOT / "configs/cf_resrl/square_closed_loop.json"
ORIGINAL = "original_resfit"
PROBE_ROWS = {"B5": "B5", "B6": "B6-joint-cap", "P1": "P1"}
MEASUREMENTS = (
    "success", "return", "discounted_return", "environment_steps", "act_calls", "wall_seconds",
    "policy_wall_seconds", "refresh_requests", "refresh_activations",
)


@dataclass(frozen=True)
class Setting:
    method: str
    act_budget: int
    parameter: int | float | None


def _seeds(config, source_seeds):
    seeds = {}
    for split in ("validation", "test"):
        first, count = [config[f"{split}_seeds"][key] for key in ("first", "count")]
        if type(first) is not int or first < 0 or type(count) is not int or count <= 0:
            raise ValueError(f"Invalid registered {split} cohort")
        if first + count > 2**32:
            raise ValueError(f"{split} seed exceeds the NumPy seed range")
        seeds[split] = tuple(range(first, first + count))
    if set(seeds["validation"]) & set(seeds["test"]) or any(
        set(cohort) & source_seeds for cohort in seeds.values()
    ):
        raise ValueError("Closed-loop validation and test seeds must be disjoint from each other and prior data")
    return seeds


def _settings(config, horizon, chunk_length):
    budgets = config["act_budgets"]
    if (
        not budgets or any(type(budget) is not int for budget in budgets)
        or budgets != sorted(set(budgets)) or budgets[0] < math.ceil(horizon / chunk_length)
    ):
        raise ValueError("ACT caps must increase from at least the nominal full-horizon call count")
    period = config["fixed_period"]
    if type(period) is not int or period <= 0:
        raise ValueError("The fixed refresh period must be a positive integer")
    for key in ("norm_thresholds", "learned_costs"):
        values = config[key]
        if (
            not values or any(type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in values)
            or values != sorted(set(values))
        ):
            raise ValueError(f"{key} must be finite, distinct, nonnegative and in ascending order")
    if type(config["bootstrap_samples"]) is not int or config["bootstrap_samples"] <= 0:
        raise ValueError("Bootstrap requires a positive integer draw count")
    if type(config["bootstrap_seed"]) is not int or config["bootstrap_seed"] < 0:
        raise ValueError("Bootstrap seed must be a nonnegative integer")
    options = {
        ORIGINAL: [None], "fixed_period_refresh": [period],
        "residual_norm_threshold": config["norm_thresholds"],
        **{method: config["learned_costs"] for method in PROBE_ROWS},
    }
    return {
        cap: {
            method: [Setting(method, budgets[0] if method == ORIGINAL else cap, value) for value in values]
            for method, values in options.items()
        }
        for cap in budgets
    }


def _probe_inputs(config):
    source = (ROOT / config["offline_attempt"]).resolve()
    status = _json(source / "status.json")
    results = _json(source / "results.json")
    digest = _sha256(source / "results.json")
    if status["status"] != "COMPLETED" or digest != status["results_sha256"] or digest != config["offline_results_sha256"]:
        raise ValueError("The offline results do not match the registered completed attempt")
    manifest = _json(source / "run_manifest.json")
    e0_attempt = Path(manifest["data"]["attempt"])
    source_manifest = e0_attempt / "run_manifest.json"
    if _sha256(source_manifest) != manifest["data"]["run_manifest_sha256"]:
        raise ValueError("E0 source manifest differs from the offline probe provenance")
    prior_seeds = {row["seed"] for row in _json(source_manifest)["source_plan"]}
    registered = {}
    for method, row_method in PROBE_ROWS.items():
        rows = [row for row in results if row["method"] == row_method and row["distribution"] == "natural"
                and row["fraction"] == 1.0 and row["status"] == "COMPLETED"]
        if len(rows) != 1:
            raise ValueError(f"Expected one completed 100% natural probe for {row_method}")
        row = rows[0]
        path = (source / row["model"]).resolve()
        if path.parent != source / "models" or _sha256(path) != row["model_sha256"]:
            raise ValueError(f"The registered local {row_method} probe is missing or changed")
        registered[method] = {
            "offline_method": row_method, "model": row["model"], "model_sha256": row["model_sha256"],
            "distribution": row["distribution"], "fraction": row["fraction"],
            "training_budget": row["training_budget"],
        }
    return source, manifest, prior_seeds, registered


def _status(out, status, **values):
    temporary = out / ".status.tmp"
    _write(temporary, {"status": status, "pid": os.getpid(), **values})
    temporary.replace(out / "status.json")


def _run_episode_record(runner, setting, seed, probe, out, index, number, split):
    from cf_resrl.closed_loop import run_episode

    result = run_episode(runner, seed=seed, **asdict(setting), probe=probe)
    if (
        result["seed"] != seed or any(result[key] != value for key, value in asdict(setting).items())
        or type(result["success"]) is not bool or type(result["act_calls"]) is not int
        or not 1 <= result["act_calls"] <= setting.act_budget
        or not 1 <= result["environment_steps"] == len(result["steps"]) <= runner.wrapper.horizon
    ):
        raise ValueError("Closed-loop episode trace does not match its seed, protocol or budget")
    episode_id = f"{number:05d}_{split}_{setting.method}_{seed}"
    path = Path("episodes") / f"{episode_id}.json.gz"
    destination = out / path
    with gzip.open(destination, "xt", compresslevel=1) as compressed:
        json.dump({"episode_id": episode_id, "split": split, **result}, compressed,
                  allow_nan=False, separators=(",", ":"))
    row = {
        "episode_id": episode_id, "seed": seed, "split": split, **asdict(setting),
        **{key: result[key] for key in MEASUREMENTS}, "trace": path.as_posix(),
        "trace_sha256": _sha256(destination),
    }
    index.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    index.flush()
    os.fsync(index.fileno())
    return row


def _summarize(rows, config):
    from cf_resrl.offline_metrics import bootstrap_mean

    if len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("A closed-loop cohort contains duplicate seeds")
    return {
        "episodes": len(rows), "successes": sum(row["success"] for row in rows),
        "estimates": {
            metric: bootstrap_mean([row[metric] for row in rows], samples=config["bootstrap_samples"],
                                   seed=config["bootstrap_seed"])
            for metric in MEASUREMENTS
        },
        "episode_ids": [row["episode_id"] for row in rows],
    }


def _choose_validation(settings, scores):
    """Lock each cap's best policy and a minimum-call policy at baseline performance."""
    original_setting = next(iter(settings.values()))[ORIGINAL][0]
    target = scores[original_setting]["successes"]

    def act_calls(setting):
        return scores[setting]["estimates"]["act_calls"]["mean"]

    fixed_budget = {
        cap: {
            method: min(options, key=lambda option: (-scores[option]["successes"], act_calls(option)))
            for method, options in methods.items()
        }
        for cap, methods in settings.items()
    }
    methods = next(iter(settings.values()))
    fixed_performance = {}
    for method in methods:
        eligible = (option for choices in settings.values() for option in choices[method]
                    if scores[option]["successes"] >= target)
        fixed_performance[method] = min(eligible, key=lambda option: (act_calls(option), option.act_budget), default=None)
    return target, fixed_budget, fixed_performance


def _experiment(out, config, settings, cohorts, evaluate):
    validation_scores = {}
    for methods in settings.values():
        for options in methods.values():
            for setting in options:
                if setting not in validation_scores:
                    validation_scores[setting] = _summarize(evaluate("validation", setting), config)
    target, fixed_budget, fixed_performance = _choose_validation(settings, validation_scores)
    locked = out / "validation_selection.json"
    _write(locked, {
        "validation_seeds": list(cohorts["validation"]), "locked_test_seeds": list(cohorts["test"]),
        "rule": "maximize validation successes, then minimize calls, then use the preregistered parameter order",
        "fixed_performance_rule": "meet the original ResFiT validation success rate, then minimize calls and use the lower cap",
        "target_validation_successes": target,
        "target_success_rate": target / len(cohorts["validation"]),
        "candidates": [{"setting": asdict(setting), **metrics} for setting, metrics in validation_scores.items()],
        "fixed_budget": {cap: {method: asdict(setting) for method, setting in rows.items()}
                         for cap, rows in fixed_budget.items()},
        "fixed_performance": {method: asdict(setting) if setting else None
                              for method, setting in fixed_performance.items()},
    })
    selection_sha = _sha256(locked)
    print(f"Validation choices locked, SHA-256 {selection_sha}", flush=True)

    def test_scores(setting):
        rows = evaluate("test", setting)
        return {"setting": asdict(setting), **_summarize(rows, config)}

    budget_scores = {cap: {method: test_scores(setting) for method, setting in methods.items()}
                     for cap, methods in fixed_budget.items()}
    performance_scores = {}
    for method, setting in fixed_performance.items():
        if setting is None:
            performance_scores[method] = {"status": "NO_VALIDATION_SETTING_MEETS_TARGET"}
            continue
        score = test_scores(setting)
        performance_scores[method] = {"status": "EVALUATED",
                                     "met_target_on_test": score["successes"] / score["episodes"] >= target / len(cohorts["validation"]),
                                     **score}
    return {
        "validation_selection_sha256": selection_sha,
        "fixed_act_budgets": budget_scores, "fixed_task_performance": performance_scores,
        "fixed_performance_success_target": target / len(cohorts["validation"]),
    }


def _physical_budget(rows):
    return {"episodes": len(rows), **{metric: sum(row[metric] for row in rows)
                                     for metric in ("environment_steps", "act_calls", "wall_seconds",
                                                    "policy_wall_seconds", "refresh_requests", "refresh_activations")}}


def _tables_and_plot(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = summary["fixed_act_budgets"]
    fields = ("act_budget", "method", "parameter", "episodes", "success_rate", "mean_return", "mean_act_calls", "mean_wall_seconds")
    with (out / "test_budget_curve.csv").open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        for cap, methods in budgets.items():
            for method, row in methods.items():
                metrics = row["estimates"]
                writer.writerow([cap, method, row["setting"]["parameter"], row["episodes"],
                                 *(metrics[field]["mean"] for field in ("success", "return", "act_calls", "wall_seconds"))])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", sharey=True)
    for method in next(iter(budgets.values())):
        points = [rows[method]["estimates"] for rows in budgets.values()]
        successes = [point["success"]["mean"] for point in points]
        axes[0].plot(list(budgets), successes, marker="o", label=method)
        axes[1].plot([point["act_calls"]["mean"] for point in points], successes, marker="o")
    axes[0].set(xlabel="ACT call cap per episode", ylabel="Test success rate")
    axes[1].set(xlabel="Measured mean ACT calls per test episode")
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(out / "test_budget_curve.png", dpi=160)
    plt.close(figure)


def _run(config, out, started):
    import torch
    from cf_resrl.models import load_probe

    torch.set_num_threads(1)
    source, offline_manifest, prior_seeds, registered = _probe_inputs(config)
    gate_config = _json(GATE_CONFIG)
    prior_seeds.update(gate_config["source_seeds"])
    m0 = _json(ROOT / "configs/cf_resrl/m0_act_training.json")
    training_seed = m0["training"]["seed"]
    prior_seeds.add(training_seed)
    prior_seeds.update(range(training_seed + 1000, training_seed + 1000 + m0["evaluation"]["num_episodes"]))
    cohorts = _seeds(config, prior_seeds)
    settings = _settings(config, gate_config["simulation_horizon_steps"], chunk_length=20)
    probes = {method: load_probe(source / entry["model"]) for method, entry in registered.items()}
    if any(probe.method != method for method, probe in probes.items()):
        raise ValueError("Loaded probe methods differ from the registered family")
    runner, provenance = load_square_runner(gate_config)
    cache = {}
    actual = []
    try:
        if provenance["identities"] != offline_manifest["data"]["identities"]:
            raise ValueError("Closed-loop ACT, ResFiT and normalization differ from the probe training data")
        (out / "episodes").mkdir()
        _write(out / "run_manifest.json", {
            **provenance, "protocol": config, "frozen_square_protocol": gate_config,
            "offline_results_sha256": config["offline_results_sha256"],
            "offline_attempt": str(source), "offline_manifest_sha256": _sha256(source / "run_manifest.json"),
            "probes": registered, "cohorts": cohorts, "prior_seed_count": len(prior_seeds),
            "accounting": "every physical validation and test rollout appears once in episode_index.jsonl",
        })
        with (out / "episode_index.jsonl").open("x") as index:
            def evaluate(split, setting):
                key = (split, setting)
                if key not in cache:
                    rows = []
                    for seed in cohorts[split]:
                        row = _run_episode_record(runner, setting, seed, probes.get(setting.method),
                                                  out, index, len(actual) + 1, split)
                        rows.append(row)
                        actual.append(row)
                    cache[key] = rows
                    _status(out, "RUNNING", phase=split, episodes_completed=len(actual), started_at_utc=started)
                    print(f"{split}: {setting.method}, cap {setting.act_budget}, parameter {setting.parameter}, "
                          f"{sum(row['success'] for row in rows)}/{len(rows)} successful", flush=True)
                return cache[key]

            summary = _experiment(out, config, settings, cohorts, evaluate)
        summary["physical_rollout_budget"] = {split: _physical_budget([row for row in actual if row["split"] == split])
                                              for split in cohorts}
        _write(out / "summary.json", summary)
        _tables_and_plot(out, summary)
    finally:
        runner.wrapper.close()
    return len(actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = _json(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.out or ROOT / config["output_root"] / "attempts" / stamp
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    _status(out, "RUNNING", started_at_utc=started, episodes_completed=0)
    print(f"Square closed-loop evidence: {out}", flush=True)
    try:
        completed = _run(config, out, started)
    except Exception as error:
        index = out / "episode_index.jsonl"
        episodes = len(index.read_bytes().splitlines()) if index.is_file() else 0
        _status(out, "FAILED", started_at_utc=started, episodes_completed=episodes,
                error=str(error), type=type(error).__name__)
        raise
    _status(out, "COMPLETED", started_at_utc=started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
            episodes_completed=completed, index_sha256=_sha256(out / "episode_index.jsonl"),
            summary_sha256=_sha256(out / "summary.json"), selection_sha256=_sha256(out / "validation_selection.json"))


if __name__ == "__main__":
    main()
