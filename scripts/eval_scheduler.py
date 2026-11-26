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


@dataclass(frozen=True)
class X1Run:
    registration: dict
    registration_sha256: str
    cell: dict
    offline_attempt: Path
    resfit_attempt: str | None


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


def _registered_probes(source, results, distribution):
    registered = {}
    for method, row_method in PROBE_ROWS.items():
        rows = [row for row in results if row["method"] == row_method and row["distribution"] == distribution
                and row["fraction"] == 1.0 and row["status"] == "COMPLETED"]
        if len(rows) != 1:
            raise ValueError(f"Expected one completed 100% {distribution} probe for {row_method}")
        row = rows[0]
        path = (source / row["model"]).resolve()
        if path.parent != source / "models" or _sha256(path) != row["model_sha256"]:
            raise ValueError(f"The registered local {row_method} probe is missing or changed")
        registered[method] = {
            "offline_method": row_method, "model": row["model"], "model_sha256": row["model_sha256"],
            "distribution": row["distribution"], "fraction": row["fraction"],
            "training_budget": row["training_budget"],
        }
    return registered


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
    return source, manifest, prior_seeds, _registered_probes(source, results, "natural")


def _x1_probe_inputs(x1):
    from cf_resrl.x1_plan import source_cohort

    source = x1.offline_attempt
    status = _json(source / "status.json")
    if status["status"] != "COMPLETED" or any(
        _sha256(source / filename) != status[f"{key}_sha256"]
        for key, filename in (("run_manifest", "run_manifest.json"), ("results", "results.json"))
    ):
        raise ValueError("X1 offline probe evidence is incomplete or changed")
    manifest = _json(source / "run_manifest.json")
    data = manifest["data"]
    cell = x1.cell
    if data["x1_cell"] != cell or data["x1_config_sha256"] != x1.registration_sha256:
        raise ValueError("X1 probes must come from the selected cell and registration")

    data_attempt = Path(data["attempt"])
    data_status = _json(data_attempt / "status.json")
    if data_status["status"] != "COMPLETED" or any(
        _sha256(data_attempt / filename) != data[f"{key}_sha256"]
        for key, filename in (
            ("status", "status.json"), ("index", "group_index.jsonl"), ("summary", "summary.json"),
            ("run_manifest", "run_manifest.json"), ("feature_schema", "feature_schema.json"),
        )
    ) or any(data_status[f"{key}_sha256"] != data[f"{key}_sha256"] for key in ("index", "summary")):
        raise ValueError("X1 source evidence differs from the verified offline probe data")
    collection = _json(data_attempt / "run_manifest.json")
    protocol = collection["protocol"]
    plan = source_cohort(x1.registration, cell["distribution"])
    if (
        protocol["x1_cell"] != cell or protocol["x1_config_sha256"] != x1.registration_sha256
        or protocol["task"] != x1.registration["task"] or protocol["simulation_horizon_steps"] != 300
        or protocol["refresh_delay_steps"] != cell["delay"] or protocol["residual_scale"] != cell["residual_scale"]
        or protocol["residual_component_limit"] != x1.registration["residual_component_limit"]
        or collection["identities"] != data["identities"] or collection["source_plan"] != plan
        or data_status["completed_groups"] != len(plan)
    ):
        raise ValueError("X1 source policies, factor settings or episode roles differ from the registered cell")

    results = _json(source / "results.json")
    distribution = cell["distribution"]
    if {row["distribution"] for row in results} != {distribution}:
        raise ValueError("X1 probes must use exactly the selected training distribution")
    registered = _registered_probes(source, results, distribution)
    provenance = {
        "x1_cell": cell,
        "x1_config_sha256": x1.registration_sha256,
        "source_index_sha256": data["index_sha256"],
        "source_manifest_sha256": data["run_manifest_sha256"],
        "offline_results_sha256": status["results_sha256"],
        "test_reset_scope": "shared locked Square resets with no natural or enriched tick selection",
    }
    return manifest, {row["seed"] for row in plan}, registered, protocol, provenance


def _x1_closed_loop_protocol(x1):
    settings = x1.registration["closed_loop"]
    if settings["norm_threshold_transform"] != "multiply_by_minimum_of_1_and_residual_scale":
        raise ValueError("Unknown registered X1 residual-norm threshold transform")
    multiplier = min(1.0, x1.cell["residual_scale"])
    return {
        **settings,
        "task": x1.registration["task"],
        "act_budgets": x1.registration["act_call_caps"],
        "norm_thresholds_nominal": settings["norm_thresholds"],
        "norm_threshold_multiplier": multiplier,
        "norm_thresholds": [threshold * multiplier for threshold in settings["norm_thresholds"]],
        **{
            f"{split}_seeds": {"first": cohort["first_seed"], "count": cohort["size"]}
            for split, cohort in x1.registration["closed_loop_cohorts"].items()
        },
    }


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


def _run(config, out, started, *, x1=None):
    import torch
    from cf_resrl.models import load_probe

    torch.set_num_threads(1)
    gate_config = _json(GATE_CONFIG)
    x1_provenance = {}
    status_fields = {}
    if x1 is None:
        source, offline_manifest, prior_seeds, registered = _probe_inputs(config)
        square_protocol = gate_config
        offline_results_sha256 = config["offline_results_sha256"]
        selection = {}
    else:
        from cf_resrl.x1_plan import resolve_cell_policies

        source = x1.offline_attempt
        offline_manifest, prior_seeds, registered, square_protocol, x1_provenance = _x1_probe_inputs(x1)
        offline_results_sha256 = x1_provenance["offline_results_sha256"]
        status_fields = {"x1_cell_id": x1.cell["cell_id"], "x1_config_sha256": x1.registration_sha256}
        attempts = {}
        if x1.resfit_attempt is not None:
            seed, separator, path = x1.resfit_attempt.partition("=")
            if not separator or seed != str(x1.cell["resfit_seed"]) or not path:
                raise ValueError("Use --resfit-attempt SEED=PATH for the selected X1 cell's training seed")
            attempts[int(seed)] = Path(path).expanduser().resolve()
        selection = {"x1_selection": resolve_cell_policies(x1.registration, x1.cell, attempts)}
    prior_seeds.update(gate_config["source_seeds"])
    m0 = _json(ROOT / "configs/cf_resrl/m0_act_training.json")
    training_seed = m0["training"]["seed"]
    prior_seeds.add(training_seed)
    prior_seeds.update(range(training_seed + 1000, training_seed + 1000 + m0["evaluation"]["num_episodes"]))
    cohorts = _seeds(config, prior_seeds)
    settings = _settings(config, square_protocol["simulation_horizon_steps"], chunk_length=20)
    probes = {method: load_probe(source / entry["model"]) for method, entry in registered.items()}
    if any(probe.method != method for method, probe in probes.items()):
        raise ValueError("Loaded probe methods differ from the registered family")
    runner, provenance = load_square_runner(square_protocol, **selection)
    cache = {}
    actual = []
    try:
        if provenance["identities"] != offline_manifest["data"]["identities"]:
            raise ValueError("Closed-loop ACT, ResFiT and normalization differ from the probe training data")
        (out / "episodes").mkdir()
        _write(out / "run_manifest.json", {
            **provenance, "protocol": config, "frozen_square_protocol": square_protocol,
            "offline_results_sha256": offline_results_sha256,
            "offline_attempt": str(source), "offline_manifest_sha256": _sha256(source / "run_manifest.json"),
            "probes": registered, "cohorts": cohorts, "prior_seed_count": len(prior_seeds),
            "accounting": "every physical validation and test rollout appears once in episode_index.jsonl",
            **x1_provenance,
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
                    _status(out, "RUNNING", phase=split, episodes_completed=len(actual), started_at_utc=started,
                            **status_fields)
                    print(f"{split}: {setting.method}, cap {setting.act_budget}, parameter {setting.parameter}, "
                          f"{sum(row['success'] for row in rows)}/{len(rows)} successful", flush=True)
                return cache[key]

            summary = _experiment(out, config, settings, cohorts, evaluate)
        summary["physical_rollout_budget"] = {split: _physical_budget([row for row in actual if row["split"] == split])
                                              for split in cohorts}
        if x1 is not None:
            summary.update(x1_provenance)
        _write(out / "summary.json", summary)
        _tables_and_plot(out, summary)
    finally:
        runner.wrapper.close()
    return len(actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--x1-config", type=Path)
    parser.add_argument("--cell")
    parser.add_argument("--offline-attempt", type=Path)
    parser.add_argument("--resfit-attempt", metavar="SEED=PATH")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    status_fields = {}
    x1 = None
    if args.x1_config is None:
        if any(value is not None for value in (args.cell, args.offline_attempt, args.resfit_attempt)):
            parser.error("X1 options require --x1-config")
        config = _json(args.config or DEFAULT_CONFIG)
        output_root = ROOT / config["output_root"]
    else:
        if args.config is not None or args.cell is None or args.offline_attempt is None:
            parser.error("X1 requires --cell and --offline-attempt and uses --x1-config alone")
        from cf_resrl.x1_plan import load_x1_config, plan_x1

        registration = load_x1_config(args.x1_config)
        cells = {cell["cell_id"]: cell for cell in plan_x1(registration)}
        if args.cell not in cells:
            parser.error("Select a registered --cell from scripts/collect_square_x1.py --list-cells")
        x1 = X1Run(registration, _sha256(args.x1_config), cells[args.cell],
                   args.offline_attempt.resolve(), args.resfit_attempt)
        config = _x1_closed_loop_protocol(x1)
        output_root = ROOT / "results/cf_resrl/expansion_square/closed_loop" / args.cell
        status_fields = {"x1_cell_id": args.cell, "x1_config_sha256": x1.registration_sha256}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.out or output_root / "attempts" / stamp
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    _status(out, "RUNNING", started_at_utc=started, episodes_completed=0, **status_fields)
    print(f"Square closed-loop evidence: {out}", flush=True)
    try:
        completed = _run(config, out, started, x1=x1)
    except Exception as error:
        index = out / "episode_index.jsonl"
        episodes = len(index.read_bytes().splitlines()) if index.is_file() else 0
        _status(out, "FAILED", started_at_utc=started, episodes_completed=episodes,
                error=str(error), type=type(error).__name__, **status_fields)
        raise
    _status(out, "COMPLETED", started_at_utc=started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
            episodes_completed=completed, index_sha256=_sha256(out / "episode_index.jsonl"),
            summary_sha256=_sha256(out / "summary.json"), selection_sha256=_sha256(out / "validation_selection.json"),
            **status_fields)


if __name__ == "__main__":
    main()
