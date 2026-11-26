"""Fit and evaluate the registered Square offline probes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/cf_resrl/square_offline_probe.json"
sys.path.insert(0, str(ROOT))
from cf_resrl.dataset import sha256_file


METHODS = (
    ("B5", "B5", "factorial"),
    ("P1", "P1", "factorial"),
    ("B6-rollouts", "B6", "b6_rollout"),
    ("B6-joint-cap", "B6", "b6_joint"),
    ("Norm", "Norm", "factorial"),
    ("Linear-full", "Linear", "factorial"),
)


def _read(path):
    return json.loads(path.read_text())


def _write(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _validate(config, source):
    source_index_sha256 = sha256_file(source / "group_index.jsonl")
    if source_index_sha256 != config["source_index_sha256"]:
        raise ValueError("Offline probe source index checksum differs from the registered E0 dataset")
    mvp = _read(ROOT / "configs/cf_resrl/mvp_square.json")
    e0 = _read(ROOT / "configs/cf_resrl/square_e0.json")
    if config["nested_fractions"] != mvp["offline_probe"]["nested_data_fractions"]:
        raise ValueError("Offline probe fractions differ from the Square MVP task")
    if config["decision_costs"] != e0["decision_costs"] or config["distributions"] != ["natural", "enriched"]:
        raise ValueError("Offline probe cohorts and decision costs differ from the registered E0 plan")


def _fit(config, out, dataset):
    from cf_resrl.models import fit_probe, predict_probe, save_probe
    from cf_resrl.offline_data import select_budget
    from cf_resrl.offline_metrics import evaluate_probe, bootstrap_mean
    import numpy as np

    models = out / "models"
    models.mkdir()
    selections, comparisons, results = [], [], []
    evaluation_options = {
        key: config[key] for key in ("critical_margin", "calibration_bins", "bootstrap_samples", "bootstrap_seed")
    }
    with (out / "predictions.jsonl").open("x") as prediction_log:
        for distribution in config["distributions"]:
            for number, fraction in enumerate(config["nested_fractions"], start=1):
                selection = select_budget(dataset, distribution, fraction, config["selection_seed"])
                selections.append(selection.manifest)
                predictions = {}
                fit_counts = {}
                for method, implementation, subset in METHODS:
                    train = getattr(selection, subset)
                    common = {
                        "method": method, "distribution": distribution, "fraction": fraction,
                        "training_budget": train.budget, "train_episode_count": len(train.source_episode_ids),
                    }
                    if not train.source_episode_ids:
                        if method != "B6-joint-cap":
                            raise RuntimeError(f"Unexpected empty training subset for {method}")
                        results.append({**common, "status": "NO_WHOLE_PAIR_WITHIN_JOINT_CAP"})
                        continue
                    started = time.perf_counter()
                    fitted = fit_probe(
                        implementation, train.context, train.residual, train.g0, train.g_delta,
                        **config["training"],
                    )
                    fit_counts[method] = fitted.parameter_count
                    relative = Path("models") / f"{distribution}-f{number}-{method}.pt"
                    destination = out / relative
                    save_probe(fitted, destination)
                    tests = {}
                    for split in ("validation", "test"):
                        evaluation = getattr(selection, split)
                        prediction = predict_probe(fitted, evaluation.context, evaluation.residual)
                        metrics, rows = evaluate_probe(
                            evaluation, prediction, costs=config["decision_costs"], **evaluation_options,
                        )
                        tests[split] = metrics
                        if split == "test":
                            predictions[method] = rows
                        for individual in rows:
                            prediction_log.write(json.dumps({
                                "method": method, "distribution": distribution, "fraction": fraction,
                                "split": split, **individual,
                            }, sort_keys=True, allow_nan=False) + "\n")
                    results.append({
                        **common, "status": "COMPLETED", "parameter_count": fitted.parameter_count,
                        "model": str(relative), "model_sha256": sha256_file(destination),
                        "fit_and_evaluate_seconds": time.perf_counter() - started, **tests,
                    })
                if fit_counts["B5"] != fit_counts["P1"]:
                    raise RuntimeError("B5 and P1 must have identical parameter counts")
                for other in ("B5", "B6-rollouts", "B6-joint-cap"):
                    if other not in predictions:
                        continue
                    p1 = predictions["P1"]
                    baseline = predictions[other]
                    assert [row["source_episode_id"] for row in p1] == [
                        row["source_episode_id"] for row in baseline
                    ]
                    differences = np.asarray([row["regret"] for row in baseline]) - np.asarray(
                        [row["regret"] for row in p1]
                    )
                    comparisons.append({
                        "distribution": distribution, "fraction": fraction,
                        "baseline": other, "metric": "test regret baseline minus P1",
                        "over_cost_grid": bootstrap_mean(
                            differences.mean(axis=1), samples=config["bootstrap_samples"],
                            seed=config["bootstrap_seed"],
                        ),
                        "by_cost": [
                            {"cost": cost, **bootstrap_mean(
                                differences[:, index], samples=config["bootstrap_samples"],
                                seed=config["bootstrap_seed"],
                            )}
                            for index, cost in enumerate(config["decision_costs"])
                        ],
                    })
                prediction_log.flush()
                _write(out / "selection_audit.json", selections)
                _write(out / "results.json", results)
                _write(out / "comparisons.json", comparisons)
                print(f"Offline probes completed: {distribution} at {fraction:.0%}", flush=True)
    return results


def _curves(out, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["distribution", "fraction", "method", "train_episodes", "branch_rollouts",
               "branch_environment_steps", "source_scout_environment_steps", "regret",
               "g_delta_mae", "interaction_mae", "calibration_error"]
    with (out / "test_curve.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(metrics)
        for row in results:
            if row["status"] != "COMPLETED":
                continue
            resources, test = row["training_budget"], row["test"]
            writer.writerow([
                row["distribution"], row["fraction"], row["method"], row["train_episode_count"],
                resources["branches"]["rollouts"], resources["branches"]["environment_steps"],
                resources["source_scout"]["environment_steps"], test["regret"]["mean"],
                test["g_delta_mae"], test["interaction_mae"], test["calibration_mean_absolute_bin_error"],
            ])
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained", sharey=True)
    for axis_row, distribution in zip(axes, ("natural", "enriched")):
        for axis, resource in zip(axis_row, ("rollouts", "environment_steps")):
            for method, _, _ in METHODS:
                series = [row for row in results if row["status"] == "COMPLETED"
                          and row["method"] == method and row["distribution"] == distribution]
                steps = [row["training_budget"]["branches"][resource] for row in series]
                regret = [row["test"]["regret"]["mean"] for row in series]
                axis.plot(steps, regret, marker="o", label=method)
            axis.set_xlabel({"rollouts": "Real branch rollouts", "environment_steps": "Real branch control steps"}[resource])
            axis.set_ylabel("Mean decision regret")
            axis.set_title(distribution.capitalize())
            axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.savefig(out / "test_regret.png", dpi=160)
    plt.close(figure)


def _run(config, out):
    import numpy as np
    import torch
    from cf_resrl.offline_data import load_e0

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = ROOT / config["data_attempt"]
    _validate(config, source)
    dataset = load_e0(source)
    _write(out / "run_manifest.json", {
        "command": [sys.executable, *sys.argv],
        "git_commit": _git("rev-parse", "HEAD"),
        "git_tracked_status_at_start": _git("status", "--porcelain", "--untracked-files=no").splitlines(),
        "config": config,
        "data": dataset.provenance,
        "feature_schema": dataset.feature_schema,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "test_estimand": "equal mean over source episodes and the registered decision costs",
    })
    _curves(out, _fit(config, out, dataset))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = _read(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.out or ROOT / config["output_root"] / "attempts" / stamp
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    print(f"Square offline probe evidence: {out}", flush=True)
    _write(out / "status.json", {"status": "RUNNING", "started_at_utc": started})
    try:
        _run(config, out)
    except Exception as error:
        _write(out / "status.json", {"status": "FAILED", "error": str(error), "type": type(error).__name__})
        raise
    _write(out / "status.json", {
        "status": "COMPLETED", "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        **{f"{stem}_sha256": sha256_file(out / filename) for stem, filename in (
            ("results", "results.json"), ("comparisons", "comparisons.json"), ("predictions", "predictions.jsonl")
        )},
    })


if __name__ == "__main__":
    main()
