"""Run PS-1 zero-training closed-loop PS-Rule evaluation on Square validation and test seeds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from collect_factorial import load_square_runner
from ps_resrl.closed_loop_phase import run_ps_closed_loop_episode


METRICS = (
    "success",
    "return",
    "discounted_return",
    "act_calls",
    "environment_steps",
    "stall_cap_triggers",
    "mean_phase_velocity",
    "residual_saturation_rate",
    "action_jerk_l2",
    "wall_seconds",
    "policy_wall_seconds",
    "wall_clock_ms_per_step",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _quantile(sorted_vals: list[float], frac: float) -> float:
    rank = (len(sorted_vals) - 1) * frac
    lo, hi = math.floor(rank), math.ceil(rank)
    v = sorted_vals[lo]
    return v + (sorted_vals[hi] - v) * (rank - lo)


def _bootstrap(values: list[float], samples: int, seed: str) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(math.fsum(rng.choices(values, k=n)) / n)
    means.sort()
    return {
        "mean": math.fsum(values) / n,
        "ci95": [_quantile(means, 0.025), _quantile(means, 0.975)],
    }


def _summarize_cohort(rows: list[dict[str, Any]], *, samples: int, seed_prefix: str) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "successes": sum(int(r["success"]) for r in rows),
        "estimates": {
            key: _bootstrap([float(r[key]) for r in rows], samples, f"{seed_prefix}:{key}")
            for key in METRICS
        },
    }


def _paired_diff(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    samples: int,
    seed_prefix: str,
) -> dict[str, Any]:
    assert [r["seed"] for r in rows_a] == [r["seed"] for r in rows_b]
    return {
        key: _bootstrap(
            [float(a[key]) - float(b[key]) for a, b in zip(rows_a, rows_b)],
            samples,
            f"{seed_prefix}:{key}",
        )
        for key in ("success", "discounted_return", "act_calls", "residual_saturation_rate", "action_jerk_l2")
    }


def _write_summary(
    out: Path,
    config: dict[str, Any],
    *,
    orig_val: list[dict[str, Any]],
    val_candidates: list[dict[str, Any]],
    best_kappa: float,
    best_nu_min: float,
    test_by_delay: dict[str, Any],
) -> None:
    samples = int(config["bootstrap_samples"])
    bseed = int(config["bootstrap_seed"])
    cf_ref = json.loads((REPO_ROOT / config["cf_closed_loop_reference"]).read_text(encoding="utf-8"))
    ref_15 = cf_ref["fixed_act_budgets"]["15"]
    summary = {
        "task": "Square",
        "selected_ps_rule": {"kappa": best_kappa, "nu_min": best_nu_min, "nu_max": config["nu_max"]},
        "validation_original_resfit": _summarize_cohort(
            orig_val, samples=samples, seed_prefix=f"{bseed}:val:orig"
        ),
        "validation_grid": [
            {"kappa": c["kappa"], "nu_min": c["nu_min"], "summary": c["summary"]}
            for c in val_candidates
        ],
        "held_out_test_by_refill_delay": test_by_delay,
        "cf_resrl_mvp_test_reference_15cap": {
            "original_resfit": ref_15["original_resfit"]["estimates"],
            "fixed_period_refresh": ref_15["fixed_period_refresh"]["estimates"],
            "B5": ref_15["B5"]["estimates"],
            "B6": ref_15["B6"]["estimates"],
            "P1": ref_15["P1"]["estimates"],
        },
    }
    _write_json(out / "summary.json", summary)

    lines = [
        "# PS-1 Closed-Loop Evaluation: PS-Rule (B2) vs Original ResFiT & CF-ResRL MVP",
        "",
        f"Validation-selected setting: `kappa={best_kappa}`, `nu_min={best_nu_min}`, `nu_max={config['nu_max']}`.",
        "",
        "## Held-Out Test (50 seeds `20269124..20269173`) Across Refill Delays",
        "",
        "| Refill Delay τ | Method | Success | Mean ACT Calls | Mean Steps | Sat Rate | Action Jerk L2 | ΔSuccess vs Orig (95% CI) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for delay in config["async_refill_delays"]:
        blk = test_by_delay[str(delay)]
        for name, label in (("original_resfit", "Original ResFiT"), ("ps_rule", "PS-Rule (B2)")):
            est = blk[name]["estimates"]
            succ = f"{blk[name]['successes']}/50 ({est['success']['mean']*100:.1f}%)"
            diff_str = "—"
            if name == "ps_rule":
                ds = blk["ps_rule_minus_original"]["success"]
                diff_str = f"{ds['mean']*100:+.1f} [{ds['ci95'][0]*100:+.1f}, {ds['ci95'][1]*100:+.1f}]"
            lines.append(
                f"| {delay} | {label} | {succ} | {est['act_calls']['mean']:.2f} | "
                f"{est['environment_steps']['mean']:.1f} | {est['residual_saturation_rate']['mean']:.3f} | "
                f"{est['action_jerk_l2']['mean']:.4f} | {diff_str} |"
            )
    lines.extend(["", f"Attempt directory: `{out}`"])
    (out / "COMPLETED_RUN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sha_lines = [
        f"{_sha256(out / name)}  {name}"
        for name in ("resolved_config.json", "episode_records.jsonl", "summary.json", "COMPLETED_RUN.md")
    ]
    (out / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    _write_json(out / "status.json", {"status": "COMPLETED", "summary_sha256": _sha256(out / "summary.json")})
    print(f"PS-1 completed: {out}", flush=True)


def _summarize_from_existing(out: Path, config: dict[str, Any]) -> None:
    rows = [
        json.loads(line)
        for line in (out / "episode_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples = int(config["bootstrap_samples"])
    bseed = int(config["bootstrap_seed"])
    orig_val = [r for r in rows if r["split"] == "validation" and r["method"] == "original_resfit"]
    val_candidates = []
    for kappa, nu_min in itertools.product(
        config["validation_grid"]["kappa"], config["validation_grid"]["nu_min"]
    ):
        sub = [
            r
            for r in rows
            if r["split"] == "validation"
            and r["method"] == "ps_rule"
            and r["kappa"] == kappa
            and r["nu_min"] == nu_min
        ]
        stats = _summarize_cohort(sub, samples=samples, seed_prefix=f"{bseed}:val:{kappa}:{nu_min}")
        val_candidates.append({
            "kappa": kappa,
            "nu_min": nu_min,
            "summary": stats,
            "key": (
                stats["successes"],
                stats["estimates"]["discounted_return"]["mean"],
                -stats["estimates"]["act_calls"]["mean"],
            ),
        })
    best = max(val_candidates, key=lambda item: item["key"])
    best_kappa = float(best["kappa"])
    best_nu_min = float(best["nu_min"])
    test_by_delay: dict[str, Any] = {}
    for delay in config["async_refill_delays"]:
        orig_rows = [
            r
            for r in rows
            if r["split"] == "test"
            and r["method"] == "original_resfit"
            and r["refill_delay_steps"] == delay
        ]
        ps_rows = [
            r
            for r in rows
            if r["split"] == "test"
            and r["method"] == "ps_rule"
            and r["refill_delay_steps"] == delay
            and r["kappa"] == best_kappa
            and r["nu_min"] == best_nu_min
        ]
        test_by_delay[str(delay)] = {
            "original_resfit": _summarize_cohort(
                orig_rows, samples=samples, seed_prefix=f"{bseed}:test_d{delay}:orig"
            ),
            "ps_rule": _summarize_cohort(
                ps_rows, samples=samples, seed_prefix=f"{bseed}:test_d{delay}:ps"
            ),
            "ps_rule_minus_original": _paired_diff(
                ps_rows, orig_rows, samples=samples, seed_prefix=f"{bseed}:test_d{delay}:diff"
            ),
        }
    _write_summary(
        out,
        config,
        orig_val=orig_val,
        val_candidates=val_candidates,
        best_kappa=best_kappa,
        best_nu_min=best_nu_min,
        test_by_delay=test_by_delay,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/ps_resrl/ps1_square_rule_eval.json",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summarize-from", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.summarize_from is not None:
        _summarize_from_existing(args.summarize_from, config)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.out or (REPO_ROOT / config["output_root"] / "attempts" / stamp)
    out.mkdir(parents=True, exist_ok=False)

    val_seeds = list(
        range(config["validation_seeds"]["first"], config["validation_seeds"]["first"] + config["validation_seeds"]["count"])
    )
    test_seeds = list(
        range(config["test_seeds"]["first"], config["test_seeds"]["first"] + config["test_seeds"]["count"])
    )
    gate_config = json.loads((REPO_ROOT / config["gate_config"]).read_text(encoding="utf-8"))
    runner, provenance = load_square_runner(gate_config)
    _write_json(out / "resolved_config.json", {**config, "provenance": provenance})
    _write_json(out / "status.json", {"status": "RUNNING"})

    samples = int(config["bootstrap_samples"])
    bseed = int(config["bootstrap_seed"])
    all_rows: list[dict[str, Any]] = []
    jsonl_path = out / "episode_records.jsonl"

    try:
        with jsonl_path.open("w", encoding="utf-8") as stream:
            def run_batch(split: str, method: str, seeds: list[int], *, kappa: float = 1.0, nu_min: float = 0.0, delay: int = 0):
                batch = []
                for seed in seeds:
                    rec = run_ps_closed_loop_episode(
                        runner,
                        seed=seed,
                        method=method,
                        kappa=kappa,
                        nu_min=nu_min,
                        nu_max=float(config["nu_max"]),
                        refill_delay_steps=delay,
                        saturation_threshold=float(config["saturation_threshold"]),
                    )
                    rec["split"] = split
                    batch.append(rec)
                    all_rows.append(rec)
                    stream.write(json.dumps(rec, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                return batch

            # 1. Parity check on validation + test for original_resfit (refill_delay=0)
            orig_val = run_batch("validation", "original_resfit", val_seeds, delay=0)
            orig_test_d0 = run_batch("test", "original_resfit", test_seeds, delay=0)
            orig_test_succ = sum(int(r["success"]) for r in orig_test_d0)
            orig_test_calls = sum(r["act_calls"] for r in orig_test_d0) / len(orig_test_d0)
            print(
                f"Parity check original_resfit test(delay=0): {orig_test_succ}/50, "
                f"mean ACT calls={orig_test_calls:.2f}",
                flush=True,
            )
            if (
                orig_test_succ != int(config["expected_original_test_successes"])
                or not math.isclose(orig_test_calls, float(config["expected_original_test_act_calls"]), abs_tol=1e-6)
            ):
                raise RuntimeError(
                    f"Original ResFiT parity mismatch: got {orig_test_succ}/50, {orig_test_calls:.4f} calls"
                )

            # 2. Validation-only grid search for PS-Rule
            val_candidates = []
            for kappa, nu_min in itertools.product(
                config["validation_grid"]["kappa"], config["validation_grid"]["nu_min"]
            ):
                rows = run_batch("validation", "ps_rule", val_seeds, kappa=kappa, nu_min=nu_min, delay=0)
                stats = _summarize_cohort(rows, samples=samples, seed_prefix=f"{bseed}:val:{kappa}:{nu_min}")
                val_candidates.append({
                    "kappa": kappa,
                    "nu_min": nu_min,
                    "summary": stats,
                    "key": (
                        stats["successes"],
                        stats["estimates"]["discounted_return"]["mean"],
                        -stats["estimates"]["act_calls"]["mean"],
                    ),
                })
                print(
                    f"Validation PS-Rule(kappa={kappa}, nu_min={nu_min}): "
                    f"{stats['successes']}/{len(val_seeds)}, "
                    f"ret={stats['estimates']['discounted_return']['mean']:.4f}, "
                    f"calls={stats['estimates']['act_calls']['mean']:.2f}",
                    flush=True,
                )

            best = max(val_candidates, key=lambda item: item["key"])
            best_kappa = float(best["kappa"])
            best_nu_min = float(best["nu_min"])
            print(f"Selected on validation: kappa={best_kappa}, nu_min={best_nu_min}", flush=True)

            # 3. Held-out test across async refill delays [0, 2, 4]
            test_by_delay: dict[str, Any] = {}
            for delay in config["async_refill_delays"]:
                if delay == 0:
                    orig_rows = orig_test_d0
                else:
                    orig_rows = run_batch("test", "original_resfit", test_seeds, delay=delay)
                ps_rows = run_batch(
                    "test", "ps_rule", test_seeds, kappa=best_kappa, nu_min=best_nu_min, delay=delay
                )
                test_by_delay[str(delay)] = {
                    "original_resfit": _summarize_cohort(
                        orig_rows, samples=samples, seed_prefix=f"{bseed}:test_d{delay}:orig"
                    ),
                    "ps_rule": _summarize_cohort(
                        ps_rows, samples=samples, seed_prefix=f"{bseed}:test_d{delay}:ps"
                    ),
                    "ps_rule_minus_original": _paired_diff(
                        ps_rows, orig_rows, samples=samples, seed_prefix=f"{bseed}:test_d{delay}:diff"
                    ),
                }
    except Exception as exc:
        _write_json(
            out / "status.json",
            {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    finally:
        runner.wrapper.close()

    _write_summary(
        out,
        config,
        orig_val=orig_val,
        val_candidates=val_candidates,
        best_kappa=best_kappa,
        best_nu_min=best_nu_min,
        test_by_delay=test_by_delay,
    )


if __name__ == "__main__":
    main()
