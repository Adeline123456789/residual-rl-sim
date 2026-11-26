"""Verify a completed Square closed-loop evaluation from its recorded evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_scheduler import (
    MEASUREMENTS,
    PROBE_ROWS,
    ORIGINAL,
    Setting,
    _choose_validation,
    _physical_budget,
    _seeds,
    _settings,
    _summarize,
)
from scripts.collect_factorial import _json, _sha256


ACT_CHUNK_LENGTH = 20
ACTION_DIMENSION = 7
CONTROL_SECONDS = 0.05
X1_RUN_EVIDENCE = (
    "x1_cell", "x1_config_sha256", "source_index_sha256", "source_manifest_sha256",
    "offline_results_sha256", "test_reset_scope",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _numeric(value):
    return type(value) in (float, int) and math.isfinite(value)


def _physical_action(values):
    return isinstance(values, list) and len(values) == ACTION_DIMENSION and all(
        _numeric(value) and -1 <= value <= 1 for value in values
    )


def _verify_steps(trace, *, horizon, delay, gamma, residual_transform=None):
    """Reconstruct the chunk clock and ACT consumption from the step ledger."""
    steps = trace["steps"]
    _require(0 < len(steps) == trace["environment_steps"] <= horizon, "Bad episode length")
    _require(trace["initial_act_calls"] == 1, "Expected exactly one initial ACT call")
    _require(1 <= trace["act_calls"] <= trace["act_budget"], "Episode exceeds its ACT cap")
    method, parameter = trace["method"], trace["parameter"]
    active_source = active_generation = 0
    pending = None
    calls, nominal_calls, dispatches, activations = 1, 0, 0, 0
    rewards = []
    for tick, row in enumerate(steps):
        _require(row["tick"] == tick and row["control_steps"] == tick + 1, f"Discontinuous step {tick}")
        _require(
            _numeric(row["simulation_time"])
            and math.isclose(row["simulation_time"], CONTROL_SECONDS * (tick + 1), abs_tol=1e-8),
            f"Bad simulation time at step {tick}",
        )
        _require(_physical_action(row["applied_env_action"]), f"Physical action out of bounds at step {tick}")
        base, residual = row["base_action"], row["proposed_residual"]
        _require(
            _physical_action(base) and isinstance(residual, list)
            and len(residual) == ACTION_DIMENSION and all(map(_numeric, residual)),
            f"Invalid normalized base or residual at step {tick}",
        )
        if residual_transform is not None:
            scale, limit = residual_transform
            original = row["actor_original_residual"]
            _require(
                isinstance(original, list) and len(original) == ACTION_DIMENSION
                and all(_numeric(value) and abs(value) <= limit + 1e-8 for value in original),
                f"Invalid original X1 actor residual at step {tick}",
            )
            scaled = [scale * value for value in original]
            _require(
                row["actor_scaled_residual"] == scaled
                and row["actor_scale_clip_mask"] == [abs(value) > limit for value in scaled]
                and residual == [max(-limit, min(value, limit)) for value in scaled],
                f"Wrong X1 residual scaling or clamp at step {tick}",
            )
        proposal = [value + delta for value, delta in zip(base, residual)]
        _require(
            row["proposed_action"] == proposal
            and row["applied_action"] == [max(-1, min(value, 1)) for value in proposal]
            and _physical_action(row["applied_normalized_action"]),
            f"Wrong normalized action limit at step {tick}",
        )
        activated = pending is not None and pending[1] == tick
        if activated:
            active_source, _, active_generation = pending
            pending = None
            activations += 1
        index = tick - active_source
        refill = pending is None and index >= ACT_CHUNK_LENGTH
        if refill:
            active_source, active_generation, index = tick, calls, 0
            calls += 1
            nominal_calls += 1
        _require(
            (row["chunk_source_tick"], row["generation"], row["chunk_index"])
            == (active_source, active_generation, index),
            f"Wrong chunk or index at step {tick}",
        )
        _require(row["nominal_refill"] is refill, f"Wrong nominal refill flag at step {tick}")
        _require(row["refresh_activated"] is activated, f"Wrong refresh activation flag at step {tick}")
        fallback = index >= ACT_CHUNK_LENGTH
        _require(row["used_fallback"] is fallback, f"Wrong fallback flag at step {tick}")
        if fallback:
            _require(
                pending is not None and base == [0.0] * ACTION_DIMENSION,
                f"Fallback did not use the registered zero base while pending at step {tick}",
            )

        source, ready = tick + 1, tick + 1 + delay
        reserve = math.ceil(max(horizon - source - ACT_CHUNK_LENGTH, 0) / ACT_CHUNK_LENGTH)
        block = (
            "pending" if pending is not None else
            "horizon" if ready >= horizon else
            "act_budget" if calls + 1 + reserve > trace["act_budget"] else None
        )
        _require(row["refresh_block"] == block, f"Wrong refresh eligibility at step {tick}")
        score, requested = row["decision_score"], False
        if block is not None or method == ORIGINAL:
            _require(score is None, f"Unexpected score while refresh was disabled at step {tick}")
        else:
            _require(_numeric(score), f"Invalid decision score at step {tick}")
            threshold = parameter
            if method == "fixed_period_refresh":
                _require(score == float(source % parameter == 0), f"Wrong period gate at step {tick}")
                threshold = 0
            elif method == "residual_norm_threshold":
                _require(
                    math.isclose(score, math.sqrt(math.fsum(value * value for value in residual))),
                    f"Wrong residual norm at step {tick}",
                )
            requested = score > threshold
        _require(row["refresh_requested_after_step"] is requested, f"Wrong refresh request at step {tick}")

        reward = row["reward"]
        _require(_numeric(reward) and reward in (0, 1), f"Invalid Square sparse reward at step {tick}")
        _require(row["terminated"] is (reward == 1), f"Reward/termination mismatch at step {tick}")
        _require(row["truncated"] is (tick + 1 == horizon), f"Wrong truncation flag at step {tick}")
        done = row["terminated"] or row["truncated"]
        _require(done is (tick + 1 == len(steps)), f"Episode continues or ends early at step {tick}")
        rewards.append(reward)

        dispatched = requested and not done
        _require(row["refresh_dispatched"] is dispatched, f"Wrong refresh dispatch at step {tick}")
        # The scheduler advances the control clock before accepting this step's request.
        expected_pending = pending
        if expected_pending is not None and expected_pending[1] == tick + 1:
            expected_pending = None
        if dispatched:
            pending = expected_pending = (source, ready, calls)
            calls += 1
            dispatches += 1
        _require(
            (row["pending_refresh_source_tick"], row["pending_refresh_ready_tick"])
            == (expected_pending[:2] if expected_pending is not None else (None, None)),
            f"Wrong pending refresh timing at step {tick}",
        )
        _require(
            row["act_calls"] == calls <= trace["act_budget"]
            and row["act_calls_since_initial_chunk"] == calls - 1,
            f"ACT ledger diverges from its cap or the inferred calls at step {tick}",
        )

    _require(
        (trace["act_calls"], trace["nominal_act_calls"], trace["refresh_requests"], trace["refresh_activations"])
        == (calls, nominal_calls, dispatches, activations),
        "Episode ACT totals disagree with its steps",
    )
    _require(trace["success"] is steps[-1]["terminated"], "Episode success disagrees with termination")
    _require(trace["return"] == sum(rewards), "Episode return disagrees with its rewards")
    _require(
        math.isclose(trace["discounted_return"], sum(gamma**i * reward for i, reward in enumerate(rewards))),
        "Discounted return disagrees with its rewards",
    )
    _require(
        _numeric(trace["wall_seconds"]) and _numeric(trace["policy_wall_seconds"])
        and 0 <= trace["policy_wall_seconds"] <= trace["wall_seconds"],
        "Invalid policy or episode wall-clock duration",
    )


def _archived_attempt(recorded, bucket):
    path = Path(recorded)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_dir():
        return path
    prefix = Path("results/cf_resrl") / bucket
    parts = path.parts
    for offset, part in enumerate(parts):
        if part == "results" and parts[offset:offset + len(prefix.parts)] == prefix.parts:
            restored = ROOT.joinpath(*parts[offset:])
            if restored.is_dir():
                return restored
    raise ValueError(f"Cannot locate archived {bucket} evidence for {recorded}")


def _verify_probe_rows(manifest, offline_results, distribution):
    _require(set(manifest["probes"]) == PROBE_ROWS.keys(), "Missing frozen probe provenance")
    for method, name in PROBE_ROWS.items():
        probe = manifest["probes"][method]
        matching = [row for row in offline_results if row["method"] == name and row["status"] == "COMPLETED"
                    and row["distribution"] == distribution and row["fraction"] == 1.0]
        _require(
            len(matching) == 1 and probe["offline_method"] == name
            and all(probe[key] == matching[0][key] for key in (
                "distribution", "fraction", "model", "model_sha256", "training_budget",
            )),
            f"Wrong frozen {method} probe provenance",
        )


def _verify_runtime(source, runtime_locks):
    _require(
        runtime_locks.keys() == {"conda-explicit.txt", "environment.lock.txt"}
        and all(_sha256(source / "runtime" / name) == entry["sha256"]
                for name, entry in runtime_locks.items()),
        "Pinned runtime lock SHA differs from the original ResFiT evidence",
    )


def _mvp_source_protocol(manifest, protocol, locked_gate):
    _require(
        protocol == _json(ROOT / "configs/cf_resrl/square_closed_loop.json"),
        "Run protocol differs from the registered closed-loop config",
    )
    frozen = manifest["frozen_square_protocol"]
    _require(
        frozen == locked_gate,
        "Gate provenance differs from the registered Square policy",
    )
    offline = ROOT / protocol["offline_attempt"]
    offline_manifest = _json(offline / "run_manifest.json")
    _require(
        _sha256(offline / "results.json") == manifest["offline_results_sha256"] == protocol["offline_results_sha256"]
        and _sha256(offline / "run_manifest.json") == manifest["offline_manifest_sha256"],
        "Offline evidence SHA differs from the closed-loop provenance",
    )
    identities = manifest["identities"]
    _require(
        identities == offline_manifest["data"]["identities"]
        and identities["policy_identity"] == manifest["act_artifacts"]["model.safetensors"]
        and identities["residual_policy_identity"] == frozen["resfit_checkpoint_sha256"],
        "Frozen model or normalization identity differs from pipeline evidence",
    )
    _verify_probe_rows(manifest, _json(offline / "results.json"), "natural")
    data = offline_manifest["data"]
    e0_path = Path(data["attempt"])
    e0_relative = Path("results/cf_resrl/mvp_square/e0_data/attempts") / e0_path.name
    _require(e0_path.parts[-len(e0_relative.parts):] == e0_relative.parts, "Invalid archived E0 path")
    e0_manifest = ROOT / e0_relative / "run_manifest.json"
    _require(_sha256(e0_manifest) == data["run_manifest_sha256"], "E0 source manifest SHA differs")
    prior = {row["seed"] for row in _json(e0_manifest)["source_plan"]}
    source = ROOT / frozen["resfit_attempt"]
    _verify_runtime(source, manifest["source_resfit_dependency_locks"])
    gamma = _json(source / "training/resolved_training_config.json")["config"]["algo"]["gamma"]
    return frozen, gamma, prior


def _x1_act_artifacts(registration, label):
    act = registration["act"]
    source = ROOT / act["attempt"]
    evidence = source / "evidence_manifest.json"
    _require(_sha256(evidence) == act["evidence_manifest_sha256"], "X1 ACT text evidence differs")
    evidence = _json(evidence)
    _require(
        evidence["status"] == "COMPLETED" and evidence["attempt_id"] == source.name
        and all(evidence["upstream"][key] == value for key, value in registration["upstream"].items()),
        "X1 ACT evidence differs from the frozen training run",
    )
    digests = {"config.json": act["config_sha256"], "model.safetensors": act["models"][str(label)]}
    for name, digest in digests.items():
        record = evidence["files"][f"training/policy_step_{label}/policy/{name}"]
        _require(record["sha256"] == digest, f"X1 ACT step {label} checksum differs for {name}")
    return digests


def _x1_resfit_source(manifest, registration, cell):
    resfit = registration["resfit"]
    seed, step = cell["resfit_seed"], cell["resfit_step"]
    is_m0 = seed == resfit["m0_seed"]
    source = _archived_attempt(
        manifest["source_resfit_attempt"], "m0/resfit_training" if is_m0 else "expansion_square/resfit_training",
    )
    if is_m0:
        _require(source.resolve() == (ROOT / resfit["m0_attempt"]).resolve(), "Wrong pinned M0 ResFiT source")
    else:
        launch = _json(source / "resolved_launcher.json")
        _require(
            launch["config_sha256"] == _sha256(ROOT / resfit["training_config"])
            and launch["training_seed"] == seed and launch["overlay_mode"] == "resumable"
            and launch["resume_checkpoint_interval_environment_steps"] == 100000
            and launch["resume_checkpoint_retention"] == 3,
            "X1 training provenance differs from the registered resumable launch",
        )

    launch_status = _json(source / "status.json")
    training = source / "training"
    checkpoints = training / "checkpoint_manifest.json"
    resolved = training / "resolved_training_config.json"
    _require(
        launch_status["status"] == "COMPLETED" and launch_status["dry_run"] is False
        and launch_status["child_returncode"] == 0 and launch_status["resfit_clean_after_cleanup"] is True
        and launch_status["lerobot_clean_after_cleanup"] is True
        and launch_status["environment_step_budget"] == resfit["registered_budget"]
        and _sha256(checkpoints) == launch_status["checkpoint_manifest_sha256"],
        "X1 ResFiT source is missing completed training evidence",
    )
    if is_m0:
        _require(
            _sha256(checkpoints) == resfit["m0_manifest_sha256"]
            and _sha256(resolved) == resfit["m0_resolved_training_sha256"],
            "Pinned M0 ResFiT training metadata differs",
        )
    _require(
        manifest["source_resfit_dependency_locks"] == launch_status["runtime"]["locks"],
        "X1 runtime lock records differ from completed training",
    )
    _verify_runtime(source, manifest["source_resfit_dependency_locks"])
    checkpoints = _json(checkpoints)
    records = checkpoints["checkpoints"]
    _require(
        checkpoints["selection"] == "fixed_environment_step"
        and checkpoints["checkpoint_interval_environment_steps"] == 10000
        and checkpoints["environment_step_budget"] == resfit["registered_budget"]
        and checkpoints["action_ledger"]["records"] == resfit["registered_budget"]["total"]
        and [record["global_step"] for record in records] == list(range(10000, 300001, 10000)),
        "X1 ResFiT checkpoint manifest differs from the fixed budget and schedule",
    )
    record = records[step // 10000 - 1]
    identity = manifest["identities"]["residual_policy_identity"]
    _require(
        record["selection"] == "fixed_environment_step" and record["sha256"] == identity
        and Path(record["path"]).parts[-3:] == ("training", "models", f"step_{step:09d}.pt")
        and (not is_m0 or identity == resfit["m0_models"][str(step)]),
        "X1 ResFiT policy digest differs from its registered fixed checkpoint",
    )

    resolved = _json(resolved)
    base_label = registration["act"]["training_base_label"]
    training_base = _x1_act_artifacts(registration, base_label)
    selected_act = _x1_act_artifacts(registration, cell["act_label"])
    _require(
        resolved["config"]["seed"] == seed and resolved["config"]["task"] == registration["task"]
        and resolved["environment_step_budget"] == resfit["registered_budget"]
        and resolved["normalization"]["action_scale"] == registration["residual_component_limit"]
        and resolved["base_policy_artifacts"] == manifest["resfit_training_base_artifacts"] == training_base
        and manifest["upstream"] == registration["upstream"] and manifest["dataset"] == resolved["dataset"]
        and manifest["act_artifacts"] == selected_act
        and manifest["identities"]["policy_identity"] == selected_act["model.safetensors"]
        and manifest["act_base_swap"] is (selected_act != training_base),
        "X1 frozen policies, dataset or upstream do not match the registered training and evaluation modes",
    )
    return resolved["config"]["algo"]["gamma"]


def _x1_source_protocol(manifest, protocol):
    from cf_resrl.x1_plan import DEFAULT_CONFIG, load_x1_config, plan_x1, source_cohort
    from scripts.eval_scheduler import X1Run, _x1_closed_loop_protocol

    registration = load_x1_config(DEFAULT_CONFIG)
    config_sha = _sha256(DEFAULT_CONFIG)
    cell = manifest["x1_cell"]
    offline = _archived_attempt(manifest["offline_attempt"], "expansion_square/offline_probe")
    x1 = X1Run(registration, config_sha, cell, offline, None)
    _require(
        manifest["x1_config_sha256"] == config_sha and cell in plan_x1(registration)
        and protocol == _x1_closed_loop_protocol(x1),
        "X1 rollout protocol differs from the registered cell",
    )
    offline_status = _json(offline / "status.json")
    _require(offline_status["status"] == "COMPLETED", "X1 offline probe attempt did not complete")
    for name, filename in (("run_manifest", "run_manifest.json"), ("results", "results.json")):
        report_key = "offline_manifest_sha256" if name == "run_manifest" else "offline_results_sha256"
        _require(
            _sha256(offline / filename) == manifest[report_key] == offline_status[f"{name}_sha256"],
            f"X1 offline provenance differs for {filename}",
        )
    offline_manifest = _json(offline / "run_manifest.json")
    data = offline_manifest["data"]
    _require(
        data["x1_cell"] == cell and data["x1_config_sha256"] == config_sha
        and data["identities"] == manifest["identities"],
        "X1 offline probes used a different source cell or policy",
    )
    offline_results = _json(offline / "results.json")
    distribution = cell["distribution"]
    _require({row["distribution"] for row in offline_results} == {distribution}, "X1 probe distributions mixed")
    _verify_probe_rows(manifest, offline_results, distribution)

    source = _archived_attempt(data["attempt"], "expansion_square/branch_data")
    source_status = _json(source / "status.json")
    plan = source_cohort(registration, distribution)
    _require(
        source_status["status"] == "COMPLETED"
        and source_status["completed_groups"] == source_status["planned_groups"] == len(plan),
        "X1 branch source is incomplete",
    )
    for name, filename in (
        ("status", "status.json"), ("index", "group_index.jsonl"), ("summary", "summary.json"),
        ("run_manifest", "run_manifest.json"), ("feature_schema", "feature_schema.json"),
    ):
        key = f"{name}_sha256"
        _require(_sha256(source / filename) == data[key], f"X1 branch evidence differs for {filename}")
        if name in ("index", "summary", "run_manifest"):
            _require(source_status[key] == data[key], f"X1 branch status checksum differs for {filename}")
    with (source / "group_index.jsonl").open() as stream:
        source_rows = [json.loads(line) for line in stream]
    _require(
        len(source_rows) == len(plan)
        and all(all(row[key] == value for key, value in assigned.items())
                for row, assigned in zip(source_rows, plan)),
        "X1 source index differs from the fixed episode roles",
    )
    collection = _json(source / "run_manifest.json")
    frozen = manifest["frozen_square_protocol"]
    _require(
        frozen == collection["protocol"] and frozen["status"] == "REGISTERED_X1_CELL"
        and frozen["x1_cell"] == cell and frozen["x1_config_sha256"] == config_sha
        and frozen["task"] == registration["task"] and frozen["simulation_horizon_steps"] == 300
        and frozen["refresh_delay_steps"] == cell["delay"]
        and frozen["residual_scale"] == manifest["residual_scale"] == cell["residual_scale"]
        and frozen["residual_component_limit"] == manifest["normalized_residual_component_limit"]
        == registration["residual_component_limit"]
        and collection["identities"] == manifest["identities"] and collection["source_plan"] == plan
        and manifest["source_index_sha256"] == data["index_sha256"]
        and manifest["source_manifest_sha256"] == data["run_manifest_sha256"]
        and manifest["test_reset_scope"] == "shared locked Square resets with no natural or enriched tick selection",
        "X1 closed-loop rollout provenance differs from the registered branch source",
    )
    mvp = frozen["mvp"]
    mvp_attempt = _archived_attempt(mvp["attempt"], "mvp_square/closed_loop")
    mvp_status = _json(mvp_attempt / "status.json")
    _require(
        mvp_status["status"] == "COMPLETED"
        and _sha256(mvp_attempt / "summary.json") == mvp_status["summary_sha256"] == mvp["summary_sha256"],
        "X1 collection does not bind to completed Square MVP evidence",
    )
    gamma = _x1_resfit_source(manifest, registration, cell)
    return frozen, gamma, {row["seed"] for row in plan}


def _source_protocol(manifest):
    protocol = manifest["protocol"]
    locked_gate = _json(ROOT / "configs/cf_resrl/m0_square_branch_gate.json")
    if "x1_cell" in manifest:
        frozen, gamma, prior = _x1_source_protocol(manifest, protocol)
    else:
        frozen, gamma, prior = _mvp_source_protocol(manifest, protocol, locked_gate)
    prior.update(locked_gate["source_seeds"])
    m0 = _json(ROOT / "configs/cf_resrl/m0_act_training.json")
    seed = m0["training"]["seed"]
    prior.add(seed)
    prior.update(range(seed + 1000, seed + 1000 + m0["evaluation"]["num_episodes"]))
    _require(len(prior) == manifest["prior_seed_count"], "Prior seed count differs from source evidence")
    cohorts = _seeds(protocol, prior)
    _require(
        {key: list(seeds) for key, seeds in cohorts.items()} == manifest["cohorts"],
        "Recorded cohorts differ from the registered disjoint split",
    )
    return protocol, frozen, cohorts, gamma


def _verify_cohorts(attempt, grouped, config, cohorts, settings, selection, summary, *, x1_evidence=None):
    if x1_evidence is not None:
        _require(
            all(summary[field] == value for field, value in x1_evidence.items()),
            "X1 test summary differs from the frozen cell evidence",
        )
    validation_plan = list(dict.fromkeys(
        setting for methods in settings.values() for options in methods.values() for setting in options
    ))
    scores = {
        setting: _summarize(grouped["validation", setting], config) for setting in validation_plan
    }
    _require(
        selection["candidates"] == [{"setting": asdict(setting), **metrics} for setting, metrics in scores.items()],
        "Locked validation candidates differ from recorded cohort outcomes",
    )
    _require(
        selection["validation_seeds"] == list(cohorts["validation"])
        and selection["locked_test_seeds"] == list(cohorts["test"]),
        "Selection cohorts disagree with the frozen protocol",
    )
    target, budgets, performance = _choose_validation(settings, scores)
    target_rate = target / len(cohorts["validation"])
    expected_budget_choices = {
        str(cap): {method: asdict(setting) for method, setting in methods.items()}
        for cap, methods in budgets.items()
    }
    expected_performance_choices = {
        method: asdict(setting) if setting else None for method, setting in performance.items()
    }
    _require(
        selection["fixed_budget"] == expected_budget_choices
        and selection["fixed_performance"] == expected_performance_choices
        and selection["target_validation_successes"] == target
        and selection["target_success_rate"] == target_rate,
        "Locked choices differ from the registered validation-only selection rule",
    )
    budget_plan = [setting for methods in budgets.values() for setting in methods.values()]
    performance_plan = [setting for setting in performance.values() if setting is not None]
    test_plan = list(dict.fromkeys(budget_plan + performance_plan))
    _require(
        list(grouped) == [(split, setting) for split, plan in (("validation", validation_plan), ("test", test_plan))
                          for setting in plan],
        "Physical rollout settings differ from the plan or repeat the same cohort",
    )
    for (split, _), rows in grouped.items():
        _require([row["seed"] for row in rows] == list(cohorts[split]), f"Incomplete {split} cohort")
    test_scores = {
        setting: {"setting": asdict(setting), **_summarize(grouped["test", setting], config)}
        for setting in test_plan
    }
    expected_budget_scores = {
        str(cap): {method: test_scores[setting] for method, setting in methods.items()}
        for cap, methods in budgets.items()
    }
    expected_performance_scores = {}
    for method, setting in performance.items():
        if setting is None:
            expected_performance_scores[method] = {"status": "NO_VALIDATION_SETTING_MEETS_TARGET"}
            continue
        test = test_scores[setting]
        expected_performance_scores[method] = {
            "status": "EVALUATED", "met_target_on_test": test["successes"] / test["episodes"] >= target_rate,
            **test,
        }
    totals = {
        split: _physical_budget([row for (key, _), rows in grouped.items() if key == split for row in rows])
        for split in cohorts
    }
    _require(
        summary["fixed_act_budgets"] == expected_budget_scores
        and summary["fixed_task_performance"] == expected_performance_scores
        and summary["fixed_performance_success_target"] == target_rate
        and summary["physical_rollout_budget"] == totals,
        "Test summaries or physical rollout costs disagree with recorded episodes",
    )
    fields = ("act_budget", "method", "parameter", "episodes", "success_rate", "mean_return",
              "mean_act_calls", "mean_wall_seconds")
    expected_csv = [list(fields)]
    # The writer emits caps and policy names in protocol order, before JSON sorts object keys.
    for cap, methods in budgets.items():
        for method, setting in methods.items():
            row = test_scores[setting]
            values = [cap, method, setting.parameter, row["episodes"], *(
                row["estimates"][metric]["mean"] for metric in ("success", "return", "act_calls", "wall_seconds")
            )]
            expected_csv.append(["" if value is None else str(value) for value in values])
    with (attempt / "test_budget_curve.csv").open(newline="") as stream:
        _require(list(csv.reader(stream)) == expected_csv, "Budget curve table disagrees with test episodes")
    return {
        "fixed_budget_test_physical_rollouts": len(set(budget_plan)) * len(cohorts["test"]),
        "fixed_performance_extra_test_physical_rollouts": (len(test_plan) - len(set(budget_plan))) * len(cohorts["test"]),
        "reused_logical_test_rollouts": (len(budget_plan) + len(performance_plan) - len(test_plan)) * len(cohorts["test"]),
        "target_success_rate": target_rate,
        "physical_rollout_budget": totals,
    }


def verify(attempt):
    attempt = Path(attempt).resolve()
    status = _json(attempt / "status.json")
    _require(status["status"] == "COMPLETED", "Only a completed Square attempt can be verified")
    checksums = {
        name: _sha256(attempt / name)
        for name in ("status.json", "run_manifest.json", "episode_index.jsonl", "summary.json",
                     "validation_selection.json", "stdout_stderr.log", "test_budget_curve.csv",
                     "test_budget_curve.png")
    }
    for field, name in (
        ("index_sha256", "episode_index.jsonl"), ("summary_sha256", "summary.json"),
        ("selection_sha256", "validation_selection.json"),
    ):
        _require(status[field] == checksums[name], f"Status checksum differs for {name}")
    manifest = _json(attempt / "run_manifest.json")
    config, frozen, cohorts, gamma = _source_protocol(manifest)
    horizon, delay = frozen["simulation_horizon_steps"], frozen["refresh_delay_steps"]
    x1_evidence, residual_transform = None, None
    if "x1_cell" in manifest:
        x1_evidence = {field: manifest[field] for field in X1_RUN_EVIDENCE}
        residual_transform = (frozen["residual_scale"], frozen["residual_component_limit"])
        _require(
            status["x1_cell_id"] == manifest["x1_cell"]["cell_id"]
            and status["x1_config_sha256"] == manifest["x1_config_sha256"],
            "Completed status differs from the registered X1 cell",
        )
    settings = _settings(config, horizon, ACT_CHUNK_LENGTH)
    selection, summary = _json(attempt / "validation_selection.json"), _json(attempt / "summary.json")
    selection_sha = checksums["validation_selection.json"]
    log = (attempt / "stdout_stderr.log").read_text(errors="replace")
    lock_line = f"Validation choices locked, SHA-256 {selection_sha}"
    _require(
        summary["validation_selection_sha256"] == selection_sha
        and log.count(lock_line) == 1 and "\ntest:" in log and log.index(lock_line) < log.index("\ntest:"),
        "Test rollout log does not follow the recorded validation lock",
    )
    grouped = defaultdict(list)
    scheduled = {
        setting for methods in settings.values() for options in methods.values() for setting in options
    }
    traces = set()
    last_key = None
    with (attempt / "episode_index.jsonl").open() as stream:
        fields = set(MEASUREMENTS) | {
            "episode_id", "seed", "split", "method", "act_budget", "parameter", "trace", "trace_sha256",
        }
        for number, line in enumerate(stream, start=1):
            row = json.loads(line)
            _require(row.keys() == fields, f"Wrong evidence fields at episode {number}")
            split, seed = row["split"], row["seed"]
            setting = Setting(row["method"], row["act_budget"], row["parameter"])
            episode_id = f"{number:05d}_{split}_{setting.method}_{seed}"
            relative = f"episodes/{episode_id}.json.gz"
            _require(
                split in cohorts and seed in cohorts[split] and setting in scheduled
                and row["episode_id"] == episode_id and row["trace"] == relative,
                f"Unregistered rollout or inconsistent evidence path at episode {number}",
            )
            key = (split, setting)
            _require(key == last_key or key not in grouped, f"Noncontiguous setting at episode {number}")
            last_key = key
            destination = attempt / relative
            _require(
                destination.resolve() == destination
                and _sha256(destination) == row["trace_sha256"],
                f"Raw trace SHA or path differs for {episode_id}",
            )
            with gzip.open(destination, "rt") as source:
                trace = json.load(source)
            _require(
                all(trace[key] == value for key, value in row.items() if key not in {"trace", "trace_sha256"}),
                f"Episode index differs from {episode_id}",
            )
            try:
                _verify_steps(trace, horizon=horizon, delay=delay, gamma=gamma, residual_transform=residual_transform)
            except ValueError as error:
                raise ValueError(f"{episode_id}: {error}") from error
            grouped[key].append(row)
            traces.add(relative)
    _require(
        len(traces) == status["episodes_completed"]
        and traces == {path.relative_to(attempt).as_posix() for path in (attempt / "episodes").iterdir()},
        "Status count or stored trace file set differs from the physical episode index",
    )
    evidence = _verify_cohorts(attempt, grouped, config, cohorts, settings, selection, summary, x1_evidence=x1_evidence)
    return {
        "status": "PASS",
        "attempt": attempt.name,
        "git_commit": manifest["git_commit"],
        "checksums": checksums,
        "trace_checksums_verified": len(traces),
        "cohort_seeds": {split: len(seeds) for split, seeds in cohorts.items()},
        "action_bounds": [-1.0, 1.0],
        "refresh_delay_steps": delay,
        **(x1_evidence or {}),
        **evidence,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.attempt)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Verified {result['trace_checksums_verified']} Square episode traces")


if __name__ == "__main__":
    main()
