"""Verify the completed Square ResFiT attempt using its original local artifacts."""

import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path


ATTEMPT = Path(__file__).resolve().parent
TRAINING = ATTEMPT / "training"
REPO = next(parent for parent in ATTEMPT.parents if (parent / ".git").exists())
BUDGET = {"warmup": 10000, "main": 290000, "total": 300000}
STEPS = tuple(range(10000, 300001, 10000))
EVAL_EPISODES = 50
EVAL_LANES = 8
EVAL_SEED = 20262124
TRAIN_SEED = 20261124
EPISODE_HORIZON = 300
ACTION_DIM = 7
RECONSTRUCTION_ATOL = 5e-7


def read_json(path):
    return json.loads(path.read_bytes())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args, directory=REPO):
    return subprocess.check_output(["git", "-C", str(directory), *args])


def verify_launch(status, manifest):
    launcher = read_json(ATTEMPT / "resolved_launcher.json")
    config_relative = "configs/cf_resrl/m0_resfit_training.json"
    source_config = git("show", f"{status['git_commit']}:{config_relative}")
    assert hashlib.sha256(source_config).hexdigest() == status["config_sha256"]
    config = json.loads(source_config)
    assert status["status"] == "COMPLETED" and status["child_returncode"] == 0
    assert status["dry_run"] is False and status["git_dirty_before_attempt"] is False
    assert status["resfit_clean_after_cleanup"] and status["lerobot_clean_after_cleanup"]
    assert status["command"] == launcher["command"]
    assert status["checkpoint_contract"] == manifest["checkpoint_contract"]
    assert sha256(ATTEMPT / "stdout_stderr.log") == status["stdout_stderr_sha256"]
    assert sha256(TRAINING / "checkpoint_manifest.json") == status["checkpoint_manifest_sha256"]
    assert "Training finished" in (ATTEMPT / "stdout_stderr.log").read_text()
    assert status["environment_step_budget"] == manifest["environment_step_budget"] == BUDGET
    for filename in ("conda-explicit.txt", "environment.lock.txt"):
        assert sha256(ATTEMPT / "runtime" / filename) == status["runtime"]["locks"][filename]["sha256"]
    req_relative = config["runtime"]["extension_requirements"]
    reqs = git("show", f"{status['git_commit']}:{req_relative}")
    assert hashlib.sha256(reqs).hexdigest() == status["runtime"]["extension_requirements_sha256"]
    for pin in config["upstream"]["patches"]:
        relative = pin["path"]
        patch = git("show", f"{status['git_commit']}:{relative}")
        assert hashlib.sha256(patch).hexdigest() == pin["sha256"]
        for stage, observed in (("before", "not_applied"), ("after", "applied")):
            recorded = next(row for row in status[f"patches_{stage}"] if row["path"].endswith(relative))
            assert recorded["sha256"] == pin["sha256"] and recorded["state"] == observed
        assert f"Reversed {Path(relative).name}" in status["patch_cleanup_stdout"]
    upstream_paths = {
        "resfit": REPO / "third_party/resfit",
        "lerobot": REPO / "third_party/resfit/deps/lerobot",
        "robosuite": REPO / "third_party/robosuite",
    }
    for name, path in upstream_paths.items():
        assert git("rev-parse", "HEAD", directory=path).decode().strip() == config["upstream"][f"{name}_commit"]
    return config


def verify_inputs(status, config, resolved):
    dataset = resolved["dataset"]
    assert dataset["name"] == config["dataset"]["name"]
    assert dataset["revision"] == config["dataset"]["revision"]
    assert dataset["split"] == "train"
    split_path = REPO / config["dataset"]["split_manifest"]
    split = read_json(split_path)
    indices = dataset["episode_indices"]
    assert len(indices) == len(set(indices)) == config["dataset"]["episodes"] == 240
    assert indices == split["splits"]["train"] == status["validation"]["split_manifest"]["episode_indices"]
    assert dataset["split_manifest_sha256"] == sha256(split_path) == config["dataset"]["split_manifest_sha256"]
    assert resolved["environment_seeds"] == {
        "training_initial_reset": TRAIN_SEED,
        "evaluation_initial_reset": EVAL_SEED,
        "evaluation_lane_seed_rule": "initial_reset_seed + env_index",
    }
    assert resolved["environment_step_budget"] == BUDGET
    training_config = resolved["config"]
    assert training_config["seed"] == TRAIN_SEED
    assert training_config["task"] == "Square"
    assert training_config["num_envs"] == 1 and training_config["eval_num_envs"] == EVAL_LANES
    assert training_config["eval_num_episodes"] == EVAL_EPISODES
    assert training_config["algo"]["learning_starts"] == BUDGET["warmup"]
    assert training_config["algo"]["total_timesteps"] == BUDGET["total"]
    assert training_config["checkpoint_interval"] == config["training"]["checkpoint_interval_environment_steps"]
    act_checksums = {}
    for filename, key in (("config.json", "config_sha256"), ("model.safetensors", "model_sha256")):
        artifact = REPO / config["base_policy"]["local_dir"] / filename
        actual = sha256(artifact)
        assert actual == status["validation"]["base_policy"][filename]["sha256"]
        assert actual == config["base_policy"][key] == resolved["base_policy_artifacts"][filename]
        act_checksums[filename] = actual
    source = status["validation"]["base_policy_source"]
    act_status_path = REPO / config["base_policy"]["source_status"]
    act_status = read_json(act_status_path)
    assert sha256(act_status_path) == source["status_sha256"]
    assert source["git_commit"] == act_status["git_commit"]
    assert act_status["status"] == "COMPLETED"
    selection = source["checkpoint_selection"]
    assert selection == act_status["checkpoint_selection"]
    assert selection["checkpoint_step_label"] == 49999
    assert selection["optimization_steps_completed"] == 50000
    return act_checksums


def verify_evaluations(manifest, act_checksums):
    assert manifest["selection"] == "fixed_environment_step"
    assert manifest["checkpoint_interval_environment_steps"] == 10000
    assert [row["global_step"] for row in manifest["checkpoints"]] == list(STEPS)
    assert sorted(path.name for path in (TRAINING / "models").glob("*.pt")) == [
        f"step_{step:09d}.pt" for step in STEPS
    ]
    assert sorted(path.name for path in (TRAINING / "outputs").glob("*_evaluation.json")) == [
        f"step_{step:09d}_evaluation.json" for step in STEPS
    ]
    summaries = []
    for record in manifest["checkpoints"]:
        step = record["global_step"]
        checkpoint = TRAINING / "models" / f"step_{step:09d}.pt"
        evaluation = TRAINING / "outputs" / f"step_{step:09d}_evaluation.json"
        assert record["selection"] == "fixed_environment_step"
        assert Path(record["path"]).name == checkpoint.name
        assert Path(record["evaluation"]["path"]).name == evaluation.name
        assert sha256(checkpoint) == record["sha256"]
        assert sha256(evaluation) == record["evaluation"]["sha256"]
        assert record["evaluation"]["episodes"] == EVAL_EPISODES
        result = read_json(evaluation)
        assert result["base_policy_artifacts"] == act_checksums
        assert result["checkpoint_global_step"] == step
        assert result["checkpoint_sha256"] == record["sha256"]
        assert result["task"] == "Square"
        assert result["initial_reset_seed"] == EVAL_SEED
        assert result["num_envs"] == EVAL_LANES
        assert result["counts_toward_training_budget"] is False
        episodes = result["episodes"]
        assert len(episodes) == EVAL_EPISODES
        lane_counts = Counter()
        for number, episode in enumerate(episodes):
            lane = episode["env_index"]
            assert type(lane) is int and 0 <= lane < EVAL_LANES
            assert episode["episode_index"] == number
            assert episode["lane_episode_index"] == lane_counts[lane]
            assert episode["lane_initial_seed"] == EVAL_SEED + lane
            assert type(episode["success"]) is bool
            assert type(episode["terminated"]) is bool and type(episode["truncated"]) is bool
            assert episode["terminated"] or episode["truncated"]
            assert 1 <= episode["environment_steps"] <= EPISODE_HORIZON
            assert math.isfinite(episode["return"])
            lane_counts[lane] += 1
        successes = sum(episode["success"] for episode in episodes)
        metrics = result["metrics"]
        assert math.isclose(metrics["eval/success_rate"], successes / EVAL_EPISODES)
        assert math.isclose(metrics["eval/mean_return"], sum(episode["return"] for episode in episodes) / EVAL_EPISODES)
        eval_steps = result["environment_steps"]
        assert eval_steps % EVAL_LANES == 0 and eval_steps == metrics["eval/environment_steps"]
        assert eval_steps >= sum(episode["environment_steps"] for episode in episodes)
        assert math.isfinite(result["wall_seconds"]) and result["wall_seconds"] > 0
        summaries.append({
            "environment_step": step,
            "checkpoint_path": str(checkpoint.relative_to(ATTEMPT)),
            "checkpoint_sha256": record["sha256"],
            "checkpoint_bytes": checkpoint.stat().st_size,
            "evaluation_path": str(evaluation.relative_to(ATTEMPT)),
            "evaluation_sha256": record["evaluation"]["sha256"],
            "completed_episodes": len(episodes),
            "successes": successes,
            "success_rate": metrics["eval/success_rate"],
            "mean_return": metrics["eval/mean_return"],
            "evaluation_environment_steps": eval_steps,
            "evaluation_wall_seconds": result["wall_seconds"],
        })
    return summaries


def clip(value, low, high):
    return min(max(value, low), high)


def verify_action_ledger(manifest, resolved):
    ledger = TRAINING / "action_ledger.jsonl"
    recorded = manifest["action_ledger"]
    expected_steps = recorded["records"]
    assert expected_steps == BUDGET["total"]
    assert Path(recorded["path"]).name == ledger.name
    dimensions = (
        "applied_env_action", "applied_normalized_action", "applied_residual_action",
        "proposed_normalized_action", "proposed_residual_action", "clip_mask",
    )
    lows = resolved["normalization"]["action_min"]
    highs = resolved["normalization"]["action_max"]
    assert len(lows) == len(highs) == ACTION_DIM and all(low < high for low, high in zip(lows, highs))
    clipped = Counter()
    clipped_rows = Counter()
    phase_rows = Counter()
    residual_rows = Counter()
    max_errors = Counter()
    max_env_action = 0.0
    digest = hashlib.sha256()
    with ledger.open("rb") as stream:
        for step, line in enumerate(stream, start=1):
            digest.update(line)
            row = json.loads(line)
            assert row["global_step"] == step
            phase = "warmup" if step <= BUDGET["warmup"] else "main"
            assert row["phase"] == phase
            assert all(len(row[name]) == 1 and len(row[name][0]) == ACTION_DIM for name in dimensions)
            env, applied, residual, proposed, candidate, mask = (row[name][0] for name in dimensions)
            assert all(type(flag) is bool for flag in mask)
            assert row["clip_count"] == sum(mask)
            assert all(math.isfinite(value) for vector in (env, applied, residual, proposed, candidate) for value in vector)
            assert all(-1 <= value <= 1 for value in env)
            assert all(-1 <= value <= 1 for value in applied)
            phase_rows[phase] += 1
            clipped[phase] += sum(mask)
            clipped_rows[phase] += any(mask)
            residual_rows[phase] += any(value != 0 for value in residual)
            max_env_action = max(max_env_action, max(map(abs, env)))
            for low, high, command, norm, delta, plan, candidate_delta, clipped_dim in zip(
                lows, highs, env, applied, residual, proposed, candidate, mask
            ):
                proposed_env = low + (clip(plan, -1, 1) + 1) * (high - low) / 2
                normalized_env = 2 * (clip(command, low, high) - low) / (high - low) - 1
                max_errors["environment_clamp"] = max(
                    max_errors["environment_clamp"], abs(command - clip(proposed_env, -1, 1))
                )
                max_errors["applied_normalization"] = max(max_errors["applied_normalization"], abs(norm - normalized_env))
                max_errors["reconstructed_base"] = max(
                    max_errors["reconstructed_base"], abs((norm - delta) - (plan - candidate_delta))
                )
                assert clipped_dim == (abs(plan - norm) > 1e-7)
    assert phase_rows == {phase: BUDGET[phase] for phase in ("warmup", "main")}
    assert digest.hexdigest() == recorded["sha256"]
    assert all(error <= RECONSTRUCTION_ATOL for error in max_errors.values()), max_errors
    return {
        "path": str(ledger.relative_to(ATTEMPT)),
        "bytes": ledger.stat().st_size,
        "records": expected_steps,
        "sha256": digest.hexdigest(),
        "environment_action_range": [-1, 1],
        "max_absolute_environment_action": max_env_action,
        "max_reconstruction_errors": dict(max_errors),
        "reconstruction_atol": RECONSTRUCTION_ATOL,
        "phase_rows": dict(phase_rows),
        "phase_clipped_dimensions": dict(clipped),
        "phase_rows_with_any_clipping": dict(clipped_rows),
        "phase_rows_with_nonzero_applied_residual": dict(residual_rows),
    }


def main():
    status = read_json(ATTEMPT / "status.json")
    manifest = read_json(TRAINING / "checkpoint_manifest.json")
    resolved = read_json(TRAINING / "resolved_training_config.json")
    config = verify_launch(status, manifest)
    act_checksums = verify_inputs(status, config, resolved)
    evaluations = verify_evaluations(manifest, act_checksums)
    ledger = verify_action_ledger(manifest, resolved)
    archive = TRAINING / "action_ledger.jsonl.zst"
    with subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE) as decoder:
        subprocess.run(["cmp", "-", str(ATTEMPT / ledger["path"])], stdin=decoder.stdout, check=True)
        assert decoder.wait() == 0
    ledger["archive"] = {
        "path": str(archive.relative_to(ATTEMPT)),
        "bytes": archive.stat().st_size,
        "sha256": sha256(archive),
        "verified_byte_identical_to_original": True,
    }
    elapsed_s = (datetime.fromisoformat(status["completed_at_utc"]) -
                 datetime.fromisoformat(status["started_at_utc"])).total_seconds()
    print(json.dumps({
        "attempt": ATTEMPT.name,
        "git_commit": status["git_commit"],
        "training_elapsed_seconds": elapsed_s,
        "training_budget": BUDGET,
        "training_seed": TRAIN_SEED,
        "evaluation_initial_seed": EVAL_SEED,
        "local_act_sha256": act_checksums,
        "upstream": config["upstream"],
        "resolved_training_config_sha256": sha256(TRAINING / "resolved_training_config.json"),
        "source_config_sha256": status["config_sha256"],
        "checkpoint_manifest_sha256": status["checkpoint_manifest_sha256"],
        "stdout_stderr_sha256": status["stdout_stderr_sha256"],
        "runtime_lock_sha256": {key: value["sha256"] for key, value in status["runtime"]["locks"].items()},
        "launcher_confirmed_overlay_cleanup": True,
        "evaluation_summary": evaluations,
        "evaluation_episodes_total": sum(row["completed_episodes"] for row in evaluations),
        "evaluation_environment_steps_total": sum(row["evaluation_environment_steps"] for row in evaluations),
        "evaluation_wall_seconds_total": sum(row["evaluation_wall_seconds"] for row in evaluations),
        "action_ledger": ledger,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
