#!/usr/bin/env python3
"""Evaluate frozen, fixed-step Square ACT checkpoints on the same seeded cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_act_training import (
    DEFAULT_CONFIG,
    LEROBOT_ROOT,
    REPO_ROOT,
    RESFIT_ROOT,
    ROBOSUITE_ROOT,
    _git,
    _sha256,
    _write_json,
    validate_spec,
)


SOURCE = REPO_ROOT / (
    "results/cf_resrl/m0/act_training/attempts/20261125T005956570423Z_3dfd8208847a"
)
OUTPUT = REPO_ROOT / "results/cf_resrl/m0/act_fixed_eval/attempts"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_plan(spec: dict, manifest: dict, requested: list[int] | None) -> list[dict]:
    final_label = spec["checkpoint_selection"]["checkpoint_step_label"]
    interval = spec["training"]["save_frequency"]
    labels = [*range(interval, final_label, interval), final_label]
    if manifest["fixed_checkpoint_labels"] != labels:
        raise ValueError("Source manifest differs from the registered ACT checkpoint schedule")
    if requested and (len(set(requested)) != len(requested) or set(requested) - set(labels)):
        raise ValueError("Request distinct registered fixed ACT checkpoint labels")
    chosen = set(requested or labels)
    original_rounds = {
        row["optimization_steps_completed"]: row for row in manifest["evaluation_rounds"]
    }
    results = []
    # Evaluate the required final ACT base first, then the remaining fixed checkpoints.
    for label in (final_label, *labels[:-1]):
        if label not in chosen:
            continue
        directory = f"training/policy_step_{label}/policy"
        original = original_rounds[label + 1 if label == final_label else label]
        results.append({
            "checkpoint_step_label": label,
            "optimization_steps_completed": label + 1,
            "policy_directory": directory,
            "config": manifest["files"][f"{directory}/config.json"],
            "model": manifest["files"][f"{directory}/model.safetensors"],
            "original_during_training_rollout": {
                **original,
                "video_evidence": manifest["files"][original["video"]],
                "same_post_update_weights": label == final_label,
            },
        })
    return results


def _lane_episode_ids(num_episodes: int, num_envs: int) -> tuple[range, ...]:
    return tuple(range(lane, num_episodes, num_envs) for lane in range(num_envs))


def _make_seeded_lane(environment: str, camera_size: int, video_key: str, num_envs: int, lane: int):
    import numpy as np
    from resfit.dexmg.environments.dexmg import RobosuiteGymWrapper

    class SeededSquareLane(RobosuiteGymWrapper):
        next_seed = None

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self.next_seed = seed
            if self.next_seed is not None:
                # Apply the same two Robosuite RNGs as the formal ResFiT overlay.
                random.seed(self.next_seed)
                np.random.seed(self.next_seed % 2**32)
                self.env.rng.bit_generator.state = np.random.default_rng(self.next_seed).bit_generator.state
                self.next_seed += num_envs
            return super().reset(seed=seed, options=options)

    env = SeededSquareLane(
        env_name=environment,
        camera_size=camera_size,
        render_gpu_device_id=0,
        env_id=lane,
    )
    env.set_wrapper_attr("video_key", video_key)
    return env


def _collect(policy, env, *, seed: int, num_episodes: int, horizon: int) -> dict:
    import numpy as np
    import torch

    width = env.num_envs
    cohort = _lane_episode_ids(num_episodes, width)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    completed = [0] * width
    lengths = np.zeros(width, dtype=np.int64)
    returns = np.zeros(width, dtype=np.float64)
    clipped = np.zeros(width, dtype=np.int64)
    raw_max = np.zeros(width, dtype=np.float64)
    actual_steps = 0
    records = []
    low, high = env.action_space.low, env.action_space.high

    while len(records) < num_episodes:
        if actual_steps >= width * max(map(len, cohort)) * horizon:
            raise RuntimeError("Square exceeded its registered episode horizon")
        with torch.inference_mode():
            proposed = policy.select_action(obs).cpu().numpy()
        if proposed.shape != low.shape or not np.isfinite(proposed).all():
            raise RuntimeError("ACT emitted an invalid Square action")
        applied = np.clip(proposed, low, high)
        lengths += 1
        clipped += np.count_nonzero(proposed != applied, axis=1)
        raw_max = np.maximum(raw_max, np.abs(proposed).max(axis=1))
        obs, reward, terminated, truncated, _ = env.step(applied)
        reward, terminated, truncated = (item.cpu().numpy() for item in (reward, terminated, truncated))
        actual_steps += width
        returns += reward

        ended = np.flatnonzero(terminated | truncated).tolist()
        if ended:
            policy.reset(env_ids=ended)
        for lane in ended:
            if completed[lane] < len(cohort[lane]):
                episode = cohort[lane][completed[lane]]
                records.append({
                    "episode_index": episode,
                    "env_index": lane,
                    "lane_episode_index": completed[lane],
                    "episode_seed": seed + episode,
                    "success": bool(reward[lane] == 1.0),
                    "return": float(returns[lane]),
                    "environment_steps": int(lengths[lane]),
                    "terminated": bool(terminated[lane]),
                    "truncated": bool(truncated[lane]),
                    "clipped_action_components": int(clipped[lane]),
                    "max_abs_proposed_action": float(raw_max[lane]),
                })
            completed[lane] += 1
            lengths[lane] = 0
            returns[lane] = 0
            clipped[lane] = 0
            raw_max[lane] = 0

    records.sort(key=lambda row: row["episode_index"])
    scored_steps = sum(row["environment_steps"] for row in records)
    successes = sum(row["success"] for row in records)
    return {
        "episode_count": num_episodes,
        "successes": successes,
        "success_rate": successes / num_episodes,
        "mean_return": sum(row["return"] for row in records) / num_episodes,
        "environment_steps": actual_steps,
        "scored_environment_steps": scored_steps,
        "unscored_environment_steps": actual_steps - scored_steps,
        "counts_toward_training_budget": False,
        "episodes": records,
    }


def _verify_file(path: Path, registered: dict) -> None:
    if path.stat().st_size != registered["bytes"] or _sha256(path) != registered["sha256"]:
        raise ValueError(f"ACT source evidence checksum mismatch: {path}")


def _worker(plan_path: Path) -> int:
    import gymnasium
    import mujoco
    import numpy as np
    import robosuite
    import torch
    from resfit.dexmg.environments.dexmg import VectorizedEnvWrapper
    from resfit.lerobot.policies.act.configuration_act import ACTConfig
    from resfit.lerobot.policies.act.modeling_act import ACTPolicy

    plan = _read_json(plan_path)
    output = plan_path.parent
    eval_spec = plan["evaluation"]
    source = Path(plan["source_attempt"])
    if torch.__version__ != plan["expected_torch"] or not torch.cuda.is_available():
        raise RuntimeError("The registered ACT CUDA runtime is unavailable")
    if mujoco.__version__ != "3.3.2" or gymnasium.__version__ != "1.1.1":
        raise RuntimeError("The registered ACT simulation runtime has changed")
    if not Path(robosuite.__file__).resolve().is_relative_to(ROBOSUITE_ROOT):
        raise RuntimeError("Robosuite must load from the registered pinned checkout")
    runtime = plan["runtime_locks"]
    for lock in ("pip_freeze_lock", "conda_explicit_lock"):
        path = output / Path(runtime[lock]["path"]).name
        if _sha256(path) != runtime[lock]["sha256"]:
            raise RuntimeError(f"Local ACT environment differs from {lock}")
    _write_json(output / "runtime.json", {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "mujoco": mujoco.__version__,
        "gymnasium": gymnasium.__version__,
        "robosuite": robosuite.__version__,
    })

    width = eval_spec["num_environments"]
    seed = eval_spec["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    vec = gymnasium.vector.AsyncVectorEnv(
        [
            partial(_make_seeded_lane, eval_spec["environment"], eval_spec["camera_size"],
                    eval_spec["video_key"], width, lane)
            for lane in range(width)
        ],
        shared_memory=True,
        copy=True,
        context="spawn",
        autoreset_mode=gymnasium.vector.AutoresetMode.SAME_STEP,
    )
    env = VectorizedEnvWrapper(vec, eval_spec["video_key"], plan["device"])
    rows = []
    try:
        horizon = vec.call("horizon")
        if len(set(horizon)) != 1:
            raise RuntimeError("Square vector lanes have different horizons")
        for fixed in plan["checkpoints"]:
            started = time.perf_counter()
            directory = source / fixed["policy_directory"]
            _verify_file(directory / "config.json", fixed["config"])
            _verify_file(directory / "model.safetensors", fixed["model"])
            original = fixed["original_during_training_rollout"]
            _verify_file(source / original["video"], original["video_evidence"])

            config = ACTConfig.from_pretrained(directory, local_files_only=True)
            original_backbone = config.pretrained_backbone_weights
            config.pretrained_backbone_weights = None
            policy = ACTPolicy.from_pretrained(directory, config=config, local_files_only=True, strict=True)
            policy.eval()
            data = _collect(policy, env, seed=seed, num_episodes=eval_spec["num_episodes"], horizon=horizon[0])
            del policy

            label = fixed["checkpoint_step_label"]
            filename = f"policy_step_{label}_evaluation.json"
            _write_json(output / filename, {
                **fixed,
                **data,
                "evaluation": eval_spec,
                "horizon": horizon[0],
                "checkpoint_backbone_initialization": original_backbone,
                "wall_seconds": time.perf_counter() - started,
            })
            rows.append({
                "checkpoint_step_label": label,
                "episode_count": data["episode_count"],
                "successes": data["successes"],
                "success_rate": data["success_rate"],
                "environment_steps": data["environment_steps"],
                "path": filename,
                "sha256": _sha256(output / filename),
            })
            _write_json(output / "summary.json", {"completed_checkpoints": rows})
            print(f"ACT checkpoint {label}: {data['successes']}/{data['episode_count']} successes", flush=True)
    finally:
        env.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-label", action="append", type=int)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_plan:
        return _worker(args.worker_plan)

    spec = _read_json(DEFAULT_CONFIG)
    episodes = validate_spec(spec)
    source = args.source.resolve()
    source_status = _read_json(source / "status.json")
    manifest = _read_json(source / "evidence_manifest.json")
    if (
        source_status["status"] != "COMPLETED"
        or source_status["returncode"] != 0
        or source_status["dry_run"]
        or source_status["episode_indices"] != episodes
        or source_status["config_sha256"] != _sha256(DEFAULT_CONFIG)
        or source_status["checkpoint_selection"] != spec["checkpoint_selection"]
        or manifest["source_commit"] != source_status["git_commit"]
        or manifest["attempt_id"] != source.name
        or manifest["dataset"] != spec["dataset"]
        or manifest["training_seed"] != spec["training"]["seed"]
    ):
        raise ValueError("Expected the completed registered formal ACT checkpoint series")
    _verify_file(source / "stdout_stderr.log", manifest["files"]["stdout_stderr.log"])
    if _sha256(source / "stdout_stderr.log") != source_status["stdout_stderr_sha256"]:
        raise ValueError("Original ACT log disagrees with formal training status")
    final_policy = f"training/{spec['checkpoint_selection']['relative_policy_path']}"
    for name, expected in source_status["fixed_final_checkpoint"].items():
        registered = manifest["files"][f"{final_policy}/{name}"]
        if (registered["bytes"], registered["sha256"]) != (expected["bytes"], expected["sha256"]):
            raise ValueError("Final ACT evidence disagrees with formal training status")
    for path, key in ((RESFIT_ROOT, "resfit_commit"), (ROBOSUITE_ROOT, "robosuite_commit"),
                      (LEROBOT_ROOT, "lerobot_commit")):
        if _git(path, "rev-parse", "HEAD") != spec["upstream"][key] or _git(path, "status", "--porcelain"):
            raise RuntimeError(f"Expected the pinned clean upstream checkout: {path}")

    git_commit = _git(REPO_ROOT, "rev-parse", "HEAD")
    started = _now()
    stamp = datetime.fromisoformat(started).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.out.resolve() / f"{stamp}_{git_commit[:12]}"
    plan_path = output / "resolved_config.json"
    worker_command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-plan", str(plan_path)]
    plan = {
        "experiment_kind": "ACT_FROZEN_FIXED_CHECKPOINT_SQUARE_EVALUATION",
        "source_attempt": str(source),
        "source_commit": source_status["git_commit"],
        "source_status_sha256": _sha256(source / "status.json"),
        "source_manifest_sha256": _sha256(source / "evidence_manifest.json"),
        "source_training_log_sha256": source_status["stdout_stderr_sha256"],
        "source_training_spec_sha256": source_status["config_sha256"],
        "git_commit": git_commit,
        "git_status_before_run": _git(REPO_ROOT, "status", "--porcelain"),
        "upstream": {k: v for k, v in spec["upstream"].items() if k != "patches"},
        "runtime_locks": spec["runtime"],
        "expected_torch": source_status["torch"],
        "training_dataset": spec["dataset"],
        "training_episode_indices": episodes,
        "device": spec["training"]["device"],
        "checkpoints": _checkpoint_plan(spec, manifest, args.checkpoint_label),
        "evaluation": {
            **{k: v for k, v in spec["evaluation"].items() if k != "rollout_frequency"},
            "seed": spec["training"]["seed"] + 1000,
            "seed_rule": "episode_seed = seed + episode_index; lane = episode_index % num_environments",
            "seed_scope": "matches the formal ResFiT evaluation base seed; original ACT workers had unseeded resets",
            "action_rule": "clip to the registered Square action space",
            "unused_lane_interactions": "step the full vector batch and count every extra transition",
        },
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "worker_command": worker_command,
        "started_at_utc": started,
    }
    output.mkdir(parents=True, exist_ok=False)
    _write_json(plan_path, plan)
    print(f"ACT fixed checkpoint evaluation attempt: {output}", flush=True)
    if args.prepare_only:
        _write_json(output / "status.json", {"status": "PLAN_ONLY", "resolved_config_sha256": _sha256(plan_path)})
        return 0

    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(map(str, [RESFIT_ROOT, LEROBOT_ROOT])),
        "PYTHONNOUSERSITE": "1",
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0",
        "WANDB_MODE": "disabled",
        "HF_HUB_OFFLINE": "1",
    }
    log = output / "stdout_stderr.log"
    status_path = output / "status.json"
    # Match the training launcher's lock capture before replacing PYTHONPATH for policy imports.
    lock_environment = {**os.environ, "PYTHONNOUSERSITE": "1"}
    for lock, command in (
        ("pip_freeze_lock", [sys.executable, "-m", "pip", "freeze", "--all"]),
        ("conda_explicit_lock", ["conda", "list", "-p", sys.prefix, "--explicit"]),
    ):
        path = output / Path(plan["runtime_locks"][lock]["path"]).name
        path.write_bytes(subprocess.check_output(command, env=lock_environment))
    with log.open("wb") as stream, subprocess.Popen(worker_command, env=env, stdout=stream,
                                                    stderr=subprocess.STDOUT, cwd=REPO_ROOT) as worker:
        _write_json(status_path, {"status": "RUNNING", "pid": worker.pid, "started_at_utc": started})
        result = worker.wait()
    _write_json(status_path, {
        "status": "COMPLETED" if result == 0 else "FAILED",
        "returncode": result,
        "started_at_utc": started,
        "finished_at_utc": _now(),
        "resolved_config_sha256": _sha256(plan_path),
        "stdout_stderr_sha256": _sha256(log),
    })
    return result


if __name__ == "__main__":
    sys.exit(main())
