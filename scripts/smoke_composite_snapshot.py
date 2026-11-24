#!/usr/bin/env python3
"""Replay a real Square suffix from one full ACT+scheduler+environment snapshot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shlex
import socket
import subprocess
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
RESFIT_ROOT = REPO_ROOT / "third_party" / "resfit"
ROBOSUITE_ROOT = REPO_ROOT / "third_party" / "robosuite"
LEROBOT_ROOT = RESFIT_ROOT / "deps" / "lerobot"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "cf_resrl" / "m0_composite_snapshot_smoke.json"
EXPECTED = {
    "resfit_commit": "66262b06a0caf159fb001504949b7b48ccda5f64",
    "robosuite_commit": "77a4751233c29456a5381209e30dd0dbf39a6557",
    "lerobot_repository": "https://github.com/huggingface/lerobot.git",
    "lerobot_commit": "69901b9b6a2300914ca3de0ea14b6fa6e0203bd4",
    "checkpoint_config_sha256": "4a6a588a50b1256560c3c74efb16c1dab1fc82ad477c5b2af97611d5e26273bb",
    "checkpoint_model_sha256": "add7b1e432ddeee2635ab09688b72fd5e1bb1eaff62a6b4b4494bc62b6e0364e",
    "checkpoint_model_bytes": 136837212,
    "torch_parameter_count": 34185089,
}
VECTOR_FIELDS = (
    "integration_state",
    "qpos",
    "qvel",
    "ctrl",
    "observation_state",
    "applied_action",
)
SCALAR_FIELDS = ("reward", "cur_time", "sim_time")
EXACT_FIELDS = (
    "suffix_index",
    "policy_action",
    "scheduler_entry",
    "scheduler_tick",
    "scheduler_counters",
    "act_queue_length",
    "act_queue_sha256",
    "image_sha256",
    "torch_rng_sha256",
    "episode_steps",
    "timestep",
    "done",
    "terminated",
    "truncated",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def validate_spec(config: dict[str, Any]) -> None:
    upstream = config["upstream"]
    checkpoint = config["checkpoint"]
    environment = config["environment"]
    task = config["task"]
    scheduler = config["scheduler"]
    for key in ("resfit_commit", "robosuite_commit", "lerobot_repository", "lerobot_commit"):
        if upstream[key] != EXPECTED[key]:
            raise ValueError(f"upstream.{key} does not match the registered value")
    if checkpoint != {
        "local_dir": "data/cf_resrl/checkpoints/resfit-act-square-b75e0805",
        "config_sha256": EXPECTED["checkpoint_config_sha256"],
        "model_sha256": EXPECTED["checkpoint_model_sha256"],
        "model_bytes": EXPECTED["checkpoint_model_bytes"],
        "torch_parameter_count": EXPECTED["torch_parameter_count"],
        "provenance": "author-owned candidate",
        "wandb_reference": "robomimic-square-bc/dzbkdpwp",
        "wandb_equivalence": "UNVERIFIED",
        "license": "NOT_PUBLISHED",
    }:
        raise ValueError("checkpoint registration does not match the audited ACT candidate")
    if environment != {
        "name": "cf-resrl-m0",
        "torch": "2.7.1+cu128",
        "gymnasium": "1.1.1",
        "lerobot": "0.1.0",
        "av": "14.2.0",
        "video_backend": "pyav",
    }:
        raise ValueError("environment does not match the registered M0 ACT runtime")
    if task != {
        "name": "Square",
        "seed": 20261124,
        "camera_size": 84,
        "render_size": 84,
        "render_gpu_device_id": 0,
        "prefix_ticks": 1,
        "suffix_ticks": 3,
        "restored_replays": 2,
    }:
        raise ValueError("task does not match the registered composite replay smoke")
    zero_action = [0.0] * 7
    if scheduler != {
        "latency_steps": 2,
        "action_low": -1.0,
        "action_high": 1.0,
        "fallback_action": zero_action,
        "residual_action": zero_action,
        "provider_identity": "registered-act-queue-no-refresh-v1",
    }:
        raise ValueError("scheduler does not match the registered composite replay smoke")
    if config["numeric_tolerance"] != {"atol": 1e-12, "rtol": 0.0}:
        raise ValueError("numeric tolerance must remain atol=1e-12 and rtol=0")
    if config["image_comparison"] != "exact_sha256":
        raise ValueError("images must be compared by exact SHA256")

    checkpoint_dir = _resolve(checkpoint["local_dir"])
    config_path = checkpoint_dir / "config.json"
    model_path = checkpoint_dir / "model.safetensors"
    if _sha256(config_path) != checkpoint["config_sha256"]:
        raise ValueError("ACT checkpoint config SHA256 mismatch")
    if model_path.stat().st_size != checkpoint["model_bytes"]:
        raise ValueError("ACT checkpoint model byte count mismatch")
    if _sha256(model_path) != checkpoint["model_sha256"]:
        raise ValueError("ACT checkpoint model SHA256 mismatch")


def _image_hash(value: Any, np: Any) -> str:
    image = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(json.dumps(list(image.shape)).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _integration_state(wrapper: Any, mujoco: Any, np: Any) -> list[float]:
    sim = wrapper.env.sim
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(sim.model._model, state_spec), dtype=np.float64)
    mujoco.mj_getState(sim.model._model, sim.data._data, state, state_spec)
    return state.tolist()


def _tensor_sequence_sha256(values: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _torch_rng_sha256(torch: Any) -> dict[str, Any]:
    return {
        "cpu": _tensor_sequence_sha256([torch.get_rng_state()]),
        "cuda": _tensor_sequence_sha256(torch.cuda.get_rng_state_all()),
    }


def _snapshot_torch_rng_sha256(snapshot: Any) -> dict[str, Any]:
    return {
        "cpu": _tensor_sequence_sha256([snapshot.cpu_state]),
        "cuda": _tensor_sequence_sha256(snapshot.cuda_states),
    }


def _perturb_torch_rng(torch: Any, device: str) -> None:
    before = _torch_rng_sha256(torch)
    torch.rand(17)
    torch.rand(17, device=device)
    after = _torch_rng_sha256(torch)
    if after["cpu"] == before["cpu"] or after["cuda"] == before["cuda"]:
        raise RuntimeError("explicit CPU/CUDA RNG perturbation did not change both states")


def _observation_batch(observation: dict[str, Any], policy: Any, torch: Any, np: Any) -> dict[str, Any]:
    return {
        key: torch.from_numpy(np.asarray(value)).unsqueeze(0).to(policy.config.device)
        for key, value in observation.items()
    }


def validate_absolute_alignment(
    wrapper: Any,
    policy: Any,
    scheduler: Any,
    *,
    expected_tick: int,
    expected_queue_length: int,
    entry: Any | None = None,
) -> None:
    if scheduler.logical_tick != expected_tick:
        raise RuntimeError("scheduler is not at the registered absolute tick")
    if wrapper.episode_steps != expected_tick:
        raise RuntimeError("environment and scheduler absolute ticks differ")
    if len(policy._action_queues[0]) != expected_queue_length:
        raise RuntimeError("ACT queue remaining length is not absolutely tick aligned")
    if entry is not None and (
        entry.tick != expected_tick
        or entry.chunk_source_tick != 0
        or entry.chunk_index != expected_tick
        or entry.generation != 0
        or entry.used_fallback
    ):
        raise RuntimeError("scheduler entry is not absolutely source-tick aligned")


def _tick_record(
    wrapper: Any,
    observation: dict[str, Any],
    policy_action: Any,
    applied_action: Any,
    reward: float,
    terminated: bool,
    truncated: bool,
    suffix_index: int,
    scheduler: Any,
    scheduler_entry: Any,
    policy: Any,
    mujoco: Any,
    np: Any,
    torch: Any,
) -> dict[str, Any]:
    env = wrapper.env
    queue = policy._action_queues[0]
    return {
        "suffix_index": suffix_index,
        "policy_action": np.asarray(policy_action, dtype=np.float32).tolist(),
        "applied_action": np.asarray(applied_action, dtype=np.float64).tolist(),
        "scheduler_entry": asdict(scheduler_entry),
        "scheduler_tick": scheduler.logical_tick,
        "scheduler_counters": asdict(scheduler.counters),
        "act_queue_length": len(queue),
        "act_queue_sha256": _tensor_sequence_sha256(list(queue)),
        "integration_state": _integration_state(wrapper, mujoco, np),
        "qpos": np.asarray(env.sim.data.qpos, dtype=np.float64).tolist(),
        "qvel": np.asarray(env.sim.data.qvel, dtype=np.float64).tolist(),
        "ctrl": np.asarray(env.sim.data.ctrl, dtype=np.float64).tolist(),
        "observation_state": np.asarray(
            observation["observation.state"], dtype=np.float64
        ).tolist(),
        "image_sha256": {
            key: _image_hash(observation[key], np)
            for key in (
                "observation.images.agentview",
                "observation.images.robot0_eye_in_hand",
            )
        },
        "torch_rng_sha256": _torch_rng_sha256(torch),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "episode_steps": int(wrapper.episode_steps),
        "timestep": int(env.timestep),
        "cur_time": float(env.cur_time),
        "sim_time": float(env.sim.data.time),
        "done": bool(env.done),
    }


def _rollout_suffix(
    wrapper: Any,
    policy: Any,
    scheduler: Any,
    observation: dict[str, Any],
    *,
    suffix_ticks: int,
    prefix_ticks: int,
    residual_action: Sequence[float],
    branch: str,
    mujoco: Any,
    np: Any,
    torch: Any,
) -> list[dict[str, Any]]:
    rows = []
    for suffix_index in range(suffix_ticks):
        expected_tick = prefix_ticks + suffix_index
        expected_queue_before = policy.config.n_action_steps - expected_tick
        validate_absolute_alignment(
            wrapper,
            policy,
            scheduler,
            expected_tick=expected_tick,
            expected_queue_length=expected_queue_before,
        )
        with torch.inference_mode():
            policy_tensor = policy.select_action(
                _observation_batch(observation, policy, torch, np)
            )
        policy_action = policy_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)
        entry = scheduler.action(residual_action)
        validate_absolute_alignment(
            wrapper,
            policy,
            scheduler,
            expected_tick=expected_tick,
            expected_queue_length=expected_queue_before - 1,
            entry=entry,
        )
        base_action = np.asarray(entry.base_action, dtype=np.float32)
        if not np.array_equal(policy_action, base_action):
            raise RuntimeError(
                f"ACT queue and scheduler chunk diverged at {branch} suffix tick {suffix_index}"
            )
        applied_action = np.asarray(entry.applied_action, dtype=np.float32)
        next_observation, reward, terminated, truncated, _ = wrapper.step(applied_action)
        scheduler.after_step(next_observation, request_refresh=False)
        validate_absolute_alignment(
            wrapper,
            policy,
            scheduler,
            expected_tick=expected_tick + 1,
            expected_queue_length=policy.config.n_action_steps - expected_tick - 1,
        )
        if terminated or truncated:
            raise RuntimeError(f"{branch} terminated at suffix tick {suffix_index}")
        rows.append(
            _tick_record(
                wrapper,
                next_observation,
                policy_action,
                applied_action,
                reward,
                terminated,
                truncated,
                suffix_index,
                scheduler,
                entry,
                policy,
                mujoco,
                np,
                torch,
            )
        )
        observation = next_observation
    return rows


def _max_abs_error(expected: Sequence[float], observed: Sequence[float]) -> float:
    if len(expected) != len(observed):
        raise ValueError("numeric vector lengths differ")
    errors = [abs(float(left) - float(right)) for left, right in zip(expected, observed)]
    if any(not math.isfinite(error) for error in errors):
        raise ValueError("numeric replay comparison produced NaN or Inf")
    return max(errors, default=0.0)


def compare_replay(
    baseline: Sequence[dict[str, Any]],
    replay: Sequence[dict[str, Any]],
    *,
    atol: float,
) -> dict[str, float]:
    if len(baseline) != len(replay):
        raise ValueError("replay tick counts differ")
    maxima = {field: 0.0 for field in (*VECTOR_FIELDS, *SCALAR_FIELDS)}
    for expected, observed in zip(baseline, replay):
        for field in EXACT_FIELDS:
            if observed[field] != expected[field]:
                raise ValueError(
                    f"exact mismatch for {field} at suffix tick {expected['suffix_index']}"
                )
        for field in VECTOR_FIELDS:
            maxima[field] = max(
                maxima[field], _max_abs_error(expected[field], observed[field])
            )
        for field in SCALAR_FIELDS:
            maxima[field] = max(
                maxima[field], abs(float(expected[field]) - float(observed[field]))
            )
    failed = {field: value for field, value in maxima.items() if value > atol}
    if failed:
        raise ValueError(f"composite replay exceeded atol={atol}: {failed}")
    return maxima


def _verify_sources() -> None:
    if _git(RESFIT_ROOT, "rev-parse", "HEAD") != EXPECTED["resfit_commit"]:
        raise RuntimeError("ResFiT checkout does not match the registered commit")
    if _git(ROBOSUITE_ROOT, "rev-parse", "HEAD") != EXPECTED["robosuite_commit"]:
        raise RuntimeError("Robosuite checkout does not match the registered commit")
    if _git(LEROBOT_ROOT, "remote", "get-url", "origin") != EXPECTED["lerobot_repository"]:
        raise RuntimeError("LeRobot remote does not match the registered repository")
    if _git(LEROBOT_ROOT, "rev-parse", "HEAD") != EXPECTED["lerobot_commit"]:
        raise RuntimeError("LeRobot checkout does not match the registered commit")
    if _git(RESFIT_ROOT, "status", "--porcelain"):
        raise RuntimeError("real composite smoke requires a clean pinned ResFiT checkout")
    if _git(LEROBOT_ROOT, "status", "--porcelain"):
        raise RuntimeError("LeRobot checkout has uncommitted changes")


def _run(
    config: dict[str, Any],
    config_path: Path,
    attempt_dir: Path,
    run: dict[str, Any],
) -> dict[str, Any]:
    validate_spec(config)
    _verify_sources()
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(config["task"]["render_gpu_device_id"])
    os.environ["WANDB_MODE"] = "disabled"
    sys.path.insert(0, str(LEROBOT_ROOT))
    sys.path.insert(0, str(RESFIT_ROOT))

    import av
    import gymnasium
    import mujoco
    import numpy as np
    import robosuite
    import torch

    from cf_resrl.composite_snapshot import (
        capture_composite_snapshot,
        restore_composite_snapshot,
    )
    from cf_resrl.scheduler import AbsoluteTickScheduler
    from resfit.dexmg.environments.dexmg import RobosuiteGymWrapper
    from resfit.lerobot.policies.act.configuration_act import ACTConfig
    from resfit.lerobot.policies.act.modeling_act import ACTPolicy

    environment = config["environment"]
    if torch.__version__ != environment["torch"]:
        raise RuntimeError("Torch version does not match the registered runtime")
    if gymnasium.__version__ != environment["gymnasium"]:
        raise RuntimeError("Gymnasium version does not match the registered runtime")
    if metadata.version("lerobot") != environment["lerobot"]:
        raise RuntimeError("LeRobot version does not match the registered runtime")
    if av.__version__ != environment["av"]:
        raise RuntimeError("PyAV version does not match the registered runtime")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the registered M0 runtime")

    run["environment"] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "mujoco": mujoco.__version__,
        "robosuite": robosuite.__version__,
        "av": av.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(config["task"]["render_gpu_device_id"]),
    }
    resolved_path = attempt_dir / "resolved_config.json"
    _write_json(
        resolved_path,
        {
            **config,
            "run": run,
            "resolved": {
                "config_path": str(config_path.resolve()),
                "checkpoint_dir": str(_resolve(config["checkpoint"]["local_dir"]).resolve()),
                "attempt_dir": str(attempt_dir.resolve()),
            },
        },
    )

    seed = config["task"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    checkpoint_dir = _resolve(config["checkpoint"]["local_dir"])
    policy_config = ACTConfig.from_pretrained(checkpoint_dir, local_files_only=True)
    policy_config.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(
        checkpoint_dir,
        config=policy_config,
        local_files_only=True,
        strict=True,
    )
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    if parameter_count != config["checkpoint"]["torch_parameter_count"]:
        raise RuntimeError("loaded ACT parameter count does not match the audited candidate")
    if policy.config.temporal_ensemble_coeff is not None:
        raise RuntimeError("registered composite smoke requires ACT action-queue mode")

    wrapper = RobosuiteGymWrapper(
        config["task"]["name"],
        camera_size=config["task"]["camera_size"],
        render_size=config["task"]["render_size"],
        render_gpu_device_id=config["task"]["render_gpu_device_id"],
    )
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        wrapper.env.rng = np.random.default_rng(seed)
        observation, _ = wrapper.reset(seed=seed)
        policy.reset()
        with torch.inference_mode():
            first_tensor = policy.select_action(
                _observation_batch(observation, policy, torch, np)
            )
        first_action = first_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)
        if len(policy._action_queues) != 1:
            raise RuntimeError("ACT did not initialize exactly one action queue")
        remaining_queue = list(policy._action_queues[0])
        if len(remaining_queue) + 1 != policy.config.n_action_steps:
            raise RuntimeError("ACT queue length does not match n_action_steps after its first action")
        if config["task"]["suffix_ticks"] > len(remaining_queue):
            raise RuntimeError("registered suffix is longer than the captured ACT queue")
        initial_chunk = [first_action.tolist()] + [
            value.detach().cpu().numpy().astype(np.float32, copy=False).tolist()
            for value in remaining_queue
        ]

        def provider_not_used(_observation: Any, _source_tick: int, _generation: int) -> Any:
            raise RuntimeError("registered no-refresh replay unexpectedly called the chunk provider")

        scheduler_config = config["scheduler"]
        scheduler = AbsoluteTickScheduler(
            initial_chunk=initial_chunk,
            provider=provider_not_used,
            latency_steps=scheduler_config["latency_steps"],
            fallback_action=scheduler_config["fallback_action"],
            action_low=scheduler_config["action_low"],
            action_high=scheduler_config["action_high"],
        )
        prefix_entry = scheduler.action(scheduler_config["residual_action"])
        validate_absolute_alignment(
            wrapper,
            policy,
            scheduler,
            expected_tick=0,
            expected_queue_length=policy.config.n_action_steps - 1,
            entry=prefix_entry,
        )
        if not np.array_equal(
            first_action,
            np.asarray(prefix_entry.base_action, dtype=np.float32),
        ):
            raise RuntimeError("ACT first action and scheduler prefix action differ")
        prefix_action = np.asarray(prefix_entry.applied_action, dtype=np.float32)
        observation, _, terminated, truncated, _ = wrapper.step(prefix_action)
        scheduler.after_step(observation, request_refresh=False)
        if terminated or truncated:
            raise RuntimeError("Square terminated during the registered prefix")
        if scheduler.logical_tick != config["task"]["prefix_ticks"]:
            raise RuntimeError("scheduler tick does not match the registered prefix boundary")
        validate_absolute_alignment(
            wrapper,
            policy,
            scheduler,
            expected_tick=config["task"]["prefix_ticks"],
            expected_queue_length=(
                policy.config.n_action_steps - config["task"]["prefix_ticks"]
            ),
        )

        snapshot = capture_composite_snapshot(
            wrapper,
            scheduler,
            policy,
            policy_identity=config["checkpoint"]["model_sha256"],
            provider_identity=scheduler_config["provider_identity"],
            observation_tick=scheduler.logical_tick,
            residual_history=(scheduler_config["residual_action"],),
        )
        if snapshot.observation_tick != wrapper.episode_steps:
            raise RuntimeError("snapshot observation tick does not match the environment boundary")
        if snapshot.scheduler.active_chunk.source_tick != 0:
            raise RuntimeError("snapshot scheduler chunk has the wrong absolute source tick")
        if len(snapshot.scheduler.ledger) != config["task"]["prefix_ticks"]:
            raise RuntimeError("snapshot ledger length does not match the prefix boundary")
        if len(snapshot.act_policy.queues[0].items) != (
            policy.config.n_action_steps - config["task"]["prefix_ticks"]
        ):
            raise RuntimeError("snapshot ACT queue length does not match its absolute tick")
        snapshot_rng_sha256 = _snapshot_torch_rng_sha256(snapshot.torch_rng)
        if _torch_rng_sha256(torch) != snapshot_rng_sha256:
            raise RuntimeError("captured Torch RNG does not match the live snapshot boundary")
        baseline = _rollout_suffix(
            wrapper,
            policy,
            scheduler,
            observation,
            suffix_ticks=config["task"]["suffix_ticks"],
            prefix_ticks=config["task"]["prefix_ticks"],
            residual_action=scheduler_config["residual_action"],
            branch="baseline",
            mujoco=mujoco,
            np=np,
            torch=torch,
        )
        branches = {"baseline": baseline}
        for replay_index in range(config["task"]["restored_replays"]):
            _perturb_torch_rng(torch, policy.config.device)
            context = restore_composite_snapshot(
                wrapper,
                scheduler,
                policy,
                snapshot,
                policy_identity=config["checkpoint"]["model_sha256"],
                provider_identity=scheduler_config["provider_identity"],
            )
            if context.observation_tick != scheduler.logical_tick:
                raise RuntimeError("restored observation and scheduler ticks differ")
            if context.residual_history != (scheduler_config["residual_action"],):
                raise RuntimeError("restored residual history differs from the snapshot")
            if _torch_rng_sha256(torch) != snapshot_rng_sha256:
                raise RuntimeError("restore did not recover the captured CPU/CUDA Torch RNG")
            if wrapper.episode_steps != scheduler.logical_tick:
                raise RuntimeError("restored environment and scheduler ticks differ")
            branch = f"restore_{replay_index + 1}"
            branches[branch] = _rollout_suffix(
                wrapper,
                policy,
                scheduler,
                context.current_observation,
                suffix_ticks=config["task"]["suffix_ticks"],
                prefix_ticks=config["task"]["prefix_ticks"],
                residual_action=scheduler_config["residual_action"],
                branch=branch,
                mujoco=mujoco,
                np=np,
                torch=torch,
            )
    finally:
        wrapper.close()

    atol = config["numeric_tolerance"]["atol"]
    comparisons = {
        branch: {
            "max_abs_error": compare_replay(baseline, rows, atol=atol),
            "image_hashes_exact": True,
            "act_queue_hashes_exact": True,
            "scheduler_ledger_exact": True,
            "torch_rng_hashes_exact": True,
        }
        for branch, rows in branches.items()
        if branch != "baseline"
    }
    tick_rows = [dict(row, branch=branch) for branch, rows in branches.items() for row in rows]
    ticks_path = attempt_dir / "tick_records.jsonl"
    _write_jsonl(ticks_path, tick_rows)
    report = {
        "status": "COMPLETED",
        "experiment_kind": "composite_snapshot_deterministic_replay_smoke",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "checkpoint": config["checkpoint"],
        "prefix_ticks": config["task"]["prefix_ticks"],
        "suffix_ticks": config["task"]["suffix_ticks"],
        "restored_replays": config["task"]["restored_replays"],
        "snapshot": {
            "schema_version": snapshot.schema_version,
            "scope": snapshot.scope,
            "observation_tick": snapshot.observation_tick,
            "act_queue_lengths": [len(queue.items) for queue in snapshot.act_policy.queues],
            "scheduler_ledger_entries": len(snapshot.scheduler.ledger),
            "torch_cuda_device_count": snapshot.torch_rng.cuda_device_count,
            "torch_rng_sha256": snapshot_rng_sha256,
            "policy_identity": snapshot.policy_identity,
            "provider_identity": snapshot.provider_identity,
        },
        "tolerance": config["numeric_tolerance"],
        "comparisons": comparisons,
        "limitations": config["limitations"],
        "artifacts": {
            "resolved_config.json": _sha256(resolved_path),
            "tick_records.jsonl": _sha256(ticks_path),
        },
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = args.out or _resolve(config["output_dir"])
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    repo_head_error: Exception | None = None
    repo_status_error: Exception | None = None
    try:
        repo_head = _git(REPO_ROOT, "rev-parse", "HEAD")
    except Exception as exc:
        repo_head = "unknown"
        repo_head_error = exc
    try:
        repo_dirty = bool(_git(REPO_ROOT, "status", "--porcelain"))
    except Exception as exc:
        repo_dirty = None
        repo_status_error = exc
    started_at = datetime.now(timezone.utc)
    attempt_id = f"{started_at.strftime('%Y%m%dT%H%M%S%fZ')}_{repo_head[:12]}"
    attempt_dir = output_root / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    run = {
        "attempt_id": attempt_id,
        "started_at_utc": started_at.isoformat(),
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]),
        "git_commit": repo_head,
        "git_dirty": repo_dirty,
        "evidence_dir": str(attempt_dir.resolve()),
    }
    status_path = attempt_dir / "status.json"
    try:
        if repo_head_error is not None:
            raise RuntimeError("cannot resolve parent repository HEAD") from repo_head_error
        if repo_status_error is not None:
            raise RuntimeError("cannot inspect parent repository status") from repo_status_error
        run["host"] = socket.gethostname()
        run["submodules"] = {
            "resfit": _git(RESFIT_ROOT, "rev-parse", "HEAD"),
            "robosuite": _git(ROBOSUITE_ROOT, "rev-parse", "HEAD"),
        }
        validate_spec(config)
        if args.dry_run:
            prepared = {
                "status": "PREPARED",
                "experiment_kind": "composite_snapshot_deterministic_replay_smoke",
                "dry_run": True,
                "run": run,
                "config_path": str(config_path.resolve()),
                "config_sha256": _sha256(config_path),
            }
            _write_json(attempt_dir / "resolved_config.json", {**config, **prepared})
            _write_json(status_path, prepared)
            print(json.dumps(prepared, indent=2, sort_keys=True))
            return 0
        if repo_dirty:
            raise RuntimeError("formal composite snapshot smoke requires a clean parent repository")
        report = _run(config, config_path, attempt_dir, run)
        report_path = attempt_dir / "smoke_report.json"
        _write_json(report_path, report)
        checksums = {
            "config": {"path": str(config_path.resolve()), "sha256": _sha256(config_path)},
            "resolved_config.json": _sha256(attempt_dir / "resolved_config.json"),
            "tick_records.jsonl": _sha256(attempt_dir / "tick_records.jsonl"),
            "smoke_report.json": _sha256(report_path),
        }
        checksums_path = attempt_dir / "checksums.json"
        _write_json(checksums_path, checksums)
        _write_json(
            status_path,
            {
                "status": "COMPLETED",
                "experiment_kind": report["experiment_kind"],
                "completed_at_utc": report["completed_at_utc"],
                "run": run,
                "checksums_sha256": _sha256(checksums_path),
            },
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failed = {
            "status": "FAILED",
            "experiment_kind": "composite_snapshot_deterministic_replay_smoke",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "run": run,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(status_path, failed)
        _write_json(attempt_dir / "smoke_report.json", failed)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
