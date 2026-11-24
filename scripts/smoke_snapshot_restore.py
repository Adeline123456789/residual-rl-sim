#!/usr/bin/env python3
"""Run the registered real-Square snapshot restore determinism gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shlex
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cf_resrl.dataset import sha256_file  # noqa: E402
from cf_resrl.snapshot import capture_square_snapshot, restore_square_snapshot  # noqa: E402


VECTOR_FIELDS = (
    "integration_state",
    "qpos",
    "qvel",
    "ctrl",
    "observation_state",
)
COUNTER_VECTOR_FIELDS = ("objects_on_pegs", "gripper_current_action")
SCALAR_FIELDS = ("reward",)
COUNTER_SCALAR_FIELDS = ("cur_time", "sim_time")
EXACT_FIELDS = ("suffix_index", "action", "terminated", "truncated")
EXACT_COUNTER_FIELDS = ("episode_steps", "timestep", "done")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _max_abs_error(expected: list[float], observed: list[float]) -> float:
    if len(expected) != len(observed):
        raise ValueError(f"numeric vector length mismatch: {len(expected)} != {len(observed)}")
    errors = [abs(float(left) - float(right)) for left, right in zip(expected, observed)]
    if any(not math.isfinite(value) for value in errors):
        raise ValueError("numeric comparison produced NaN or Inf")
    return max(errors, default=0.0)


def compare_replay(
    baseline: list[dict[str, Any]],
    replay: list[dict[str, Any]],
    *,
    atol: float,
) -> dict[str, float]:
    """Compare one restored suffix against the baseline and return observed maxima."""
    if len(baseline) != len(replay):
        raise ValueError(f"tick count mismatch: {len(baseline)} != {len(replay)}")
    maxima = {
        **{field: 0.0 for field in VECTOR_FIELDS},
        **{f"counters.{field}": 0.0 for field in COUNTER_VECTOR_FIELDS},
        **{field: 0.0 for field in SCALAR_FIELDS},
        **{f"counters.{field}": 0.0 for field in COUNTER_SCALAR_FIELDS},
    }
    for expected, observed in zip(baseline, replay):
        for field in EXACT_FIELDS:
            if observed[field] != expected[field]:
                raise ValueError(f"exact mismatch for {field} at tick {expected['suffix_index']}")
        if observed["image_sha256"] != expected["image_sha256"]:
            raise ValueError(f"image hash mismatch at tick {expected['suffix_index']}")
        for field in EXACT_COUNTER_FIELDS:
            if observed["counters"][field] != expected["counters"][field]:
                raise ValueError(
                    f"counter mismatch for {field} at tick {expected['suffix_index']}"
                )
        for field in VECTOR_FIELDS:
            maxima[field] = max(
                maxima[field], _max_abs_error(expected[field], observed[field])
            )
        for field in COUNTER_VECTOR_FIELDS:
            key = f"counters.{field}"
            maxima[key] = max(
                maxima[key],
                _max_abs_error(expected["counters"][field], observed["counters"][field]),
            )
        for field in SCALAR_FIELDS:
            maxima[field] = max(
                maxima[field], abs(float(expected[field]) - float(observed[field]))
            )
        for field in COUNTER_SCALAR_FIELDS:
            key = f"counters.{field}"
            maxima[key] = max(
                maxima[key],
                abs(
                    float(expected["counters"][field])
                    - float(observed["counters"][field])
                ),
            )
    failed = {field: value for field, value in maxima.items() if value > atol}
    if failed:
        raise ValueError(f"snapshot replay exceeded atol={atol}: {failed}")
    return maxima


def _image_hash(value: Any, np: Any) -> str:
    image = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(json.dumps(list(image.shape)).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _integration_state(env: Any, mujoco: Any, np: Any) -> list[float]:
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    size = mujoco.mj_stateSize(env.sim.model._model, spec)
    state = np.empty(size, dtype=np.float64)
    mujoco.mj_getState(env.sim.model._model, env.sim.data._data, state, spec)
    return state.tolist()


def _tick_record(
    wrapper: Any,
    observation: dict[str, Any],
    action: Any,
    reward: float,
    terminated: bool,
    truncated: bool,
    suffix_index: int,
    mujoco: Any,
    np: Any,
) -> dict[str, Any]:
    env = wrapper.env
    image_keys = (
        "observation.images.agentview",
        "observation.images.robot0_eye_in_hand",
    )
    return {
        "suffix_index": suffix_index,
        "action": np.asarray(action, dtype=np.float64).tolist(),
        "integration_state": _integration_state(env, mujoco, np),
        "qpos": np.asarray(env.sim.data.qpos, dtype=np.float64).tolist(),
        "qvel": np.asarray(env.sim.data.qvel, dtype=np.float64).tolist(),
        "ctrl": np.asarray(env.sim.data.ctrl, dtype=np.float64).tolist(),
        "observation_state": np.asarray(
            observation["observation.state"], dtype=np.float64
        ).tolist(),
        "image_sha256": {
            key: _image_hash(observation[key], np) for key in image_keys
        },
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "counters": {
            "episode_steps": int(wrapper.episode_steps),
            "timestep": int(env.timestep),
            "cur_time": float(env.cur_time),
            "sim_time": float(env.sim.data.time),
            "done": bool(env.done),
            "objects_on_pegs": np.asarray(
                env.objects_on_pegs, dtype=np.float64
            ).tolist(),
            "gripper_current_action": np.asarray(
                env.robots[0].gripper["right"].current_action, dtype=np.float64
            ).tolist(),
        },
    }


def _rollout_suffix(
    wrapper: Any,
    actions: list[list[float]],
    branch: str,
    mujoco: Any,
    np: Any,
) -> list[dict[str, Any]]:
    rows = []
    for index, values in enumerate(actions):
        action = np.asarray(values, dtype=np.float32)
        observation, reward, terminated, truncated, _ = wrapper.step(action)
        if terminated or truncated:
            raise RuntimeError(f"{branch} terminated during suffix tick {index}")
        row = _tick_record(
            wrapper,
            observation,
            action,
            reward,
            terminated,
            truncated,
            index,
            mujoco,
            np,
        )
        row["branch"] = branch
        rows.append(row)
    return rows


def _run(config: dict[str, Any], config_path: Path, output_dir: Path, run: dict[str, Any]) -> dict[str, Any]:
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(config["render_gpu_device_id"])
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "resfit"))

    import gymnasium
    import mujoco
    import numpy as np
    import robosuite
    import torch

    from resfit.dexmg.environments.dexmg import RobosuiteGymWrapper

    run["environment"] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "mujoco": mujoco.__version__,
        "robosuite": robosuite.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    resolved_path = output_dir / "resolved_config.json"
    _write_json(
        resolved_path,
        {
            **config,
            "run": run,
            "resolved": {
                "config_path": str(config_path.resolve()),
                "output_dir": str(output_dir.resolve()),
            },
        },
    )

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    wrapper = RobosuiteGymWrapper(
        "Square",
        camera_size=int(config["camera_size"]),
        render_size=int(config["render_size"]),
        render_gpu_device_id=int(config["render_gpu_device_id"]),
    )
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        wrapper.env.rng = np.random.default_rng(seed)
        wrapper.reset(seed=seed)

        for index, values in enumerate(config["prefix_actions"]):
            action = np.asarray(values, dtype=np.float32)
            _, _, terminated, truncated, _ = wrapper.step(action)
            if terminated or truncated:
                raise RuntimeError(f"prefix terminated at tick {index}")
        if wrapper.episode_steps != len(config["prefix_actions"]):
            raise RuntimeError("wrapper prefix counter does not match the registered prefix")

        snapshot = capture_square_snapshot(wrapper)
        branches = {
            "baseline": _rollout_suffix(
                wrapper, config["suffix_actions"], "baseline", mujoco, np
            )
        }
        for replay_index in range(int(config["restored_replays"])):
            branch = f"restore_{replay_index + 1}"
            restore_square_snapshot(wrapper, snapshot)
            branches[branch] = _rollout_suffix(
                wrapper, config["suffix_actions"], branch, mujoco, np
            )
    finally:
        wrapper.close()

    atol = float(config["numeric_tolerance"]["atol"])
    rtol = float(config["numeric_tolerance"]["rtol"])
    if rtol != 0.0:
        raise ValueError("registered snapshot smoke requires rtol=0.0")
    comparisons = {
        branch: {
            "max_abs_error": compare_replay(branches["baseline"], rows, atol=atol),
            "image_hashes_exact": True,
            "integer_and_boolean_counters_exact": True,
        }
        for branch, rows in branches.items()
        if branch != "baseline"
    }
    tick_rows = [row for rows in branches.values() for row in rows]
    ticks_path = output_dir / "tick_records.jsonl"
    _write_jsonl(ticks_path, tick_rows)
    report = {
        "status": "COMPLETED",
        "experiment_kind": "snapshot_restore_smoke",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "task": config["task"],
        "seed": seed,
        "prefix_ticks": len(config["prefix_actions"]),
        "suffix_ticks": len(config["suffix_actions"]),
        "restored_replays": int(config["restored_replays"]),
        "snapshot": {
            "schema_version": snapshot.schema_version,
            "scope": snapshot.scope,
            "model": snapshot.model.__dict__,
            "capture_episode_steps": snapshot.wrapper_state["episode_steps"],
            "capture_timestep": snapshot.environment_state["timestep"],
        },
        "tolerance": {"atol": atol, "rtol": rtol},
        "comparisons": comparisons,
        "artifacts": {
            "resolved_config.json": sha256_file(resolved_path),
            "tick_records.jsonl": sha256_file(ticks_path),
        },
        "limitations": [
            "This gate covers the Square environment snapshot only; ACT queues and the CF-ResRL scheduler are excluded.",
            "Image comparison hashes the two processed CHW float32 observation tensors exactly."
        ],
    }
    _write_json(output_dir / "smoke_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/cf_resrl/m0_snapshot_smoke.json",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = args.out or REPO_ROOT / config["output_dir"]
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    repo_head = _git("rev-parse", "HEAD")
    started_at = datetime.now(timezone.utc)
    attempt_id = f"{started_at.strftime('%Y%m%dT%H%M%S%fZ')}_{repo_head[:12]}"
    output_dir = output_root / "attempts" / attempt_id
    output_dir.mkdir(parents=True, exist_ok=False)
    run = {
        "attempt_id": attempt_id,
        "started_at_utc": started_at.isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "git_commit": repo_head,
        "git_dirty": bool(_git("status", "--porcelain")),
        "submodules": {
            "resfit": _git("-C", "third_party/resfit", "rev-parse", "HEAD"),
            "robosuite": _git("-C", "third_party/robosuite", "rev-parse", "HEAD"),
        },
        "evidence_dir": str(output_dir.resolve()),
    }
    try:
        report = _run(config, config_path, output_dir, run)
    except Exception as exc:
        failed = {
            "status": "FAILED",
            "experiment_kind": "snapshot_restore_smoke",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "run": run,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(output_dir / "smoke_report.json", failed)
        print(f"failed evidence: {output_dir / 'smoke_report.json'}", file=sys.stderr)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
