#!/usr/bin/env python3
"""Run the registered Square environment smoke test and record its evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_UPSTREAM = {
    "commit": "66262b06a0caf159fb001504949b7b48ccda5f64",
    "robosuite_commit": "77a4751233c29456a5381209e30dd0dbf39a6557",
    "mujoco_version": "3.3.2",
}


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _observation_shapes(obs: dict[str, object], np: object) -> dict[str, list[int]]:
    expected = {
        "observation.state": (9,),
        "observation.images.agentview": (3, 84, 84),
        "observation.images.robot0_eye_in_hand": (3, 84, 84),
    }
    shapes: dict[str, list[int]] = {}
    for key, shape in expected.items():
        value = np.asarray(obs[key])
        if value.shape != shape:
            raise RuntimeError(f"{key} shape is {value.shape}, expected {shape}")
        if not np.isfinite(value).all():
            raise RuntimeError(f"{key} contains NaN or Inf")
        shapes[key] = list(value.shape)
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["square"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config_path = _resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    upstream = config["upstream"]
    for key, expected in EXPECTED_UPSTREAM.items():
        if upstream.get(key) != expected:
            raise RuntimeError(f"config upstream.{key} is {upstream.get(key)!r}, expected {expected!r}")

    resfit_dir = REPO_ROOT / "third_party" / "resfit"
    robosuite_dir = REPO_ROOT / "third_party" / "robosuite"
    resfit_sha = _git(resfit_dir, "rev-parse", "HEAD")
    robosuite_sha = _git(robosuite_dir, "rev-parse", "HEAD")
    if resfit_sha != EXPECTED_UPSTREAM["commit"]:
        raise RuntimeError(f"ResFiT submodule is {resfit_sha}, expected {EXPECTED_UPSTREAM['commit']}")
    if robosuite_sha != EXPECTED_UPSTREAM["robosuite_commit"]:
        raise RuntimeError(
            f"Robosuite submodule is {robosuite_sha}, expected {EXPECTED_UPSTREAM['robosuite_commit']}"
        )

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    sys.path.insert(0, str(resfit_dir))

    import gymnasium
    import mujoco
    import numpy as np
    import robosuite
    import torch

    from resfit.dexmg.environments.dexmg import RobosuiteGymWrapper

    if mujoco.__version__ != EXPECTED_UPSTREAM["mujoco_version"]:
        raise RuntimeError(
            f"MuJoCo package is {mujoco.__version__}, expected {EXPECTED_UPSTREAM['mujoco_version']}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the M0 environment")

    env = None
    try:
        env = RobosuiteGymWrapper(
            "Square",
            camera_size=84,
            render_size=84,
            render_gpu_device_id=0,
        )
        if tuple(env.action_space.shape) != (7,):
            raise RuntimeError(f"action shape is {env.action_space.shape}, expected (7,)")

        observation, _ = env.reset()
        reset_shapes = _observation_shapes(observation, np)
        observation, reward, terminated, truncated, _ = env.step(np.zeros(7, dtype=np.float32))
        step_shapes = _observation_shapes(observation, np)
        if not np.isfinite(reward):
            raise RuntimeError("zero-action reward is NaN or Inf")

        frame = env.render()
        if frame.shape != (84, 84, 3) or frame.dtype != np.uint8 or frame.size == 0:
            raise RuntimeError(
                f"rendered frame is shape={frame.shape}, dtype={frame.dtype}; expected (84, 84, 3), uint8"
            )
    finally:
        if env is not None:
            env.close()

    repo_head = _git(REPO_ROOT, "rev-parse", "HEAD")
    dirty = bool(_git(REPO_ROOT, "status", "--porcelain"))
    manifest = {
        "status": "COMPLETED",
        "experiment_kind": "smoke",
        "task": "Square",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "config": str(config_path.relative_to(REPO_ROOT)),
        "git": {"commit": repo_head, "dirty": dirty},
        "submodules": {"resfit": resfit_sha, "robosuite": robosuite_sha},
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "mujoco": mujoco.__version__,
            "robosuite": robosuite.__version__,
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "checks": {
            "action_shape": list(env.action_space.shape),
            "reset_observation_shapes": reset_shapes,
            "step_observation_shapes": step_shapes,
            "render_shape": list(frame.shape),
            "render_dtype": str(frame.dtype),
            "zero_action_reward": float(reward),
            "zero_action_terminated": bool(terminated),
            "zero_action_truncated": bool(truncated),
        },
    }

    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path = out_path.with_suffix(out_path.suffix + ".tmp")
    pending_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending_path.replace(out_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
