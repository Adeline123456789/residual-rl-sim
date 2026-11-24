#!/usr/bin/env python3
"""Strictly load the pinned ACT candidate and run a short Square rollout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import random
import shlex
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESFIT_DIR = REPO_ROOT / "third_party" / "resfit"
ROBOSUITE_DIR = REPO_ROOT / "third_party" / "robosuite"
LEROBOT_DIR = RESFIT_DIR / "deps" / "lerobot"
EXPECTED = {
    "resfit_commit": "66262b06a0caf159fb001504949b7b48ccda5f64",
    "robosuite_commit": "77a4751233c29456a5381209e30dd0dbf39a6557",
    "lerobot_repository": "https://github.com/huggingface/lerobot.git",
    "lerobot_commit": "69901b9b6a2300914ca3de0ea14b6fa6e0203bd4",
    "config_sha256": "4a6a588a50b1256560c3c74efb16c1dab1fc82ad477c5b2af97611d5e26273bb",
    "model_sha256": "add7b1e432ddeee2635ab09688b72fd5e1bb1eaff62a6b4b4494bc62b6e0364e",
    "model_bytes": 136837212,
    "safetensors_tensor_element_count": 34204289,
    "torch_parameter_count": 34185089,
}
EXPECTED_PIP_CHECK_DEVIATIONS = [
    "cmake not installed",
    "diffusers not installed; registered ResFiT patch removes the unused eager import",
    "flask not installed",
    "gdown not installed",
    "h5py not installed",
    "omegaconf not installed",
    "opencv-python-headless not installed",
    "pymunk not installed",
    "pyzmq not installed",
    "rerun-sdk not installed",
    "torchcodec not installed; video_backend is pyav",
    "zarr not installed",
    "gymnasium metadata requires 0.29.1; Square M0 intentionally keeps 1.1.1",
]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_spec(config: dict[str, Any]) -> None:
    upstream = config["upstream"]
    checkpoint = config["checkpoint"]
    environment = config["environment"]
    task = config["task"]
    for key in ("resfit_commit", "robosuite_commit", "lerobot_repository", "lerobot_commit"):
        if upstream[key] != EXPECTED[key]:
            raise ValueError(f"upstream.{key} is {upstream[key]!r}, expected {EXPECTED[key]!r}")
    for key in (
        "config_sha256",
        "model_sha256",
        "model_bytes",
        "safetensors_tensor_element_count",
        "torch_parameter_count",
    ):
        if checkpoint[key] != EXPECTED[key]:
            raise ValueError(f"checkpoint.{key} is {checkpoint[key]!r}, expected {EXPECTED[key]!r}")
    if checkpoint["provenance"] != "author-owned candidate":
        raise ValueError("checkpoint provenance must remain author-owned candidate")
    if checkpoint["wandb_equivalence"] != "UNVERIFIED":
        raise ValueError("W&B equivalence must remain UNVERIFIED")
    if checkpoint["license"] != "NOT_PUBLISHED":
        raise ValueError("checkpoint license must remain NOT_PUBLISHED")
    if (
        environment["gymnasium"] != "1.1.1"
        or environment["lerobot"] != "0.1.0"
        or environment["av"] != "14.2.0"
        or environment["video_backend"] != "pyav"
    ):
        raise ValueError("ACT smoke requires the registered M0 runtime versions")
    if environment["runtime_scope"] != "minimal_act_square_path":
        raise ValueError("ACT runtime scope must remain minimal_act_square_path")
    if environment["intentional_pip_check_deviations"] != EXPECTED_PIP_CHECK_DEVIATIONS:
        raise ValueError("ACT runtime pip-check deviations do not match the registered list")
    if task != {
        "name": "Square",
        "seed": 20261124,
        "camera_size": 84,
        "render_gpu_device_id": 0,
        "rollout_steps": 4,
        "action_abs_limit": 1.0,
        "action_handling": "clip_to_environment_action_space_and_record",
    }:
        raise ValueError("task settings do not match the registered Square ACT smoke")


def _array_observation(np: Any, observation: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "observation.state": (9,),
        "observation.images.agentview": (3, 84, 84),
        "observation.images.robot0_eye_in_hand": (3, 84, 84),
    }
    summary = {}
    for key, shape in expected.items():
        value = np.asarray(observation[key])
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise RuntimeError(f"invalid {key}: shape={value.shape}, dtype={value.dtype}")
        summary[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
    return summary


def _run(config: dict[str, Any], config_path: Path, attempt_dir: Path, run: dict[str, Any]) -> dict[str, Any]:
    validate_spec(config)
    if _git(RESFIT_DIR, "rev-parse", "HEAD") != EXPECTED["resfit_commit"]:
        raise RuntimeError("ResFiT checkout does not match the registered commit")
    if _git(ROBOSUITE_DIR, "rev-parse", "HEAD") != EXPECTED["robosuite_commit"]:
        raise RuntimeError("Robosuite checkout does not match the registered commit")
    if _git(LEROBOT_DIR, "remote", "get-url", "origin") != EXPECTED["lerobot_repository"]:
        raise RuntimeError("LeRobot remote does not match the registered repository")
    if _git(LEROBOT_DIR, "rev-parse", "HEAD") != EXPECTED["lerobot_commit"]:
        raise RuntimeError("LeRobot checkout does not match the registered commit")
    if _git(LEROBOT_DIR, "status", "--porcelain"):
        raise RuntimeError("LeRobot checkout has uncommitted source changes")

    checkpoint_dir = REPO_ROOT / config["checkpoint"]["local_dir"]
    config_file = checkpoint_dir / "config.json"
    model_file = checkpoint_dir / "model.safetensors"
    if _sha256_file(config_file) != EXPECTED["config_sha256"]:
        raise RuntimeError("checkpoint config SHA256 mismatch")
    if model_file.stat().st_size != EXPECTED["model_bytes"]:
        raise RuntimeError("checkpoint model byte count mismatch")
    if _sha256_file(model_file) != EXPECTED["model_sha256"]:
        raise RuntimeError("checkpoint model SHA256 mismatch")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(config["task"]["render_gpu_device_id"])
    os.environ["WANDB_MODE"] = "disabled"
    sys.path.insert(0, str(LEROBOT_DIR))
    sys.path.insert(0, str(RESFIT_DIR))

    import av
    import gymnasium
    import mujoco
    import numpy as np
    import robosuite
    import torch

    from resfit.dexmg.environments.dexmg import RobosuiteGymWrapper
    from resfit.lerobot.policies.act.configuration_act import ACTConfig
    from resfit.lerobot.policies.act.modeling_act import ACTPolicy

    if torch.__version__ != config["environment"]["torch"]:
        raise RuntimeError(f"torch is {torch.__version__}, expected {config['environment']['torch']}")
    if gymnasium.__version__ != config["environment"]["gymnasium"]:
        raise RuntimeError(
            f"gymnasium is {gymnasium.__version__}, expected {config['environment']['gymnasium']}"
        )
    if metadata.version("lerobot") != config["environment"]["lerobot"]:
        raise RuntimeError("installed LeRobot version does not match the registered runtime")
    if av.__version__ != config["environment"]["av"]:
        raise RuntimeError(f"av is {av.__version__}, expected {config['environment']['av']}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the M0 environment")

    resolved = {
        **config,
        "run": run,
        "resolved": {
            "config_path": str(config_path.resolve()),
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "output_dir": str(attempt_dir.resolve()),
        },
    }
    resolved_path = attempt_dir / "resolved_config.json"
    _write_json(resolved_path, resolved)

    seed = config["task"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()

    policy_config = ACTConfig.from_pretrained(checkpoint_dir, local_files_only=True)
    source_backbone_weights = policy_config.pretrained_backbone_weights
    policy_config.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(
        checkpoint_dir,
        config=policy_config,
        local_files_only=True,
        strict=True,
    )
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    if parameter_count != EXPECTED["torch_parameter_count"]:
        raise RuntimeError(f"loaded parameter count is {parameter_count}")
    state_dict_element_count = sum(tensor.numel() for tensor in policy.state_dict().values())
    if state_dict_element_count != EXPECTED["safetensors_tensor_element_count"]:
        raise RuntimeError(f"loaded state-dict tensor element count is {state_dict_element_count}")

    env = None
    rollout = []
    try:
        env = RobosuiteGymWrapper(
            config["task"]["name"],
            camera_size=config["task"]["camera_size"],
            render_size=config["task"]["camera_size"],
            render_gpu_device_id=config["task"]["render_gpu_device_id"],
        )
        observation, _ = env.reset()
        policy.reset()
        for step in range(config["task"]["rollout_steps"]):
            observation_summary = _array_observation(np, observation)
            batch = {
                key: torch.from_numpy(np.asarray(value)).unsqueeze(0).to(policy_config.device)
                for key, value in observation.items()
            }
            with torch.inference_mode():
                action_tensor = policy.select_action(batch)
            action = action_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            if action.shape != (1, 7) or not np.isfinite(action).all():
                raise RuntimeError(f"invalid ACT action: shape={action.shape}")
            raw_action = action[0]
            applied_action = np.clip(raw_action, env.action_space.low, env.action_space.high)
            raw_max_abs = float(np.abs(raw_action).max())
            applied_max_abs = float(np.abs(applied_action).max())
            if applied_max_abs > config["task"]["action_abs_limit"]:
                raise RuntimeError(f"applied ACT action exceeds registered limit: {applied_max_abs}")
            next_observation, reward, terminated, truncated, _ = env.step(applied_action)
            rollout.append(
                {
                    "step": step,
                    "observation": observation_summary,
                    "raw_action": raw_action.tolist(),
                    "raw_action_sha256": hashlib.sha256(raw_action.tobytes()).hexdigest(),
                    "raw_action_max_abs": raw_max_abs,
                    "applied_action": applied_action.tolist(),
                    "applied_action_sha256": hashlib.sha256(applied_action.tobytes()).hexdigest(),
                    "applied_action_max_abs": applied_max_abs,
                    "clipping_mask": (raw_action != applied_action).tolist(),
                    "clipped_dimensions": int(np.count_nonzero(raw_action != applied_action)),
                    "max_clip_delta": float(np.abs(raw_action - applied_action).max()),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            observation = next_observation
            if terminated or truncated:
                break
        final_observation = _array_observation(np, observation)
    finally:
        if env is not None:
            env.close()

    report = {
        "status": "COMPLETED",
        "experiment_kind": "act_policy_smoke",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "checkpoint": {
            "local_dir": str(checkpoint_dir.resolve()),
            "config_sha256": EXPECTED["config_sha256"],
            "model_sha256": EXPECTED["model_sha256"],
            "model_bytes": EXPECTED["model_bytes"],
            "torch_parameter_count": parameter_count,
            "safetensors_tensor_element_count": state_dict_element_count,
            "provenance": config["checkpoint"]["provenance"],
            "wandb_reference": config["checkpoint"]["wandb_reference"],
            "wandb_equivalence": config["checkpoint"]["wandb_equivalence"],
            "license": config["checkpoint"]["license"],
        },
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": metadata.version("torchvision"),
            "gymnasium": gymnasium.__version__,
            "mujoco": mujoco.__version__,
            "robosuite": robosuite.__version__,
            "lerobot": metadata.version("lerobot"),
            "av": av.__version__,
            "video_backend": config["environment"]["video_backend"],
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
            "runtime_scope": config["environment"]["runtime_scope"],
            "intentional_pip_check_deviations": config["environment"][
                "intentional_pip_check_deviations"
            ],
        },
        "policy_load": {
            "strict": True,
            "local_files_only": True,
            "source_pretrained_backbone_weights": source_backbone_weights,
            "constructor_pretrained_backbone_weights": None,
        },
        "rollout": rollout,
        "final_observation": final_observation,
        "limitations": [
            "This is a load-and-execute smoke test, not a policy quality evaluation.",
            "The author-owned candidate remains unverified as equivalent to the unavailable W&B checkpoint.",
            "The publisher did not provide a checkpoint license or model card.",
        ],
        "artifacts": {"resolved_config.json": _sha256_file(resolved_path)},
    }
    _write_json(attempt_dir / "smoke_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/cf_resrl/m0_act_policy_smoke.json",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repo_head = _git(REPO_ROOT, "rev-parse", "HEAD")
    started_at = datetime.now(timezone.utc)
    attempt_id = f"{started_at.strftime('%Y%m%dT%H%M%S%fZ')}_{repo_head[:12]}"
    attempt_dir = REPO_ROOT / config["output_dir"] / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    run = {
        "attempt_id": attempt_id,
        "started_at_utc": started_at.isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "git_commit": repo_head,
        "git_dirty": bool(_git(REPO_ROOT, "status", "--porcelain")),
        "evidence_dir": str(attempt_dir.resolve()),
    }
    try:
        report = _run(config, config_path, attempt_dir, run)
    except Exception as exc:
        failed = {
            "status": "FAILED",
            "experiment_kind": "act_policy_smoke",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "run": run,
            "checkpoint": {
                "provenance": config["checkpoint"]["provenance"],
                "wandb_equivalence": config["checkpoint"]["wandb_equivalence"],
                "license": config["checkpoint"]["license"],
            },
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(attempt_dir / "smoke_report.json", failed)
        print(f"failed evidence: {attempt_dir / 'smoke_report.json'}", file=sys.stderr)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
