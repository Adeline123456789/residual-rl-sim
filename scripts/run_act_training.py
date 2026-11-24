#!/usr/bin/env python3
"""Launch the pinned 50k-step Square ACT reproduction in an immutable attempt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RESFIT_ROOT = REPO_ROOT / "third_party" / "resfit"
ROBOSUITE_ROOT = REPO_ROOT / "third_party" / "robosuite"
LEROBOT_ROOT = RESFIT_ROOT / "deps" / "lerobot"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "cf_resrl" / "m0_act_training.json"
EXPECTED = {
    "resfit_commit": "66262b06a0caf159fb001504949b7b48ccda5f64",
    "robosuite_commit": "77a4751233c29456a5381209e30dd0dbf39a6557",
    "lerobot_repository": "https://github.com/huggingface/lerobot.git",
    "lerobot_commit": "69901b9b6a2300914ca3de0ea14b6fa6e0203bd4",
    "dataset_revision": "4e5cee7ea7d63b920ff76c2529b781c2474697da",
    "split_manifest_sha256": "629d01857156c691f297218e0cda3e3b3a802a91122f006a1ca70fd31163afcd",
    "runtime_environment": "cf-resrl-m0",
    "pip_freeze_lock_path": "results/cf_resrl/m0/act_runtime/environment.lock.txt",
    "pip_freeze_lock_sha256": "ce0fd3f7d80633ba96054f17a046fc3a4cc806ca945fd8e70928c9ec2a424eda",
    "conda_explicit_lock_path": "results/cf_resrl/m0/act_runtime/conda-explicit.txt",
    "conda_explicit_lock_sha256": "2b5538025ba240838ad5e1798a4a1556748b191c0790bce16a2e169afc4e7f7d",
    "backbone_weights_enum": "ResNet18_Weights.IMAGENET1K_V1",
    "backbone_weights_filename": "resnet18-f37072fd.pth",
    "backbone_weights_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def validate_spec(config: dict[str, Any]) -> list[int]:
    upstream = config["upstream"]
    runtime = config["runtime"]
    backbone_weights = config["act_backbone_weights"]
    dataset = config["dataset"]
    training = config["training"]
    evaluation = config["evaluation"]
    selection = config["checkpoint_selection"]

    for key in ("resfit_commit", "robosuite_commit", "lerobot_repository", "lerobot_commit"):
        if upstream[key] != EXPECTED[key]:
            raise ValueError(f"upstream.{key} does not match the registered value")
    if upstream["required_patches"] != [
        "patches/resfit/train_bc_dexmg_cf_resrl.patch",
        "patches/resfit/residual_td3_cf_resrl.patch",
    ]:
        raise ValueError("ACT training requires the two registered ResFiT overlays")
    expected_runtime = {
        "conda_environment": EXPECTED["runtime_environment"],
        "pip_freeze_lock": {
            "path": EXPECTED["pip_freeze_lock_path"],
            "sha256": EXPECTED["pip_freeze_lock_sha256"],
        },
        "conda_explicit_lock": {
            "path": EXPECTED["conda_explicit_lock_path"],
            "sha256": EXPECTED["conda_explicit_lock_sha256"],
        },
    }
    if runtime != expected_runtime:
        raise ValueError("ACT runtime lock registration does not match the pinned environment")
    for lock_name in ("pip_freeze_lock", "conda_explicit_lock"):
        lock = runtime[lock_name]
        if _sha256(_resolve(lock["path"])) != lock["sha256"]:
            raise ValueError(f"registered {lock_name} file SHA256 mismatch")
    if backbone_weights != {
        "enum": EXPECTED["backbone_weights_enum"],
        "filename": EXPECTED["backbone_weights_filename"],
        "sha256": EXPECTED["backbone_weights_sha256"],
    }:
        raise ValueError("ACT backbone weights do not match the registered torchvision artifact")
    if dataset["repo_id"] != "ankile/robomimic-mh-square-image":
        raise ValueError("ACT training dataset repo does not match the registered value")
    if dataset["root"] != "data/cf_resrl/square":
        raise ValueError("ACT training dataset root does not match the registered value")
    if dataset["split_manifest"] != (
        "results/cf_resrl/m0/dataset_audit/attempts/"
        "20261124T131023495428Z_a9dcb4b558af/split_manifest.json"
    ):
        raise ValueError("ACT training split manifest path does not match the registered audit")
    if dataset["revision"] != EXPECTED["dataset_revision"]:
        raise ValueError("dataset revision does not match the registered value")
    if dataset["split_manifest_sha256"] != EXPECTED["split_manifest_sha256"]:
        raise ValueError("split manifest SHA256 does not match the registered value")

    split_path = _resolve(dataset["split_manifest"])
    if _sha256(split_path) != dataset["split_manifest_sha256"]:
        raise ValueError("split manifest file SHA256 mismatch")
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if split_manifest["dataset_revision"] != dataset["revision"]:
        raise ValueError("split manifest dataset revision mismatch")
    episode_indices = split_manifest["splits"][dataset["split"]]
    if len(episode_indices) != dataset["episode_count"] or len(episode_indices) != 240:
        raise ValueError("registered ACT training split must contain 240 episodes")

    expected_training = {
        "policy": "act",
        "seed": 20261124,
        "device": "cuda",
        "steps": 50000,
        "batch_size": 256,
        "num_workers": 4,
        "grad_clip_norm": 10.0,
        "log_frequency": 100,
        "save_frequency": 1000,
        "wandb_enabled": False,
        "output_root": "results/cf_resrl/m0/act_training",
    }
    if training != expected_training:
        raise ValueError("ACT training settings do not match the registered 50k reproduction")
    if evaluation != {
        "environment": "Square",
        "rollout_frequency": 1000,
        "num_environments": 16,
        "num_episodes": 100,
        "camera_size": 84,
        "video_key": "observation.images.agentview",
    }:
        raise ValueError("ACT evaluation settings do not match the registered reproduction")
    if selection != {
        "selection": "final_fixed_optimization_step",
        "optimization_steps_completed": 50000,
        "checkpoint_step_label": 49999,
        "relative_policy_path": "policy_step_49999/policy",
        "best_checkpoint_is_not_selected": True,
    }:
        raise ValueError("ACT checkpoint selection must remain the fixed final step")
    if dataset["video_backend"] != "pyav" or dataset["split"] != "train":
        raise ValueError("ACT training must use the registered train split and PyAV backend")
    return episode_indices


def patch_states(config: dict[str, Any]) -> list[dict[str, str]]:
    states = []
    for relative in config["upstream"]["required_patches"]:
        patch = _resolve(relative)
        reverse = subprocess.run(
            ["git", "-C", str(RESFIT_ROOT), "apply", "--reverse", "--check", str(patch)],
            capture_output=True,
            text=True,
        )
        forward = subprocess.run(
            ["git", "-C", str(RESFIT_ROOT), "apply", "--check", str(patch)],
            capture_output=True,
            text=True,
        )
        if reverse.returncode == 0:
            state = "applied"
        elif forward.returncode == 0:
            state = "not_applied"
        else:
            raise RuntimeError(f"registered patch cannot be applied or reversed: {patch}")
        states.append({"path": str(patch.resolve()), "sha256": _sha256(patch), "state": state})
    return states


def build_command(config: dict[str, Any], attempt_dir: Path) -> list[str]:
    dataset = config["dataset"]
    training = config["training"]
    evaluation = config["evaluation"]
    output_dir = attempt_dir / "training"
    cache_dir = attempt_dir / "cache"
    python_path = os.pathsep.join((str(RESFIT_ROOT), str(LEROBOT_ROOT)))
    policy_kwargs = json.dumps(
        {"pretrained_backbone_weights": config["act_backbone_weights"]["enum"]},
        separators=(",", ":"),
        sort_keys=True,
    )
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "cf-resrl-m0",
        "env",
        "PYTHONNOUSERSITE=1",
        "PYTHONUNBUFFERED=1",
        f"PYTHONPATH={python_path}",
        f"CACHE_DIR={cache_dir}",
        "WANDB_MODE=disabled",
        "HF_HUB_DISABLE_XET=1",
        "python",
        "-m",
        "resfit.lerobot.scripts.train_bc_dexmg",
        "--dataset",
        dataset["repo_id"],
        "--dataset_revision",
        dataset["revision"],
        "--dataset_root",
        str(_resolve(dataset["root"]).resolve()),
        "--episode_split_manifest",
        str(_resolve(dataset["split_manifest"]).resolve()),
        "--episode_split",
        dataset["split"],
        "--video_backend",
        dataset["video_backend"],
        "--policy",
        training["policy"],
        "--policy_kwargs",
        policy_kwargs,
        "--seed",
        str(training["seed"]),
        "--device",
        training["device"],
        "--steps",
        str(training["steps"]),
        "--batch_size",
        str(training["batch_size"]),
        "--num_workers",
        str(training["num_workers"]),
        "--grad_clip_norm",
        str(training["grad_clip_norm"]),
        "--log_freq",
        str(training["log_frequency"]),
        "--save_freq",
        str(training["save_frequency"]),
        "--output_dir",
        str(output_dir.resolve()),
        "--eval_env",
        evaluation["environment"],
        "--rollout_freq",
        str(evaluation["rollout_frequency"]),
        "--eval_num_envs",
        str(evaluation["num_environments"]),
        "--eval_num_episodes",
        str(evaluation["num_episodes"]),
        "--eval_camera_size",
        str(evaluation["camera_size"]),
        "--eval_video_key",
        evaluation["video_key"],
    ]


def capture_and_verify_runtime(config: dict[str, Any], attempt_dir: Path) -> dict[str, Any]:
    runtime = config["runtime"]
    environment = runtime["conda_environment"]
    runtime_dir = attempt_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=False)

    pip_command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        environment,
        "env",
        "PYTHONNOUSERSITE=1",
        "python",
        "-m",
        "pip",
        "freeze",
        "--all",
    ]
    explicit_command = ["conda", "list", "-n", environment, "--explicit"]
    metadata_program = """
import json
import platform
from pathlib import Path
import sys
import torch
import torchvision.models as models
from urllib.parse import urlparse

enum_text = sys.argv[1]
enum_class_name, enum_member_name = enum_text.split(".", maxsplit=1)
weights = getattr(getattr(models, enum_class_name), enum_member_name)
weights.get_state_dict(progress=False, check_hash=True)
weights_filename = Path(urlparse(weights.url).path).name
weights_path = Path(torch.hub.get_dir()) / "checkpoints" / weights_filename

gpus = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    gpus.append({
        "index": index,
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(index)),
    })
print(json.dumps({
    "host": platform.node(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpus": gpus,
    "backbone_weights": {
        "enum": f"{enum_class_name}.{weights.name}",
        "filename": weights_filename,
        "path": str(weights_path.resolve()),
        "torchvision_check_hash": True,
    },
}, sort_keys=True))
""".strip()
    metadata_command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        environment,
        "env",
        "PYTHONNOUSERSITE=1",
        "python",
        "-c",
        metadata_program,
        config["act_backbone_weights"]["enum"],
    ]

    pip_bytes = subprocess.run(pip_command, check=True, capture_output=True).stdout
    explicit_bytes = subprocess.run(explicit_command, check=True, capture_output=True).stdout
    metadata_stdout = subprocess.run(
        metadata_command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    metadata = json.loads(metadata_stdout)
    backbone_path = Path(metadata["backbone_weights"]["path"])
    if backbone_path.is_file():
        metadata["backbone_weights"]["bytes"] = backbone_path.stat().st_size
        metadata["backbone_weights"]["sha256"] = _sha256(backbone_path)
    else:
        metadata["backbone_weights"]["bytes"] = None
        metadata["backbone_weights"]["sha256"] = None

    pip_path = runtime_dir / "environment.lock.txt"
    explicit_path = runtime_dir / "conda-explicit.txt"
    pip_path.write_bytes(pip_bytes)
    explicit_path.write_bytes(explicit_bytes)
    observed = {
        "pip_freeze_lock": {
            "path": str(pip_path.resolve()),
            "sha256": _sha256(pip_path),
            "expected_sha256": runtime["pip_freeze_lock"]["sha256"],
        },
        "conda_explicit_lock": {
            "path": str(explicit_path.resolve()),
            "sha256": _sha256(explicit_path),
            "expected_sha256": runtime["conda_explicit_lock"]["sha256"],
        },
    }
    verification = {
        "environment": environment,
        "metadata": metadata,
        "locks": observed,
        "commands": {
            "pip_freeze": pip_command,
            "conda_explicit": explicit_command,
            "metadata": metadata_command,
        },
    }
    _write_json(runtime_dir / "verification.json", verification)
    if not metadata["cuda_available"] or not metadata["gpus"]:
        raise RuntimeError("registered ACT training requires an available CUDA GPU")
    for lock_name, lock in observed.items():
        if lock["sha256"] != lock["expected_sha256"]:
            raise RuntimeError(f"current {lock_name} SHA256 does not match the registered runtime")
    backbone = metadata["backbone_weights"]
    registered_backbone = config["act_backbone_weights"]
    if backbone["enum"] != registered_backbone["enum"]:
        raise RuntimeError("torchvision backbone weights enum mismatch")
    if backbone["filename"] != registered_backbone["filename"]:
        raise RuntimeError("torchvision backbone weights filename mismatch")
    if backbone["sha256"] != registered_backbone["sha256"]:
        raise RuntimeError("torchvision backbone weights full-file SHA256 mismatch")
    return verification


def _attempt_dir(output_root: Path, repo_head: str) -> tuple[Path, datetime]:
    started = datetime.now(timezone.utc)
    attempt_id = f"{started.strftime('%Y%m%dT%H%M%S%fZ')}_{repo_head[:12]}"
    attempt = output_root / "attempts" / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt, started


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    output_root = args.out or _resolve("results/cf_resrl/m0/act_training")
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    repo_error: Exception | None = None
    parent_status_error: Exception | None = None
    try:
        repo_head = _git(REPO_ROOT, "rev-parse", "HEAD")
    except Exception as exc:
        repo_head = "unknown"
        repo_error = exc
    try:
        parent_status = _git(REPO_ROOT, "status", "--porcelain")
    except Exception as exc:
        parent_status = ""
        parent_status_error = exc
    attempt_dir, started = _attempt_dir(output_root, repo_head)
    status_path = attempt_dir / "status.json"
    log_path = attempt_dir / "stdout_stderr.log"
    launcher_pid: int | None = None
    host: str | None = None
    gpus: list[dict[str, Any]] = []
    python_version: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    process: subprocess.Popen[bytes] | None = None
    returncode: int | None = None
    resolved: dict[str, Any] | None = None

    try:
        launcher_pid = os.getpid()
        host = socket.gethostname()
        if repo_error is not None:
            raise RuntimeError("cannot resolve parent repository HEAD") from repo_error
        if parent_status_error is not None:
            raise RuntimeError("cannot inspect parent repository status") from parent_status_error
        config = json.loads(config_path.read_text(encoding="utf-8"))
        episode_indices = validate_spec(config)
        if not args.dry_run and parent_status:
            raise RuntimeError("formal ACT training launcher requires a clean parent repository")
        if _git(RESFIT_ROOT, "rev-parse", "HEAD") != EXPECTED["resfit_commit"]:
            raise RuntimeError("ResFiT base commit mismatch")
        if _git(ROBOSUITE_ROOT, "rev-parse", "HEAD") != EXPECTED["robosuite_commit"]:
            raise RuntimeError("Robosuite base commit mismatch")
        if _git(LEROBOT_ROOT, "remote", "get-url", "origin") != EXPECTED["lerobot_repository"]:
            raise RuntimeError("LeRobot repository mismatch")
        if _git(LEROBOT_ROOT, "rev-parse", "HEAD") != EXPECTED["lerobot_commit"]:
            raise RuntimeError("LeRobot commit mismatch")
        if _git(LEROBOT_ROOT, "status", "--porcelain"):
            raise RuntimeError("LeRobot checkout has uncommitted changes")

        states_before = patch_states(config)
        if not args.dry_run:
            if _git(RESFIT_ROOT, "status", "--porcelain"):
                raise RuntimeError("formal ACT training must start from the clean pinned ResFiT checkout")
            runtime_verification = capture_and_verify_runtime(config, attempt_dir)
            metadata = runtime_verification["metadata"]
            host = metadata["host"]
            gpus = metadata["gpus"]
            python_version = metadata["python"]
            torch_version = metadata["torch"]
            cuda_version = metadata["cuda"]
            apply_result = subprocess.run(
                ["bash", str(REPO_ROOT / "scripts" / "apply_resfit_cf_resrl_patch.sh")],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            states_after = patch_states(config)
            if any(row["state"] != "applied" for row in states_after):
                raise RuntimeError("registered ResFiT overlays were not applied")
        else:
            runtime_verification = {
                "status": "NOT_EXECUTED_DRY_RUN",
                "registered_locks": config["runtime"],
            }
            states_after = states_before
            apply_result = None

        command = build_command(config, attempt_dir)
        resolved = {
            "status": "PREPARED",
            "experiment_kind": "act_training",
            "dry_run": args.dry_run,
            "started_at_utc": started.isoformat(),
            "git_commit": repo_head,
            "config_path": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
            "attempt_dir": str(attempt_dir.resolve()),
            "episode_indices": episode_indices,
            "patches_before": states_before,
            "patches_after": states_after,
            "patch_apply_stdout": apply_result.stdout if apply_result is not None else None,
            "runtime_verification": runtime_verification,
            "command": command,
            "command_shell": shlex.join(command),
            "checkpoint_selection": config["checkpoint_selection"],
            "launcher_pid": launcher_pid,
            "child_pid": None,
            "log_path": str(log_path.resolve()),
            "returncode": None,
            "host": host,
            "python": python_version,
            "torch": torch_version,
            "cuda": cuda_version,
            "gpus": gpus,
        }
        _write_json(attempt_dir / "resolved_launcher.json", resolved)
        print(resolved["command_shell"])
        if args.dry_run:
            _write_json(status_path, resolved)
            return 0

        with log_path.open("wb") as log:
            process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            _write_json(
                status_path,
                {
                    **resolved,
                    "status": "RUNNING",
                    "child_pid": process.pid,
                },
            )
            returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"ACT training exited with status {returncode}")

        policy_dir = attempt_dir / "training" / config["checkpoint_selection"]["relative_policy_path"]
        checkpoint_files = {}
        for name in ("config.json", "model.safetensors"):
            path = policy_dir / name
            if not path.is_file():
                raise RuntimeError(f"fixed final ACT checkpoint is missing {name}")
            checkpoint_files[name] = {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        completed = {
            **resolved,
            "status": "COMPLETED",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "child_pid": process.pid,
            "returncode": returncode,
            "stdout_stderr_sha256": _sha256(log_path),
            "fixed_final_checkpoint": checkpoint_files,
        }
        _write_json(status_path, completed)
        return 0
    except Exception as exc:
        verification_path = attempt_dir / "runtime" / "verification.json"
        if verification_path.is_file():
            runtime_verification = json.loads(verification_path.read_text(encoding="utf-8"))
            metadata = runtime_verification["metadata"]
            host = metadata["host"]
            gpus = metadata["gpus"]
            python_version = metadata["python"]
            torch_version = metadata["torch"]
            cuda_version = metadata["cuda"]
        if process is not None:
            returncode = process.returncode
        failed: dict[str, Any] = {
            **(resolved or {}),
            "status": "FAILED",
            "experiment_kind": "act_training",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": repo_head,
            "config_path": str(config_path.resolve()),
            "attempt_dir": str(attempt_dir.resolve()),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "launcher_pid": launcher_pid,
            "child_pid": process.pid if process is not None else None,
            "log_path": str(log_path.resolve()),
            "returncode": returncode,
            "host": host,
            "python": python_version,
            "torch": torch_version,
            "cuda": cuda_version,
            "gpus": gpus,
        }
        if log_path.is_file():
            failed["stdout_stderr_sha256"] = _sha256(log_path)
        _write_json(status_path, failed)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
