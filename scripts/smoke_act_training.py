#!/usr/bin/env python3
"""Run one real CUDA ACT optimizer step and verify its saved checkpoint."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import re
import runpy
import shlex
import subprocess
import sys
import time
import traceback
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RESFIT_ROOT = REPO_ROOT / "third_party" / "resfit"
ROBOSUITE_ROOT = REPO_ROOT / "third_party" / "robosuite"
LEROBOT_ROOT = RESFIT_ROOT / "deps" / "lerobot"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "cf_resrl" / "m0_act_training_smoke.json"
PATCH_ROOTS = {"resfit": RESFIT_ROOT, "lerobot": LEROBOT_ROOT}
EXPECTED = {
    "resfit_commit": "66262b06a0caf159fb001504949b7b48ccda5f64",
    "robosuite_commit": "77a4751233c29456a5381209e30dd0dbf39a6557",
    "lerobot_commit": "69901b9b6a2300914ca3de0ea14b6fa6e0203bd4",
    "dataset_revision": "4e5cee7ea7d63b920ff76c2529b781c2474697da",
    "split_manifest_sha256": "629d01857156c691f297218e0cda3e3b3a802a91122f006a1ca70fd31163afcd",
    "pip_freeze_sha256": "ce0fd3f7d80633ba96054f17a046fc3a4cc806ca945fd8e70928c9ec2a424eda",
    "conda_explicit_sha256": "2b5538025ba240838ad5e1798a4a1556748b191c0790bce16a2e169afc4e7f7d",
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


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def validate_spec(config: dict[str, Any]) -> list[int]:
    upstream = config["upstream"]
    runtime = config["runtime"]
    dataset = config["dataset"]
    training = config["training"]
    checkpoint = config["checkpoint"]

    for name in ("resfit_commit", "robosuite_commit", "lerobot_commit"):
        if upstream[name] != EXPECTED[name]:
            raise ValueError(f"upstream.{name} does not match the registered value")
    expected_patches = [
        {
            "path": "patches/resfit/train_bc_dexmg_cf_resrl.patch",
            "sha256": "63283baa4fd93a5802ae1e83170b4c0a18616df36ef9fcb36dff56b5def2d975",
            "target_root": "resfit",
        },
        {
            "path": "patches/resfit/residual_td3_cf_resrl.patch",
            "sha256": "441a6070a30d2c2de8ff286c008e948a16d573febe07d72041255b42a5d9b35f",
            "target_root": "resfit",
        },
        {
            "path": "patches/lerobot/noncontiguous_episode_subset.patch",
            "sha256": "4d20bc4ec899087891018fd8360eac31ec06774279c05581fada3827db967429",
            "target_root": "lerobot",
        },
    ]
    if upstream["patches"] != expected_patches:
        raise ValueError("ACT training smoke patch registration changed")
    for row in upstream["patches"]:
        if _sha256(_resolve(row["path"])) != row["sha256"]:
            raise ValueError(f"registered patch SHA256 mismatch: {row['path']}")

    expected_runtime = {
        "conda_environment": "cf-resrl-m0",
        "pip_freeze_lock": {
            "path": "results/cf_resrl/m0/act_runtime/environment.lock.txt",
            "sha256": EXPECTED["pip_freeze_sha256"],
        },
        "conda_explicit_lock": {
            "path": "results/cf_resrl/m0/act_runtime/conda-explicit.txt",
            "sha256": EXPECTED["conda_explicit_sha256"],
        },
    }
    if runtime != expected_runtime:
        raise ValueError("ACT training smoke runtime registration changed")
    for name in ("pip_freeze_lock", "conda_explicit_lock"):
        row = runtime[name]
        if _sha256(_resolve(row["path"])) != row["sha256"]:
            raise ValueError(f"registered runtime lock SHA256 mismatch: {name}")

    if dataset != {
        "repo_id": "ankile/robomimic-mh-square-image",
        "revision": EXPECTED["dataset_revision"],
        "root": "data/cf_resrl/square",
        "split_manifest": (
            "results/cf_resrl/m0/dataset_audit/attempts/"
            "20261124T131023495428Z_a9dcb4b558af/split_manifest.json"
        ),
        "split_manifest_sha256": EXPECTED["split_manifest_sha256"],
        "split": "train",
        "episode_count": 240,
        "video_backend": "pyav",
    }:
        raise ValueError("ACT training smoke dataset registration changed")
    split_path = _resolve(dataset["split_manifest"])
    if _sha256(split_path) != dataset["split_manifest_sha256"]:
        raise ValueError("ACT training smoke split manifest SHA256 mismatch")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    episode_indices = split["splits"][dataset["split"]]
    if split["dataset_revision"] != dataset["revision"] or len(episode_indices) != 240:
        raise ValueError("ACT training smoke split manifest content changed")

    if training != {
        "policy": "act",
        "seed": 20261124,
        "device": "cuda",
        "steps": 1,
        "batch_size": 256,
        "num_workers": 0,
        "grad_clip_norm": 10.0,
        "log_frequency": 1,
        "save_frequency": 1,
        "wandb_enabled": False,
        "rollout_enabled": False,
        "policy_kwargs": {"pretrained_backbone_weights": None},
        "output_root": "results/cf_resrl/m0/act_training_smoke",
    }:
        raise ValueError("ACT training smoke optimizer-step contract changed")
    if checkpoint != {
        "optimization_steps_completed": 1,
        "step_label": 0,
        "relative_policy_path": "training/policy_step_0/policy",
        "relative_trainer_state_path": "training/latest/trainer_state.pt",
        "reload": "strict_local_only",
    }:
        raise ValueError("ACT training smoke checkpoint contract changed")
    return episode_indices


def patch_states(config: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for registered in config["upstream"]["patches"]:
        patch = _resolve(registered["path"])
        target_name = registered["target_root"]
        target_root = PATCH_ROOTS[target_name]
        forward = subprocess.run(
            ["git", "-C", str(target_root), "apply", "--check", str(patch)],
            capture_output=True,
            text=True,
        )
        reverse = subprocess.run(
            ["git", "-C", str(target_root), "apply", "--reverse", "--check", str(patch)],
            capture_output=True,
            text=True,
        )
        if forward.returncode == 0:
            state = "not_applied"
        elif reverse.returncode == 0:
            state = "applied"
        else:
            raise RuntimeError(f"registered patch cannot be applied or reversed: {patch}")
        rows.append(
            {
                "path": str(patch.resolve()),
                "sha256": registered["sha256"],
                "target_root": target_name,
                "target_path": str(target_root.resolve()),
                "state": state,
            }
        )
    return rows


def build_training_argv(config: dict[str, Any], attempt_dir: Path) -> list[str]:
    dataset = config["dataset"]
    training = config["training"]
    return [
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
        json.dumps(training["policy_kwargs"], sort_keys=True, separators=(",", ":")),
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
        str((attempt_dir / "training").resolve()),
    ]


def _capture_runtime(config: dict[str, Any], attempt_dir: Path) -> dict[str, Any]:
    import torch

    registered_environment = config["runtime"]["conda_environment"]
    if os.environ.get("CONDA_DEFAULT_ENV") != registered_environment:
        raise RuntimeError(
            f"ACT training smoke must run inside {registered_environment}, "
            f"got {os.environ.get('CONDA_DEFAULT_ENV')!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("ACT training smoke requires CUDA")
    runtime_dir = attempt_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=False)
    pip_path = runtime_dir / "environment.lock.txt"
    conda_path = runtime_dir / "conda-explicit.txt"
    pip_path.write_bytes(
        subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=True,
            capture_output=True,
        ).stdout
    )
    conda_path.write_bytes(
        subprocess.run(
            ["conda", "list", "-n", registered_environment, "--explicit"],
            check=True,
            capture_output=True,
        ).stdout
    )
    observed = {
        "pip_freeze_lock": _sha256(pip_path),
        "conda_explicit_lock": _sha256(conda_path),
    }
    expected = {
        "pip_freeze_lock": config["runtime"]["pip_freeze_lock"]["sha256"],
        "conda_explicit_lock": config["runtime"]["conda_explicit_lock"]["sha256"],
    }
    if observed != expected:
        raise RuntimeError(f"current ACT runtime lock mismatch: {observed}")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "host": platform.node(),
        "gpu": {
            "index": device,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "locks": {
            "pip_freeze_lock": {
                "path": str(pip_path.resolve()),
                "sha256": observed["pip_freeze_lock"],
            },
            "conda_explicit_lock": {
                "path": str(conda_path.resolve()),
                "sha256": observed["conda_explicit_lock"],
            },
        },
    }


def _apply_patches(rows: list[dict[str, str]], applied: list[dict[str, str]]) -> None:
    if any(row["state"] != "not_applied" for row in rows):
        raise RuntimeError("ACT training smoke requires all overlays to start unapplied")
    for row in rows:
        patch = Path(row["path"])
        target_root = PATCH_ROOTS[row["target_root"]]
        subprocess.run(
            ["git", "-C", str(target_root), "apply", "--check", str(patch)],
            check=True,
        )
    for row in rows:
        patch = Path(row["path"])
        target_root = PATCH_ROOTS[row["target_root"]]
        subprocess.run(
            ["git", "-C", str(target_root), "apply", str(patch)],
            check=True,
        )
        applied.append(row)


def _reverse_patches(applied: list[dict[str, str]]) -> None:
    for row in reversed(applied):
        patch = Path(row["path"])
        target_root = PATCH_ROOTS[row["target_root"]]
        subprocess.run(
            ["git", "-C", str(target_root), "apply", "--reverse", str(patch)],
            check=True,
        )
    if _git(LEROBOT_ROOT, "status", "--porcelain"):
        raise RuntimeError("LeRobot checkout is not clean after ACT training smoke cleanup")
    if _git(RESFIT_ROOT, "status", "--porcelain"):
        raise RuntimeError("ResFiT checkout is not clean after ACT training smoke cleanup")


def _run_training(argv: list[str], attempt_dir: Path) -> dict[str, Any]:
    import torch

    log_path = attempt_dir / "stdout_stderr.log"
    original_argv = sys.argv[:]
    original_path = sys.path[:]
    env_updates = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "disabled",
        "HF_HUB_DISABLE_XET": "1",
        "CACHE_DIR": str((attempt_dir / "cache").resolve()),
    }
    original_env = {name: os.environ.get(name) for name in env_updates}
    for name, value in env_updates.items():
        os.environ[name] = value
    sys.path.insert(0, str(LEROBOT_ROOT))
    sys.path.insert(0, str(RESFIT_ROOT))
    sys.argv = argv
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8") as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                try:
                    runpy.run_module(argv[0], run_name="__main__")
                except Exception:
                    traceback.print_exc(file=stream)
                    raise
        torch.cuda.synchronize()
        duration = time.perf_counter() - started
        result = {
            "status": "COMPLETED",
            "duration_seconds": duration,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "stdout_stderr_log": str(log_path.resolve()),
            "stdout_stderr_sha256": _sha256(log_path),
        }
        _write_json(attempt_dir / "training_runtime.json", result)
        return result
    except Exception as exc:
        failed = {
            "status": "FAILED",
            "duration_seconds": time.perf_counter() - started,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "stdout_stderr_log": str(log_path.resolve()),
            "stdout_stderr_sha256": _sha256(log_path) if log_path.is_file() else None,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(attempt_dir / "training_runtime.json", failed)
        raise
    finally:
        logging.basicConfig(level=logging.WARNING, force=True)
        sys.argv = original_argv
        sys.path[:] = original_path
        for name, value in original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _verify_checkpoint(config: dict[str, Any], attempt_dir: Path) -> dict[str, Any]:
    import torch

    sys.path.insert(0, str(LEROBOT_ROOT))
    sys.path.insert(0, str(RESFIT_ROOT))
    from resfit.lerobot.policies.act.configuration_act import ACTConfig
    from resfit.lerobot.policies.act.modeling_act import ACTPolicy

    policy_dir = attempt_dir / config["checkpoint"]["relative_policy_path"]
    latest_policy_dir = attempt_dir / "training" / "latest" / "policy"
    trainer_state_path = attempt_dir / config["checkpoint"]["relative_trainer_state_path"]
    resolved_training_path = attempt_dir / "training" / "resolved_training_config.json"
    required = [
        policy_dir / "config.json",
        policy_dir / "model.safetensors",
        latest_policy_dir / "config.json",
        latest_policy_dir / "model.safetensors",
        trainer_state_path,
        resolved_training_path,
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"ACT training smoke is missing {path}")
    trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
    if trainer_state["step"] != config["checkpoint"]["step_label"]:
        raise RuntimeError("ACT training smoke trainer step label mismatch")
    optimizer_state_entries = len(trainer_state["optimizer"]["state"])
    if optimizer_state_entries == 0:
        raise RuntimeError("ACT training smoke optimizer checkpoint has no updated state")

    policy_config = ACTConfig.from_pretrained(policy_dir, local_files_only=True)
    policy_config.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(
        policy_dir,
        config=policy_config,
        local_files_only=True,
        strict=True,
    )
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    state_dict_elements = sum(value.numel() for value in policy.state_dict().values())
    del policy
    gc.collect()
    torch.cuda.empty_cache()

    latest_policy_config = ACTConfig.from_pretrained(latest_policy_dir, local_files_only=True)
    latest_policy_config.pretrained_backbone_weights = None
    latest_policy = ACTPolicy.from_pretrained(
        latest_policy_dir,
        config=latest_policy_config,
        local_files_only=True,
        strict=True,
    )
    latest_parameter_count = sum(parameter.numel() for parameter in latest_policy.parameters())
    latest_state_dict_elements = sum(value.numel() for value in latest_policy.state_dict().values())
    del latest_policy
    gc.collect()
    torch.cuda.empty_cache()
    if (latest_parameter_count, latest_state_dict_elements) != (
        parameter_count,
        state_dict_elements,
    ):
        raise RuntimeError("ACT training smoke latest checkpoint differs from fixed step-0 checkpoint")

    log_text = (attempt_dir / "stdout_stderr.log").read_text(encoding="utf-8")
    loss_match = re.search(r"\[step\s+0/1\]\s+loss:\s+([0-9.eE+-]+)", log_text)
    if loss_match is None:
        raise RuntimeError("ACT training smoke log has no step-0 loss")
    artifacts = {
        str(path.relative_to(attempt_dir)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "retained": path.name not in {"model.safetensors", "trainer_state.pt"},
        }
        for path in required
    }
    for name, row in artifacts.items():
        if not row["retained"]:
            row["discard_reason"] = "short-lived smoke payload exceeds ordinary Git artifact limits"
    discarded = _discard_heavy_smoke_artifacts(attempt_dir)
    for name, row in discarded.items():
        if artifacts[name]["sha256"] != row["sha256"] or artifacts[name]["bytes"] != row["bytes"]:
            raise RuntimeError("discarded ACT smoke artifact metadata changed before removal")
    return {
        "strict_local_reload": True,
        "strict_local_reload_latest": True,
        "optimization_steps_completed": config["checkpoint"]["optimization_steps_completed"],
        "step_label": trainer_state["step"],
        "loss": float(loss_match.group(1)),
        "optimizer_state_entries": optimizer_state_entries,
        "parameter_count": parameter_count,
        "state_dict_tensor_elements": state_dict_elements,
        "artifacts": artifacts,
    }


def _discard_heavy_smoke_artifacts(attempt_dir: Path) -> dict[str, dict[str, Any]]:
    relative_paths = (
        "training/policy_step_0/policy/model.safetensors",
        "training/latest/policy/model.safetensors",
        "training/latest/trainer_state.pt",
    )
    discarded = {}
    for name in relative_paths:
        path = attempt_dir / name
        if path.is_file():
            discarded[name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "retained": False,
                "discard_reason": "short-lived smoke payload exceeds ordinary Git artifact limits",
            }
            path.unlink()
    return discarded


def _evidence_checksums(attempt_dir: Path) -> dict[str, Any]:
    names = (
        "resolved_config.json",
        "status.json",
        "stdout_stderr.log",
        "training_runtime.json",
        "runtime/environment.lock.txt",
        "runtime/conda-explicit.txt",
        "training/resolved_training_config.json",
        "training/policy_step_0/policy/config.json",
        "training/latest/policy/config.json",
    )
    return {
        "algorithm": "sha256",
        "files": {name: _sha256(attempt_dir / name) for name in names},
    }


def _existing_evidence_checksums(attempt_dir: Path) -> dict[str, Any]:
    names = (
        "resolved_config.json",
        "status.json",
        "stdout_stderr.log",
        "training_runtime.json",
        "runtime/environment.lock.txt",
        "runtime/conda-explicit.txt",
        "training/resolved_training_config.json",
        "training/policy_step_0/policy/config.json",
        "training/latest/policy/config.json",
    )
    return {
        "algorithm": "sha256",
        "files": {
            name: _sha256(attempt_dir / name)
            for name in names
            if (attempt_dir / name).is_file()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = args.out or _resolve(config["training"]["output_root"])
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    repo_head = _git(REPO_ROOT, "rev-parse", "HEAD")
    parent_status = _git(REPO_ROOT, "status", "--porcelain")
    started = datetime.now(timezone.utc)
    attempt_id = f"{started.strftime('%Y%m%dT%H%M%S%fZ')}_{repo_head[:12]}"
    attempt_dir = output_root / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    status_path = attempt_dir / "status.json"
    applied: list[dict[str, str]] = []
    training_result: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    resolved: dict[str, Any] | None = None

    try:
        (attempt_dir / "cache").mkdir()
        episode_indices = validate_spec(config)
        if not args.dry_run and parent_status:
            raise RuntimeError("formal ACT training smoke requires a clean parent repository")
        if _git(RESFIT_ROOT, "rev-parse", "HEAD") != EXPECTED["resfit_commit"]:
            raise RuntimeError("ResFiT commit mismatch")
        if _git(ROBOSUITE_ROOT, "rev-parse", "HEAD") != EXPECTED["robosuite_commit"]:
            raise RuntimeError("Robosuite commit mismatch")
        if _git(LEROBOT_ROOT, "rev-parse", "HEAD") != EXPECTED["lerobot_commit"]:
            raise RuntimeError("LeRobot commit mismatch")
        if _git(RESFIT_ROOT, "status", "--porcelain"):
            raise RuntimeError("ACT training smoke requires a clean ResFiT checkout")
        if _git(LEROBOT_ROOT, "status", "--porcelain"):
            raise RuntimeError("ACT training smoke requires a clean LeRobot checkout")
        patches = patch_states(config)
        if any(row["state"] != "not_applied" for row in patches):
            raise RuntimeError("ACT training smoke requires unapplied registered overlays")
        training_argv = build_training_argv(config, attempt_dir)
        resolved = {
            **config,
            "run": {
                "attempt_id": attempt_id,
                "started_at_utc": started.isoformat(),
                "git_commit": repo_head,
                "git_dirty_before_attempt": bool(parent_status),
                "dry_run": args.dry_run,
                "evidence_dir": str(attempt_dir.resolve()),
                "command": shlex.join([sys.executable, "-m", *training_argv]),
            },
            "resolved": {
                "config_path": str(config_path.resolve()),
                "dataset_root": str(_resolve(config["dataset"]["root"]).resolve()),
                "split_manifest": str(_resolve(config["dataset"]["split_manifest"]).resolve()),
                "episode_indices": episode_indices,
                "patches": patches,
                "training_argv": training_argv,
            },
        }
        _write_json(attempt_dir / "resolved_config.json", resolved)
        prepared = {
            "status": "PREPARED" if args.dry_run else "RUNNING_PREFLIGHT",
            "experiment_kind": "act_training_smoke",
            "run": resolved["run"],
            "patches": patches,
        }
        _write_json(status_path, prepared)
        if args.dry_run:
            print(json.dumps({"attempt_dir": str(attempt_dir.resolve()), "training_argv": training_argv}))
            return 0

        runtime = _capture_runtime(config, attempt_dir)
        _apply_patches(patches, applied)
        _write_json(
            status_path,
            {
                **prepared,
                "status": "RUNNING",
                "runtime": runtime,
                "patches_applied": applied,
            },
        )
        try:
            training_result = _run_training(training_argv, attempt_dir)
            checkpoint = _verify_checkpoint(config, attempt_dir)
        finally:
            _reverse_patches(applied)
            applied.clear()

        completed = {
            "status": "COMPLETED",
            "experiment_kind": "act_training_smoke",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "run": resolved["run"],
            "runtime": runtime,
            "dataset": {
                "repo_id": config["dataset"]["repo_id"],
                "revision": config["dataset"]["revision"],
                "split": config["dataset"]["split"],
                "split_manifest_sha256": config["dataset"]["split_manifest_sha256"],
                "episode_count": len(episode_indices),
            },
            "patches": patches,
            "training": training_result,
            "checkpoint": checkpoint,
            "resfit_clean_after_cleanup": not bool(_git(RESFIT_ROOT, "status", "--porcelain")),
            "lerobot_clean_after_cleanup": not bool(_git(LEROBOT_ROOT, "status", "--porcelain")),
            "limitations": config["limitations"],
        }
        _write_json(status_path, completed)
        _write_json(attempt_dir / "evidence_checksums.json", _evidence_checksums(attempt_dir))
        print(json.dumps(completed, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        discarded_partial_artifacts = _discard_heavy_smoke_artifacts(attempt_dir)
        if applied:
            try:
                _reverse_patches(applied)
                applied.clear()
            except Exception as cleanup_exc:
                cleanup_error = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            else:
                cleanup_error = None
        else:
            cleanup_error = None
        failed = {
            "status": "FAILED",
            "experiment_kind": "act_training_smoke",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "run": (resolved or {}).get(
                "run",
                {
                    "attempt_id": attempt_id,
                    "git_commit": repo_head,
                    "evidence_dir": str(attempt_dir.resolve()),
                },
            ),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "cleanup_error": cleanup_error,
            "resfit_status": _git(RESFIT_ROOT, "status", "--porcelain"),
            "lerobot_status": _git(LEROBOT_ROOT, "status", "--porcelain"),
            "discarded_partial_artifacts": discarded_partial_artifacts,
        }
        if training_result is not None:
            failed["training"] = training_result
        elif (attempt_dir / "training_runtime.json").is_file():
            failed["training"] = json.loads(
                (attempt_dir / "training_runtime.json").read_text(encoding="utf-8")
            )
        if runtime is not None:
            failed["runtime"] = runtime
        _write_json(status_path, failed)
        _write_json(
            attempt_dir / "evidence_checksums.json",
            _existing_evidence_checksums(attempt_dir),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
