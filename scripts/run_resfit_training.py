#!/usr/bin/env python3
"""Launch the pinned 300k-environment-step Square ResFiT run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RESFIT_ROOT = REPO_ROOT / "third_party" / "resfit"
ROBOSUITE_ROOT = REPO_ROOT / "third_party" / "robosuite"
LEROBOT_ROOT = RESFIT_ROOT / "deps" / "lerobot"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "cf_resrl" / "m0_resfit_training.json"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_spec(config: dict[str, Any]) -> dict[str, Any]:
    upstream = config["upstream"]
    dataset = config["dataset"]
    base = config["base_policy"]
    training = config["training"]

    expected_upstream = {
        "resfit_commit": "66262b06a0caf159fb001504949b7b48ccda5f64",
        "robosuite_commit": "77a4751233c29456a5381209e30dd0dbf39a6557",
        "lerobot_commit": "69901b9b6a2300914ca3de0ea14b6fa6e0203bd4",
    }
    for key, value in expected_upstream.items():
        if upstream[key] != value:
            raise ValueError(f"upstream.{key} does not match the registered value")
    if upstream["required_patches"] != [
        "patches/resfit/train_bc_dexmg_cf_resrl.patch",
        "patches/resfit/residual_td3_cf_resrl.patch",
    ]:
        raise ValueError("ResFiT training requires the two registered overlays")

    split_path = _resolve(dataset["split_manifest"])
    split_sha = _sha256(split_path)
    if split_sha != dataset["split_manifest_sha256"]:
        raise ValueError("split manifest SHA-256 mismatch")
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if split_manifest["dataset_revision"] != dataset["revision"]:
        raise ValueError("split manifest dataset revision mismatch")
    episode_indices = split_manifest["splits"][dataset["split"]]
    if dataset["split"] != "train" or dataset["episodes"] != 240 or len(episode_indices) != 240:
        raise ValueError("ResFiT must use the registered 240-episode train split")

    base_dir = _resolve(base["local_dir"])
    base_files = {
        "config.json": {
            "path": str((base_dir / "config.json").resolve()),
            "sha256": _sha256(base_dir / "config.json"),
            "expected_sha256": base["config_sha256"],
        },
        "model.safetensors": {
            "path": str((base_dir / "model.safetensors").resolve()),
            "sha256": _sha256(base_dir / "model.safetensors"),
            "expected_sha256": base["model_sha256"],
        },
    }
    if any(row["sha256"] != row["expected_sha256"] for row in base_files.values()):
        raise ValueError("base policy artifact SHA-256 mismatch")

    expected_training = {
        "config_name": "residual_td3_square_config",
        "seed": 20261124,
        "warmup_environment_steps": 10000,
        "main_environment_steps": 290000,
        "total_environment_steps": 300000,
        "checkpoint_interval_environment_steps": 10000,
        "checkpoint_contract": "evaluation_loadable_not_training_resumable",
        "wandb_mode": "disabled",
        "selection": "fixed_environment_step",
        "output_root": "results/cf_resrl/m0/resfit_training",
    }
    if training != expected_training:
        raise ValueError("ResFiT settings do not match the registered fixed-step run")
    if training["warmup_environment_steps"] + training["main_environment_steps"] != training["total_environment_steps"]:
        raise ValueError("warm-up plus main environment-step budget must equal total")

    return {
        "split_manifest": {
            "path": str(split_path.resolve()),
            "sha256": split_sha,
            "episode_indices": episode_indices,
        },
        "base_policy": base_files,
    }


def patch_states(config: dict[str, Any], *, strict: bool = True) -> list[dict[str, str]]:
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
        elif strict:
            raise RuntimeError(f"registered patch cannot be applied or reversed: {patch}")
        else:
            state = "modified_or_conflicted"
        states.append({"path": str(patch.resolve()), "sha256": _sha256(patch), "state": state})
    return states


def build_command(config: dict[str, Any], attempt_dir: Path) -> list[str]:
    dataset = config["dataset"]
    base = config["base_policy"]
    training = config["training"]
    output_dir = (attempt_dir / "training").resolve()
    cache_dir = (attempt_dir / "cache").resolve()
    python_path = os.pathsep.join((str(RESFIT_ROOT), str(LEROBOT_ROOT)))
    overrides = [
        f"seed={training['seed']}",
        f"base_policy.local_dir={_resolve(base['local_dir']).resolve()}",
        f"base_policy.expected_config_sha256={base['config_sha256']}",
        f"base_policy.expected_model_sha256={base['model_sha256']}",
        f"offline_data.name={dataset['name']}",
        f"offline_data.revision={dataset['revision']}",
        f"offline_data.root={_resolve(dataset['root']).resolve()}",
        f"offline_data.split_manifest={_resolve(dataset['split_manifest']).resolve()}",
        f"offline_data.expected_split_manifest_sha256={dataset['split_manifest_sha256']}",
        f"offline_data.split={dataset['split']}",
        f"offline_data.video_backend={dataset['video_backend']}",
        f"offline_data.num_episodes={dataset['episodes']}",
        f"algo.learning_starts={training['warmup_environment_steps']}",
        f"algo.total_timesteps={training['total_environment_steps']}",
        f"checkpoint_interval={training['checkpoint_interval_environment_steps']}",
        f"output_dir={output_dir}",
        f"wandb.mode={training['wandb_mode']}",
        f"hydra.run.dir={(attempt_dir / 'hydra').resolve()}",
        "hydra.job.chdir=false",
    ]
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
        "resfit.rl_finetuning.scripts.train_residual_td3",
        f"--config-name={training['config_name']}",
        *overrides,
    ]


def _attempt_dir(output_root: Path, repo_head: str) -> tuple[Path, datetime]:
    started = datetime.now(timezone.utc)
    attempt_id = f"{started.strftime('%Y%m%dT%H%M%S%fZ')}_{repo_head[:12]}"
    attempt = output_root / "attempts" / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "cache").mkdir()
    return attempt, started


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

    attempt_dir, started = _attempt_dir(output_root, repo_head)
    status_path = attempt_dir / "status.json"

    try:
        if not args.dry_run and _git(REPO_ROOT, "status", "--porcelain"):
            raise RuntimeError("formal ResFiT training requires a clean parent repository")
        validation = validate_spec(config)
        if _git(RESFIT_ROOT, "rev-parse", "HEAD") != config["upstream"]["resfit_commit"]:
            raise RuntimeError("ResFiT base commit mismatch")
        if _git(ROBOSUITE_ROOT, "rev-parse", "HEAD") != config["upstream"]["robosuite_commit"]:
            raise RuntimeError("Robosuite base commit mismatch")
        if _git(LEROBOT_ROOT, "rev-parse", "HEAD") != config["upstream"]["lerobot_commit"]:
            raise RuntimeError("LeRobot runtime checkout mismatch")
        if _git(LEROBOT_ROOT, "status", "--porcelain"):
            raise RuntimeError("LeRobot runtime checkout has uncommitted changes")

        states_before = patch_states(config)
        apply_result = None
        if not args.dry_run:
            if _git(RESFIT_ROOT, "status", "--porcelain"):
                raise RuntimeError("formal ResFiT training must start from the clean pinned checkout")
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
            states_after = states_before

        command = build_command(config, attempt_dir)
        resolved = {
            "status": "DRY_RUN_VALIDATED" if args.dry_run else "RUNNING",
            "experiment_kind": "resfit_training",
            "dry_run": args.dry_run,
            "started_at_utc": started.isoformat(),
            "launcher_pid": os.getpid(),
            "git_commit": repo_head,
            "config_path": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
            "attempt_dir": str(attempt_dir.resolve()),
            "cache_dir": str((attempt_dir / "cache").resolve()),
            "training_output_dir": str((attempt_dir / "training").resolve()),
            "stdout_stderr_log": str((attempt_dir / "stdout_stderr.log").resolve()),
            "validation": validation,
            "environment_step_budget": {
                "warmup": config["training"]["warmup_environment_steps"],
                "main": config["training"]["main_environment_steps"],
                "total": config["training"]["total_environment_steps"],
            },
            "checkpoint_contract": config["training"]["checkpoint_contract"],
            "patches_before": states_before,
            "patches_after": states_after,
            "patch_apply_stdout": apply_result.stdout if apply_result is not None else None,
            "command": command,
            "command_shell": shlex.join(command),
        }
        _write_json(attempt_dir / "resolved_launcher.json", resolved)
        print(json.dumps({"attempt_dir": resolved["attempt_dir"], "command": command}, sort_keys=True))
        if args.dry_run:
            _write_json(status_path, resolved)
            return 0

        log_path = attempt_dir / "stdout_stderr.log"
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            _write_json(status_path, {**resolved, "child_pid": process.pid})
            returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"ResFiT training exited with status {returncode}")

        checkpoint_manifest = attempt_dir / "training" / "checkpoint_manifest.json"
        if not checkpoint_manifest.is_file():
            raise RuntimeError("completed ResFiT run has no checkpoint manifest")
        completed = {
            **resolved,
            "status": "COMPLETED",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "child_pid": process.pid,
            "child_returncode": returncode,
            "stdout_stderr_sha256": _sha256(log_path),
            "checkpoint_manifest_sha256": _sha256(checkpoint_manifest),
        }
        _write_json(status_path, completed)
        return 0
    except Exception as exc:
        failed = {
            "status": "FAILED",
            "experiment_kind": "resfit_training",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": repo_head,
            "config_path": str(config_path.resolve()),
            "attempt_dir": str(attempt_dir.resolve()),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "launcher_pid": os.getpid(),
        }
        log_path = attempt_dir / "stdout_stderr.log"
        if log_path.is_file():
            failed["stdout_stderr_sha256"] = _sha256(log_path)
        if "process" in locals():
            failed["child_pid"] = process.pid
        if "returncode" in locals():
            failed["child_returncode"] = returncode
        _write_json(status_path, failed)
        _write_json(attempt_dir / "failure.json", failed)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
