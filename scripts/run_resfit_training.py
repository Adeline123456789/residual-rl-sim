#!/usr/bin/env python3
"""Launch a pinned 300k-environment-step Square ResFiT run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RESFIT_ROOT = REPO_ROOT / "third_party" / "resfit"
ROBOSUITE_ROOT = REPO_ROOT / "third_party" / "robosuite"
LEROBOT_ROOT = RESFIT_ROOT / "deps" / "lerobot"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "cf_resrl" / "m0_resfit_training.json"
PATCH_ROOTS = {"resfit": RESFIT_ROOT, "lerobot": LEROBOT_ROOT}
X1_SEEDS = (20261125, 20261126)
X1_OUTPUT_ROOT = "results/cf_resrl/expansion_square/resfit_training"
RESUMABLE_PATCH = "patches/resfit/residual_td3_cf_resrl_resume.patch"
EXPECTED_PATCHES = [
    {
        "path": "patches/resfit/train_bc_dexmg_cf_resrl.patch",
        "sha256": "63283baa4fd93a5802ae1e83170b4c0a18616df36ef9fcb36dff56b5def2d975",
        "target_root": "resfit",
    },
    {
        "path": "patches/resfit/residual_td3_cf_resrl_eval.patch",
        "sha256": "7fba47e4ee1fb75d22d162020df202f598f0117b549dbc0fa26159843722df0f",
        "target_root": "resfit",
    },
    {
        "path": "patches/lerobot/noncontiguous_episode_subset.patch",
        "sha256": "4d20bc4ec899087891018fd8360eac31ec06774279c05581fada3827db967429",
        "target_root": "lerobot",
    },
]


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


def _overlay_mode(config: dict[str, Any]) -> str:
    if "resume_checkpoint_interval_environment_steps" in config["training"]:
        return "resumable"
    return "fixed-evaluation"


def validate_spec(config: dict[str, Any]) -> dict[str, Any]:
    upstream = config["upstream"]
    dataset = config["dataset"]
    base = config["base_policy"]
    runtime = config["runtime"]
    training = config["training"]

    expected_upstream = {
        "resfit_commit": "66262b06a0caf159fb001504949b7b48ccda5f64",
        "robosuite_commit": "77a4751233c29456a5381209e30dd0dbf39a6557",
        "lerobot_commit": "69901b9b6a2300914ca3de0ea14b6fa6e0203bd4",
    }
    for key, value in expected_upstream.items():
        if upstream[key] != value:
            raise ValueError(f"upstream.{key} does not match the registered value")
    expected_patches = EXPECTED_PATCHES
    resumable = _overlay_mode(config) == "resumable"
    if resumable:
        expected_patches = [
            EXPECTED_PATCHES[0],
            {"path": RESUMABLE_PATCH, "sha256": _sha256(_resolve(RESUMABLE_PATCH)), "target_root": "resfit"},
            EXPECTED_PATCHES[2],
        ]
    if upstream["patches"] != expected_patches:
        raise ValueError("ResFiT training patch registration changed")
    for registered in upstream["patches"]:
        patch = _resolve(registered["path"])
        if _sha256(patch) != registered["sha256"]:
            raise ValueError(f"registered patch SHA256 mismatch: {registered['path']}")

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

    source_status_path = _resolve(base["source_status"])
    source_status = json.loads(source_status_path.read_text(encoding="utf-8"))
    selection = source_status["checkpoint_selection"]
    base_dir = _resolve(base["local_dir"]).resolve()
    if (
        source_status["status"] != "COMPLETED"
        or source_status["returncode"] != 0
        or source_status["dry_run"]
        or selection["selection"] != "final_fixed_optimization_step"
        or selection["optimization_steps_completed"] != 50000
        or selection["checkpoint_step_label"] != 49999
        or base_dir != source_status_path.resolve().parent / "training" / selection["relative_policy_path"]
    ):
        raise ValueError("ResFiT must start from the completed ACT 50k fixed final checkpoint")
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
    if any(
        row["sha256"] != row["expected_sha256"]
        or row["sha256"] != source_status["fixed_final_checkpoint"][name]["sha256"]
        for name, row in base_files.items()
    ):
        raise ValueError("base policy artifact SHA-256 mismatch")
    if runtime != {
        "conda_environment": "cf-resrl-resfit",
        "extension_requirements": "configs/cf_resrl/m0_resfit_runtime_requirements.txt",
    }:
        raise ValueError("ResFiT must use the registered runtime")

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
    if resumable:
        if training["seed"] not in X1_SEEDS:
            raise ValueError("X1 ResFiT seed must be one of the registered replication seeds")
        expected_training.update(
            seed=training["seed"],
            output_root=X1_OUTPUT_ROOT,
            resume_checkpoint_interval_environment_steps=100000,
            resume_checkpoint_retention=3,
        )
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
        "base_policy_source": {
            "status_path": base["source_status"],
            "status_sha256": _sha256(source_status_path),
            "git_commit": source_status["git_commit"],
            "checkpoint_selection": selection,
        },
    }


def patch_states(config: dict[str, Any], *, strict: bool = True) -> list[dict[str, str]]:
    states = []
    for registered in config["upstream"]["patches"]:
        patch = _resolve(registered["path"])
        target_name = registered["target_root"]
        target_root = PATCH_ROOTS[target_name]
        reverse = subprocess.run(
            ["git", "-C", str(target_root), "apply", "--reverse", "--check", str(patch)],
            capture_output=True,
            text=True,
        )
        forward = subprocess.run(
            ["git", "-C", str(target_root), "apply", "--check", str(patch)],
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
        states.append(
            {
                "path": str(patch.resolve()),
                "sha256": _sha256(patch),
                "target_root": target_name,
                "target_path": str(target_root.resolve()),
                "state": state,
            }
        )
    return states


def _patch_command(action: str, overlay: str) -> list[str]:
    return ["bash", str(REPO_ROOT / "scripts" / "apply_resfit_cf_resrl_patch.sh"), action, overlay]


def restore_patch_checkouts(overlay: str = "fixed-evaluation") -> str:
    result = subprocess.run(
        _patch_command("reverse", overlay),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if _git(LEROBOT_ROOT, "status", "--porcelain"):
        raise RuntimeError("LeRobot checkout is not clean after patch cleanup")
    if _git(RESFIT_ROOT, "status", "--porcelain"):
        raise RuntimeError("ResFiT checkout is not clean after patch cleanup")
    return result.stdout


def _lock_overlay() -> tuple[int, Path]:
    marker = Path(_git(REPO_ROOT, "rev-parse", "--path-format=absolute", "--git-path",
                       "cf_resrl_resfit_overlay_owner.json"))
    descriptor = os.open(marker.with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise RuntimeError("another ResFiT launcher holds the checkout overlay lock") from exc
    except OSError:
        os.close(descriptor)
        raise
    return descriptor, marker


def _record_overlay_owner(marker: Path, owner: dict[str, Any]) -> None:
    temporary = marker.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(owner, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(marker)
    directory = os.open(marker.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _ensure_training_is_inactive(attempt: Path) -> None:
    trainer = b"resfit.rl_finetuning.scripts.train_residual_td3"
    output = os.fsencode(f"output_dir={attempt / 'training'}")
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if trainer in argv and output in argv:
            raise RuntimeError(f"the prior ResFiT process is still live: PID {proc.name}")


def _check_exact_overlay(target: Path, applied: list[Path]) -> None:
    """Prove that reversing the registered patches alone yields HEAD."""
    if _git(target, "ls-files", "--others", "--exclude-standard") or _git(target, "diff", "--cached", "--name-only"):
        raise RuntimeError(f"checkout has unrelated staged or untracked changes: {target}")
    with tempfile.TemporaryDirectory(prefix="cf_resrl_index_") as directory:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(directory) / "index")}
        git = ["git", "-C", str(target)]
        commands = [["read-tree", "HEAD"], ["add", "-u"]]
        commands.extend(["apply", "--cached", "--reverse", str(path)] for path in reversed(applied))
        commands.append(["diff", "--cached", "--exit-code", "--quiet", "HEAD"])
        for command in commands:
            if subprocess.run([*git, *command], env=env, capture_output=True).returncode:
                raise RuntimeError(f"checkout contains changes outside the registered overlay: {target}")


def _recover_stale_overlay(
    marker: Path, config: dict[str, Any], config_sha256: str, seed: int
) -> dict[str, Any] | None:
    if not marker.exists():
        return None
    owner = json.loads(marker.read_text(encoding="utf-8"))
    prior = Path(owner["attempt_dir"])
    if (
        owner["mode"] != "resumable"
        or owner["config_sha256"] != config_sha256
        or owner["training_seed"] != seed
        or not prior.is_absolute()
        or prior.parent.name != "attempts"
        or not prior.is_dir()
    ):
        raise RuntimeError("stale ResFiT overlay ownership does not match this X1 run")
    _ensure_training_is_inactive(prior)
    states = patch_states(config)
    for name, target in PATCH_ROOTS.items():
        _check_exact_overlay(target, [
            Path(row["path"]) for row in states if row["target_root"] == name and row["state"] == "applied"
        ])
    cleanup = restore_patch_checkouts("resumable")
    marker.unlink()
    return {"attempt_dir": str(prior), "launcher_pid": owner["launcher_pid"], "cleanup_stdout": cleanup}


def _check_parent_checkout(resumable: bool) -> None:
    if resumable:
        status = _git(
            REPO_ROOT, "status", "--porcelain=v1", "-z", "--no-renames",
            "--ignore-submodules=dirty", "--untracked-files=all",
        )
        if all(row.startswith(f"?? {X1_OUTPUT_ROOT}/") for row in status.split("\0") if row):
            return
    elif not _git(REPO_ROOT, "status", "--porcelain"):
        return
    raise RuntimeError("formal ResFiT training requires a clean parent repository")


def _resume_provenance(
    generation: Path, config: dict[str, Any], config_sha256: str, seed: int
) -> dict[str, Any]:
    """Bind a continuation to its completed generation and its recorded launch."""
    generation = generation.expanduser().resolve()
    manifest_path = generation / "resume.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    interval = config["training"]["resume_checkpoint_interval_environment_steps"]
    total = config["training"]["total_environment_steps"]
    step = manifest["step"]
    if (
        manifest["version"] != 1
        or type(step) is not int
        or not interval <= step < total
        or step % interval
        or generation.name != f"step_{step:09d}"
        or generation.parent.name != "resume"
        or generation.parent.parent.name != "training"
        or not (generation / "state.pt").is_file()
    ):
        raise ValueError("resume path is not an eligible local fixed-step ResFiT generation")
    origin = generation.parents[2]
    ledger = Path(manifest["action_ledger"])
    producer = ledger.resolve().parents[1]
    if (
        not ledger.is_absolute()
        or ledger.resolve() != producer / "training" / "action_ledger.jsonl"
        or not ledger.is_file()
        or origin.parent.name != "attempts"
        or producer.parent.name != "attempts"
    ):
        raise ValueError("resume action ledger is outside its recorded training attempt")
    if any(
        path.name > generation.name
        and len(path.name) == len(generation.name)
        and path.name[5:].isdecimal()
        and (path / "resume.json").is_file()
        for path in generation.parent.glob("step_*")
    ):
        raise ValueError("resume from the latest completed generation in the training lineage")
    budget = {
        label: config["training"][f"{label}_environment_steps"] for label in ("warmup", "main", "total")
    }
    runtime_locks = None
    for attempt in dict.fromkeys((origin, producer)):
        receipt = json.loads((attempt / "resolved_launcher.json").read_text(encoding="utf-8"))
        if (
            receipt["dry_run"]
            or receipt["config_sha256"] != config_sha256
            or receipt.get("training_seed") != seed
            or receipt.get("overlay_mode") != "resumable"
            or receipt["attempt_dir"] != str(attempt)
            or receipt["environment_step_budget"] != budget
            or receipt.get("resume_checkpoint_interval_environment_steps") != interval
            or receipt.get("resume_checkpoint_retention") != config["training"]["resume_checkpoint_retention"]
        ):
            raise ValueError("resume generation config and seed do not match its recorded launch")
        prior_resume = receipt.get("resume")
        if (attempt == origin and prior_resume is not None) or (
            attempt != origin and (not prior_resume or prior_resume["origin_attempt"] != str(origin))
        ):
            raise ValueError("resume producer does not belong to the original training lineage")
        locks = {name: row["sha256"] for name, row in receipt["runtime"]["locks"].items()}
        if locks.keys() != {"conda-explicit.txt", "environment.lock.txt"}:
            raise ValueError("resume lineage is missing the recorded runtime locks")
        if runtime_locks is not None and locks != runtime_locks:
            raise ValueError("resume lineage dependency locks disagree")
        runtime_locks = locks
        _ensure_training_is_inactive(attempt)
    return {
        "generation": str(generation),
        "manifest_sha256": _sha256(manifest_path),
        "origin_attempt": str(origin),
        "producer_attempt": str(producer),
        "restored_environment_steps": step,
        "runtime_lock_sha256": runtime_locks,
    }


def build_command(
    config: dict[str, Any], attempt_dir: Path, resume_from: Path | None = None
) -> list[str]:
    dataset = config["dataset"]
    base = config["base_policy"]
    training = config["training"]
    output_dir = (attempt_dir / "training").resolve()
    cache_dir = (attempt_dir / "cache").resolve()
    resumable = _overlay_mode(config) == "resumable"
    python_roots = (RESFIT_ROOT, LEROBOT_ROOT, REPO_ROOT) if resumable else (RESFIT_ROOT, LEROBOT_ROOT)
    python_path = os.pathsep.join(map(str, python_roots))
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
    if resumable:
        overrides.extend([
            f"resume_checkpoint_interval={training['resume_checkpoint_interval_environment_steps']}",
            f"resume_checkpoint_retention={training['resume_checkpoint_retention']}",
        ])
        if resume_from is not None:
            overrides.append(f"resume_from={resume_from}")
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        config["runtime"]["conda_environment"],
        "env",
        "PYTHONNOUSERSITE=1",
        "PYTHONUNBUFFERED=1",
        # Evaluation writes plots from a headless trainer with replay worker threads.
        "MPLBACKEND=Agg",
        f"PYTHONPATH={python_path}",
        f"CACHE_DIR={cache_dir}",
        "WANDB_MODE=disabled",
        "HF_HUB_DISABLE_XET=1",
        *(["HF_HUB_OFFLINE=1"] if resumable else []),
        "python",
        "-m",
        "resfit.rl_finetuning.scripts.train_residual_td3",
        f"--config-name={training['config_name']}",
        *overrides,
    ]


def capture_runtime(config: dict[str, Any], attempt_dir: Path) -> dict[str, Any]:
    runtime = config["runtime"]
    name = runtime["conda_environment"]
    runtime_dir = attempt_dir / "runtime"
    runtime_dir.mkdir()
    commands = {
        "environment.lock.txt": [
            "conda", "run", "-n", name, "env", "PYTHONNOUSERSITE=1", "python", "-m", "pip", "freeze", "--all",
        ],
        "conda-explicit.txt": ["conda", "list", "-n", name, "--explicit"],
    }
    files = {}
    for filename, command in commands.items():
        path = runtime_dir / filename
        path.write_bytes(subprocess.check_output(command, cwd=REPO_ROOT))
        files[filename] = {"path": str(path.resolve()), "sha256": _sha256(path)}
    requirements = _resolve(runtime["extension_requirements"])
    frozen = set((runtime_dir / "environment.lock.txt").read_text().lower().replace("_", "-").splitlines())
    for line in requirements.read_text().splitlines() + ["torch==2.7.1+cu128", "mujoco==3.3.2", "gymnasium==1.1.1"]:
        if line.lower() not in frozen:
            raise RuntimeError(f"ResFiT runtime is missing pinned distribution {line}")
    return {
        **runtime,
        "extension_requirements_sha256": _sha256(requirements),
        "locks": files,
    }


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
    parser.add_argument("--seed", type=int, choices=X1_SEEDS)
    parser.add_argument("--resume-from", type=Path)
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
    config_sha256 = _sha256(config_path)
    overlay = _overlay_mode(config)
    resumable = overlay == "resumable"
    patch_apply_attempted = False
    patch_cleanup_stdout: str | None = None
    lock_descriptor = None
    marker = None
    owner_recorded = False
    resolved = {}

    try:
        if not args.dry_run:
            lock_descriptor, marker = _lock_overlay()
            _check_parent_checkout(resumable)
        if not resumable and (args.seed is not None or args.resume_from is not None):
            raise ValueError("--seed and --resume-from require the registered X1 ResFiT config")
        validation = validate_spec(config)
        training = config["training"]
        if args.seed is not None:
            training["seed"] = args.seed
        seed = training["seed"]
        resume = None
        if args.resume_from is not None:
            resume = _resume_provenance(args.resume_from, config, config_sha256, seed)
        if _git(RESFIT_ROOT, "rev-parse", "HEAD") != config["upstream"]["resfit_commit"]:
            raise RuntimeError("ResFiT base commit mismatch")
        if _git(ROBOSUITE_ROOT, "rev-parse", "HEAD") != config["upstream"]["robosuite_commit"]:
            raise RuntimeError("Robosuite base commit mismatch")
        if _git(LEROBOT_ROOT, "rev-parse", "HEAD") != config["upstream"]["lerobot_commit"]:
            raise RuntimeError("LeRobot runtime checkout mismatch")
        stale_overlay = None
        if not args.dry_run and resumable:
            stale_overlay = _recover_stale_overlay(marker, config, config_sha256, seed)
        if _git(LEROBOT_ROOT, "status", "--porcelain"):
            raise RuntimeError("LeRobot runtime checkout has uncommitted changes")

        states_before = patch_states(config)
        apply_result = None
        runtime = None
        if not args.dry_run:
            if _git(RESFIT_ROOT, "status", "--porcelain"):
                raise RuntimeError("formal ResFiT training must start from the clean pinned checkout")
            runtime = capture_runtime(config, attempt_dir)
            if resume is not None and {
                name: row["sha256"] for name, row in runtime["locks"].items()
            } != resume["runtime_lock_sha256"]:
                raise RuntimeError("local ResFiT dependency locks differ from the resume lineage")
            if resumable:
                _record_overlay_owner(marker, {
                    "mode": overlay,
                    "config_sha256": config_sha256,
                    "training_seed": seed,
                    "attempt_dir": str(attempt_dir.resolve()),
                    "launcher_pid": os.getpid(),
                })
                owner_recorded = True
            patch_apply_attempted = True
            apply_result = subprocess.run(
                _patch_command("apply", overlay),
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            states_after = patch_states(config)
            if any(row["state"] != "applied" for row in states_after):
                raise RuntimeError("registered training overlays were not applied")
        else:
            states_after = states_before

        command = build_command(config, attempt_dir, Path(resume["generation"]) if resume else None)
        resolved = {
            "status": "DRY_RUN_VALIDATED" if args.dry_run else "RUNNING",
            "experiment_kind": "resfit_training",
            "dry_run": args.dry_run,
            "started_at_utc": started.isoformat(),
            "launcher_pid": os.getpid(),
            "git_commit": repo_head,
            "config_path": str(config_path.resolve()),
            "config_sha256": config_sha256,
            "attempt_dir": str(attempt_dir.resolve()),
            "cache_dir": str((attempt_dir / "cache").resolve()),
            "training_output_dir": str((attempt_dir / "training").resolve()),
            "stdout_stderr_log": str((attempt_dir / "stdout_stderr.log").resolve()),
            "validation": validation,
            "runtime": runtime,
            "git_dirty_before_attempt": False if not args.dry_run else None,
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
        if resumable:
            resolved.update({
                "overlay_mode": overlay,
                "training_seed": seed,
                "resume_checkpoint_interval_environment_steps": training["resume_checkpoint_interval_environment_steps"],
                "resume_checkpoint_retention": training["resume_checkpoint_retention"],
                "resume": resume,
                "remaining_environment_step_budget_at_launch": (
                    training["total_environment_steps"] - resume["restored_environment_steps"]
                    if resume else training["total_environment_steps"]
                ),
                "stale_overlay_recovery": stale_overlay,
            })
        _write_json(attempt_dir / "resolved_launcher.json", resolved)
        print(json.dumps({"attempt_dir": resolved["attempt_dir"], "command": command}, sort_keys=True))
        if args.dry_run:
            _write_json(status_path, resolved)
            return 0

        log_path = attempt_dir / "stdout_stderr.log"
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock_descriptor,)
            )
            _write_json(status_path, {**resolved, "child_pid": process.pid})
            returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"ResFiT training exited with status {returncode}")

        checkpoint_manifest = attempt_dir / "training" / "checkpoint_manifest.json"
        if not checkpoint_manifest.is_file():
            raise RuntimeError("completed ResFiT run has no checkpoint manifest")
        patch_cleanup_stdout = restore_patch_checkouts(overlay)
        patch_apply_attempted = False
        if owner_recorded:
            marker.unlink()
            owner_recorded = False
        completed = {
            **resolved,
            "status": "COMPLETED",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "child_pid": process.pid,
            "child_returncode": returncode,
            "stdout_stderr_sha256": _sha256(log_path),
            "checkpoint_manifest_sha256": _sha256(checkpoint_manifest),
            "patch_cleanup_stdout": patch_cleanup_stdout,
            "resfit_clean_after_cleanup": True,
            "lerobot_clean_after_cleanup": True,
        }
        _write_json(status_path, completed)
        return 0
    except Exception as exc:
        cleanup_error: str | None = None
        if patch_apply_attempted and "process" in locals() and process.poll() is None:
            cleanup_error = "training process is still live; left its overlays in place"
        elif patch_apply_attempted:
            try:
                patch_cleanup_stdout = restore_patch_checkouts(overlay)
                patch_apply_attempted = False
                if owner_recorded:
                    marker.unlink()
                    owner_recorded = False
            except Exception as cleanup_exc:
                cleanup_error = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
        failed = {
            **(resolved if resumable else {}),
            "status": "FAILED",
            "experiment_kind": "resfit_training",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": repo_head,
            "config_path": str(config_path.resolve()),
            "attempt_dir": str(attempt_dir.resolve()),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "launcher_pid": os.getpid(),
            "patch_cleanup_stdout": patch_cleanup_stdout,
            "cleanup_error": cleanup_error,
            "resfit_status": _git(RESFIT_ROOT, "status", "--porcelain"),
            "lerobot_status": _git(LEROBOT_ROOT, "status", "--porcelain"),
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
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
