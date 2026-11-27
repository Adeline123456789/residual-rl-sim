"""Registered Square X1 cells and local ACT/ResFiT checkpoint provenance."""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path
from typing import Any, Mapping

from .dataset import make_episode_splits, sha256_file


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs/cf_resrl/square_x1.json"
_AXES = {
    "resfit_seed": [20261124, 20261125, 20261126],
    "resfit_step": [150000, 300000],
    "act_label": [25000, 49999],
    "delay": [0, 2, 4],
    "residual_scale": [0.5, 1.0, 1.5],
    "distribution": ["natural", "enriched"],
}
_FIXED_SELECTION = "fixed_environment_step"
_RUNTIME_LOCKS = {"conda-explicit.txt", "environment.lock.txt"}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cohort_seeds(cohort: dict[str, int]) -> range:
    return range(cohort["first_seed"], cohort["first_seed"] + cohort["size"])


def load_x1_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read and validate the registration without reading any training artifacts."""
    config = _json(Path(path))
    if (
        config["status"] != "SPECIFICATION_NOT_A_RUN_RESULT"
        or config["task"] != "Square"
        or config["axes"] != _AXES
        or config["act_call_caps"] != [15, 20, 25]
        or config["residual_component_limit"] != 0.1
        or config["factorial_split"] != {"train": 60, "validation": 20, "test": 20}
        or config["act"]["training_base_label"] != 49999
        or config["resfit"]["registered_budget"] != {"warmup": 10000, "main": 290000, "total": 300000}
    ):
        raise ValueError("Square X1 registration axes, budgets, or split changed")
    cohort_sizes = {
        "factorial_cohorts": {"natural": 100, "enriched": 100},
        "pair_training_cohorts": {"natural": 60, "enriched": 60},
        "closed_loop_cohorts": {"validation": 20, "test": 50},
    }
    assigned: set[int] = set()
    for name, sizes in cohort_sizes.items():
        cohorts = config[name]
        if {label: row["size"] for label, row in cohorts.items()} != sizes:
            raise ValueError(f"Square X1 cohort sizes changed for {name}")
        for row in cohorts.values():
            seeds = set(_cohort_seeds(row))
            if min(seeds) < 20270124 or seeds & assigned:
                raise ValueError("Square X1 cohorts must be disjoint from one another and M0/E0/MVP")
            assigned.update(seeds)
    return config


def plan_x1(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate policy, delay, residual, and distribution cells in registered order."""
    axes = config["axes"]
    cells = []
    for seed, step, label, delay, scale, distribution in product(*(axes[name] for name in _AXES)):
        cells.append({
            "cell_id": f"r{seed}_s{step}_a{label}_d{delay}_rscale{scale:g}_{distribution}",
            "resfit_seed": seed,
            "resfit_step": step,
            "act_label": label,
            "act_evaluation_mode": config["act_evaluation_modes"][str(label)],
            "delay": delay,
            "residual_scale": scale,
            "distribution": distribution,
        })
    return cells


def source_cohort(config: dict[str, Any], distribution: str) -> list[dict[str, Any]]:
    """Use the same source seeds and split roles in every cell of a distribution."""
    if distribution not in config["axes"]["distribution"]:
        raise ValueError(f"Unknown Square X1 distribution: {distribution}")
    factorial = config["factorial_cohorts"][distribution]
    sizes = config["factorial_split"]
    splits = make_episode_splits(
        factorial["size"], sizes["train"], sizes["validation"], sizes["test"],
        config["split_seed"] + factorial["first_seed"],
    )
    split_by_seed = {
        factorial["first_seed"] + index: split
        for split, indices in splits.items()
        for index in indices
    }
    sources = []
    for bank, cohort in (
        ("factorial", factorial),
        ("pair", config["pair_training_cohorts"][distribution]),
    ):
        for seed in _cohort_seeds(cohort):
            sources.append({
                "source_episode_id": f"square_x1_{seed}",
                "seed": seed,
                "distribution": distribution,
                "bank": bank,
                "split": split_by_seed.get(seed, "train"),
                "selector_seed": config["selector_seed"] + seed,
            })
    return sources


def _verify_sha(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def _local_act(
    config: dict[str, Any], label: int
) -> tuple[dict[str, str], Path, dict[str, str]]:
    act = config["act"]
    attempt = (REPO_ROOT / act["attempt"]).resolve()
    path = attempt / "evidence_manifest.json"
    _verify_sha(path, act["evidence_manifest_sha256"])
    manifest = _json(path)
    final_label = act["training_base_label"]
    status = _json(attempt / "status.json")
    if (
        manifest["status"] != "COMPLETED"
        or manifest["attempt_id"] != attempt.name
        or status["status"] != "COMPLETED"
        or status["dry_run"]
        or status["returncode"] != 0
        or status["checkpoint_selection"]["checkpoint_step_label"] != final_label
        or any(manifest["upstream"][key] != value for key, value in config["upstream"].items())
    ):
        raise ValueError("Registered ACT training evidence does not match the completed fixed run")

    artifacts = {}
    for fixed_label in dict.fromkeys((final_label, label)):
        rel = Path("training") / f"policy_step_{fixed_label}" / "policy"
        digests = {"config.json": act["config_sha256"], "model.safetensors": act["models"][str(fixed_label)]}
        for name, digest in digests.items():
            file = attempt / rel / name
            record = manifest["files"][(rel / name).as_posix()]
            if record["sha256"] != digest or record["bytes"] != file.stat().st_size:
                raise ValueError(f"ACT file differs from the original evidence: {file}")
            _verify_sha(file, digest)
            if fixed_label == final_label and status["fixed_final_checkpoint"][name]["sha256"] != digest:
                raise ValueError("ACT final checkpoint differs from ACT training status")
        artifacts[fixed_label] = (attempt / rel, digests)
    selected_dir, selected_files = artifacts[label]
    return artifacts[final_label][1], selected_dir, selected_files


def _launch_lineage(
    config: dict[str, Any], attempt: Path, status: dict[str, Any], seed: int,
) -> dict[Path, tuple[int, int]]:
    """Bind inherited checkpoints to the actual producer and its restored prefix."""
    registered = (REPO_ROOT / config["resfit"]["training_config"]).resolve()
    config_sha = sha256_file(registered)
    training_config = _json(registered)
    if (
        any(training_config["upstream"][key] != value for key, value in config["upstream"].items())
        or training_config["base_policy"]["model_sha256"] != config["act"]["models"]["49999"]
    ):
        raise ValueError("X1 training config changed the upstream or ACT training base")

    root = Path(status["resume"]["origin_attempt"]) if status.get("resume") else attempt
    locks = {name: row["sha256"] for name, row in status["runtime"]["locks"].items()}
    if locks.keys() != _RUNTIME_LOCKS:
        raise ValueError("X1 training status is missing its two dependency locks")
    allowed = {}
    current = attempt
    last_step = config["resfit"]["registered_budget"]["total"]
    while current not in allowed:
        receipt = _json(current / "resolved_launcher.json")
        receipt_locks = {name: row["sha256"] for name, row in receipt["runtime"]["locks"].items()}
        if (
            receipt["dry_run"]
            or receipt["attempt_dir"] != str(current)
            or Path(receipt["config_path"]).resolve() != registered
            or receipt["config_sha256"] != config_sha
            or receipt["training_seed"] != seed
            or receipt["overlay_mode"] != "resumable"
            or receipt["environment_step_budget"] != config["resfit"]["registered_budget"]
            or receipt["resume_checkpoint_interval_environment_steps"] != 100000
            or receipt["resume_checkpoint_retention"] != 3
            or receipt_locks != locks
            or (current == attempt and any(status.get(key) != value for key, value in receipt.items() if key != "status"))
        ):
            raise ValueError("X1 ResFiT checkpoint lineage differs from its registered launch")
        for name, digest in locks.items():
            _verify_sha(current / "runtime" / name, digest)
        resume = receipt.get("resume")
        previous_step = resume["restored_environment_steps"] if resume else 0
        allowed[current] = (previous_step, last_step)
        if not resume and current == root:
            return allowed
        if not resume or current == root:
            break
        previous = Path(resume["producer_attempt"])
        generation = Path(resume["generation"])
        if (
            Path(resume["origin_attempt"]) != root
            or generation != root / "training" / "resume" / f"step_{previous_step:09d}"
            or previous.parent != attempt.parent
            or previous.resolve() != previous
            or resume["runtime_lock_sha256"] != locks
            or not 0 < previous_step < last_step
            or previous_step % 100000
        ):
            break
        _verify_sha(generation / "resume.json", resume["manifest_sha256"])
        current = previous
        last_step = previous_step
    raise ValueError("X1 ResFiT checkpoint path is outside the recorded resume lineage")


def resolve_cell_policies(
    config: dict[str, Any], cell: dict[str, Any], resfit_attempts: Mapping[int, Path],
) -> dict[str, Any]:
    """Verify completed local policies before evaluating a single registered X1 cell."""
    seed = cell["resfit_seed"]
    step = cell["resfit_step"]
    act_label = cell["act_label"]
    if (
        seed not in config["axes"]["resfit_seed"]
        or step not in config["axes"]["resfit_step"]
        or act_label not in config["axes"]["act_label"]
    ):
        raise ValueError("Square X1 policy selection is outside the registered axes")

    resfit = config["resfit"]
    is_m0 = seed == resfit["m0_seed"]
    if is_m0:
        attempt = (REPO_ROOT / resfit["m0_attempt"]).resolve()
        if seed in resfit_attempts and Path(resfit_attempts[seed]).resolve() != attempt:
            raise ValueError("Square X1 requires the pinned M0 attempt for seed 20261124")
    else:
        if seed not in resfit_attempts:
            raise ValueError(f"A completed X1 ResFiT attempt is required for seed {seed}")
        attempt = Path(resfit_attempts[seed]).resolve()

    status = _json(attempt / "status.json")
    budget = resfit["registered_budget"]
    if (
        status["status"] != "COMPLETED"
        or status["dry_run"]
        or status["child_returncode"] != 0
        or status["attempt_dir"] != str(attempt)
        or status["environment_step_budget"] != budget
        or not status["resfit_clean_after_cleanup"]
        or not status["lerobot_clean_after_cleanup"]
    ):
        raise ValueError("Square X1 requires a completed formal ResFiT attempt")
    if is_m0 and status["checkpoint_manifest_sha256"] != resfit["m0_manifest_sha256"]:
        raise ValueError("M0 ResFiT checkpoint manifest differs from the registered digest")
    manifest_path = attempt / "training" / "checkpoint_manifest.json"
    _verify_sha(manifest_path, status["checkpoint_manifest_sha256"])
    manifest = _json(manifest_path)
    records = manifest["checkpoints"]
    if (
        manifest["selection"] != _FIXED_SELECTION
        or manifest["checkpoint_interval_environment_steps"] != 10000
        or manifest["environment_step_budget"] != budget
        or manifest["action_ledger"]["records"] != budget["total"]
        or [record["global_step"] for record in records] != list(range(10000, 300001, 10000))
    ):
        raise ValueError("ResFiT manifest does not cover the registered fixed-step schedule")
    record = records[step // 10000 - 1]
    if record["selection"] != _FIXED_SELECTION:
        raise ValueError(f"ResFiT checkpoint {step} does not use fixed-step selection")

    allowed = {attempt: (0, budget["total"])} if is_m0 else _launch_lineage(config, attempt, status, seed)
    model = Path(record["path"])
    if model.resolve() != model or not any(
        first_step < step <= last_step and model == producer / "training" / "models" / f"step_{step:09d}.pt"
        for producer, (first_step, last_step) in allowed.items()
    ):
        raise ValueError(f"ResFiT checkpoint {step} is outside its recorded attempt lineage")
    digest = record["sha256"]
    if is_m0 and digest != resfit["m0_models"][str(step)]:
        raise ValueError(f"M0 ResFiT checkpoint {step} differs from the registered digest")
    _verify_sha(model, digest)

    resolved_path = attempt / "training" / "resolved_training_config.json"
    if is_m0:
        _verify_sha(resolved_path, resfit["m0_resolved_training_sha256"])
    resolved = _json(resolved_path)
    final_label = config["act"]["training_base_label"]
    final_dir = (REPO_ROOT / config["act"]["attempt"] / "training" / f"policy_step_{final_label}" / "policy").resolve()
    training_base = {
        "config.json": config["act"]["config_sha256"],
        "model.safetensors": config["act"]["models"][str(final_label)],
    }
    if (
        resolved["config"]["seed"] != seed
        or resolved["config"]["task"] != config["task"]
        or resolved["environment_step_budget"] != budget
        or resolved["normalization"]["action_scale"] != config["residual_component_limit"]
        or Path(resolved["base_policy_dir"]).resolve() != final_dir
        or resolved["base_policy_artifacts"] != training_base
    ):
        raise ValueError("ResFiT seed, task, budget, normalization or ACT training base changed")
    training_base_artifacts, act_dir, act_artifacts = _local_act(config, act_label)
    return {
        "resfit_attempt": attempt,
        "resfit_status": status,
        "resfit_resolved": resolved,
        "resfit_model": model,
        "resfit_model_sha256": digest,
        "training_base_artifacts": training_base_artifacts,
        "act_dir": act_dir,
        "act_artifacts": act_artifacts,
        "upstream": config["upstream"],
    }
