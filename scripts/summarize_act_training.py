#!/usr/bin/env python3
"""Index and verify evidence from a completed formal ACT training attempt."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cf_resrl.dataset import sha256_file  # noqa: E402


def summarize(attempt: Path) -> dict:
    status = json.loads((attempt / "status.json").read_text())
    if status["status"] != "COMPLETED" or status["returncode"] != 0 or status["dry_run"]:
        raise ValueError("Expected a completed formal ACT run with exit code 0")
    if not status["resfit_clean_after_cleanup"] or not status["lerobot_clean_after_cleanup"]:
        raise ValueError("Run did not verify restoration of ResFiT and LeRobot")

    files = {}

    def record(path: Path, kind: str = "text_evidence", expected_sha: str | None = None) -> dict:
        entry = {"kind": kind, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if expected_sha is not None and entry["sha256"] != expected_sha:
            raise ValueError(f"Checksum mismatch: {path}")
        files[path.relative_to(attempt).as_posix()] = entry
        return entry

    original_attempt = Path(status["attempt_dir"])
    log = attempt / "stdout_stderr.log"
    record(log, expected_sha=status["stdout_stderr_sha256"])
    for name in (
        "status.json", "resolved_launcher.json", "runtime/verification.json",
        "training/resolved_training_config.json",
    ):
        record(attempt / name)
    for lock in status["runtime_verification"]["locks"].values():
        record(attempt / Path(lock["path"]).relative_to(original_attempt), expected_sha=lock["expected_sha256"])

    config = ROOT / "configs/cf_resrl/m0_act_training.json"
    if sha256_file(config) != status["config_sha256"]:
        raise ValueError("ACT registered config differs from the completed attempt")
    spec = json.loads(config.read_text())
    dataset = spec["dataset"]
    split_path = ROOT / dataset["split_manifest"]
    split = json.loads(split_path.read_text())
    if (
        sha256_file(split_path) != dataset["split_manifest_sha256"]
        or split["splits"]["train"] != status["episode_indices"]
    ):
        raise ValueError("Episode split differs from the verified attempt")

    training = attempt / "training"
    final = status["checkpoint_selection"]
    save_interval = spec["training"]["save_frequency"]
    fixed_labels = list(range(save_interval, final["checkpoint_step_label"], save_interval))
    fixed_labels.append(final["checkpoint_step_label"])
    checkpoints = {int(path.name.removeprefix("policy_step_")): path for path in training.glob("policy_step_*")}
    if sorted(checkpoints) != fixed_labels:
        raise ValueError("Fixed checkpoint directories differ from the registered schedule")
    directories = list(checkpoints.values()) + sorted(training.glob("best_step_*"))
    for directory in directories + [training / "best", training / "latest"]:
        record(directory / "policy/config.json")
        record(directory / "policy/model.safetensors", "checkpoint_payload")
    for directory in (training / "best", training / "latest"):
        record(directory / "trainer_state.pt", "checkpoint_payload")
    final_policy = f"training/{final['relative_policy_path']}"
    for name, saved in status["fixed_final_checkpoint"].items():
        entry = files[f"{final_policy}/{name}"]
        if (entry["bytes"], entry["sha256"]) != (saved["bytes"], saved["sha256"]):
            raise ValueError(f"Fixed final checkpoint changed: {name}")

    content = log.read_text()
    counts = re.findall(r"Evaluation completed: (\d+) episodes, (\d+) successes", content)
    performance = re.findall(r"Performance: (\d+) total steps in [\d.]+s", content)
    metrics = re.findall(
        r"\[step\s+(\d+)\] eval success-rate: ([\d.]+)% \| rollout: ([\d.]+) s \| ([\d.]+) fps",
        content,
    )
    videos = re.findall(r"Video saved with annotated frames: (.*?\.mp4)", content)
    evaluations = []
    for (episodes, successes), steps, (label, rate, seconds, fps), video_path in zip(
        counts, performance, metrics, videos, strict=True
    ):
        step = int(label)
        video = Path(video_path).relative_to(original_attempt)
        if not video.name.startswith(f"eval_step_{step}_") or int(episodes) != spec["evaluation"]["num_episodes"]:
            raise ValueError(f"Evaluation at step {step} has unexpected evidence")
        if abs(int(successes) / int(episodes) * 100 - float(rate)) > 0.051:
            raise ValueError(f"Evaluation at step {step} has inconsistent success counts")
        record(attempt / video, "rollout_video")
        evaluations.append({
            "optimization_steps_completed": step, "episodes": int(episodes), "successes": int(successes),
            "vectorized_environment_steps": int(steps), "rollout_seconds": float(seconds),
            "environment_steps_per_second": float(fps), "video": video.as_posix(),
        })
    interval = spec["evaluation"]["rollout_frequency"]
    expected_rounds = list(range(interval, spec["training"]["steps"] + 1, interval))
    if [row["optimization_steps_completed"] for row in evaluations] != expected_rounds:
        raise ValueError("Evaluation rounds differ from the registered schedule")

    duration = datetime.fromisoformat(status["completed_at_utc"]) - datetime.fromisoformat(status["started_at_utc"])
    return {
        "status": status["status"], "attempt_id": attempt.name, "source_commit": status["git_commit"],
        "started_at_utc": status["started_at_utc"], "completed_at_utc": status["completed_at_utc"],
        "wall_clock_seconds": duration.total_seconds(),
        "dataset": dataset, "training_seed": spec["training"]["seed"], "upstream": spec["upstream"],
        "checkpoint_selection": final, "fixed_checkpoint_labels": fixed_labels,
        "final_policy_directory": final_policy, "evaluation_rounds": evaluations,
        "evidence_notes": [
            "The formal launcher requires clean parent, ResFiT and LeRobot checkouts before starting; its terminal status confirms overlay cleanup.",
            "The upstream trainer saved fixed intermediate policy_step_n files after update n (zero-based), while the periodic evaluation at n followed n completed updates. Those rounds are not independent exact-checkpoint evaluations.",
            "The evaluation at 50000 follows the final fixed policy_step_49999 save after the same 50000th update. Best-policy checkpoints are not the selected base.",
            "The trainer recorded aggregate evaluations and annotated video. No structured per-episode records or independent fixed-checkpoint re-evaluations were emitted by this attempt.",
            "Payload entries identify retained files relative to this attempt. The checksum inventory does not assert remote payload availability.",
        ],
        "files": files,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    attempt = parser.parse_args().attempt
    manifest = summarize(attempt)
    (attempt / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
