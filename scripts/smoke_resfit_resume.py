"""Interrupt and restore a real short Square ResFiT run in isolated checkouts."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RESFIT = ROOT / "third_party/resfit"
LEROBOT = RESFIT / "deps/lerobot"
RESUME_PATCH = ROOT / "patches/resfit/residual_td3_cf_resrl_resume.patch"
TRAINING_CONFIG = ROOT / "configs/cf_resrl/m0_resfit_training.json"
INTERRUPT_STEP = 96
TOTAL_STEPS = 160
INTERVAL = 32
SMOKE_TIMEOUT = 600
sys.path.insert(0, str(ROOT))
from cf_resrl.dataset import sha256_file


def _json(path):
    return json.loads(path.read_text())


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _git(repository, *argv):
    subprocess.run(["git", "-C", str(repository), *map(str, argv)], check=True, stdout=subprocess.PIPE)


def _isolated_worktree(stack, repository, destination, revision):
    _git(repository, "worktree", "add", "--detach", destination, revision)
    stack.callback(_git, repository, "worktree", "remove", "--force", destination)


def _command(spec, attempt, split, restored=None):
    dataset, act = spec["dataset"], spec["base_policy"]
    overrides = {
        "seed": spec["training"]["seed"],
        "base_policy.local_dir": ROOT / act["local_dir"],
        "base_policy.expected_config_sha256": act["config_sha256"],
        "base_policy.expected_model_sha256": act["model_sha256"],
        "offline_data.name": dataset["name"],
        "offline_data.revision": dataset["revision"],
        "offline_data.root": ROOT / dataset["root"],
        "offline_data.split_manifest": split,
        "offline_data.expected_split_manifest_sha256": sha256_file(split),
        "offline_data.split": dataset["split"],
        "offline_data.video_backend": dataset["video_backend"],
        "offline_data.num_episodes": 2,
        "algo.learning_starts": INTERVAL,
        "algo.total_timesteps": TOTAL_STEPS,
        "algo.critic_warmup_steps": 2,
        "algo.batch_size": 8,
        "algo.buffer_size": 96,
        "eval_num_envs": 1,
        "eval_num_episodes": 1,
        "save_video": "false",
        "log_freq": 8,
        "checkpoint_interval": INTERVAL,
        "resume_checkpoint_interval": INTERVAL,
        "resume_checkpoint_retention": 3,
        "output_dir": attempt / "training",
        "hydra.run.dir": attempt / "hydra",
        "hydra.job.chdir": "false",
        "wandb.mode": "disabled",
    }
    if restored is not None:
        overrides["resume_from"] = restored
    return [
        sys.executable, "-u", "-m", "resfit.rl_finetuning.scripts.train_residual_td3",
        "--config-name=residual_td3_square_config",
        *(f"{key}={value}" for key, value in overrides.items()),
    ]


def _stop_group(process, signal_to_send):
    if process.poll() is None:
        os.killpg(process.pid, signal_to_send)
    try:
        return process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        return process.wait(timeout=20)


def _run(attempt, argv, pythonpath, interrupt=None):
    attempt.mkdir()
    log = attempt / "stdout_stderr.log"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(map(str, pythonpath)),
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0",
        "HF_HUB_OFFLINE": "1",
        "WANDB_MODE": "disabled",
        "CACHE_DIR": str(attempt / "cache"),
    }
    for name in ("HF_OFFLINE_BUFFER_REPO", "HF_ONLINE_BUFFER_REPO"):
        env.pop(name, None)
    _write(attempt / "command.json", {"argv": argv, "pythonpath": env["PYTHONPATH"]})
    with log.open("wb") as sink:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=sink, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            if interrupt is None:
                if process.wait(timeout=SMOKE_TIMEOUT) != 0:
                    raise RuntimeError(f"Resumed trainer exited with {process.returncode}. See {log}")
                return process.returncode
            deadline = time.monotonic() + SMOKE_TIMEOUT
            while time.monotonic() < deadline and process.poll() is None:
                with log.open("rb") as source:
                    source.seek(max(0, log.stat().st_size - 65536))
                    wrote = f"[Resume checkpoint step {INTERRUPT_STEP}]".encode() in source.read()
                if wrote and (interrupt / "resume.json").is_file():
                    code = _stop_group(process, signal.SIGTERM)
                    if code == 0:
                        raise RuntimeError("First trainer exited before intentional interruption")
                    return code
                time.sleep(0.1)
            raise RuntimeError(f"First trainer did not publish step {INTERRUPT_STEP}. See {log}")
        finally:
            if process.poll() is None:
                _stop_group(process, signal.SIGTERM)


def _verify(out, first, resumed, checkpoint, returncodes):
    first_ledger = (first / "training/action_ledger.jsonl").read_bytes().splitlines(keepends=True)
    ledger = (resumed / "training/action_ledger.jsonl").read_bytes().splitlines(keepends=True)
    if ledger[:INTERRUPT_STEP] != first_ledger[:INTERRUPT_STEP]:
        raise RuntimeError("The restored ledger changed the saved prefix")
    for step, line in enumerate(ledger, start=1):
        row = json.loads(line)
        if row["global_step"] != step or row["phase"] != ("warmup" if step <= INTERVAL else "main"):
            raise RuntimeError(f"Duplicated or missing training interaction at step {step}")
    if len(ledger) != TOTAL_STEPS:
        raise RuntimeError("The resumed trainer did not stop at its original total budget")
    if f"Restored ResFiT training at environment step {INTERRUPT_STEP}." not in (
        resumed / "stdout_stderr.log"
    ).read_text(errors="replace"):
        raise RuntimeError("The restarted process did not report live state restoration")
    manifest = _json(resumed / "training/checkpoint_manifest.json")
    if [row["global_step"] for row in manifest["checkpoints"]] != list(
        range(INTERVAL, TOTAL_STEPS + 1, INTERVAL)
    ):
        raise RuntimeError("Fixed evaluations were lost or repeated in the new attempt")
    for row in manifest["checkpoints"]:
        for artifact in (row, row["evaluation"]):
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError("Restored evaluation artifact digest mismatch")
    retained = sorted(int(item.name.split("_")[1]) for item in checkpoint.parent.glob("step_*"))
    if retained != [96, 128, 160]:
        raise RuntimeError(f"The rotating checkpoint store did not keep the three newest: {retained}")
    _write(out / "verification.json", {
        "status": "PASS",
        "scope": "isolated real Square stop and restart test",
        "returncodes": returncodes,
        "intentionally_interrupted_after_saved_step": INTERRUPT_STEP,
        "resumed_steps": TOTAL_STEPS - INTERRUPT_STEP,
        "verified_prefix_lines": INTERRUPT_STEP,
        "training_ledger_lines": len(ledger),
        "fixed_evaluation_steps": [row["global_step"] for row in manifest["checkpoints"]],
        "retained_full_generations": retained,
        "overlay_sha256": sha256_file(RESUME_PATCH),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    _write(out / "status.json", {
        "status": "RUNNING", "pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"Real resume smoke evidence: {out}", flush=True)
    try:
        spec = _json(TRAINING_CONFIG)
        pin = spec["upstream"]
        full_split = _json(ROOT / spec["dataset"]["split_manifest"])
        full_split["splits"]["train"] = full_split["splits"]["train"][:2]
        split = out / "smoke_train_split.json"
        _write(split, full_split)
        first, resumed = out / "interrupted", out / "resumed"
        checkpoint = first / "training/resume" / f"step_{INTERRUPT_STEP:09d}"
        with ExitStack() as stack:
            scratch_resfit, scratch_lerobot = out / "resfit", out / "lerobot"
            _isolated_worktree(stack, RESFIT, scratch_resfit, pin["resfit_commit"])
            _isolated_worktree(stack, LEROBOT, scratch_lerobot, pin["lerobot_commit"])
            _git(scratch_resfit, "apply", ROOT / pin["patches"][0]["path"])
            _git(scratch_resfit, "apply", RESUME_PATCH)
            _git(scratch_lerobot, "apply", ROOT / pin["patches"][2]["path"])
            pythonpath = (ROOT, scratch_resfit, scratch_lerobot, ROOT / "third_party/robosuite")
            before = _run(first, _command(spec, first, split), pythonpath, interrupt=checkpoint)
            after = _run(resumed, _command(spec, resumed, split, restored=checkpoint), pythonpath)
            _verify(out, first, resumed, checkpoint, {"interrupted": before, "resumed": after})
    except Exception as error:
        _write(out / "status.json", {"status": "FAILED", "error": str(error), "type": type(error).__name__})
        raise
    _write(out / "status.json", {"status": "COMPLETED", "verification_sha256": sha256_file(out / "verification.json")})


if __name__ == "__main__":
    main()
