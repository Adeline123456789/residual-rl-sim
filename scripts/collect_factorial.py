"""Run the registered eight-group Square branch semantics gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
RESFIT_ROOT = REPO_ROOT / "third_party/resfit"
LEROBOT_ROOT = RESFIT_ROOT / "deps/lerobot"
ROBOSUITE_ROOT = REPO_ROOT / "third_party/robosuite"
DEFAULT_CONFIG = REPO_ROOT / "configs/cf_resrl/m0_square_branch_gate.json"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _source(config):
    attempt = REPO_ROOT / config["resfit_attempt"]
    status = _json(attempt / "status.json")
    if status["status"] != "COMPLETED" or status["environment_step_budget"]["total"] != 300000:
        raise ValueError("The registered formal ResFiT 300k run must be completed")
    training = attempt / "training"
    manifest_path = training / "checkpoint_manifest.json"
    if _sha256(manifest_path) != status["checkpoint_manifest_sha256"]:
        raise ValueError("ResFiT checkpoint manifest checksum mismatch")
    manifest = _json(manifest_path)
    record = manifest["checkpoints"][-1]
    if record["global_step"] != config["resfit_environment_step"] or record["global_step"] != 300000:
        raise ValueError("The Square branch gate requires the fixed 300k checkpoint")
    model = training / "models" / Path(record["path"]).name
    if _sha256(model) != record["sha256"] or record["sha256"] != config["resfit_checkpoint_sha256"]:
        raise ValueError("ResFiT weights do not match the registered SHA-256")
    resolved = _json(training / "resolved_training_config.json")
    act = Path(resolved["base_policy_dir"])
    for name, checksum in resolved["base_policy_artifacts"].items():
        if _sha256(act / name) != checksum:
            raise ValueError(f"ACT artifact checksum mismatch: {name}")
    pins = _json(Path(status["config_path"]))["upstream"]
    for name, root in (("resfit", RESFIT_ROOT), ("lerobot", LEROBOT_ROOT), ("robosuite", ROBOSUITE_ROOT)):
        if _git(root, "rev-parse", "HEAD") != pins[f"{name}_commit"] or _git(root, "status", "--porcelain"):
            raise ValueError(f"{name} checkout does not match the clean registered revision")
    return attempt, status, resolved, act, model, pins


def _run(config, out):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path[:0] = [str(path) for path in (REPO_ROOT, RESFIT_ROOT, LEROBOT_ROOT, ROBOSUITE_ROOT)]
    import mujoco
    from omegaconf import OmegaConf
    import torch
    from cf_resrl.branch_runner import BRANCHES, PROVIDER_ID, SquareBranchRunner
    from resfit.dexmg.environments.dexmg import RobosuiteGymWrapper
    from resfit.lerobot.policies.act.configuration_act import ACTConfig
    from resfit.lerobot.policies.act.modeling_act import ACTPolicy
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent

    if not torch.cuda.is_available() or mujoco.__version__ != "3.3.2":
        raise RuntimeError("Square branch gate requires the pinned CUDA and MuJoCo runtime")
    if config["task"] != "Square" or tuple(config["branch_labels"]) != BRANCHES:
        raise ValueError("Unexpected registered branch treatments")
    expected_pairs = list(zip(config["source_seeds"], config["source_ticks"], strict=True))
    if len(expected_pairs) != 8 or len(set(config["source_seeds"])) != 8 or config["refresh_delay_steps"] != 2:
        raise ValueError("Square gate requires eight independent seeds and two-tick refresh latency")
    attempt, status, resolved, act, model, pins = _source(config)
    device = torch.device("cuda:0")
    act_cfg = ACTConfig.from_pretrained(act, local_files_only=True)
    act_cfg.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(act, config=act_cfg, local_files_only=True, strict=True).to(device).eval()
    if policy.config.temporal_ensemble_coeff is not None or policy.config.n_action_steps != 20:
        raise ValueError("The gate uses the original ACT 20-step native chunk")
    agent_config = OmegaConf.create(resolved["config"]["agent"])
    actor_images = resolved["config"]["rl_camera"]
    agent = QAgent((3, 84, 84), (9,), 7, actor_images, agent_config, residual_actor=True)
    checkpoint = torch.load(model, map_location="cpu", weights_only=False)
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=True)
    agent.eval()
    normalization = resolved["normalization"]
    for key, value in normalization.items():
        stored = checkpoint["normalization"][key]
        if (stored.tolist() if isinstance(stored, torch.Tensor) else stored) != value:
            raise ValueError(f"Normalization differs between the trainer and the checkpoint: {key}")
    if checkpoint["base_policy_artifacts"] != resolved["base_policy_artifacts"]:
        raise ValueError("ResFiT was trained against a different ACT base")
    del checkpoint
    identities = {
        "policy_identity": resolved["base_policy_artifacts"]["model.safetensors"],
        "residual_policy_identity": config["resfit_checkpoint_sha256"],
        "provider_identity": PROVIDER_ID,
        "normalization_identity": hashlib.sha256(json.dumps(normalization, sort_keys=True).encode()).hexdigest(),
    }
    _write(out / "run_manifest.json", {
        "command": [sys.executable, *sys.argv],
        "git_commit": _git(REPO_ROOT, "rev-parse", "HEAD"),
        "git_status_at_start": _git(REPO_ROOT, "status", "--porcelain", "--untracked-files=no").splitlines(),
        "upstream": pins,
        "source_resfit_attempt": str(attempt.relative_to(REPO_ROOT)),
        "source_resfit_dependency_locks": status["runtime"]["locks"],
        "dataset": resolved["dataset"],
        "act_artifacts": resolved["base_policy_artifacts"],
        "identities": identities,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "mujoco": mujoco.__version__,
            "gpu": torch.cuda.get_device_name(device),
        },
        "protocol": config,
        "fallback_when_pending_and_old_chunk_exhausted": "zero residual-normalized base action",
        "after_intervention": "original frozen residual actor per step and nominal ACT inference at chunk exhaustion",
        "time": "logical control ticks measure refresh latency, wall time includes inference",
    })

    wrapper = RobosuiteGymWrapper("Square", camera_size=84, render_gpu_device_id=0)
    if wrapper.horizon != config["simulation_horizon_steps"]:
        raise RuntimeError("Environment horizon disagrees with the registered gate")
    runner = SquareBranchRunner(
        wrapper, policy, agent, normalization, identities, device,
        delay=config["refresh_delay_steps"], gamma=resolved["config"]["algo"]["gamma"],
    )
    groups = []
    try:
        for index, (seed, tick) in enumerate(expected_pairs, start=1):
            group = runner.group(seed, tick, verify_replay=index == 1)
            group["source_episode_id"] = f"square_gate_{seed}"
            path = out / f"group_{index:02d}.json"
            _write(path, group)
            groups.append({
                "source_episode_id": group["source_episode_id"],
                "seed": seed,
                "source_tick": tick,
                "path": path.name,
                "sha256": _sha256(path),
                "prefix_control_steps": group["source_prefix_steps"],
                "prefix_act_calls": group["source_prefix_act_calls"],
                "replay_verification": group["replay_verification"],
                "branches": {
                    label: {key: value for key, value in row.items() if key != "steps"}
                    for label, row in group["branches"].items()
                },
                "factorial_values": group["factorial_values"],
            })
            print(f"Square gate {index}/8: source {seed} at tick {tick}, saved four real branches", flush=True)
    finally:
        wrapper.close()
    all_branches = [row for group in groups for row in group["branches"].values()]
    _write(out / "summary.json", {
        "status": "PASS",
        "groups": groups,
        "branch_rollouts": len(all_branches),
        "branch_environment_steps": sum(branch["environment_steps"] for branch in all_branches),
        "branch_act_calls": sum(branch["act_calls"] for branch in all_branches),
        "source_prefix_environment_steps": sum(group["prefix_control_steps"] for group in groups),
        "source_prefix_act_calls": sum(group["prefix_act_calls"] for group in groups),
        "separate_replay_verification": groups[0]["replay_verification"],
        "checks": ["exact paired first actions and waiting trajectories", "post-action request",
                   "absolute-tick chunk activation", "physical action bounds", "deterministic replay"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = _json(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.out or REPO_ROOT / config["output_root"] / "attempts" / stamp
    out.mkdir(parents=True, exist_ok=False)
    _write(out / "status.json", {"status": "RUNNING", "pid": os.getpid()})
    print(f"Square branch gate evidence: {out}", flush=True)
    try:
        _run(config, out)
    except Exception as error:
        _write(out / "status.json", {"status": "FAILED", "error": str(error), "type": type(error).__name__})
        raise
    _write(out / "status.json", {"status": "COMPLETED", "summary_sha256": _sha256(out / "summary.json")})


if __name__ == "__main__":
    main()
