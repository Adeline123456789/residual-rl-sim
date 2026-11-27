"""Standalone PS-2 trainer & smoke runner for PS-ResRL (8-D) and B1-ctx (7-D).

Keeps third_party/resfit untouched and only temporarily applies the registered
lerobot noncontiguous episode-subset patch during dataset loading, reversing it
in a finally block so all submodules remain clean.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RESFIT_ROOT = REPO_ROOT / "third_party" / "resfit"
LEROBOT_ROOT = RESFIT_ROOT / "deps" / "lerobot"
ROBOSUITE_ROOT = REPO_ROOT / "third_party" / "robosuite"
LEROBOT_PATCH = REPO_ROOT / "patches" / "lerobot" / "noncontiguous_episode_subset.patch"

for p in (REPO_ROOT, RESFIT_ROOT, LEROBOT_ROOT, ROBOSUITE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _apply_lerobot_patch() -> bool:
    applied = (
        subprocess.run(
            ["git", "-C", str(LEROBOT_ROOT), "apply", "--reverse", "--check", str(LEROBOT_PATCH)],
            capture_output=True,
        ).returncode
        == 0
    )
    if applied:
        return False
    subprocess.run(["git", "-C", str(LEROBOT_ROOT), "apply", str(LEROBOT_PATCH)], check=True)
    return True


def _reverse_lerobot_patch() -> None:
    subprocess.run(
        ["git", "-C", str(LEROBOT_ROOT), "apply", "--reverse", str(LEROBOT_PATCH)],
        check=True,
    )


def _populate_phase_offline_buffer(
    *,
    dataset,
    rb,
    base_policy,
    action_scaler,
    state_standardizer,
    image_keys: list[str],
    mode: str,
    max_chunk_wall_steps: int,
    device: torch.device,
    max_episodes: int | None = None,
) -> int:
    from tensordict import TensorDict
    from torch.utils.data import DataLoader
    from resfit.rl_finetuning.utils.dtype import to_uint8
    from ps_resrl.phase_vec_env import (
        build_phase_context_tensor,
        interpolate_batched_chunk,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    chunk_length = int(base_policy.config.n_action_steps)
    episode_cache: dict[int, dict[str, Any]] = {}
    transitions = 0
    prev_ep = None
    step_in_ep = 0
    current_chunk: torch.Tensor | None = None
    seen_episodes: set[int] = set()

    for sample in loader:
        ep_idx = int(sample["episode_index"].item())
        if max_episodes is not None and ep_idx not in seen_episodes and len(seen_episodes) >= max_episodes:
            break
        seen_episodes.add(ep_idx)
        if ep_idx != prev_ep:
            step_in_ep = 0
            current_chunk = None
            prev_ep = ep_idx

        if current_chunk is None or step_in_ep % chunk_length == 0:
            raw_obs = {
                k: v.to(device)
                for k, v in sample.items()
                if k.startswith("observation")
            }
            with torch.no_grad():
                base_policy.eval()
                batch = base_policy.normalize_inputs(dict(raw_obs))
                if base_policy.config.image_features:
                    batch = dict(batch)
                    batch["observation.images"] = [
                        batch[k] for k in base_policy.config.image_features
                    ]
                seq = base_policy.model(batch)[0][:, :chunk_length]
                seq = base_policy.unnormalize_outputs({"action": seq})["action"]
                current_chunk = action_scaler.scale(seq).cpu()  # (1, H, 7)

        chunk_pos = step_in_ep % chunk_length
        phi_t = torch.tensor([float(chunk_pos)], dtype=torch.float32)
        wall_t = torch.tensor([chunk_pos], dtype=torch.long)
        base_7d = interpolate_batched_chunk(current_chunk, phi_t).squeeze(0)
        phase_ctx = build_phase_context_tensor(
            current_chunk, phi_t, wall_t, max_chunk_wall_steps
        ).squeeze(0)

        gt_7d = action_scaler.scale(sample["action"].float().squeeze(0)).cpu()
        std_state = state_standardizer.standardize(
            sample["observation.state"].float().squeeze(0).to(device)
        ).cpu()
        full_state = torch.cat([std_state, phase_ctx], dim=-1)

        if mode == "ps_resrl":
            base_obs_act = torch.cat([base_7d, torch.zeros(1, dtype=torch.float32)], dim=-1)
            target_act = torch.cat([gt_7d, torch.zeros(1, dtype=torch.float32)], dim=-1)
        else:
            base_obs_act = base_7d
            target_act = gt_7d

        curr_obs = {
            "observation.state": full_state,
            "observation.base_action": base_obs_act,
        }
        for k in image_keys:
            curr_obs[k] = sample[k].squeeze(0)
        to_uint8(curr_obs, image_keys)

        done_flag = bool(sample["next.done"].item())
        if ep_idx in episode_cache:
            prev_entry = episode_cache[ep_idx]
            td = TensorDict(
                {
                    "obs": TensorDict(prev_entry["obs"], batch_size=[]),
                    "action": prev_entry["action"],
                    "next": TensorDict(
                        {
                            "obs": TensorDict(curr_obs, batch_size=[]),
                            "done": torch.tensor(done_flag, dtype=torch.bool),
                            "reward": torch.tensor(float(done_flag), dtype=torch.float32),
                        },
                        batch_size=[],
                    ),
                    "_priority": torch.tensor(10.0, dtype=torch.float32),
                },
                batch_size=[],
            ).unsqueeze(0)
            rb.add(td)
            transitions += 1

        episode_cache[ep_idx] = {"obs": curr_obs, "action": target_act}
        step_in_ep += 1

    return transitions


def _add_online_transitions(
    *,
    obs: dict[str, torch.Tensor],
    next_obs: dict[str, torch.Tensor],
    actions: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    info: dict[str, Any],
    device: torch.device,
    image_keys: list[str],
    lowdim_keys: list[str],
    num_envs: int,
    online_rb,
) -> None:
    from tensordict import TensorDict
    from resfit.rl_finetuning.utils.dtype import to_uint8

    obs_keys_set = set(image_keys) | set(lowdim_keys)
    for i in range(num_envs):
        if done[i] and "final_obs" in info and info["final_obs"][i] is not None:
            next_obs_i = {
                k: torch.as_tensor(v, device=device)
                for k, v in info["final_obs"][i].items()
                if k in obs_keys_set
            }
        else:
            next_obs_i = {k: v[i] for k, v in next_obs.items() if k in obs_keys_set}
        curr_obs_i = {k: v[i] for k, v in obs.items() if k in obs_keys_set}
        to_uint8(curr_obs_i, image_keys)
        to_uint8(next_obs_i, image_keys)

        td = TensorDict(
            {
                "obs": TensorDict(curr_obs_i, batch_size=[]),
                "next": TensorDict(
                    {
                        "obs": TensorDict(next_obs_i, batch_size=[]),
                        "done": done[i],
                        "reward": reward[i],
                    },
                    batch_size=[],
                ),
                "action": actions[i],
                "_priority": torch.tensor(10.0, dtype=torch.float32),
            },
            batch_size=[],
        ).unsqueeze(0)
        online_rb.add(td)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["WANDB_MODE"] = "disabled"

    config = json.loads(args.config.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    kind = "smoke" if args.smoke else "formal"
    out = args.out or (
        REPO_ROOT / config["training"]["output_root"] / config["experiment_name"] / kind / stamp
    )
    out.mkdir(parents=True, exist_ok=False)
    models_dir = out / "models"
    outputs_dir = out / "outputs"
    models_dir.mkdir()
    outputs_dir.mkdir()

    upstream = config["upstream"]
    for name, root in (("resfit", RESFIT_ROOT), ("lerobot", LEROBOT_ROOT), ("robosuite", ROBOSUITE_ROOT)):
        if _git(root, "rev-parse", "HEAD") != upstream[f"{name}_commit"] or _git(root, "status", "--porcelain"):
            raise RuntimeError(f"Submodule {name} must start clean at pinned commit")

    patch_applied = False
    env = None
    eval_env = None
    try:
        patch_applied = _apply_lerobot_patch()

        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
        from resfit.dexmg.environments.dexmg import create_vectorized_env
        from resfit.lerobot.policies.act.configuration_act import ACTConfig
        from resfit.lerobot.policies.act.modeling_act import ACTPolicy
        from resfit.rl_finetuning.config.residual_td3 import ResidualTD3SquareConfig
        from resfit.rl_finetuning.off_policy.common_utils import utils
        from resfit.rl_finetuning.utils.checkpoint import save_checkpoint
        from resfit.rl_finetuning.utils.evaluate_dexmg import run_dexmg_evaluation
        from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
        from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
        from ps_resrl.phase_td3_agent import build_phase_q_agent
        from ps_resrl.phase_vec_env import PhaseSymbioticVecEnvWrapper

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed = int(config["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        base_dir = REPO_ROOT / config["base_policy"]["local_dir"]
        if _sha256(base_dir / "config.json") != config["base_policy"]["config_sha256"]:
            raise ValueError("ACT config.json SHA256 mismatch")
        if _sha256(base_dir / "model.safetensors") != config["base_policy"]["model_sha256"]:
            raise ValueError("ACT model.safetensors SHA256 mismatch")

        act_cfg = ACTConfig.from_pretrained(base_dir, local_files_only=True)
        act_cfg.pretrained_backbone_weights = None
        base_policy = ACTPolicy.from_pretrained(
            base_dir, config=act_cfg, local_files_only=True, strict=True
        ).to(device).eval()
        eval_base_policy = ACTPolicy.from_pretrained(
            base_dir, config=act_cfg, local_files_only=True, strict=True
        ).to(device).eval()

        split_path = REPO_ROOT / config["dataset"]["split_manifest"]
        if _sha256(split_path) != config["dataset"]["split_manifest_sha256"]:
            raise ValueError("Split manifest SHA256 mismatch")
        split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
        ep_indices = split_manifest["splits"][config["dataset"]["split"]]
        dataset = LeRobotDataset(
            config["dataset"]["name"],
            root=REPO_ROOT / config["dataset"]["root"],
            episodes=ep_indices,
            revision=config["dataset"]["revision"],
            video_backend=config["dataset"]["video_backend"],
        )
        stats = dataset.stats if dataset.episodes is not None else dataset.meta.stats

        rl_cfg = ResidualTD3SquareConfig()
        action_scaler = ActionScaler.from_dataset_stats(
            action_stats=stats["action"],
            action_scale=rl_cfg.agent.actor.action_scale,
            min_range_per_dim=rl_cfg.offline_data.min_action_range,
            device=device,
        )
        state_standardizer = StateStandardizer.from_dataset_stats(
            state_stats=stats["observation.state"],
            min_std=rl_cfg.offline_data.min_state_std,
            device=device,
        )

        mode = config["mode"]
        phase_cfg = config["phase"]
        train_cfg = config["training"]
        warmup_steps = int(train_cfg["smoke_warmup_steps"] if args.smoke else train_cfg["warmup_environment_steps"])
        total_steps = int(train_cfg["smoke_total_steps"] if args.smoke else train_cfg["total_environment_steps"])
        eval_episodes = int(train_cfg["smoke_eval_episodes"] if args.smoke else train_cfg["eval_episodes"])
        eval_num_envs = int(min(2 if args.smoke else train_cfg["eval_num_envs"], eval_episodes))
        ckpt_interval = int(total_steps if args.smoke else train_cfg["checkpoint_interval"])
        eval_interval = int(total_steps if args.smoke else train_cfg["eval_interval"])

        vec_env = create_vectorized_env(
            env_name=config["task"],
            num_envs=1,
            device=str(device),
            video_key=rl_cfg.video_key,
            debug=False,
        )
        env = PhaseSymbioticVecEnvWrapper(
            vec_env=vec_env,
            base_policy=base_policy,
            action_scaler=action_scaler,
            state_standardizer=state_standardizer,
            mode=mode,
            nu_min=float(phase_cfg["nu_min"]),
            nu_max=float(phase_cfg["nu_max"]),
            max_chunk_wall_steps=int(phase_cfg["max_chunk_wall_steps"]),
        )
        eval_vec_env = create_vectorized_env(
            env_name=config["task"],
            num_envs=eval_num_envs,
            device=str(device),
            video_key=rl_cfg.video_key,
            debug=False,
        )
        eval_env = PhaseSymbioticVecEnvWrapper(
            vec_env=eval_vec_env,
            base_policy=eval_base_policy,
            action_scaler=action_scaler,
            state_standardizer=state_standardizer,
            mode=mode,
            nu_min=float(phase_cfg["nu_min"]),
            nu_max=float(phase_cfg["nu_max"]),
            max_chunk_wall_steps=int(phase_cfg["max_chunk_wall_steps"]),
        )

        image_keys = list(rl_cfg.rl_camera)
        lowdim_dim = env.observation_space["observation.state"].shape[1]
        img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
        action_dim = env.action_space.shape[1]
        lowdim_keys = ["observation.state", "observation.base_action"]

        agent = build_phase_q_agent(
            obs_shape=(img_c, img_h, img_w),
            prop_shape=(lowdim_dim,),
            action_dim=action_dim,
            rl_cameras=image_keys,
            cfg=rl_cfg.agent,
            mode=mode,
            phase_action_scale=float(phase_cfg["phase_action_scale"]),
        )

        # Zero-init degeneracy verification before any training
        obs_reset, _ = env.reset()
        with torch.no_grad(), utils.eval_mode(agent):
            init_act = agent.act(obs_reset, eval_mode=True, stddev=0.0, cpu=True)
        init_max_abs = float(init_act.abs().max().item())
        if init_max_abs != 0.0:
            raise RuntimeError(f"Actor step-0 deterministic output must be 0.0, got {init_max_abs}")

        resolved_payload = {
            **config,
            "smoke": args.smoke,
            "git_commit": _git(REPO_ROOT, "rev-parse", "HEAD"),
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "lowdim_dim": lowdim_dim,
            "action_dim": action_dim,
            "zero_init_max_abs": init_max_abs,
        }
        _write_json(out / "resolved_config.json", resolved_payload)
        _write_json(out / "status.json", {"status": "RUNNING", "smoke": args.smoke})

        online_batch_size = int(rl_cfg.algo.batch_size * (1 - rl_cfg.algo.offline_fraction))
        offline_batch_size = int(rl_cfg.algo.batch_size * rl_cfg.algo.offline_fraction)
        online_rb = TensorDictPrioritizedReplayBuffer(
            storage=LazyTensorStorage(max_size=rl_cfg.algo.buffer_size, device="cpu"),
            alpha=0.0,
            beta=0.0,
            eps=1e-6,
            priority_key="_priority",
            transform=MultiStepTransform(n_steps=rl_cfg.algo.n_step, gamma=rl_cfg.algo.gamma),
            pin_memory=True,
            prefetch=rl_cfg.algo.prefetch_batches,
            batch_size=online_batch_size,
        )
        offline_rb = TensorDictPrioritizedReplayBuffer(
            storage=LazyTensorStorage(max_size=100_000, device="cpu"),
            alpha=0.0,
            beta=0.0,
            eps=1e-6,
            priority_key="_priority",
            transform=MultiStepTransform(n_steps=rl_cfg.algo.n_step, gamma=rl_cfg.algo.gamma),
            pin_memory=True,
            prefetch=rl_cfg.algo.prefetch_batches,
            batch_size=offline_batch_size,
        )

        offline_count = _populate_phase_offline_buffer(
            dataset=dataset,
            rb=offline_rb,
            base_policy=base_policy,
            action_scaler=action_scaler,
            state_standardizer=state_standardizer,
            image_keys=image_keys,
            mode=mode,
            max_chunk_wall_steps=int(phase_cfg["max_chunk_wall_steps"]),
            device=device,
            max_episodes=4 if args.smoke else None,
        )
        print(f"Populated offline phase buffer: {offline_count} transitions", flush=True)

        # Warm-up collection
        obs, _ = env.reset()
        phase_noise = float(phase_cfg["phase_warmup_noise_scale"])
        while len(online_rb) < warmup_steps:
            res_noise = (torch.rand((1, 7), device=device) * 2.0 - 1.0) * rl_cfg.algo.random_action_noise_scale
            if mode == "ps_resrl":
                p_noise = (torch.rand((1, 1), device=device) * 2.0 - 1.0) * phase_noise
                rand_act = torch.cat([res_noise, p_noise], dim=-1)
            else:
                rand_act = res_noise
            next_obs, reward, terminated, truncated, info = env.step(rand_act)
            done = terminated | truncated
            _add_online_transitions(
                obs=obs,
                next_obs=next_obs,
                actions=info["scaled_action"],
                reward=reward,
                done=done,
                info=info,
                device=device,
                image_keys=image_keys,
                lowdim_keys=lowdim_keys,
                num_envs=1,
                online_rb=online_rb,
            )
            obs = next_obs
        print(f"Warm-up completed: online_rb={len(online_rb)}", flush=True)

        # Critic warmup
        critic_warmup = 20 if args.smoke else rl_cfg.algo.critic_warmup_steps
        for _ in range(critic_warmup):
            b_on = online_rb.sample(online_batch_size).to(device, non_blocking=True)
            b_off = offline_rb.sample(offline_batch_size).to(device, non_blocking=True)
            batch = torch.cat([b_on, b_off], dim=0)
            m = agent.update(batch, stddev=0.0, update_actor=False, bc_batch=None, ref_agent=agent)

        obs, _ = env.reset()
        actor_updates = 0
        actor_update_cadence = rl_cfg.algo.num_updates_per_iteration // rl_cfg.algo.actor_updates_per_iteration
        checkpoints = []
        last_metrics: dict[str, Any] = {}
        eval_metrics: dict[str, Any] = {}
        nu_samples: list[float] = []
        stall_triggers = 0

        for global_step in range(1, total_steps + 1):
            with torch.no_grad(), utils.eval_mode(agent):
                stddev = utils.schedule(rl_cfg.algo.stddev_schedule, global_step)
                act = agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)
            next_obs, reward, terminated, truncated, info = env.step(act)
            done = terminated | truncated
            nu_samples.append(float(info["phase_velocity"][0].item()))
            stall_triggers += int(info["stall_cap_triggered"][0].item())
            _add_online_transitions(
                obs=obs,
                next_obs=next_obs,
                actions=info["scaled_action"],
                reward=reward,
                done=done,
                info=info,
                device=device,
                image_keys=image_keys,
                lowdim_keys=lowdim_keys,
                num_envs=1,
                online_rb=online_rb,
            )
            obs = next_obs

            for u_idx in range(rl_cfg.algo.num_updates_per_iteration):
                b_on = online_rb.sample(online_batch_size).to(device, non_blocking=True)
                b_off = offline_rb.sample(offline_batch_size).to(device, non_blocking=True)
                batch = torch.cat([b_on, b_off], dim=0)
                update_actor = (u_idx + 1) % actor_update_cadence == 0
                if update_actor:
                    if rl_cfg.algo.actor_lr_warmup_steps > 0:
                        prog = min(1.0, actor_updates / rl_cfg.algo.actor_lr_warmup_steps)
                        for pg in agent.actor_opt.param_groups:
                            pg["lr"] = rl_cfg.agent.actor_lr * prog
                    actor_updates += 1
                last_metrics = agent.update(batch, stddev, update_actor, bc_batch=None, ref_agent=agent)

            if global_step % eval_interval == 0:
                eval_calls_before = eval_env.act_call_count
                eval_metrics = run_dexmg_evaluation(
                    env=eval_env,
                    agent=agent,
                    num_episodes=eval_episodes,
                    device=device,
                    global_step=global_step,
                    save_video=False,
                    save_q_plots=False,
                    run_name=config["experiment_name"],
                    output_dir=outputs_dir,
                )
                eval_calls = eval_env.act_call_count - eval_calls_before
                eval_metrics["eval/act_calls_total"] = eval_calls
                agent.train(True)

            if global_step % ckpt_interval == 0:
                ckpt_path = models_dir / f"step_{global_step:09d}.pt"
                save_checkpoint(
                    agent=agent,
                    checkpoint_path=ckpt_path,
                    global_step=global_step,
                    config=config,
                    success_rate=eval_metrics.get("eval/success_rate"),
                    base_policy_artifacts=config["base_policy"],
                    normalization={
                        "action_min": action_scaler.limits.min.detach().cpu().tolist(),
                        "action_max": action_scaler.limits.max.detach().cpu().tolist(),
                        "state_mean": state_standardizer._mean.detach().cpu().tolist(),
                        "state_std": state_standardizer._std.detach().cpu().tolist(),
                    },
                )
                checkpoints.append({
                    "global_step": global_step,
                    "path": str(ckpt_path.relative_to(out)),
                    "sha256": _sha256(ckpt_path),
                    "eval_success_rate": eval_metrics.get("eval/success_rate"),
                    "eval_act_calls_total": eval_metrics.get("eval/act_calls_total"),
                })

        summary = {
            "status": "COMPLETED",
            "smoke": args.smoke,
            "mode": mode,
            "offline_transitions": offline_count,
            "online_transitions": len(online_rb),
            "total_steps": total_steps,
            "zero_init_max_abs": init_max_abs,
            "mean_phase_velocity_train": sum(nu_samples) / len(nu_samples),
            "stall_cap_triggers_train": stall_triggers,
            "final_critic_loss": float(last_metrics["train/critic_loss"]),
            "final_actor_loss_base": float(last_metrics.get("train/actor_loss_base", 0.0)),
            "checkpoints": checkpoints,
        }
        _write_json(out / "summary.json", summary)
        _write_json(out / "status.json", {"status": "COMPLETED", "summary_sha256": _sha256(out / "summary.json")})
        print(json.dumps(summary, indent=2))
    except Exception as exc:
        _write_json(
            out / "status.json",
            {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    finally:
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()
        if patch_applied:
            _reverse_lerobot_patch()
        for name, root in (("resfit", RESFIT_ROOT), ("lerobot", LEROBOT_ROOT), ("robosuite", ROBOSUITE_ROOT)):
            if _git(root, "status", "--porcelain"):
                raise RuntimeError(f"Submodule {name} dirty after run")


if __name__ == "__main__":
    main()
