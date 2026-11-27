"""Real Square MuJoCo verification for ContinuousPhaseScheduler and PhaseCompositeSnapshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cf_resrl.scheduler import AbsoluteTickScheduler
from collect_factorial import DEFAULT_CONFIG as GATE_CONFIG, load_square_runner
from ps_resrl.phase_scheduler import ContinuousPhaseScheduler
from ps_resrl.phase_snapshot import (
    capture_phase_composite_snapshot,
    restore_phase_composite_snapshot,
)


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _max_abs_diff(a: Any, b: Any) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _integration_state(wrapper: Any) -> np.ndarray:
    import mujoco

    sim = wrapper.env.sim
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    buf = np.empty(mujoco.mj_stateSize(sim.model._model, spec), dtype=np.float64)
    mujoco.mj_getState(sim.model._model, sim.data._data, buf, spec)
    return buf


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=GATE_CONFIG)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.out or (REPO_ROOT / "results/ps_resrl/smoke_phase_scheduler/attempts" / stamp)
    out.mkdir(parents=True, exist_ok=False)

    runner, provenance = load_square_runner(config)
    seed = 20263124
    try:
        # Check A: Bit-exact zero-modulation parity vs AbsoluteTickScheduler on 30 real Square steps.
        def run_nominal_30(use_phase: bool):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            runner.wrapper.env.rng = np.random.default_rng(seed)
            obs, _ = runner.wrapper.reset(seed=seed)
            init_chunk = runner._chunk(obs, 0, 0)
            if use_phase:
                sched = ContinuousPhaseScheduler(
                    initial_chunk=init_chunk,
                    provider=runner._chunk,
                    latency_steps=runner.delay,
                    fallback_action=[0.0] * len(init_chunk[0]),
                )
            else:
                sched = AbsoluteTickScheduler(
                    initial_chunk=init_chunk,
                    provider=runner._chunk,
                    latency_steps=runner.delay,
                    fallback_action=[0.0] * len(init_chunk[0]),
                )
            trace = []
            for tick in range(30):
                sched.refill_if_exhausted(obs)
                actor_obs = runner._actor_observation(sched, obs)
                residual = runner._propose(actor_obs)
                entry = sched.action(residual, phase_velocity=1.0) if use_phase else sched.action(residual)
                phys = runner._physical(entry.applied_action)
                obs, reward, terminated, truncated, _ = runner.wrapper.step(phys.cpu().numpy())
                sched.after_step(obs, request_refresh=(tick == 6))
                sim_data = runner.wrapper.env.sim.data
                trace.append({
                    "tick": tick,
                    "applied_env_action": phys.cpu().tolist(),
                    "qpos": sim_data.qpos.copy(),
                    "qvel": sim_data.qvel.copy(),
                    "ctrl": sim_data.ctrl.copy(),
                    "reward": float(reward),
                    "act_calls": sched.counters.act_calls,
                    "img_sha": {
                        k: _sha256_array(obs[k]) for k in runner.policy.config.image_features
                    },
                })
                if terminated or truncated:
                    break
            return trace

        abs_trace = run_nominal_30(use_phase=False)
        phase_trace = run_nominal_30(use_phase=True)
        if len(abs_trace) != len(phase_trace):
            raise RuntimeError("Zero-modulation rollout length mismatch")
        parity_errors = {
            "applied_env_action": max(_max_abs_diff(a["applied_env_action"], b["applied_env_action"]) for a, b in zip(abs_trace, phase_trace)),
            "qpos": max(_max_abs_diff(a["qpos"], b["qpos"]) for a, b in zip(abs_trace, phase_trace)),
            "qvel": max(_max_abs_diff(a["qvel"], b["qvel"]) for a, b in zip(abs_trace, phase_trace)),
            "ctrl": max(_max_abs_diff(a["ctrl"], b["ctrl"]) for a, b in zip(abs_trace, phase_trace)),
            "reward": max(abs(a["reward"] - b["reward"]) for a, b in zip(abs_trace, phase_trace)),
            "act_calls_equal": all(a["act_calls"] == b["act_calls"] for a, b in zip(abs_trace, phase_trace)),
            "images_equal": all(a["img_sha"] == b["img_sha"] for a, b in zip(abs_trace, phase_trace)),
        }

        # Check B: PhaseCompositeSnapshot capture at tick 12 and double deterministic replay under fractional nu.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        runner.wrapper.env.rng = np.random.default_rng(seed)
        collector_rng = np.random.default_rng(seed)
        obs, _ = runner.wrapper.reset(seed=seed)
        init_chunk = runner._chunk(obs, 0, 0)
        sched = ContinuousPhaseScheduler(
            initial_chunk=init_chunk,
            provider=runner._chunk,
            latency_steps=runner.delay,
            fallback_action=[0.0] * len(init_chunk[0]),
        )
        observations = [obs]
        ledger = []
        for tick in range(12):
            sched.refill_if_exhausted(obs)
            actor_obs = runner._actor_observation(sched, obs)
            residual = runner._propose(actor_obs)
            nu = 0.75 if tick % 2 else 1.0
            entry = sched.action(residual, phase_velocity=nu)
            phys = runner._physical(entry.applied_action)
            obs, _, _, _, _ = runner.wrapper.step(phys.cpu().numpy())
            sched.after_step(obs, request_refresh=False)
            observations.append(obs)
            ledger.append(entry)

        sched.refill_if_exhausted(obs)
        actor_obs = runner._actor_observation(sched, obs)
        snap = capture_phase_composite_snapshot(
            runner.wrapper,
            sched,
            runner.policy,
            observation_tick=12,
            residual_history=[e.proposed_residual for e in ledger],
            observation_history=observations,
            applied_action_ledger=ledger,
            actor_observation=actor_obs,
            collector_rng=collector_rng,
            **runner.identities,
        )

        frac_seq = [0.0, 0.25, 0.7, 1.4, 0.5, 0.0, 1.1, 0.35, 1.8, 0.9, 0.4, 1.2, 0.6, 1.0, 0.2]

        def replay_15():
            restored = restore_phase_composite_snapshot(
                runner.wrapper, sched, runner.policy, snap, collector_rng=collector_rng, **runner.identities
            )
            cur_obs = restored.current_observation
            rows = []
            for offset, nu in enumerate(frac_seq):
                sched.refill_if_exhausted(cur_obs)
                a_obs = runner._actor_observation(sched, cur_obs)
                res = runner._propose(a_obs)
                entry = sched.action(res, phase_velocity=nu)
                phys = runner._physical(entry.applied_action)
                cur_obs, rew, term, trunc, _ = runner.wrapper.step(phys.cpu().numpy())
                sched.after_step(cur_obs, request_refresh=False)
                sim_data = runner.wrapper.env.sim.data
                rows.append({
                    "applied_env_action": phys.cpu().tolist(),
                    "phase_cursor": entry.phase_cursor,
                    "qpos": sim_data.qpos.copy(),
                    "qvel": sim_data.qvel.copy(),
                    "ctrl": sim_data.ctrl.copy(),
                    "integration_state": _integration_state(runner.wrapper),
                    "reward": float(rew),
                    "img_sha": {
                        k: _sha256_array(cur_obs[k]) for k in runner.policy.config.image_features
                    },
                })
                if term or trunc:
                    break
            return rows

        rep1 = replay_15()
        rep2 = replay_15()
        replay_errors = {
            "applied_env_action": max(_max_abs_diff(a["applied_env_action"], b["applied_env_action"]) for a, b in zip(rep1, rep2)),
            "phase_cursor": max(abs(a["phase_cursor"] - b["phase_cursor"]) for a, b in zip(rep1, rep2)),
            "qpos": max(_max_abs_diff(a["qpos"], b["qpos"]) for a, b in zip(rep1, rep2)),
            "qvel": max(_max_abs_diff(a["qvel"], b["qvel"]) for a, b in zip(rep1, rep2)),
            "ctrl": max(_max_abs_diff(a["ctrl"], b["ctrl"]) for a, b in zip(rep1, rep2)),
            "integration_state": max(_max_abs_diff(a["integration_state"], b["integration_state"]) for a, b in zip(rep1, rep2)),
            "reward": max(abs(a["reward"] - b["reward"]) for a, b in zip(rep1, rep2)),
            "images_equal": all(a["img_sha"] == b["img_sha"] for a, b in zip(rep1, rep2)),
        }

        report = {
            "status": "COMPLETED",
            "seed": seed,
            "provenance": provenance,
            "zero_modulation_parity_errors": parity_errors,
            "fractional_snapshot_replay_errors": replay_errors,
        }
        (out / "smoke_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (out / "status.json").write_text(json.dumps({"status": "COMPLETED"}, indent=2) + "\n")
        print(json.dumps(report["zero_modulation_parity_errors"], indent=2))
        print(json.dumps(report["fractional_snapshot_replay_errors"], indent=2))
        assert parity_errors["applied_env_action"] == 0.0
        assert parity_errors["qpos"] == 0.0 and parity_errors["images_equal"]
        assert replay_errors["integration_state"] == 0.0 and replay_errors["images_equal"]
    finally:
        runner.wrapper.close()


if __name__ == "__main__":
    main()
