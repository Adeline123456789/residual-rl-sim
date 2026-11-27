"""Closed-loop evaluation for PS-Rule (B2) vs Original ResFiT under nominal & async refill."""

from __future__ import annotations

import math
import random
import time
from typing import Any

import numpy as np
import torch

from ps_resrl.frenet import phase_hold_velocity
from ps_resrl.phase_scheduler import ContinuousPhaseScheduler


@torch.inference_mode()
def run_ps_closed_loop_episode(
    runner: Any,
    *,
    seed: int,
    method: str,
    kappa: float = 1.0,
    nu_min: float = 0.0,
    nu_max: float = 1.0,
    refill_delay_steps: int = 0,
    saturation_threshold: float = 0.095,
) -> dict[str, Any]:
    """Execute one closed-loop Square rollout with ContinuousPhaseScheduler."""
    if method not in ("original_resfit", "ps_rule"):
        raise ValueError(f"Unknown PS-1 closed-loop method: {method}")
    horizon = runner.wrapper.horizon
    chunk_length = runner.policy.config.n_action_steps

    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    runner.wrapper.env.rng = np.random.default_rng(seed)
    observation, _ = runner.wrapper.reset(seed=seed)

    policy_seconds = 0.0

    def timed_chunk(obs: Any, source_tick: int, generation: int):
        nonlocal policy_seconds
        t0 = time.perf_counter()
        chunk = runner._chunk(obs, source_tick, generation)
        policy_seconds += time.perf_counter() - t0
        if len(chunk) != chunk_length:
            raise RuntimeError("ACT chunk length changed during episode")
        return chunk

    initial = timed_chunk(observation, 0, 0)
    scheduler = ContinuousPhaseScheduler(
        initial_chunk=initial,
        provider=timed_chunk,
        latency_steps=refill_delay_steps if refill_delay_steps > 0 else runner.delay,
        fallback_action=[0.0] * len(initial[0]),
        async_refill=(refill_delay_steps > 0),
    )

    velocities: list[float] = []
    sat_flags: list[bool] = []
    applied_actions: list[list[float]] = []
    total_return = 0.0
    discounted_return = 0.0
    success = False
    steps = 0

    for tick in range(horizon):
        scheduler.refill_if_exhausted(observation)
        t_pol = time.perf_counter()
        actor_obs = runner._actor_observation(scheduler, observation)
        residual = runner._propose(actor_obs)
        if method == "original_resfit":
            nu = 1.0
        else:
            nu = phase_hold_velocity(
                scheduler.current_base_action(),
                residual,
                kappa=kappa,
                nu_min=nu_min,
                nu_max=nu_max,
            )
        policy_seconds += time.perf_counter() - t_pol

        entry = scheduler.action(residual, phase_velocity=nu)
        phys = runner._physical(entry.applied_action)
        phys_list = phys.cpu().tolist()
        observation, reward, terminated, truncated, _ = runner.wrapper.step(
            phys.cpu().numpy()
        )
        scheduler.after_step(observation, request_refresh=False)

        velocities.append(float(entry.applied_phase_velocity))
        sat_flags.append(max(abs(float(x)) for x in residual) >= saturation_threshold)
        applied_actions.append(phys_list)

        rew = float(reward)
        total_return += rew
        discounted_return += (runner.gamma**tick) * rew
        steps += 1
        if terminated or truncated:
            success = bool(terminated)
            break

    jerks = [
        math.sqrt(sum((a - b) ** 2 for a, b in zip(curr, prev)))
        for prev, curr in zip(applied_actions[:-1], applied_actions[1:])
    ]
    wall_seconds = time.perf_counter() - started
    return {
        "seed": seed,
        "method": method,
        "kappa": kappa if method == "ps_rule" else None,
        "nu_min": nu_min if method == "ps_rule" else None,
        "refill_delay_steps": refill_delay_steps,
        "success": success,
        "return": total_return,
        "discounted_return": discounted_return,
        "act_calls": 1 + scheduler.counters.act_calls,
        "environment_steps": steps,
        "stall_cap_triggers": scheduler.counters.stall_cap_triggers,
        "mean_phase_velocity": float(sum(velocities) / len(velocities)),
        "min_phase_velocity": float(min(velocities)),
        "residual_saturation_rate": float(sum(sat_flags) / len(sat_flags)),
        "action_jerk_l2": float(sum(jerks) / len(jerks)) if jerks else 0.0,
        "wall_seconds": wall_seconds,
        "policy_wall_seconds": policy_seconds,
        "wall_clock_ms_per_step": 1000.0 * wall_seconds / steps,
    }
