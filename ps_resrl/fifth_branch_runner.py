"""Fifth-branch continuous-phase causal diagnostic on E0 factorial snapshots."""

from __future__ import annotations

import math
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from ps_resrl.frenet import frenet_decompose, phase_hold_velocity
from ps_resrl.phase_scheduler import ContinuousPhaseScheduler
from ps_resrl.phase_snapshot import (
    PhaseCompositeSnapshot,
    capture_phase_composite_snapshot,
    restore_phase_composite_snapshot,
)


PHASE_BRANCHES = ("V10_PhaseFreeze1", "V10_PhaseHold4", "V10_PhaseRuleSuffix")


def _branch_velocity(
    branch_mode: str,
    offset: int,
    base_action: Sequence[float],
    residual: Sequence[float],
    *,
    hold_steps: int,
    kappa: float,
    nu_min: float,
) -> float:
    if branch_mode == "V10_Replay":
        return 1.0
    if branch_mode == "V10_PhaseFreeze1":
        return 0.0 if offset == 0 else 1.0
    if branch_mode == "V10_PhaseHold4":
        if offset < hold_steps:
            return phase_hold_velocity(
                base_action, residual, kappa=kappa, nu_min=nu_min, nu_max=1.0
            )
        return 1.0
    if branch_mode == "V10_PhaseRuleSuffix":
        return phase_hold_velocity(
            base_action, residual, kappa=kappa, nu_min=nu_min, nu_max=1.0
        )
    raise ValueError(f"Unknown PS-0 branch mode: {branch_mode}")


def _run_phase_branch(
    snapshot: PhaseCompositeSnapshot,
    scheduler: ContinuousPhaseScheduler,
    runner: Any,
    *,
    branch_mode: str,
    candidate: list[float],
    collector_rng: Any,
    remaining: int,
    hold_steps: int,
    kappa: float,
    nu_min: float,
    saturation_threshold: float,
    saturation_window: int,
) -> dict[str, Any]:
    restored = restore_phase_composite_snapshot(
        runner.wrapper,
        scheduler,
        runner.policy,
        snapshot,
        collector_rng=collector_rng,
        **runner.identities,
    )
    observation = restored.current_observation
    start_calls = scheduler.counters.act_calls
    start_stall = scheduler.counters.stall_cap_triggers
    velocities: list[float] = []
    next_sat_flags: list[bool] = []
    total_return = 0.0
    discounted_return = 0.0
    success = False
    steps = 0
    started = time.perf_counter()

    for offset in range(remaining):
        if offset > 0:
            scheduler.refill_if_exhausted(observation)
            actor_obs = runner._actor_observation(scheduler, observation)
            residual = runner._propose(actor_obs)
        else:
            residual = candidate

        base_action = scheduler.current_base_action()
        nu = _branch_velocity(
            branch_mode,
            offset,
            base_action,
            residual,
            hold_steps=hold_steps,
            kappa=kappa,
            nu_min=nu_min,
        )
        entry = scheduler.action(residual, phase_velocity=nu)
        phys = runner._physical(entry.applied_action)
        observation, reward, terminated, truncated, _ = runner.wrapper.step(
            phys.cpu().numpy()
        )
        scheduler.after_step(observation, request_refresh=False)

        velocities.append(float(entry.applied_phase_velocity))
        if 1 <= offset <= saturation_window:
            max_abs_res = max(abs(float(x)) for x in residual)
            next_sat_flags.append(max_abs_res >= saturation_threshold)

        rew = float(reward)
        total_return += rew
        discounted_return += (runner.gamma**offset) * rew
        steps += 1
        if terminated or truncated:
            success = bool(terminated)
            break

    return {
        "label": branch_mode,
        "remaining_horizon": remaining,
        "environment_steps": steps,
        "act_calls": scheduler.counters.act_calls - start_calls,
        "stall_cap_triggers": scheduler.counters.stall_cap_triggers - start_stall,
        "mean_phase_velocity": float(sum(velocities) / len(velocities)),
        "min_phase_velocity": float(min(velocities)),
        "next10_residual_saturation_rate": (
            float(sum(next_sat_flags) / len(next_sat_flags)) if next_sat_flags else 0.0
        ),
        "next10_steps_observed": len(next_sat_flags),
        "wall_seconds": time.perf_counter() - started,
        "success": success,
        "return": total_return,
        "discounted_return": discounted_return,
    }


@torch.inference_mode()
def evaluate_e0_source_episode(
    runner: Any,
    e0_row: Mapping[str, Any],
    *,
    hold_steps: int = 4,
    kappa: float = 1.0,
    nu_min: float = 0.0,
    saturation_threshold: float = 0.095,
    saturation_window: int = 10,
    verify_v10: bool = False,
) -> dict[str, Any]:
    """Replay one E0 factorial source episode to source_tick and run PS-0 phase branches."""
    seed = int(e0_row["seed"])
    source_tick = int(e0_row["source_tick"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    runner.wrapper.env.rng = np.random.default_rng(seed)
    collector_rng = np.random.default_rng(seed)

    observation, _ = runner.wrapper.reset(seed=seed)
    initial = runner._chunk(observation, 0, 0)
    scheduler = ContinuousPhaseScheduler(
        initial_chunk=initial,
        provider=runner._chunk,
        latency_steps=runner.delay,
        fallback_action=[0.0] * len(initial[0]),
    )
    observations = [observation]
    ledger = []
    for _ in range(source_tick):
        scheduler.refill_if_exhausted(observation)
        actor_obs = runner._actor_observation(scheduler, observation)
        residual = runner._propose(actor_obs)
        entry = scheduler.action(residual, phase_velocity=1.0)
        phys = runner._physical(entry.applied_action)
        observation, _, terminated, truncated, _ = runner.wrapper.step(phys.cpu().numpy())
        scheduler.after_step(observation, request_refresh=False)
        if terminated or truncated:
            raise RuntimeError(
                f"E0 source episode {seed} terminated before source_tick {source_tick}"
            )
        ledger.append(entry)
        observations.append(observation)

    scheduler.refill_if_exhausted(observation)
    actor_obs = runner._actor_observation(scheduler, observation)
    candidate = runner._propose(actor_obs)
    base_at_source = scheduler.current_base_action()
    frenet = frenet_decompose(base_at_source, candidate)
    snapshot = capture_phase_composite_snapshot(
        runner.wrapper,
        scheduler,
        runner.policy,
        observation_tick=source_tick,
        residual_history=[entry.proposed_residual for entry in ledger],
        observation_history=observations,
        applied_action_ledger=ledger,
        actor_observation=actor_obs,
        collector_rng=collector_rng,
        **runner.identities,
    )
    remaining = runner.wrapper.horizon - source_tick

    if verify_v10:
        v10_replay = _run_phase_branch(
            snapshot,
            scheduler,
            runner,
            branch_mode="V10_Replay",
            candidate=candidate,
            collector_rng=collector_rng,
            remaining=remaining,
            hold_steps=hold_steps,
            kappa=kappa,
            nu_min=nu_min,
            saturation_threshold=saturation_threshold,
            saturation_window=saturation_window,
        )
        recorded_v10 = e0_row["branches"]["V10"]
        if (
            v10_replay["success"] != recorded_v10["success"]
            or v10_replay["act_calls"] != recorded_v10["act_calls"]
            or v10_replay["environment_steps"] != recorded_v10["environment_steps"]
            or not math.isclose(
                v10_replay["discounted_return"],
                float(recorded_v10["discounted_return"]),
                abs_tol=1e-9,
                rel_tol=0.0,
            )
        ):
            raise RuntimeError(
                f"V10 parity failed for seed={seed}: replayed={v10_replay} vs e0={recorded_v10}"
            )

    phase_branches = {
        branch_mode: _run_phase_branch(
            snapshot,
            scheduler,
            runner,
            branch_mode=branch_mode,
            candidate=candidate,
            collector_rng=collector_rng,
            remaining=remaining,
            hold_steps=hold_steps,
            kappa=kappa,
            nu_min=nu_min,
            saturation_threshold=saturation_threshold,
            saturation_window=saturation_window,
        )
        for branch_mode in PHASE_BRANCHES
    }

    return {
        "source_episode_id": e0_row["source_episode_id"],
        "seed": seed,
        "distribution": e0_row["distribution"],
        "split": e0_row["split"],
        "source_tick": source_tick,
        "remaining_horizon": remaining,
        "v10_parity_verified": verify_v10,
        "source_frenet": {
            "chunk_index": int(math.floor(scheduler.phase_cursor)),
            "brake_ratio": frenet.brake_ratio,
            "parallel_signed": frenet.parallel_signed,
            "perp_norm": frenet.perp_norm,
            "residual_l2": math.sqrt(sum(float(x) * float(x) for x in candidate)),
            "base_trans_l2": math.sqrt(sum(float(x) * float(x) for x in base_at_source[:3])),
        },
        "e0_branches": {
            label: {
                "success": bool(e0_row["branches"][label]["success"]),
                "return": float(e0_row["branches"][label]["return"]),
                "discounted_return": float(e0_row["branches"][label]["discounted_return"]),
                "act_calls": int(e0_row["branches"][label]["act_calls"]),
                "environment_steps": int(e0_row["branches"][label]["environment_steps"]),
            }
            for label in ("V00", "V01", "V10", "V11")
        },
        "phase_branches": phase_branches,
    }


def _quantile(sorted_values: list[float], fraction: float) -> float:
    rank = (len(sorted_values) - 1) * fraction
    lo, hi = math.floor(rank), math.ceil(rank)
    v = sorted_values[lo]
    return v + (sorted_values[hi] - v) * (rank - lo)


def _bootstrap_ci(
    values: list[float], *, samples: int, seed: str
) -> dict[str, float | list[float]]:
    if not values:
        return {"mean": 0.0, "ci95": [0.0, 0.0]}
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(samples):
        draw = rng.choices(values, k=n)
        means.append(math.fsum(draw) / n)
    means.sort()
    return {
        "mean": math.fsum(values) / n,
        "ci95": [_quantile(means, 0.025), _quantile(means, 0.975)],
    }


def _summarize_subset(
    rows: list[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: str,
) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"episodes": 0}
    v10_succ = [float(r["e0_branches"]["V10"]["success"]) for r in rows]
    v11_succ = [float(r["e0_branches"]["V11"]["success"]) for r in rows]
    v10_ret = [float(r["e0_branches"]["V10"]["discounted_return"]) for r in rows]
    v11_ret = [float(r["e0_branches"]["V11"]["discounted_return"]) for r in rows]
    v10_calls = [float(r["e0_branches"]["V10"]["act_calls"]) for r in rows]
    v11_calls = [float(r["e0_branches"]["V11"]["act_calls"]) for r in rows]

    refresh_rescued_idx = [
        i for i in range(n) if v10_succ[i] == 0.0 and v11_succ[i] == 1.0
    ]
    keep_fail_idx = [i for i in range(n) if v10_succ[i] == 0.0]

    branch_stats: dict[str, Any] = {}
    for branch_mode in PHASE_BRANCHES:
        pb_succ = [float(r["phase_branches"][branch_mode]["success"]) for r in rows]
        pb_ret = [
            float(r["phase_branches"][branch_mode]["discounted_return"]) for r in rows
        ]
        pb_calls = [float(r["phase_branches"][branch_mode]["act_calls"]) for r in rows]
        pb_sat = [
            float(r["phase_branches"][branch_mode]["next10_residual_saturation_rate"])
            for r in rows
        ]
        pb_nu = [
            float(r["phase_branches"][branch_mode]["mean_phase_velocity"]) for r in rows
        ]
        pb_stall = [
            float(r["phase_branches"][branch_mode]["stall_cap_triggers"]) for r in rows
        ]

        rescued_from_v10_fail = sum(
            1 for i in keep_fail_idx if pb_succ[i] == 1.0
        )
        regressed_from_v10_succ = sum(
            1 for i in range(n) if v10_succ[i] == 1.0 and pb_succ[i] == 0.0
        )
        recovered_of_refresh_rescued = sum(
            1 for i in refresh_rescued_idx if pb_succ[i] == 1.0
        )
        recovery_rate = (
            recovered_of_refresh_rescued / len(refresh_rescued_idx)
            if refresh_rescued_idx
            else None
        )

        # Stratified by brake_ratio < 0
        brake_neg = [i for i in range(n) if r_brake(rows[i]) < 0.0]
        brake_pos = [i for i in range(n) if r_brake(rows[i]) >= 0.0]

        branch_stats[branch_mode] = {
            "success_rate": _bootstrap_ci(
                pb_succ, samples=bootstrap_samples, seed=f"{bootstrap_seed}:{branch_mode}:succ"
            ),
            "discounted_return": _bootstrap_ci(
                pb_ret, samples=bootstrap_samples, seed=f"{bootstrap_seed}:{branch_mode}:ret"
            ),
            "act_calls": _bootstrap_ci(
                pb_calls, samples=bootstrap_samples, seed=f"{bootstrap_seed}:{branch_mode}:calls"
            ),
            "delta_success_vs_V10": _bootstrap_ci(
                [p - b for p, b in zip(pb_succ, v10_succ)],
                samples=bootstrap_samples,
                seed=f"{bootstrap_seed}:{branch_mode}:dsucc10",
            ),
            "delta_success_vs_V11": _bootstrap_ci(
                [p - b for p, b in zip(pb_succ, v11_succ)],
                samples=bootstrap_samples,
                seed=f"{bootstrap_seed}:{branch_mode}:dsucc11",
            ),
            "delta_discounted_return_vs_V10": _bootstrap_ci(
                [p - b for p, b in zip(pb_ret, v10_ret)],
                samples=bootstrap_samples,
                seed=f"{bootstrap_seed}:{branch_mode}:dret10",
            ),
            "delta_discounted_return_vs_V11": _bootstrap_ci(
                [p - b for p, b in zip(pb_ret, v11_ret)],
                samples=bootstrap_samples,
                seed=f"{bootstrap_seed}:{branch_mode}:dret11",
            ),
            "delta_act_calls_vs_V11": _bootstrap_ci(
                [p - b for p, b in zip(pb_calls, v11_calls)],
                samples=bootstrap_samples,
                seed=f"{bootstrap_seed}:{branch_mode}:dcalls11",
            ),
            "rescued_from_V10_failure": rescued_from_v10_fail,
            "regressed_from_V10_success": regressed_from_v10_succ,
            "refresh_rescued_count": len(refresh_rescued_idx),
            "recovered_of_refresh_rescued_count": recovered_of_refresh_rescued,
            "phase_recovery_rate": recovery_rate,
            "mean_next10_residual_saturation_rate": sum(pb_sat) / n,
            "mean_phase_velocity": sum(pb_nu) / n,
            "total_stall_cap_triggers": int(sum(pb_stall)),
            "by_brake_sign": {
                "brake_lt_0": {
                    "count": len(brake_neg),
                    "delta_return_vs_V10": (
                        sum(pb_ret[i] - v10_ret[i] for i in brake_neg) / len(brake_neg)
                        if brake_neg
                        else None
                    ),
                    "delta_success_vs_V10": (
                        sum(pb_succ[i] - v10_succ[i] for i in brake_neg) / len(brake_neg)
                        if brake_neg
                        else None
                    ),
                },
                "brake_ge_0": {
                    "count": len(brake_pos),
                    "delta_return_vs_V10": (
                        sum(pb_ret[i] - v10_ret[i] for i in brake_pos) / len(brake_pos)
                        if brake_pos
                        else None
                    ),
                    "delta_success_vs_V10": (
                        sum(pb_succ[i] - v10_succ[i] for i in brake_pos) / len(brake_pos)
                        if brake_pos
                        else None
                    ),
                },
            },
        }

    return {
        "episodes": n,
        "baselines": {
            "V10": {
                "success_rate": _bootstrap_ci(
                    v10_succ, samples=bootstrap_samples, seed=f"{bootstrap_seed}:V10:succ"
                ),
                "discounted_return": _bootstrap_ci(
                    v10_ret, samples=bootstrap_samples, seed=f"{bootstrap_seed}:V10:ret"
                ),
                "act_calls": _bootstrap_ci(
                    v10_calls, samples=bootstrap_samples, seed=f"{bootstrap_seed}:V10:calls"
                ),
            },
            "V11": {
                "success_rate": _bootstrap_ci(
                    v11_succ, samples=bootstrap_samples, seed=f"{bootstrap_seed}:V11:succ"
                ),
                "discounted_return": _bootstrap_ci(
                    v11_ret, samples=bootstrap_samples, seed=f"{bootstrap_seed}:V11:ret"
                ),
                "act_calls": _bootstrap_ci(
                    v11_calls, samples=bootstrap_samples, seed=f"{bootstrap_seed}:V11:calls"
                ),
            },
            "V11_minus_V10_return": _bootstrap_ci(
                [b - a for a, b in zip(v10_ret, v11_ret)],
                samples=bootstrap_samples,
                seed=f"{bootstrap_seed}:Gdelta",
            ),
        },
        "v10_failures": len(keep_fail_idx),
        "v10_fail_v11_success": len(refresh_rescued_idx),
        "phase_branches": branch_stats,
    }


def r_brake(row: Mapping[str, Any]) -> float:
    return float(row["source_frenet"]["brake_ratio"])


def summarize_ps0(
    rows: Iterable[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 20261327,
) -> dict[str, Any]:
    all_rows = list(rows)
    by_dist: dict[str, list[Mapping[str, Any]]] = {"natural": [], "enriched": []}
    for row in all_rows:
        dist = row["distribution"]
        if dist in by_dist:
            by_dist[dist].append(row)

    return {
        "total_episodes": len(all_rows),
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "unit": "source_episode",
            "interval": "percentile_95",
        },
        "distributions": {
            dist: _summarize_subset(
                subset,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=f"{bootstrap_seed}:{dist}",
            )
            for dist, subset in by_dist.items()
            if subset
        },
    }
