"""Select one E0 source state per nominal episode and branch from its snapshot."""

from __future__ import annotations

from dataclasses import asdict
import base64
import hashlib
import math
import random
import time
import zlib
from typing import TYPE_CHECKING

from cf_resrl.composite_snapshot import capture_composite_snapshot
from cf_resrl.factorial import FactorialOutcome
from cf_resrl.scheduler import AbsoluteTickScheduler

if TYPE_CHECKING:
    import numpy as np


PAIR_LABELS = ("V10", "V11")


def _should_replace(distribution, tick, rng, residual_norm_sq, best_norm_sq):
    if distribution == "natural":
        return rng.randrange(tick + 1) == 0
    return residual_norm_sq > best_norm_sq


def visible_features(runner, scheduler, actor_observation, residual) -> dict[str, np.ndarray]:
    """Extract the same deployment-visible inputs for every offline probe."""
    import numpy as np
    import torch

    active = scheduler.active_chunk
    tick = scheduler.logical_tick
    index = tick - active.source_tick
    chunk_length = runner.policy.config.n_action_steps
    if index < 0 or len(active.actions) != chunk_length:
        raise RuntimeError("Source does not have a registered ACT action chunk")
    suffix = active.actions[index:]
    chunk = np.zeros((chunk_length, len(residual)), dtype=np.float32)
    mask = np.zeros(chunk_length, dtype=np.float32)
    if suffix:
        chunk[:len(suffix)] = suffix
        mask[:len(suffix)] = 1
    with torch.inference_mode():
        visual = runner.agent._encode(actor_observation, augment=False)
        visual = runner.agent.actor.compress(visual.flatten(1, -1))
    features = {
        "visual": visual[0].detach().cpu().numpy().astype(np.float32, copy=True),
        "state": actor_observation["observation.state"][0].detach().cpu().numpy().astype(np.float32, copy=True),
        "base": actor_observation["observation.base_action"][0].detach().cpu().numpy().astype(np.float32, copy=True),
        "chunk_suffix": chunk,
        "chunk_mask": mask,
        "tick": np.array([tick], dtype=np.float32),
        "chunk_index": np.array([index], dtype=np.float32),
        "residual": np.array(residual, dtype=np.float32),
    }
    if not all(np.isfinite(value).all() for value in features.values()):
        raise RuntimeError("A source feature contains NaN or infinity")
    return features


def _encode_actor_observation(observation):
    import numpy as np

    evidence = {}
    for key, tensor in observation.items():
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        raw = array.tobytes()
        evidence[key] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "encoding": "zlib+base64",
            "data": base64.b64encode(zlib.compress(raw, level=3)).decode("ascii"),
        }
    return evidence


def _terminal(row):
    return row["terminated"] or row["truncated"]


def _validate_branches(results, *, scheduler_snapshot, delay, remaining, candidate, low, high):
    """Validate treatment and waiting even when physics terminates a suffix early."""
    source_tick = scheduler_snapshot.logical_tick
    start_calls = scheduler_snapshot.counters.act_calls
    old_chunk = scheduler_snapshot.active_chunk
    old_end = old_chunk.source_tick + len(old_chunk.actions)
    wait_end = source_tick + 1 + delay
    for label, branch in results.items():
        rows = branch["steps"]
        if not rows or len(rows) != branch["environment_steps"] or branch["remaining_horizon"] != remaining:
            raise RuntimeError(f"Inconsistent {label} branch accounting")
        if len(rows) > remaining or (len(rows) < remaining and not _terminal(rows[-1])):
            raise RuntimeError(f"Incorrect {label} branch horizon")
        first = rows[0]
        treated_residual = candidate if label[1] == "1" else [0.0] * len(candidate)
        if list(first["proposed_residual"]) != treated_residual:
            raise RuntimeError(f"Incorrect {label} first residual intervention")
        request = label[2] == "1"
        dispatched = request and not _terminal(first)
        request_source = source_tick + 1 if dispatched else None
        request_ready = wait_end if dispatched else None
        if (
            first["refresh_requested_after_step"] != request
            or first["pending_refresh_source_tick"] != request_source
            or first["pending_refresh_ready_tick"] != request_ready
            or first["act_calls_since_initial_chunk"] - start_calls != int(dispatched)
        ):
            raise RuntimeError(f"Incorrect {label} refresh dispatch")
        for offset, row in enumerate(rows):
            tick = source_tick + offset
            if row["tick"] != tick or row["control_steps"] != tick + 1:
                raise RuntimeError(f"Discontinuous {label} control ticks")
            if offset and row["refresh_requested_after_step"]:
                raise RuntimeError(f"Unexpected extra {label} refresh request")
            if offset != len(rows) - 1 and _terminal(row):
                raise RuntimeError(f"Continued {label} after termination")
            if len(row["applied_env_action"]) != len(candidate) or not all(
                lo <= action <= hi for action, lo, hi in zip(row["applied_env_action"], low, high)
            ):
                raise RuntimeError(f"Out-of-range {label} environment action")
            waiting = dispatched and source_tick < tick < wait_end
            if row["used_fallback"] != (waiting and tick >= old_end):
                raise RuntimeError(f"Incorrect {label} fallback timing")
            if waiting and (
                row["generation"] != old_chunk.generation
                or row["chunk_source_tick"] != old_chunk.source_tick
            ):
                raise RuntimeError(f"Premature {label} new chunk while waiting")
        if rows[-1]["act_calls_since_initial_chunk"] - start_calls != branch["act_calls"]:
            raise RuntimeError(f"Inconsistent {label} ACT-call ledger")
        activated = dispatched and len(rows) > delay + 1
        if activated:
            active = rows[delay + 1]
            if (
                active["tick"] != wait_end
                or active["generation"] != scheduler_snapshot.next_generation
                or active["chunk_source_tick"] != request_source
                or active["chunk_index"] != delay
            ):
                raise RuntimeError(f"Incorrect {label} refresh activation or absolute index")
        branch["refresh_dispatched"] = dispatched
        branch["refresh_activation_tick"] = wait_end if activated else None
        branch["waiting_fallback_steps"] = sum(row["used_fallback"] for row in rows)

    shared = (
        "tick", "generation", "chunk_index", "applied_env_action",
        "reward", "terminated", "truncated", "simulation_time", "observation_state",
        "observation_image_sha256", "integration_state_sha256",
    )
    for keep_label, refresh_label in (("V00", "V01"), PAIR_LABELS):
        if keep_label not in results:
            continue
        keep = results[keep_label]["steps"]
        refresh = results[refresh_label]["steps"]
        prefix_length = min(delay + 1, old_end - source_tick)
        for a, b in zip(keep[:prefix_length], refresh[:prefix_length]):
            for key in shared:
                if a[key] != b[key]:
                    raise RuntimeError(f"{keep_label}/{refresh_label} diverged before refresh could affect {key}")


def collect_episode(
    runner, *, seed: int, distribution: str, labels: tuple[str, ...], selector_seed: int
) -> tuple[dict, dict[str, np.ndarray]]:
    """Collect a nominal episode, then evaluate the selected real branch pair or quartet."""
    import numpy as np
    import torch

    from cf_resrl.branch_runner import BRANCHES, _array_sha256

    labels = tuple(labels)
    if distribution not in ("natural", "enriched") or labels not in (BRANCHES, PAIR_LABELS):
        raise ValueError("Use a registered E0 distribution and branch set")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    runner.wrapper.env.rng = np.random.default_rng(seed)
    selector_rng = random.Random(selector_seed)
    started = time.perf_counter()
    with torch.inference_mode():
        observation, _ = runner.wrapper.reset(seed=seed)
        initial = runner._chunk(observation, 0, 0)
        scheduler = AbsoluteTickScheduler(
            initial_chunk=initial,
            provider=runner._chunk,
            latency_steps=runner.delay,
            fallback_action=[0.0] * len(initial[0]),
        )
        rows = []
        observations = [observation]
        best_norm_sq = -math.inf
        snapshot = None
        for tick in range(runner.wrapper.horizon):
            scheduler.refill_if_exhausted(observation)
            actor_observation = runner._actor_observation(scheduler, observation)
            residual = runner._propose(actor_observation)
            norm_sq = math.fsum(component * component for component in residual)
            if _should_replace(distribution, tick, selector_rng, norm_sq, best_norm_sq):
                snapshot = capture_composite_snapshot(
                    runner.wrapper, scheduler, runner.policy,
                    observation_tick=tick,
                    residual_history=[row["proposed_residual"] for row in rows],
                    observation_history=observations,
                    applied_action_ledger=rows,
                    actor_observation=actor_observation,
                    collector_rng=selector_rng,
                    **runner.identities,
                )
                candidate = residual
                best_norm_sq = norm_sq
            observation, row = runner._step(scheduler, residual, False)
            rows.append(row)
            observations.append(observation)
            if _terminal(row):
                break

        scout = {
            "environment_steps": len(rows),
            "act_calls": 1 + scheduler.counters.act_calls,
            "wall_seconds": time.perf_counter() - started,
            "success": rows[-1]["terminated"],
            "return": sum(row["reward"] for row in rows),
            "steps": rows,
        }
        source_tick = snapshot.observation_tick
        remaining = runner.wrapper.horizon - source_tick
        results = {
            label: runner._branch(snapshot, scheduler, label, candidate, selector_rng, remaining)
            for label in labels
        }
        _validate_branches(
            results, scheduler_snapshot=snapshot.scheduler, delay=runner.delay,
            remaining=remaining, candidate=candidate, low=runner.wrapper.action_space.low,
            high=runner.wrapper.action_space.high,
        )
        scheduler.restore(snapshot.scheduler)
        features = visible_features(runner, scheduler, snapshot.actor_observation, candidate)
    v = {label.lower(): branch["discounted_return"] for label, branch in results.items()}
    v["G_delta"] = v["v11"] - v["v10"]
    if labels == BRANCHES:
        factorial = FactorialOutcome(*(v[label.lower()] for label in BRANCHES))
        v.update(G0=factorial.base_refresh_gain, I_delta=factorial.interaction)
    return {
        "source_seed": seed,
        "source_tick": source_tick,
        "source_episode_length": len(rows),
        "remaining_horizon": remaining,
        "source_prefix_steps": source_tick,
        "source_prefix_act_calls": 1 + snapshot.scheduler.counters.act_calls,
        "source_state_sha256": _array_sha256(np.asarray(snapshot.environment.integration_state)),
        "source_integration_state_size": snapshot.environment.model.integration_state_size,
        "candidate_residual": candidate,
        "active_chunk": asdict(snapshot.scheduler.active_chunk),
        "actor_observation": _encode_actor_observation(snapshot.actor_observation),
        "selection": {
            "distribution": distribution,
            "selector_seed": selector_seed,
            "rule": "uniform_reservoir" if distribution == "natural" else "max_residual_l2_earliest_tie",
            "within_episode_probability": 1.0 / len(rows) if distribution == "natural" else None,
            "selected_residual_l2": math.sqrt(best_norm_sq),
        },
        "source_prefix": list(snapshot.applied_action_ledger),
        "nominal_scout": scout,
        "branches": results,
        "factorial_values": v,
    }, features
