"""Evaluate refresh decisions with the frozen ACT and ResFiT policies."""

from __future__ import annotations

import math
import random
import time

import numpy as np
import torch

from cf_resrl.e0_collector import visible_features
from cf_resrl.models import predict_probe
from cf_resrl.offline_data import pack_features
from cf_resrl.scheduler import AbsoluteTickScheduler


_PROBES = ("B5", "B6", "P1")
_BASELINES = ("original_resfit", "fixed_period_refresh", "residual_norm_threshold")


def _nominal_calls(horizon: int, chunk_length: int) -> int:
    return (max(horizon, 0) + chunk_length - 1) // chunk_length


def _refresh_block(scheduler, *, horizon, chunk_length, delay, act_budget):
    if scheduler.pending_refresh is not None:
        return "pending"
    source_tick = scheduler.logical_tick + 1
    if source_tick + delay >= horizon:
        return "horizon"
    # Charge the initial chunk, this request, and nominal refills after its expiry.
    reserve = _nominal_calls(horizon - source_tick - chunk_length, chunk_length)
    if 2 + scheduler.counters.act_calls + reserve > act_budget:
        return "act_budget"
    return None


def _decision_score(method, parameter, probe, runner, scheduler, actor_observation, residual):
    if method == "fixed_period_refresh":
        return float((scheduler.logical_tick + 1) % parameter == 0)
    if method == "residual_norm_threshold":
        return math.sqrt(math.fsum(value * value for value in residual))
    features = visible_features(runner, scheduler, actor_observation, residual)
    context, candidate = pack_features(features)
    return float(predict_probe(probe, context[None, :], candidate[None, :])["g_delta"][0])


def _validate_method(method, parameter, probe):
    if method not in (*_BASELINES, *_PROBES):
        raise ValueError(f"Unknown closed-loop method: {method}")
    if method in _PROBES:
        if probe is None or probe.method != method:
            raise ValueError(f"{method} requires its matching fitted probe")
    elif probe is not None:
        raise ValueError(f"{method} does not use a fitted probe")
    if method == "original_resfit":
        return None
    if method == "fixed_period_refresh":
        if type(parameter) is not int or parameter <= 0:
            raise ValueError("The refresh period must be a positive integer")
        return 0.0
    if type(parameter) not in (int, float) or not math.isfinite(parameter):
        raise ValueError("The decision threshold must be finite")
    if method == "residual_norm_threshold" and parameter < 0:
        raise ValueError("The residual norm threshold must be nonnegative")
    return float(parameter)


@torch.inference_mode()
def run_episode(runner, *, seed, act_budget, method, parameter, probe=None) -> dict:
    """Execute one real rollout with an episode-level cap on ACT inference calls."""
    threshold = _validate_method(method, parameter, probe)
    horizon = runner.wrapper.horizon
    chunk_length = runner.policy.config.n_action_steps
    if horizon <= 0 or chunk_length <= 0 or not 0 <= runner.delay < chunk_length:
        raise ValueError("Expected a positive horizon and a delay shorter than the ACT chunk")
    if type(act_budget) is not int or act_budget < _nominal_calls(horizon, chunk_length):
        raise ValueError("The ACT budget must cover all nominal refills through the horizon")

    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    runner.wrapper.env.rng = np.random.default_rng(seed)
    observation, _ = runner.wrapper.reset(seed=seed)
    policy_seconds = 0.0

    def timed_chunk(observation, source_tick, generation):
        nonlocal policy_seconds
        start = time.perf_counter()
        chunk = runner._chunk(observation, source_tick, generation)
        policy_seconds += time.perf_counter() - start
        if len(chunk) != chunk_length:
            raise RuntimeError("ACT chunk length changed during a budgeted episode")
        return chunk

    initial = timed_chunk(observation, 0, 0)
    scheduler = AbsoluteTickScheduler(
        initial_chunk=initial,
        provider=timed_chunk,
        latency_steps=runner.delay,
        fallback_action=[0.0] * len(initial[0]),
    )
    nominal_calls = 0
    awaiting_activation = set()
    activations = 0
    rows = []
    for _ in range(horizon):
        nominal_refill = scheduler.refill_if_exhausted(observation)
        nominal_calls += int(nominal_refill)
        generation = scheduler.active_chunk.generation
        activated = generation in awaiting_activation
        if activated:
            awaiting_activation.remove(generation)
            activations += 1

        policy_start = time.perf_counter()
        actor_observation = runner._actor_observation(scheduler, observation)
        residual = runner._propose(actor_observation)
        blocked = _refresh_block(
            scheduler, horizon=horizon, chunk_length=chunk_length, delay=runner.delay, act_budget=act_budget,
        )
        score = None
        request = False
        if threshold is not None and blocked is None:
            score = _decision_score(method, parameter, probe, runner, scheduler, actor_observation, residual)
            if not math.isfinite(score):
                raise RuntimeError("Closed-loop probe returned a non-finite score")
            request = score > threshold
        policy_seconds += time.perf_counter() - policy_start

        requests_before = scheduler.counters.refresh_requests
        observation, row = runner._step(scheduler, residual, request)
        dispatched = scheduler.counters.refresh_requests > requests_before
        if dispatched:
            awaiting_activation.add(scheduler.pending_refresh.chunk.generation)
        act_calls = 1 + scheduler.counters.act_calls
        if act_calls > act_budget or act_calls != 1 + nominal_calls + scheduler.counters.refresh_requests:
            raise RuntimeError("Closed-loop ACT call accounting exceeded the registered budget")
        row.update(
            nominal_refill=nominal_refill,
            refresh_block=blocked,
            decision_score=score,
            refresh_dispatched=dispatched,
            refresh_activated=activated,
            act_calls=act_calls,
        )
        rows.append(row)
        if row["terminated"] or row["truncated"]:
            break

    return {
        "seed": seed,
        "method": method,
        "parameter": parameter,
        "act_budget": act_budget,
        "act_calls": act_calls,
        "initial_act_calls": 1,
        "nominal_act_calls": nominal_calls,
        "refresh_requests": scheduler.counters.refresh_requests,
        "refresh_activations": activations,
        "environment_steps": len(rows),
        "return": sum(row["reward"] for row in rows),
        "discounted_return": sum(runner.gamma**tick * row["reward"] for tick, row in enumerate(rows)),
        "success": rows[-1]["terminated"],
        "wall_seconds": time.perf_counter() - started,
        "policy_wall_seconds": policy_seconds,
        "steps": rows,
    }
