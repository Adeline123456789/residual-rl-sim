"""Real Square rollouts with one residual and one refresh intervention."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import random
import time

import mujoco
import numpy as np
import torch

from cf_resrl.composite_snapshot import capture_composite_snapshot, restore_composite_snapshot
from cf_resrl.factorial import FactorialOutcome
from cf_resrl.scheduler import AbsoluteTickScheduler


BRANCHES = ("V00", "V01", "V10", "V11")
PROVIDER_ID = "resfit_native_act_queue__absolute_tick"


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class SquareBranchRunner:
    """Give both policies the observations and action normalization from training."""

    def __init__(self, wrapper, policy, agent, normalization, identities, device, delay, gamma):
        self.wrapper = wrapper
        self.policy = policy
        self.agent = agent
        self.identities = identities
        self.device = device
        self.delay = delay
        self.gamma = gamma
        self.low = self._tensor(wrapper.action_space.low)
        self.high = self._tensor(wrapper.action_space.high)
        self.action_min = self._tensor(normalization["action_min"])
        self.action_max = self._tensor(normalization["action_max"])
        self.action_range = (self.action_max - self.action_min).clamp_min(1e-8)
        self.state_mean = self._tensor(normalization["state_mean"])
        self.state_std = self._tensor(normalization["state_std"])

    def _tensor(self, values):
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _batch(self, observation):
        return {key: torch.as_tensor(value, device=self.device).unsqueeze(0) for key, value in observation.items()}

    def _scale(self, action):
        clamped = action.clamp(self.action_min, self.action_max)
        return 2.0 * (clamped - self.action_min) / self.action_range - 1.0

    def _physical(self, action):
        normalized = self._tensor(action).clamp(-1, 1)
        return (self.action_min + (normalized + 1.0) * self.action_range / 2.0).clamp(self.low, self.high)

    def _chunk(self, observation, _source_tick, _generation):
        # ACT's native first action and remaining queue come from one inference.
        self.policy.reset()
        first = self.policy.select_action(self._batch(observation))[0]
        (queue,) = self.policy._action_queues
        if len(queue) + 1 != self.policy.config.n_action_steps:
            raise RuntimeError("ACT did not return a complete action chunk")
        return self._scale(torch.stack([first, *queue])).cpu().tolist()

    def _actor_observation(self, scheduler, observation):
        batch = self._batch(observation)
        batch["observation.state"] = (batch["observation.state"] - self.state_mean) / self.state_std
        batch["observation.base_action"] = self._tensor(scheduler.current_base_action()).unsqueeze(0)
        return batch

    def _propose(self, actor_observation):
        return self.agent.act(actor_observation, eval_mode=True, cpu=True)[0].tolist()

    def _step(self, scheduler, residual, refresh):
        entry = scheduler.action(residual)
        physical = self._physical(entry.applied_action)
        applied_normalized = self._scale(physical)
        proposed_normalized = self._tensor(entry.base_action) + self._tensor(residual)
        observation, reward, terminated, truncated, _ = self.wrapper.step(physical.cpu().numpy())
        scheduler.after_step(observation, request_refresh=refresh and not (terminated or truncated))
        pending = scheduler.pending_refresh
        sim = self.wrapper.env.sim
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(sim.model._model, state_spec), dtype=np.float64)
        mujoco.mj_getState(sim.model._model, sim.data._data, state, state_spec)
        row = {
            **asdict(entry),
            "applied_normalized_action": applied_normalized.cpu().tolist(),
            "applied_normalized_residual": (applied_normalized - self._tensor(entry.base_action)).cpu().tolist(),
            "action_clip_mask": (~torch.isclose(proposed_normalized, applied_normalized, rtol=0.0, atol=1e-7)).cpu().tolist(),
            "applied_env_action": physical.cpu().tolist(),
            "refresh_requested_after_step": refresh,
            "pending_refresh_source_tick": pending.chunk.source_tick if pending else None,
            "pending_refresh_ready_tick": pending.ready_tick if pending else None,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "simulation_time": float(self.wrapper.env.sim.data.time),
            "integration_state_sha256": _array_sha256(state),
            "observation_state": observation["observation.state"].tolist(),
            "observation_image_sha256": {
                key: _array_sha256(observation[key]) for key in self.policy.config.image_features
            },
            "control_steps": scheduler.counters.control_steps,
            "act_calls_since_initial_chunk": scheduler.counters.act_calls,
        }
        return observation, row

    def _branch(self, snapshot, scheduler, branch, candidate, collector_rng, remaining):
        restored = restore_composite_snapshot(
            self.wrapper, scheduler, self.policy, snapshot, collector_rng=collector_rng, **self.identities
        )
        observation = restored.current_observation
        current_actor_input = self._actor_observation(scheduler, observation)
        if not all(torch.equal(value, restored.actor_observation[key]) for key, value in current_actor_input.items()):
            raise RuntimeError("Restored residual actor observation differs from captured input")
        if self._propose(current_actor_input) != candidate:
            raise RuntimeError("Restored candidate residual differs from captured proposal")

        start_calls = scheduler.counters.act_calls
        rows = []
        started = time.perf_counter()
        for index in range(remaining):
            if index:
                scheduler.refill_if_exhausted(observation)
                residual = self._propose(self._actor_observation(scheduler, observation))
            else:
                residual = candidate if branch[1] == "1" else [0.0] * len(candidate)
            observation, row = self._step(scheduler, residual, index == 0 and branch[2] == "1")
            rows.append(row)
            if row["terminated"] or row["truncated"]:
                break
        return {
            "label": branch,
            "remaining_horizon": remaining,
            "environment_steps": len(rows),
            "act_calls": scheduler.counters.act_calls - start_calls,
            "wall_seconds": time.perf_counter() - started,
            "success": rows[-1]["terminated"],
            "return": sum(row["reward"] for row in rows),
            "discounted_return": sum(self.gamma**i * row["reward"] for i, row in enumerate(rows)),
            "steps": rows,
        }

    @torch.inference_mode()
    def group(self, seed, source_tick, *, verify_replay=False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.wrapper.env.rng = np.random.default_rng(seed)
        collector_rng = np.random.default_rng(seed)
        observation, _ = self.wrapper.reset(seed=seed)
        initial = self._chunk(observation, 0, 0)
        scheduler = AbsoluteTickScheduler(
            initial_chunk=initial,
            provider=self._chunk,
            latency_steps=self.delay,
            fallback_action=[0.0] * len(initial[0]),
        )
        observations = [observation]
        prefix = []
        for _ in range(source_tick):
            scheduler.refill_if_exhausted(observation)
            residual = self._propose(self._actor_observation(scheduler, observation))
            observation, row = self._step(scheduler, residual, False)
            if row["terminated"] or row["truncated"]:
                raise RuntimeError(f"Source episode {seed} finished before registered tick {source_tick}")
            prefix.append(row)
            observations.append(observation)
        scheduler.refill_if_exhausted(observation)
        actor_obs = self._actor_observation(scheduler, observation)
        candidate = self._propose(actor_obs)
        snapshot = capture_composite_snapshot(
            self.wrapper,
            scheduler,
            self.policy,
            observation_tick=source_tick,
            residual_history=[row["proposed_residual"] for row in prefix],
            observation_history=observations,
            applied_action_ledger=prefix,
            actor_observation=actor_obs,
            collector_rng=collector_rng,
            **self.identities,
        )
        remaining = self.wrapper.horizon - source_tick
        results = {
            branch: self._branch(snapshot, scheduler, branch, candidate, collector_rng, remaining)
            for branch in BRANCHES
        }
        self._check_pairs(results, source_tick)
        if verify_replay:
            repeated = self._branch(snapshot, scheduler, "V11", candidate, collector_rng, remaining)
            if repeated["steps"] != results["V11"]["steps"]:
                raise RuntimeError("Restored V11 rollout failed exact deterministic replay")
        values = FactorialOutcome(*(results[branch]["discounted_return"] for branch in BRANCHES))
        return {
            "source_seed": seed,
            "source_tick": source_tick,
            "remaining_horizon": remaining,
            "source_prefix_steps": source_tick,
            "source_prefix_act_calls": 1 + snapshot.scheduler.counters.act_calls,
            "source_state_sha256": _array_sha256(np.asarray(snapshot.environment.integration_state)),
            "source_integration_state_size": snapshot.environment.model.integration_state_size,
            "candidate_residual": candidate,
            "active_chunk": asdict(snapshot.scheduler.active_chunk),
            "pending_refresh": asdict(snapshot.scheduler.pending_refresh) if snapshot.scheduler.pending_refresh else None,
            "source_prefix": prefix,
            "branches": results,
            "factorial_values": {
                **asdict(values),
                "G0": values.base_refresh_gain,
                "G_delta": values.residual_refresh_gain,
                "I_delta": values.interaction,
            },
            "replay_verification": (
                {
                    "label": "V11",
                    "comparison": "exact",
                    "environment_steps": repeated["environment_steps"],
                    "act_calls": repeated["act_calls"],
                    "wall_seconds": repeated["wall_seconds"],
                }
                if verify_replay else None
            ),
        }

    def _check_pairs(self, results, source_tick):
        for keep_label, refresh_label in (("V00", "V01"), ("V10", "V11")):
            keep = results[keep_label]["steps"]
            refresh = results[refresh_label]["steps"]
            if len(keep) <= self.delay + 1 or len(refresh) <= self.delay + 1:
                raise RuntimeError("Branch ended before refresh activation could be observed")
            if keep[0]["refresh_requested_after_step"] or not refresh[0]["refresh_requested_after_step"]:
                raise RuntimeError("The first-step refresh treatment was not applied")
            if any(row["refresh_requested_after_step"] for row in [*keep[1:], *refresh[1:]]):
                raise RuntimeError("A later control step issued an extra experimental refresh")
            for a, b in zip(keep[: self.delay + 1], refresh[: self.delay + 1]):
                for key in ("tick", "generation", "chunk_index", "applied_env_action", "reward", "simulation_time",
                            "observation_state", "observation_image_sha256", "integration_state_sha256"):
                    if a[key] != b[key]:
                        raise RuntimeError(f"Paired branches diverged before refresh activation: {key}")
            active = refresh[self.delay + 1]
            if (
                active["tick"] != source_tick + 1 + self.delay
                or active["chunk_source_tick"] != source_tick + 1
                or active["chunk_index"] != self.delay
                or active["used_fallback"]
            ):
                raise RuntimeError("Refreshed ACT chunk used the wrong activation tick or index")
            if (
                refresh[0]["pending_refresh_source_tick"] != source_tick + 1
                or refresh[0]["pending_refresh_ready_tick"] != source_tick + 1 + self.delay
            ):
                raise RuntimeError("Refresh request is not anchored to the post-action observation")
        for branch in results.values():
            for offset, row in enumerate(branch["steps"]):
                if row["tick"] != source_tick + offset:
                    raise RuntimeError("Branch control ticks are discontinuous")
                if not all(lo <= action <= hi for action, lo, hi in zip(
                    row["applied_env_action"], self.wrapper.action_space.low, self.wrapper.action_space.high
                )):
                    raise RuntimeError("Physical action exceeded the environment action limits")
