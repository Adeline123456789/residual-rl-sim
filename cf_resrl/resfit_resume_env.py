"""Capture the live control state of the single Square ResFiT training lane."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from cf_resrl.composite_snapshot import (
    ActPolicySnapshot,
    _capture_act_policy,
    _restore_act_policy,
)
from cf_resrl.snapshot import SquareEnvSnapshot


@dataclass(frozen=True)
class ResfitEnvSnapshot:
    environment: SquareEnvSnapshot
    act_policy: ActPolicySnapshot
    base_normalized_action: Any
    observation: dict[str, Any]


def _training_vector(env: Any) -> Any:
    vector = env.vec_env.vec_env
    if vector.num_envs != 1 or vector.autoreset_mode.value != "SameStep":
        raise ValueError("Square training resume requires one SAME_STEP environment lane")
    return vector


def _move_act_queue(snapshot: ActPolicySnapshot, device: Any) -> ActPolicySnapshot:
    # Only the pending actions travel with the snapshot. The frozen ACT model
    # is reloaded from the same pinned local path by the trainer.
    queues = []
    for queue in snapshot.queues:
        items = tuple(
            replace(item, value=item.value.detach().to(device, copy=True))
            for item in queue.items
        )
        queues.append(replace(queue, items=items))
    return replace(snapshot, queues=tuple(queues))


def capture_resfit_env(env: Any, observation: dict[str, Any]) -> ResfitEnvSnapshot:
    """Capture worker state and parent-side tensors after a complete env step."""
    (worker_snapshot,) = _training_vector(env).call("capture_training_state")
    action = env._last_base_naction
    return ResfitEnvSnapshot(
        environment=worker_snapshot,
        act_policy=_move_act_queue(_capture_act_policy(env.base_policy), "cpu"),
        base_normalized_action=action.detach().to("cpu", copy=True),
        observation={key: value.detach().to("cpu", copy=True) for key, value in observation.items()},
    )


def restore_resfit_env(
    env: Any, snapshot: ResfitEnvSnapshot, device: Any
) -> dict[str, Any]:
    """Restore a new worker and return the observation for the next action."""
    vector = _training_vector(env)
    policy = env.base_policy
    policy._ensure_action_queues(vector.num_envs)
    _restore_act_policy(policy, _move_act_queue(snapshot.act_policy, device))
    vector.call("restore_training_state", snapshot.environment)

    env._last_base_naction = snapshot.base_normalized_action.to(device, copy=True)
    return {key: value.to(device, copy=True) for key, value in snapshot.observation.items()}
