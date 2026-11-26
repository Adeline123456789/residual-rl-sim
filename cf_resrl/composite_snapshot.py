"""Composite Square snapshots for environment and frozen policy control state.

The environment snapshot remains owned by :mod:`cf_resrl.snapshot`; this module
adds the control state that must travel with it at a branch boundary. Restoring
never calls ``reset``, ACT inference, or the scheduler's chunk provider.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

from cf_resrl.scheduler import SchedulerSnapshot
from cf_resrl.snapshot import (
    SquareEnvSnapshot,
    capture_square_snapshot,
    restore_square_snapshot,
)


@dataclass(frozen=True)
class ActQueueItemSnapshot:
    value: Any
    shape: tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True)
class ActQueueSnapshot:
    maxlen: int
    items: tuple[ActQueueItemSnapshot, ...]


@dataclass(frozen=True)
class ActPolicySnapshot:
    n_action_steps: int
    config_device: str
    queues: tuple[ActQueueSnapshot, ...]


@dataclass(frozen=True)
class TorchRngSnapshot:
    cuda_available: bool
    cuda_device_count: int
    cpu_state: Any
    cuda_states: tuple[Any, ...]


@dataclass(frozen=True)
class CompositeSnapshot:
    schema_version: int
    scope: str
    policy_identity: str
    provider_identity: str
    environment: SquareEnvSnapshot
    scheduler: SchedulerSnapshot
    act_policy: ActPolicySnapshot
    torch_rng: TorchRngSnapshot
    observation_tick: int
    current_observation: Any
    residual_history: tuple[Any, ...]
    observation_history: tuple[Any, ...] = ()
    applied_action_ledger: tuple[Any, ...] = ()
    actor_observation: Any = None
    residual_policy_identity: str | None = None
    normalization_identity: str | None = None
    collector_rng_kind: str | None = None
    collector_rng_state: Any = None


@dataclass(frozen=True)
class RestoredControlContext:
    observation_tick: int
    current_observation: Any
    residual_history: tuple[Any, ...]
    observation_history: tuple[Any, ...] = ()
    applied_action_ledger: tuple[Any, ...] = ()
    actor_observation: Any = None


def _torch(torch_api: Any | None) -> Any:
    if torch_api is None:
        import torch as torch_api

    return torch_api


def _clone(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return deepcopy(value)


def _identity(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty registered identity")
    return value


def _collector_state(rng: Any | None) -> tuple[str | None, Any]:
    if rng is None:
        return None, None
    if hasattr(rng, "bit_generator"):
        generator = rng.bit_generator
        return type(generator).__qualname__, deepcopy(generator.state)
    return "python_random", deepcopy(rng.getstate())


def _validate_resfit_context(
    residual_policy_identity: str | None,
    normalization_identity: str | None,
    actor_observation: Any,
) -> None:
    if (
        residual_policy_identity is None
        and normalization_identity is None
        and actor_observation is None
    ):
        return
    _identity(residual_policy_identity, name="residual_policy_identity")
    _identity(normalization_identity, name="normalization_identity")
    if actor_observation is None:
        raise ValueError("ResFiT snapshot must include the normalized actor observation")


def _item_snapshot(value: Any) -> ActQueueItemSnapshot:
    if not all(hasattr(value, name) for name in ("shape", "dtype", "device")):
        raise TypeError("ACT queue entries must be tensor-like values")
    return ActQueueItemSnapshot(
        value=_clone(value),
        shape=tuple(int(size) for size in value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
    )


def _validate_act_mode(policy: Any) -> None:
    if getattr(policy.config, "temporal_ensemble_coeff", None) is not None:
        raise NotImplementedError(
            "composite snapshots support the registered ACT action-queue mode only"
        )


def _capture_act_policy(policy: Any) -> ActPolicySnapshot:
    _validate_act_mode(policy)
    if not hasattr(policy, "_action_queues"):
        raise ValueError("ACT action queues have not been initialized")
    n_action_steps = int(policy.config.n_action_steps)
    queues = []
    for queue in policy._action_queues:
        if queue.maxlen != n_action_steps:
            raise ValueError("ACT queue maxlen does not match n_action_steps")
        queues.append(
            ActQueueSnapshot(
                maxlen=int(queue.maxlen),
                items=tuple(_item_snapshot(item) for item in queue),
            )
        )
    if not queues:
        raise ValueError("ACT policy has no environment queues")
    return ActPolicySnapshot(
        n_action_steps=n_action_steps,
        config_device=str(policy.config.device),
        queues=tuple(queues),
    )


def _metadata(value: Any) -> tuple[tuple[int, ...], str, str]:
    return (
        tuple(int(size) for size in value.shape),
        str(value.dtype),
        str(value.device),
    )


def _restore_act_policy(policy: Any, snapshot: ActPolicySnapshot) -> None:
    _validate_act_mode(policy)
    if int(policy.config.n_action_steps) != snapshot.n_action_steps:
        raise ValueError("ACT n_action_steps does not match the snapshot")
    if str(policy.config.device) != snapshot.config_device:
        raise ValueError("ACT config device does not match the snapshot")
    if not hasattr(policy, "_action_queues"):
        raise ValueError("ACT action queues have not been initialized")
    if len(policy._action_queues) != len(snapshot.queues):
        raise ValueError("ACT queue topology does not match the snapshot")

    restored_queues = []
    for current, saved in zip(policy._action_queues, snapshot.queues):
        if current.maxlen != saved.maxlen:
            raise ValueError("ACT queue maxlen does not match the snapshot")
        expected_metadata = {
            (item.shape, item.dtype, item.device) for item in saved.items
        }
        for item in current:
            if expected_metadata and _metadata(item) not in expected_metadata:
                raise ValueError("ACT queue tensor shape, dtype, or device mismatch")
        restored_queues.append(
            deque((_clone(item.value) for item in saved.items), maxlen=saved.maxlen)
        )
    policy._action_queues = restored_queues


def _capture_torch_rng(torch_api: Any) -> TorchRngSnapshot:
    cuda_available = bool(torch_api.cuda.is_available())
    cuda_device_count = int(torch_api.cuda.device_count())
    if cuda_device_count < 0:
        raise ValueError("Torch CUDA device count must be nonnegative")
    cuda_states = (
        tuple(_clone(state) for state in torch_api.cuda.get_rng_state_all())
        if cuda_available
        else ()
    )
    if len(cuda_states) != cuda_device_count:
        raise ValueError("Torch CUDA RNG state count does not match device count")
    return TorchRngSnapshot(
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cpu_state=_clone(torch_api.get_rng_state()),
        cuda_states=cuda_states,
    )


def _restore_torch_rng(torch_api: Any, snapshot: TorchRngSnapshot) -> None:
    cuda_available = bool(torch_api.cuda.is_available())
    cuda_device_count = int(torch_api.cuda.device_count())
    if cuda_available != snapshot.cuda_available:
        raise ValueError("Torch CUDA availability does not match the snapshot")
    if cuda_device_count != snapshot.cuda_device_count:
        raise ValueError("Torch CUDA device count does not match the snapshot")
    if len(snapshot.cuda_states) != snapshot.cuda_device_count:
        raise ValueError("snapshot CUDA RNG state count does not match device count")
    torch_api.set_rng_state(_clone(snapshot.cpu_state))
    if snapshot.cuda_available:
        torch_api.cuda.set_rng_state_all(
            [_clone(state) for state in snapshot.cuda_states]
        )


def capture_composite_snapshot(
    wrapper: Any,
    scheduler: Any,
    act_policy: Any,
    *,
    policy_identity: str,
    provider_identity: str,
    observation_tick: int,
    residual_history: Sequence[Any] = (),
    observation_history: Sequence[Any] = (),
    applied_action_ledger: Sequence[Any] = (),
    actor_observation: Any = None,
    residual_policy_identity: str | None = None,
    normalization_identity: str | None = None,
    collector_rng: Any | None = None,
    mujoco_api: Any | None = None,
    torch_api: Any | None = None,
) -> CompositeSnapshot:
    """Capture a reusable full control-boundary snapshot.

    The authoritative current observation is always the environment wrapper's
    ``_last_obs`` value captured in the same snapshot operation. No caller-
    supplied raw observation is accepted. A frozen ResFiT caller also supplies
    the normalized actor observation and the identities of the weights and
    normalization parameters used to construct it.
    """
    policy_identity = _identity(policy_identity, name="policy_identity")
    provider_identity = _identity(provider_identity, name="provider_identity")
    _validate_resfit_context(
        residual_policy_identity, normalization_identity, actor_observation
    )
    collector_rng_kind, collector_rng_state = _collector_state(collector_rng)
    environment_snapshot = capture_square_snapshot(wrapper, mujoco_api=mujoco_api)
    if "_last_obs" not in environment_snapshot.wrapper_state:
        raise ValueError("environment snapshot is missing wrapper _last_obs")
    scheduler_snapshot = scheduler.snapshot()
    if int(observation_tick) != scheduler_snapshot.logical_tick:
        raise ValueError("observation tick must match the scheduler logical tick")
    torch_api = _torch(torch_api)
    return CompositeSnapshot(
        schema_version=1,
        scope=(
            "square_environment__act_queue__absolute_tick_scheduler__resfit_context"
            if residual_policy_identity is not None
            else "square_environment__act_queue__absolute_tick_scheduler"
        ),
        policy_identity=policy_identity,
        provider_identity=provider_identity,
        environment=environment_snapshot,
        scheduler=scheduler_snapshot,
        act_policy=_capture_act_policy(act_policy),
        torch_rng=_capture_torch_rng(torch_api),
        observation_tick=int(observation_tick),
        current_observation=_clone(environment_snapshot.wrapper_state["_last_obs"]),
        residual_history=tuple(_clone(value) for value in residual_history),
        observation_history=tuple(_clone(value) for value in observation_history),
        applied_action_ledger=tuple(_clone(value) for value in applied_action_ledger),
        actor_observation=_clone(actor_observation),
        residual_policy_identity=residual_policy_identity,
        normalization_identity=normalization_identity,
        collector_rng_kind=collector_rng_kind,
        collector_rng_state=collector_rng_state,
    )


def restore_composite_snapshot(
    wrapper: Any,
    scheduler: Any,
    act_policy: Any,
    snapshot: CompositeSnapshot,
    *,
    policy_identity: str,
    provider_identity: str,
    residual_policy_identity: str | None = None,
    normalization_identity: str | None = None,
    collector_rng: Any | None = None,
    mujoco_api: Any | None = None,
    torch_api: Any | None = None,
) -> RestoredControlContext:
    """Restore environment, ACT queues, scheduler, RNG, then control context."""
    if snapshot.schema_version != 1:
        raise ValueError("unsupported composite snapshot schema")
    if snapshot.observation_tick != snapshot.scheduler.logical_tick:
        raise ValueError("snapshot observation and scheduler ticks are inconsistent")
    policy_identity = _identity(policy_identity, name="policy_identity")
    provider_identity = _identity(provider_identity, name="provider_identity")
    if policy_identity != snapshot.policy_identity:
        raise ValueError("ACT policy identity does not match the snapshot")
    if provider_identity != snapshot.provider_identity:
        raise ValueError("scheduler provider identity does not match the snapshot")
    if residual_policy_identity != snapshot.residual_policy_identity:
        raise ValueError("residual policy identity does not match the snapshot")
    if normalization_identity != snapshot.normalization_identity:
        raise ValueError("normalization identity does not match the snapshot")
    _validate_resfit_context(
        residual_policy_identity, normalization_identity, snapshot.actor_observation
    )
    collector_rng_kind, _ = _collector_state(collector_rng)
    if collector_rng_kind != snapshot.collector_rng_kind:
        raise ValueError("collector RNG kind does not match the snapshot")
    torch_api = _torch(torch_api)

    restore_square_snapshot(wrapper, snapshot.environment, mujoco_api=mujoco_api)
    _restore_act_policy(act_policy, snapshot.act_policy)
    scheduler.restore(snapshot.scheduler)
    _restore_torch_rng(torch_api, snapshot.torch_rng)
    if collector_rng_kind == "python_random":
        collector_rng.setstate(deepcopy(snapshot.collector_rng_state))
    elif collector_rng_kind is not None:
        collector_rng.bit_generator.state = deepcopy(snapshot.collector_rng_state)
    return RestoredControlContext(
        observation_tick=snapshot.observation_tick,
        current_observation=_clone(snapshot.current_observation),
        residual_history=tuple(_clone(value) for value in snapshot.residual_history),
        observation_history=tuple(_clone(value) for value in snapshot.observation_history),
        applied_action_ledger=tuple(_clone(value) for value in snapshot.applied_action_ledger),
        actor_observation=_clone(snapshot.actor_observation),
    )
