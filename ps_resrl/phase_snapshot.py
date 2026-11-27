"""Composite Square snapshots for continuous-phase PS-ResRL branches."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

from cf_resrl.composite_snapshot import (
    ActPolicySnapshot,
    RestoredControlContext,
    TorchRngSnapshot,
    _capture_act_policy,
    _capture_torch_rng,
    _clone,
    _collector_state,
    _identity,
    _restore_act_policy,
    _restore_torch_rng,
    _torch,
    _validate_resfit_context,
)
from cf_resrl.snapshot import (
    SquareEnvSnapshot,
    capture_square_snapshot,
    restore_square_snapshot,
)
from ps_resrl.phase_scheduler import PhaseSchedulerSnapshot


@dataclass(frozen=True)
class PhaseCompositeSnapshot:
    schema_version: int
    scope: str
    policy_identity: str
    provider_identity: str
    environment: SquareEnvSnapshot
    scheduler: PhaseSchedulerSnapshot
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


def capture_phase_composite_snapshot(
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
) -> PhaseCompositeSnapshot:
    """Capture a reusable full control-boundary snapshot including phase cursor."""
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
    return PhaseCompositeSnapshot(
        schema_version=1,
        scope="square_environment__act_queue__continuous_phase_scheduler__resfit_context",
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


def restore_phase_composite_snapshot(
    wrapper: Any,
    scheduler: Any,
    act_policy: Any,
    snapshot: PhaseCompositeSnapshot,
    *,
    policy_identity: str,
    provider_identity: str,
    residual_policy_identity: str | None = None,
    normalization_identity: str | None = None,
    collector_rng: Any | None = None,
    mujoco_api: Any | None = None,
    torch_api: Any | None = None,
) -> RestoredControlContext:
    """Restore environment, ACT queues, continuous phase scheduler, RNG, and context."""
    if snapshot.schema_version != 1:
        raise ValueError("unsupported phase composite snapshot schema")
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
