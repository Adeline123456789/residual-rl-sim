"""Continuous-phase action-chunk scheduling for PS-ResRL.

Generalizes :class:`cf_resrl.scheduler.AbsoluteTickScheduler` with a continuous
chunk phase cursor ``phi_t`` and wall-step anti-stall cap. Pose dimensions are
linearly interpolated while gripper dimensions strictly use nearest-neighbor
discrete lookup. Under ``phase_velocity == 1.0``, behavior is bit-exact with
``AbsoluteTickScheduler``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

from cf_resrl.scheduler import Action, ActionChunk, PendingRefresh, _action


ChunkProvider = Callable[[Any, int, int], Sequence[Sequence[float]]]


@dataclass(frozen=True)
class PhaseActionLedgerEntry:
    tick: int
    generation: int
    chunk_source_tick: int
    chunk_index: int
    phase_cursor: float
    chunk_wall_steps: int
    proposed_phase_velocity: float
    applied_phase_velocity: float
    used_fallback: bool
    base_action: Action
    applied_base_action: Action
    proposed_residual: Action
    applied_residual: Action
    proposed_action: Action
    applied_action: Action


@dataclass(frozen=True)
class PhaseSchedulerCounters:
    control_steps: int = 0
    refresh_attempts: int = 0
    refresh_requests: int = 0
    ignored_refresh_requests: int = 0
    act_calls: int = 0
    natural_phase_refills: int = 0
    stall_cap_triggers: int = 0


@dataclass(frozen=True)
class PhaseSchedulerSnapshot:
    schema_version: int
    latency_steps: int
    repeat_rule: str
    action_low: float
    action_high: float
    fallback_action: Action
    pose_dims: tuple[int, ...]
    gripper_dims: tuple[int, ...]
    nu_min: float
    nu_max: float
    max_chunk_wall_steps: int
    logical_tick: int
    phase_cursor: float
    chunk_wall_steps: int
    active_chunk: ActionChunk
    pending_refresh: PendingRefresh | None
    next_generation: int
    counters: PhaseSchedulerCounters
    ledger: tuple[PhaseActionLedgerEntry, ...]


def interpolate_chunk_action(
    actions: Sequence[Action],
    phase_cursor: float,
    *,
    pose_dims: Sequence[int] = (0, 1, 2, 3, 4, 5),
    gripper_dims: Sequence[int] = (6,),
) -> Action:
    """Evaluate a chunk at continuous phase ``phase_cursor`` with discrete gripper lookup."""
    k = len(actions)
    if k == 0:
        raise ValueError("action chunk must not be empty")
    if not math.isfinite(phase_cursor) or not (0.0 <= phase_cursor < k):
        raise ValueError(f"phase_cursor {phase_cursor} out of bounds [0, {k})")

    u = int(math.floor(phase_cursor))
    alpha = phase_cursor - u
    if alpha == 0.0:
        return actions[u]

    u_next = min(u + 1, k - 1)
    dim = len(actions[0])
    out = list(actions[u])
    for d in pose_dims:
        if 0 <= d < dim:
            out[d] = (1.0 - alpha) * actions[u][d] + alpha * actions[u_next][d]
    nearest_idx = min(int(math.floor(phase_cursor + 0.5)), k - 1)
    for d in gripper_dims:
        if 0 <= d < dim:
            out[d] = actions[nearest_idx][d]
    return tuple(out)


class ContinuousPhaseScheduler:
    """Execute cached chunks with continuous phase progression and stall protection."""

    REPEAT_RULE = "ignore_while_pending"

    def __init__(
        self,
        *,
        initial_chunk: Sequence[Sequence[float]],
        provider: ChunkProvider,
        latency_steps: int,
        fallback_action: Sequence[float],
        start_tick: int = 0,
        action_low: float = -1.0,
        action_high: float = 1.0,
        pose_dims: Sequence[int] = (0, 1, 2, 3, 4, 5),
        gripper_dims: Sequence[int] = (6,),
        nu_min: float = 0.0,
        nu_max: float = 2.0,
        max_chunk_wall_steps: int | None = None,
        async_refill: bool = False,
    ) -> None:
        if isinstance(latency_steps, bool) or latency_steps < 0:
            raise ValueError("latency_steps must be a nonnegative integer")
        if isinstance(start_tick, bool) or start_tick < 0:
            raise ValueError("start_tick must be a nonnegative integer")
        self._dimension = len(fallback_action)
        if self._dimension == 0:
            raise ValueError("actions must have at least one dimension")
        self._fallback_action = _action(
            fallback_action, dimension=self._dimension, name="fallback_action"
        )
        self._action_low = float(action_low)
        self._action_high = float(action_high)
        if not self._action_low < self._action_high:
            raise ValueError("action_low must be less than action_high")

        self._pose_dims = tuple(int(d) for d in pose_dims)
        self._gripper_dims = tuple(int(d) for d in gripper_dims)
        if set(self._pose_dims) & set(self._gripper_dims):
            raise ValueError("pose_dims and gripper_dims must be disjoint")
        if sorted((*self._pose_dims, *self._gripper_dims)) != list(range(self._dimension)):
            raise ValueError("pose_dims and gripper_dims must partition action dimensions")

        if not (math.isfinite(nu_min) and math.isfinite(nu_max)) or nu_min < 0.0 or nu_min > nu_max:
            raise ValueError("nu_min and nu_max must satisfy 0 <= nu_min <= nu_max")
        self._nu_min = float(nu_min)
        self._nu_max = float(nu_max)

        self._latency_steps = int(latency_steps)
        self._provider = provider
        self._logical_tick = int(start_tick)
        self._active_chunk = self._make_chunk(initial_chunk, start_tick, 0)
        self._phase_cursor = 0.0
        self._chunk_wall_steps = 0
        self._max_chunk_wall_steps = (
            int(max_chunk_wall_steps)
            if max_chunk_wall_steps is not None
            else 2 * len(self._active_chunk.actions)
        )
        if self._max_chunk_wall_steps <= 0:
            raise ValueError("max_chunk_wall_steps must be positive")

        self._pending_refresh: PendingRefresh | None = None
        self._next_generation = 1
        self._counters = PhaseSchedulerCounters()
        self._ledger: tuple[PhaseActionLedgerEntry, ...] = ()
        self._awaiting_after_step = False
        self._pending_applied_nu = 1.0
        self._async_refill = bool(async_refill)

    def _make_chunk(
        self,
        values: Sequence[Sequence[float]],
        source_tick: int,
        generation: int,
    ) -> ActionChunk:
        actions = tuple(
            _action(value, dimension=self._dimension, name="chunk action")
            for value in values
        )
        if not actions:
            raise ValueError("action chunks must not be empty")
        return ActionChunk(actions, int(source_tick), int(generation))

    @property
    def logical_tick(self) -> int:
        return self._logical_tick

    @property
    def phase_cursor(self) -> float:
        return self._phase_cursor

    @property
    def chunk_wall_steps(self) -> int:
        return self._chunk_wall_steps

    @property
    def max_chunk_wall_steps(self) -> int:
        return self._max_chunk_wall_steps

    @property
    def active_chunk(self) -> ActionChunk:
        return self._active_chunk

    @property
    def pending_refresh(self) -> PendingRefresh | None:
        return self._pending_refresh

    @property
    def counters(self) -> PhaseSchedulerCounters:
        return self._counters

    @property
    def ledger(self) -> tuple[PhaseActionLedgerEntry, ...]:
        return self._ledger

    def _activate_ready_chunk(self) -> None:
        pending = self._pending_refresh
        if pending is not None and self._logical_tick >= pending.ready_tick:
            self._active_chunk = pending.chunk
            self._pending_refresh = None
            self._phase_cursor = float(self._logical_tick - pending.chunk.source_tick)
            self._chunk_wall_steps = 0

    def refill_if_exhausted(self, observation: Any) -> bool:
        """Run nominal ACT inference at phase expiry or stall cap unless refresh pending."""
        if self._awaiting_after_step:
            raise RuntimeError("chunk refill requires a completed environment step")
        self._activate_ready_chunk()
        in_bounds = 0.0 <= self._phase_cursor < len(self._active_chunk.actions)
        within_stall_cap = self._chunk_wall_steps < self._max_chunk_wall_steps
        if self._pending_refresh is not None or (in_bounds and within_stall_cap):
            return False

        generation = self._next_generation
        values = self._provider(observation, self._logical_tick, generation)
        chunk = self._make_chunk(values, self._logical_tick, generation)
        self._next_generation += 1
        if self._async_refill and self._latency_steps > 0:
            self._pending_refresh = PendingRefresh(
                chunk=chunk,
                ready_tick=self._logical_tick + self._latency_steps,
            )
        else:
            self._active_chunk = chunk
            self._phase_cursor = 0.0
            self._chunk_wall_steps = 0
        stalled = in_bounds and not within_stall_cap
        counters = self._counters
        self._counters = PhaseSchedulerCounters(
            control_steps=counters.control_steps,
            refresh_attempts=counters.refresh_attempts,
            refresh_requests=counters.refresh_requests,
            ignored_refresh_requests=counters.ignored_refresh_requests,
            act_calls=counters.act_calls + 1,
            natural_phase_refills=counters.natural_phase_refills + int(not stalled),
            stall_cap_triggers=counters.stall_cap_triggers + int(stalled),
        )
        return True

    def current_base_action(self) -> Action:
        if self._awaiting_after_step:
            raise RuntimeError("current base action requires a completed environment step")
        self._activate_ready_chunk()
        if 0.0 <= self._phase_cursor < len(self._active_chunk.actions):
            return interpolate_chunk_action(
                self._active_chunk.actions,
                self._phase_cursor,
                pose_dims=self._pose_dims,
                gripper_dims=self._gripper_dims,
            )
        return self._fallback_action

    def chunk_phase_context(self) -> tuple[float, ...]:
        """Return 9-D deployment-visible context: [phi/H, wall/W_max, b(phi+1)-b(phi)]."""
        if self._awaiting_after_step:
            raise RuntimeError("chunk_phase_context requires a completed environment step")
        self._activate_ready_chunk()
        k = len(self._active_chunk.actions)
        phi_norm = min(max(self._phase_cursor, 0.0), float(k)) / float(k)
        wall_norm = min(max(self._chunk_wall_steps, 0), self._max_chunk_wall_steps) / float(
            self._max_chunk_wall_steps
        )
        if 0.0 <= self._phase_cursor < k:
            curr = interpolate_chunk_action(
                self._active_chunk.actions,
                self._phase_cursor,
                pose_dims=self._pose_dims,
                gripper_dims=self._gripper_dims,
            )
            next_phi = min(self._phase_cursor + 1.0, float(k - 1))
            nxt = interpolate_chunk_action(
                self._active_chunk.actions,
                next_phi,
                pose_dims=self._pose_dims,
                gripper_dims=self._gripper_dims,
            )
            diff = tuple(b_next - b_curr for b_next, b_curr in zip(nxt, curr))
        else:
            diff = (0.0,) * self._dimension
        return (phi_norm, wall_norm, *diff)

    def action(
        self,
        proposed_residual: Sequence[float],
        *,
        phase_velocity: float = 1.0,
    ) -> PhaseActionLedgerEntry:
        """Return and record the action and phase velocity for the current step."""
        if self._awaiting_after_step:
            raise RuntimeError("after_step must follow each scheduled action")
        proposed_nu = float(phase_velocity)
        if not math.isfinite(proposed_nu):
            raise ValueError("phase_velocity must be finite")
        applied_nu = min(self._nu_max, max(self._nu_min, proposed_nu))

        base_action = self.current_base_action()
        in_bounds = 0.0 <= self._phase_cursor < len(self._active_chunk.actions)
        used_fallback = not in_bounds
        chunk_index = (
            int(math.floor(self._phase_cursor))
            if in_bounds
            else self._logical_tick - self._active_chunk.source_tick
        )
        residual = _action(
            proposed_residual,
            dimension=self._dimension,
            name="proposed_residual",
        )
        proposed_action = tuple(
            base + delta for base, delta in zip(base_action, residual)
        )
        applied_base_action = tuple(
            min(self._action_high, max(self._action_low, value))
            for value in base_action
        )
        applied_action = tuple(
            min(self._action_high, max(self._action_low, value))
            for value in proposed_action
        )
        applied_residual = tuple(
            applied - base
            for applied, base in zip(applied_action, applied_base_action)
        )
        entry = PhaseActionLedgerEntry(
            tick=self._logical_tick,
            generation=self._active_chunk.generation,
            chunk_source_tick=self._active_chunk.source_tick,
            chunk_index=chunk_index,
            phase_cursor=self._phase_cursor,
            chunk_wall_steps=self._chunk_wall_steps,
            proposed_phase_velocity=proposed_nu,
            applied_phase_velocity=applied_nu,
            used_fallback=used_fallback,
            base_action=base_action,
            applied_base_action=applied_base_action,
            proposed_residual=residual,
            applied_residual=applied_residual,
            proposed_action=proposed_action,
            applied_action=applied_action,
        )
        self._ledger = (*self._ledger, entry)
        self._pending_applied_nu = applied_nu
        self._awaiting_after_step = True
        return entry

    def after_step(self, observation: Any, *, request_refresh: bool) -> None:
        """Advance logical tick, phase cursor, wall counter, and optionally request refresh."""
        if not self._awaiting_after_step:
            raise RuntimeError("an action must be submitted before after_step")
        self._awaiting_after_step = False
        self._logical_tick += 1
        self._phase_cursor += self._pending_applied_nu
        self._chunk_wall_steps += 1
        self._activate_ready_chunk()
        self._counters = PhaseSchedulerCounters(
            control_steps=self._counters.control_steps + 1,
            refresh_attempts=self._counters.refresh_attempts,
            refresh_requests=self._counters.refresh_requests,
            ignored_refresh_requests=self._counters.ignored_refresh_requests,
            act_calls=self._counters.act_calls,
            natural_phase_refills=self._counters.natural_phase_refills,
            stall_cap_triggers=self._counters.stall_cap_triggers,
        )
        if not request_refresh:
            return

        counters = self._counters
        if self._pending_refresh is not None:
            self._counters = PhaseSchedulerCounters(
                control_steps=counters.control_steps,
                refresh_attempts=counters.refresh_attempts + 1,
                refresh_requests=counters.refresh_requests,
                ignored_refresh_requests=counters.ignored_refresh_requests + 1,
                act_calls=counters.act_calls,
                natural_phase_refills=counters.natural_phase_refills,
                stall_cap_triggers=counters.stall_cap_triggers,
            )
            return

        source_tick = self._logical_tick
        generation = self._next_generation
        self._next_generation += 1
        values = self._provider(observation, source_tick, generation)
        chunk = self._make_chunk(values, source_tick, generation)
        self._pending_refresh = PendingRefresh(
            chunk=chunk,
            ready_tick=source_tick + self._latency_steps,
        )
        self._counters = PhaseSchedulerCounters(
            control_steps=counters.control_steps,
            refresh_attempts=counters.refresh_attempts + 1,
            refresh_requests=counters.refresh_requests + 1,
            ignored_refresh_requests=counters.ignored_refresh_requests,
            act_calls=counters.act_calls + 1,
            natural_phase_refills=counters.natural_phase_refills,
            stall_cap_triggers=counters.stall_cap_triggers,
        )

    def snapshot(self) -> PhaseSchedulerSnapshot:
        """Capture phase-scheduler timing state at a control-step boundary."""
        if self._awaiting_after_step:
            raise RuntimeError("scheduler snapshots require a completed env step")
        return PhaseSchedulerSnapshot(
            schema_version=1,
            latency_steps=self._latency_steps,
            repeat_rule=self.REPEAT_RULE,
            action_low=self._action_low,
            action_high=self._action_high,
            fallback_action=self._fallback_action,
            pose_dims=self._pose_dims,
            gripper_dims=self._gripper_dims,
            nu_min=self._nu_min,
            nu_max=self._nu_max,
            max_chunk_wall_steps=self._max_chunk_wall_steps,
            logical_tick=self._logical_tick,
            phase_cursor=self._phase_cursor,
            chunk_wall_steps=self._chunk_wall_steps,
            active_chunk=self._active_chunk,
            pending_refresh=self._pending_refresh,
            next_generation=self._next_generation,
            counters=self._counters,
            ledger=self._ledger,
        )

    def restore(self, snapshot: PhaseSchedulerSnapshot) -> None:
        """Restore a reusable phase-scheduler snapshot without calling ACT."""
        expected = (
            1,
            self._latency_steps,
            self.REPEAT_RULE,
            self._action_low,
            self._action_high,
            self._fallback_action,
            self._pose_dims,
            self._gripper_dims,
            self._nu_min,
            self._nu_max,
            self._max_chunk_wall_steps,
        )
        observed = (
            snapshot.schema_version,
            snapshot.latency_steps,
            snapshot.repeat_rule,
            snapshot.action_low,
            snapshot.action_high,
            snapshot.fallback_action,
            snapshot.pose_dims,
            snapshot.gripper_dims,
            snapshot.nu_min,
            snapshot.nu_max,
            snapshot.max_chunk_wall_steps,
        )
        if observed != expected:
            raise ValueError("phase scheduler snapshot configuration mismatch")
        self._logical_tick = snapshot.logical_tick
        self._phase_cursor = snapshot.phase_cursor
        self._chunk_wall_steps = snapshot.chunk_wall_steps
        self._active_chunk = snapshot.active_chunk
        self._pending_refresh = snapshot.pending_refresh
        self._next_generation = snapshot.next_generation
        self._counters = snapshot.counters
        self._ledger = snapshot.ledger
        self._awaiting_after_step = False
        self._pending_applied_nu = 1.0
