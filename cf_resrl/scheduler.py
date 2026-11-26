"""Absolute-tick action-chunk scheduling for CF-ResRL branch rollouts.

The scheduler owns policy timing only. The environment is stepped by the caller,
which must request refresh through :meth:`after_step` after the current action has
actually been submitted. This keeps KEEP and REFRESH branches identical at the
intervention tick and makes inference latency advance logical control time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence


Action = tuple[float, ...]
ChunkProvider = Callable[[Any, int, int], Sequence[Sequence[float]]]


def _action(values: Sequence[float], *, dimension: int, name: str) -> Action:
    result = tuple(float(value) for value in values)
    if len(result) != dimension:
        raise ValueError(f"{name} must have dimension {dimension}")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class ActionChunk:
    actions: tuple[Action, ...]
    source_tick: int
    generation: int


@dataclass(frozen=True)
class PendingRefresh:
    chunk: ActionChunk
    ready_tick: int


@dataclass(frozen=True)
class ActionLedgerEntry:
    tick: int
    generation: int
    chunk_source_tick: int
    chunk_index: int
    used_fallback: bool
    base_action: Action
    applied_base_action: Action
    proposed_residual: Action
    applied_residual: Action
    proposed_action: Action
    applied_action: Action


@dataclass(frozen=True)
class SchedulerCounters:
    control_steps: int = 0
    refresh_attempts: int = 0
    refresh_requests: int = 0
    ignored_refresh_requests: int = 0
    act_calls: int = 0


@dataclass(frozen=True)
class SchedulerSnapshot:
    schema_version: int
    latency_steps: int
    repeat_rule: str
    action_low: float
    action_high: float
    fallback_action: Action
    logical_tick: int
    active_chunk: ActionChunk
    pending_refresh: PendingRefresh | None
    next_generation: int
    counters: SchedulerCounters
    ledger: tuple[ActionLedgerEntry, ...]


class AbsoluteTickScheduler:
    """Execute cached chunks and install refreshes at their absolute indices.

    ``provider(observation, source_tick, generation)`` returns a complete action
    chunk aligned to ``source_tick``. A refresh with latency ``tau`` becomes
    available at ``source_tick + tau``. Therefore, for ``tau=2``, the first
    action used from that chunk is index 2 rather than index 0.
    """

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
        self._latency_steps = int(latency_steps)
        self._provider = provider
        self._logical_tick = int(start_tick)
        self._active_chunk = self._make_chunk(initial_chunk, start_tick, 0)
        self._pending_refresh: PendingRefresh | None = None
        self._next_generation = 1
        self._counters = SchedulerCounters()
        self._ledger: tuple[ActionLedgerEntry, ...] = ()
        self._awaiting_after_step = False

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
    def active_chunk(self) -> ActionChunk:
        return self._active_chunk

    @property
    def pending_refresh(self) -> PendingRefresh | None:
        return self._pending_refresh

    @property
    def counters(self) -> SchedulerCounters:
        return self._counters

    @property
    def ledger(self) -> tuple[ActionLedgerEntry, ...]:
        return self._ledger

    def _activate_ready_chunk(self) -> None:
        pending = self._pending_refresh
        if pending is not None and self._logical_tick >= pending.ready_tick:
            self._active_chunk = pending.chunk
            self._pending_refresh = None

    def refill_if_exhausted(self, observation: Any) -> bool:
        """Run the nominal ACT inference at chunk expiry unless a refresh is pending."""
        if self._awaiting_after_step:
            raise RuntimeError("chunk refill requires a completed environment step")
        self._activate_ready_chunk()
        index = self._logical_tick - self._active_chunk.source_tick
        if self._pending_refresh is not None or 0 <= index < len(self._active_chunk.actions):
            return False
        generation = self._next_generation
        values = self._provider(observation, self._logical_tick, generation)
        self._active_chunk = self._make_chunk(values, self._logical_tick, generation)
        self._next_generation += 1
        counters = self._counters
        self._counters = SchedulerCounters(
            control_steps=counters.control_steps,
            refresh_attempts=counters.refresh_attempts,
            refresh_requests=counters.refresh_requests,
            ignored_refresh_requests=counters.ignored_refresh_requests,
            act_calls=counters.act_calls + 1,
        )
        return True

    def current_base_action(self) -> Action:
        if self._awaiting_after_step:
            raise RuntimeError("current base action requires a completed environment step")
        self._activate_ready_chunk()
        index = self._logical_tick - self._active_chunk.source_tick
        if 0 <= index < len(self._active_chunk.actions):
            return self._active_chunk.actions[index]
        return self._fallback_action

    def action(self, proposed_residual: Sequence[float]) -> ActionLedgerEntry:
        """Return and record the action that the caller must submit now."""
        if self._awaiting_after_step:
            raise RuntimeError("after_step must follow each scheduled action")
        base_action = self.current_base_action()
        chunk_index = self._logical_tick - self._active_chunk.source_tick
        used_fallback = not 0 <= chunk_index < len(self._active_chunk.actions)
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
        entry = ActionLedgerEntry(
            tick=self._logical_tick,
            generation=self._active_chunk.generation,
            chunk_source_tick=self._active_chunk.source_tick,
            chunk_index=chunk_index,
            used_fallback=used_fallback,
            base_action=base_action,
            applied_base_action=applied_base_action,
            proposed_residual=residual,
            applied_residual=applied_residual,
            proposed_action=proposed_action,
            applied_action=applied_action,
        )
        self._ledger = (*self._ledger, entry)
        self._awaiting_after_step = True
        return entry

    def after_step(self, observation: Any, *, request_refresh: bool) -> None:
        """Advance one control tick and optionally request a new chunk.

        The caller invokes this only after ``env.step(applied_action)`` returns;
        the supplied observation is consequently aligned to the new source tick.
        """
        if not self._awaiting_after_step:
            raise RuntimeError("an action must be submitted before after_step")
        self._awaiting_after_step = False
        self._logical_tick += 1
        self._activate_ready_chunk()
        self._counters = SchedulerCounters(
            control_steps=self._counters.control_steps + 1,
            refresh_attempts=self._counters.refresh_attempts,
            refresh_requests=self._counters.refresh_requests,
            ignored_refresh_requests=self._counters.ignored_refresh_requests,
            act_calls=self._counters.act_calls,
        )
        if not request_refresh:
            return

        counters = self._counters
        if self._pending_refresh is not None:
            self._counters = SchedulerCounters(
                control_steps=counters.control_steps,
                refresh_attempts=counters.refresh_attempts + 1,
                refresh_requests=counters.refresh_requests,
                ignored_refresh_requests=counters.ignored_refresh_requests + 1,
                act_calls=counters.act_calls,
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
        self._counters = SchedulerCounters(
            control_steps=counters.control_steps,
            refresh_attempts=counters.refresh_attempts + 1,
            refresh_requests=counters.refresh_requests + 1,
            ignored_refresh_requests=counters.ignored_refresh_requests,
            act_calls=counters.act_calls + 1,
        )

    def snapshot(self) -> SchedulerSnapshot:
        """Capture policy timing state at a control-step boundary."""
        if self._awaiting_after_step:
            raise RuntimeError("scheduler snapshots require a completed env step")
        return SchedulerSnapshot(
            schema_version=1,
            latency_steps=self._latency_steps,
            repeat_rule=self.REPEAT_RULE,
            action_low=self._action_low,
            action_high=self._action_high,
            fallback_action=self._fallback_action,
            logical_tick=self._logical_tick,
            active_chunk=self._active_chunk,
            pending_refresh=self._pending_refresh,
            next_generation=self._next_generation,
            counters=self._counters,
            ledger=self._ledger,
        )

    def restore(self, snapshot: SchedulerSnapshot) -> None:
        """Restore one reusable scheduler snapshot without calling ACT."""
        expected = (
            1,
            self._latency_steps,
            self.REPEAT_RULE,
            self._action_low,
            self._action_high,
            self._fallback_action,
        )
        observed = (
            snapshot.schema_version,
            snapshot.latency_steps,
            snapshot.repeat_rule,
            snapshot.action_low,
            snapshot.action_high,
            snapshot.fallback_action,
        )
        if observed != expected:
            raise ValueError("scheduler snapshot configuration mismatch")
        self._logical_tick = snapshot.logical_tick
        self._active_chunk = snapshot.active_chunk
        self._pending_refresh = snapshot.pending_refresh
        self._next_generation = snapshot.next_generation
        self._counters = snapshot.counters
        self._ledger = snapshot.ledger
        self._awaiting_after_step = False
