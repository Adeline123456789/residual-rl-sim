"""Small, dependency-free execution contracts for PA-ResWM experiments.

These utilities do NOT implement ResFiT, train a policy, or certify safety.
Tick t's applied action produces observation t+1. Chunk actions are indexed
relative to the SOURCE observation tick, not the inference completion tick.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

Vector = tuple[float, ...]


def vector(values: Sequence[float]) -> Vector:
    result = tuple(float(x) for x in values)
    if not result or not all(math.isfinite(x) for x in result):
        raise ValueError("A vector must be nonempty and finite")
    return result


@dataclass(frozen=True)
class AppliedAction:
    episode_id: str
    tick: int
    base_version: str
    base: Vector
    proposed: Vector
    applied: Vector

    def __post_init__(self) -> None:
        if self.tick < 0 or not self.episode_id or not self.base_version:
            raise ValueError("Invalid identity or tick")
        for field in ("base", "proposed", "applied"):
            object.__setattr__(self, field, vector(getattr(self, field)))
        if not (len(self.base) == len(self.proposed) == len(self.applied)):
            raise ValueError("Action dimensions differ")


class ActionLedger:
    """Append-only actual command history, scoped to ONE episode.

    `applied` means the command passed to env.step / accepted by the actuator
    interface, after clipping and filters. It is not measured joint motion.
    """

    def __init__(self, episode_id: str, start_tick: int = 0) -> None:
        if not episode_id or start_tick < 0:
            raise ValueError("Invalid episode or starting tick")
        self.episode_id = episode_id
        self.start_tick = start_tick
        self._entries: list[AppliedAction] = []
        self._dimension: int | None = None

    def append(self, entry: AppliedAction) -> None:
        if entry.episode_id != self.episode_id:
            raise ValueError("Cross-episode action")
        if entry.tick != self.start_tick + len(self._entries):
            raise ValueError("Actions must be contiguous, ordered, and unique")
        if self._dimension is not None and len(entry.applied) != self._dimension:
            raise ValueError("Action dimension changed within episode")
        self._dimension = len(entry.applied)
        self._entries.append(entry)

    def window(self, start: int, stop: int) -> tuple[AppliedAction, ...]:
        """Return [start, stop); fail closed on missing/future history."""
        end = self.start_tick + len(self._entries)
        if not self.start_tick <= start <= stop <= end:
            raise ValueError("Requested action history is unavailable")
        return tuple(self._entries[start-self.start_tick:stop-self.start_tick])


def align_prediction(
    stale_state: Sequence[float], source_tick: int, now_tick: int,
    ledger: ActionLedger,
    forward: Callable[[Vector, Vector], Sequence[float]],
) -> Vector:
    """Roll a MODEL estimate forward with APPLIED, never planned, actions.

    This predicts an estimate, not the true current state. A deployment must
    additionally handle uncertainty, current proprioception, and late frames.
    """
    estimate = vector(stale_state)
    for entry in ledger.window(source_tick, now_tick):
        estimate = vector(forward(estimate, entry.applied))
    return estimate


@dataclass(frozen=True)
class ActionChunk:
    episode_id: str
    base_version: str
    generation: int
    source_tick: int
    ready_tick: int
    actions: tuple[Vector, ...]

    def __post_init__(self) -> None:
        if (not self.episode_id or not self.base_version or self.generation < 0
                or self.source_tick < 0 or self.ready_tick < self.source_tick):
            raise ValueError("Invalid chunk metadata")
        actions = tuple(vector(a) for a in self.actions)
        if not actions or any(len(a) != len(actions[0]) for a in actions):
            raise ValueError("Empty chunk or inconsistent action dimensions")
        object.__setattr__(self, "actions", actions)


class ChunkQueue:
    """Timestamped base queue with immutable already-committed prefix.

    Physical actuator FIFO delay is NOT simulated by this class. The caller
    supplies committed_until and must model actuator delay separately.
    """

    def __init__(self, episode_id: str, base_version: str) -> None:
        self.episode_id = episode_id
        self.base_version = base_version
        self.last_generation = -1
        self.last_now = -1
        self._dimension: int | None = None
        self._actions: dict[int, Vector] = {}

    def install(self, chunk: ActionChunk, now: int,
                committed_until: int | None = None) -> int:
        cutoff = now if committed_until is None else committed_until
        if now < 0 or now < self.last_now or cutoff < now:
            raise ValueError("Nonmonotonic clock or invalid committed prefix")
        if chunk.episode_id != self.episode_id:
            raise ValueError("Chunk belongs to a different episode")
        if chunk.base_version != self.base_version:
            raise ValueError("Chunk uses an unpromoted base version")
        if chunk.ready_tick > now:
            raise ValueError("Cannot use unfinished inference")
        if self._dimension is not None and len(chunk.actions[0]) != self._dimension:
            raise ValueError("Action dimension changed")
        self.last_now = now
        if chunk.generation <= self.last_generation:
            return 0  # A late, out-of-order result cannot replace newer work.
        self.last_generation = chunk.generation
        self._dimension = len(chunk.actions[0])
        # A new chunk invalidates only the uncommitted suffix. No silent reuse
        # beyond its horizon: missing slots must trigger an explicit fallback.
        self._actions = {t: a for t, a in self._actions.items() if now <= t < cutoff}
        inserted = 0
        for index, action in enumerate(chunk.actions):
            tick = chunk.source_tick + index
            if tick >= cutoff:
                self._actions[tick] = action
                inserted += 1
        return inserted

    def action_at(self, tick: int) -> Vector:
        if tick < self.last_now or tick not in self._actions:
            raise ValueError("No valid action: request refresh or explicit fallback")
        return self._actions[tick]


def recenter_residual(teacher_applied: Sequence[float],
                      new_base: Sequence[float],
                      radius: float) -> Vector:
    """Construct an exact coordinate-change label, rejecting infeasible labels.

    Do not zero the residual after base distillation. Do not silently clip this
    target: clipping would destroy the claimed action-preserving identity.
    """
    teacher, base = vector(teacher_applied), vector(new_base)
    if len(teacher) != len(base):
        raise ValueError("Action dimensions differ")
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("Radius must be finite and nonnegative")
    residual = tuple(t-b for t, b in zip(teacher, base))
    if any(abs(x) > radius + 1e-12 for x in residual):
        raise ValueError("New base cannot preserve teacher within residual bound")
    return residual


def bootstrap_target(rewards: Sequence[float], gamma: float,
                     next_q: float, *, terminated: bool,
                     truncated: bool, bootstrap_on_truncation: bool) -> float:
    """n-step target for ONE pre-segmented, episode-contained reward window.

    Caller must stop this window at the first termination/truncation; the flags
    refer to its last transition. Finite-horizon task timeouts normally use
    bootstrap_on_truncation=False. Artificial collector cutoffs may use True
    with a genuine final observation (never an auto-reset observation).
    """
    rs = vector(rewards)
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("Discount must be in [0, 1]")
    if not math.isfinite(next_q):
        raise ValueError("Nonfinite Q")
    allow = not terminated and (not truncated or bootstrap_on_truncation)
    return sum(gamma**i * r for i, r in enumerate(rs)) + (
        gamma**len(rs) * next_q if allow else 0.0)
