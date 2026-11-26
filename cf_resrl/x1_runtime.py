"""Apply X1 residual amplitudes before the registered Square action clamps."""

from __future__ import annotations

import math

from cf_resrl.branch_runner import SquareBranchRunner


class ScaledSquareBranchRunner(SquareBranchRunner):
    """Use the same bounded candidate for scouting, intervention and deployment."""

    def __init__(self, *args, residual_scale, **kwargs):
        super().__init__(*args, **kwargs)
        if residual_scale not in (0.5, 1.0, 1.5):
            raise ValueError("Unregistered X1 residual scale")
        self.residual_scale = float(residual_scale)
        self.residual_limit = float(self.agent.cfg.actor.action_scale)
        if not math.isfinite(self.residual_limit) or self.residual_limit <= 0:
            raise ValueError("The learned actor must define a finite normalized residual bound")

    def _propose(self, actor_observation):
        original = super()._propose(actor_observation)
        if not all(map(math.isfinite, original)):
            raise RuntimeError("ResFiT produced a non-finite normalized residual")
        self._original_residual = original
        self._scaled_residual = [self.residual_scale * value for value in original]
        return [max(-self.residual_limit, min(self.residual_limit, value)) for value in self._scaled_residual]

    def _step(self, scheduler, residual, refresh):
        observation, row = super()._step(scheduler, residual, refresh)
        row["actor_original_residual"] = self._original_residual
        row["actor_scaled_residual"] = self._scaled_residual
        row["actor_scale_clip_mask"] = [abs(value) > self.residual_limit for value in self._scaled_residual]
        return observation, row
