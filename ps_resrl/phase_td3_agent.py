"""8-D Phase-Symbiotic Actor & QAgent builder on top of pinned ResFiT."""

from __future__ import annotations

import copy
from typing import Sequence

import torch

from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.rl.actor import Actor, SpatialEmb
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent


class PhaseSymbioticActor(Actor):
    """Residual actor with per-dimension scaling for 7-D residual + 1-D phase delta."""

    def __init__(
        self,
        repr_dim: int,
        patch_repr_dim: int,
        prop_dim: int,
        action_dim: int,
        cfg,
        *,
        action_scale_vec: Sequence[float],
        residual_actor: bool = True,
    ) -> None:
        super().__init__(
            repr_dim,
            patch_repr_dim,
            prop_dim,
            action_dim,
            cfg,
            residual_actor=residual_actor,
        )
        if len(action_scale_vec) != action_dim:
            raise ValueError(
                f"action_scale_vec length {len(action_scale_vec)} != action_dim {action_dim}"
            )
        self.register_buffer(
            "action_scale_vec",
            torch.tensor(action_scale_vec, dtype=torch.float32).unsqueeze(0),
        )

    def forward(self, obs: dict[str, torch.Tensor], std: float):
        if isinstance(self.compress, SpatialEmb):
            assert not self.residual_actor, "Not implemented"
            feat = self.compress.forward(obs["feat"], obs["observation.state"])
        else:
            feat = obs["feat"].flatten(1, -1)
            feat = self.compress(feat)

        all_input = [feat]
        if self.prop_dim > 0:
            all_input.append(obs["observation.state"])
            if self.residual_actor:
                all_input.append(obs["observation.base_action"])

        policy_input = torch.cat(all_input, dim=-1)
        mu: torch.Tensor = self.policy(policy_input)
        scaled_mu = mu * self.action_scale_vec.to(device=mu.device, dtype=mu.dtype)
        return utils.TruncatedNormal(scaled_mu, std)


def build_phase_q_agent(
    obs_shape: tuple[int, int, int],
    prop_shape: tuple[int],
    action_dim: int,
    rl_cameras: list[str] | str,
    cfg,
    *,
    mode: str = "ps_resrl",
    phase_action_scale: float = 1.0,
) -> QAgent:
    """Build QAgent for either 8-D PS-ResRL or 7-D B1-ctx ablation."""
    if mode not in ("ps_resrl", "b1_ctx"):
        raise ValueError(f"Unsupported PS-2 mode: {mode}")

    agent = QAgent(
        obs_shape=obs_shape,
        prop_shape=prop_shape,
        action_dim=action_dim,
        rl_cameras=rl_cameras,
        cfg=cfg,
        residual_actor=True,
    )
    if mode == "ps_resrl":
        if action_dim != 8:
            raise ValueError("ps_resrl mode requires action_dim == 8")
        sample_encoder = agent.encoders[0]
        repr_dim = int(sample_encoder.repr_dim) * len(agent.rl_cameras)
        patch_repr_dim = int(sample_encoder.patch_repr_dim)
        prop_dim = prop_shape[0] if cfg.use_prop else 0
        scale_vec = [float(cfg.actor.action_scale)] * 7 + [float(phase_action_scale)]
        agent.actor = PhaseSymbioticActor(
            repr_dim,
            patch_repr_dim,
            prop_dim,
            action_dim,
            cfg.actor,
            action_scale_vec=scale_vec,
            residual_actor=True,
        )
        agent.actor_target = copy.deepcopy(agent.actor)
        agent.to(cfg.device)
        agent.actor_opt = torch.optim.AdamW(agent.actor.parameters(), lr=cfg.actor_lr)
        if cfg.lr_warmup_steps > 0:
            warmup_start = cfg.lr_warmup_start
            actor_start_factor = max(
                warmup_start / cfg.actor_lr if cfg.actor_lr > 0 else 1e-8, 1e-8
            )
            agent.actor_scheduler = torch.optim.lr_scheduler.LinearLR(
                agent.actor_opt,
                start_factor=actor_start_factor,
                total_iters=cfg.lr_warmup_steps,
            )
        agent.critic_target.train(False)
        agent.train(True)
    return agent
