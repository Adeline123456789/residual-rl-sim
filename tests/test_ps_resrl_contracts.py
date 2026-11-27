"""Hard-contract unit tests for PS-ResRL continuous-phase scheduling and Frenet rule."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import random
import sys
import unittest

from cf_resrl.scheduler import AbsoluteTickScheduler
from ps_resrl.frenet import frenet_decompose, phase_hold_velocity
from ps_resrl.phase_scheduler import ContinuousPhaseScheduler, interpolate_chunk_action

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from ps_resrl.phase_vec_env import (
        PHASE_CONTEXT_DIM,
        build_phase_context_tensor,
        interpolate_batched_chunk,
    )


def _make_chunk(seed: int, length: int = 20, dim: int = 7) -> list[list[float]]:
    rng = random.Random(seed)
    rows = []
    for idx in range(length):
        pose = [rng.uniform(-0.8, 0.8) for _ in range(dim - 1)]
        grip = -1.0 if idx < length // 2 else 1.0
        rows.append([*pose, grip])
    return rows


class ContinuousPhaseSchedulerContractTests(unittest.TestCase):
    def test_zero_modulation_is_bit_exact_with_absolute_tick_scheduler(self) -> None:
        rng = random.Random(2026)
        initial = _make_chunk(100)
        provider_calls_abs: list[tuple[int, int]] = []
        provider_calls_phase: list[tuple[int, int]] = []

        def make_provider(log: list[tuple[int, int]]):
            def provider(_obs, source_tick: int, generation: int):
                log.append((source_tick, generation))
                return _make_chunk(1000 + generation)

            return provider

        abs_sched = AbsoluteTickScheduler(
            initial_chunk=initial,
            provider=make_provider(provider_calls_abs),
            latency_steps=2,
            fallback_action=[0.0] * 7,
        )
        phase_sched = ContinuousPhaseScheduler(
            initial_chunk=initial,
            provider=make_provider(provider_calls_phase),
            latency_steps=2,
            fallback_action=[0.0] * 7,
        )

        for tick in range(60):
            abs_refill = abs_sched.refill_if_exhausted(f"obs_{tick}")
            phase_refill = phase_sched.refill_if_exhausted(f"obs_{tick}")
            self.assertEqual(abs_refill, phase_refill)
            self.assertEqual(abs_sched.current_base_action(), phase_sched.current_base_action())

            residual = [0.0] * 7 if tick % 3 == 0 else [rng.uniform(-0.1, 0.1) for _ in range(7)]
            abs_entry = abs_sched.action(residual)
            phase_entry = phase_sched.action(residual, phase_velocity=1.0)

            self.assertEqual(abs_entry.tick, phase_entry.tick)
            self.assertEqual(abs_entry.generation, phase_entry.generation)
            self.assertEqual(abs_entry.chunk_source_tick, phase_entry.chunk_source_tick)
            self.assertEqual(abs_entry.chunk_index, phase_entry.chunk_index)
            self.assertEqual(abs_entry.used_fallback, phase_entry.used_fallback)
            self.assertEqual(abs_entry.base_action, phase_entry.base_action)
            self.assertEqual(abs_entry.applied_action, phase_entry.applied_action)
            self.assertEqual(abs_entry.applied_residual, phase_entry.applied_residual)

            request = tick in (5, 22, 41)
            abs_sched.after_step(f"obs_{tick + 1}", request_refresh=request)
            phase_sched.after_step(f"obs_{tick + 1}", request_refresh=request)

            self.assertEqual(abs_sched.counters.act_calls, phase_sched.counters.act_calls)
            self.assertEqual(abs_sched.counters.refresh_requests, phase_sched.counters.refresh_requests)
            self.assertEqual(provider_calls_abs, provider_calls_phase)

    def test_gripper_dimension_remains_strictly_discrete(self) -> None:
        chunk = [tuple(row) for row in _make_chunk(77, length=10, dim=7)]
        for step in range(100):
            phi = 9.99 * step / 99.0
            action = interpolate_chunk_action(chunk, phi)
            self.assertIn(action[6], (-1.0, 1.0))

    def test_anti_stall_cap_triggers_at_exact_wall_limit(self) -> None:
        initial = _make_chunk(12, length=10)
        calls = 0

        def provider(_obs, _tick: int, gen: int):
            nonlocal calls
            calls += 1
            return _make_chunk(12 + gen, length=10)

        sched = ContinuousPhaseScheduler(
            initial_chunk=initial,
            provider=provider,
            latency_steps=0,
            fallback_action=[0.0] * 7,
            max_chunk_wall_steps=20,
        )
        for step in range(20):
            refilled = sched.refill_if_exhausted("obs")
            self.assertFalse(refilled)
            entry = sched.action([0.0] * 7, phase_velocity=0.0)
            self.assertEqual(entry.phase_cursor, 0.0)
            self.assertEqual(entry.chunk_wall_steps, step)
            sched.after_step("obs", request_refresh=False)

        self.assertEqual(calls, 0)
        self.assertTrue(sched.refill_if_exhausted("obs"))
        self.assertEqual(calls, 1)
        self.assertEqual(sched.counters.stall_cap_triggers, 1)
        self.assertEqual(sched.counters.natural_phase_refills, 0)
        self.assertEqual(sched.phase_cursor, 0.0)
        self.assertEqual(sched.chunk_wall_steps, 0)

    def test_snapshot_restore_exact_under_fractional_velocities(self) -> None:
        initial = _make_chunk(42, length=20)
        sched = ContinuousPhaseScheduler(
            initial_chunk=initial,
            provider=lambda _o, _t, g: _make_chunk(42 + g, length=20),
            latency_steps=2,
            fallback_action=[0.0] * 7,
        )
        for step, nu in enumerate((0.25, 0.7, 1.4, 0.0)):
            sched.refill_if_exhausted(f"obs_{step}")
            sched.action([0.01 * step] * 7, phase_velocity=nu)
            sched.after_step(f"obs_{step + 1}", request_refresh=False)

        snap = sched.snapshot()
        seq = (0.3, 1.25, 0.0, 0.8, 1.5)
        run1 = []
        for idx, nu in enumerate(seq):
            sched.refill_if_exhausted(f"r_{idx}")
            entry = sched.action([0.02] * 7, phase_velocity=nu)
            run1.append((entry.base_action, entry.applied_action, entry.phase_cursor, entry.chunk_wall_steps))
            sched.after_step(f"r_{idx + 1}", request_refresh=idx == 1)

        sched.restore(snap)
        run2 = []
        for idx, nu in enumerate(seq):
            sched.refill_if_exhausted(f"r_{idx}")
            entry = sched.action([0.02] * 7, phase_velocity=nu)
            run2.append((entry.base_action, entry.applied_action, entry.phase_cursor, entry.chunk_wall_steps))
            sched.after_step(f"r_{idx + 1}", request_refresh=idx == 1)

        self.assertEqual(run1, run2)

    def test_frenet_decomposition_and_phase_hold_velocity(self) -> None:
        base = [0.0, 0.0, -0.4, 0.0, 0.0, 0.0, -1.0]
        opposing_with_perp = [0.08, -0.06, 0.10, 0.0, 0.0, 0.0, 0.0]
        aligned = [0.0, 0.0, -0.05, 0.0, 0.0, 0.0, 0.0]
        zero_base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

        dec = frenet_decompose(base, opposing_with_perp)
        self.assertLess(dec.brake_ratio, 0.0)
        self.assertAlmostEqual(dec.perp_norm, 0.10, places=6)
        nu_brake = phase_hold_velocity(base, opposing_with_perp, kappa=1.0, nu_min=0.0, nu_max=1.0)
        self.assertTrue(0.0 <= nu_brake < 1.0)

        nu_aligned = phase_hold_velocity(base, aligned, kappa=1.0, nu_min=0.0, nu_max=1.0)
        self.assertEqual(nu_aligned, 1.0)

        nu_zero = phase_hold_velocity(zero_base, opposing_with_perp, kappa=1.0, nu_min=0.0, nu_max=1.0)
        self.assertTrue(math.isfinite(nu_zero))
        self.assertEqual(nu_zero, 1.0)


@unittest.skipUnless(HAS_TORCH, "PyTorch tests require torch")
class PhaseSymbioticRLContractTests(unittest.TestCase):
    def test_batched_chunk_interpolation_and_context(self) -> None:
        chunk_list = _make_chunk(55, length=20, dim=7)
        chunks = torch.tensor([chunk_list], dtype=torch.float32)
        for step in range(20):
            phi = torch.tensor([float(step)], dtype=torch.float32)
            batched = interpolate_batched_chunk(chunks, phi)[0].tolist()
            expected = chunks[0, step].tolist()
            self.assertEqual(batched, expected)

        for frac in (0.25, 4.7, 9.5, 13.8):
            phi = torch.tensor([frac], dtype=torch.float32)
            out = interpolate_batched_chunk(chunks, phi)[0]
            self.assertIn(float(out[6].item()), (-1.0, 1.0))

        ctx = build_phase_context_tensor(
            chunks, torch.tensor([3.5]), torch.tensor([10]), max_chunk_wall_steps=40
        )
        self.assertEqual(tuple(ctx.shape), (1, PHASE_CONTEXT_DIM))
        self.assertAlmostEqual(float(ctx[0, 0].item()), 3.5 / 20.0, places=6)
        self.assertAlmostEqual(float(ctx[0, 1].item()), 10.0 / 40.0, places=6)

    def test_phase_symbiotic_actor_zero_init_and_per_dim_scale(self) -> None:
        resfit_root = Path(__file__).resolve().parents[1] / "third_party" / "resfit"
        if str(resfit_root) not in sys.path:
            sys.path.insert(0, str(resfit_root))
        from resfit.rl_finetuning.config.rlpd import ActorConfig
        from ps_resrl.phase_td3_agent import PhaseSymbioticActor

        cfg = ActorConfig(action_scale=0.1, actor_last_layer_init_scale=0.0)
        actor = PhaseSymbioticActor(
            repr_dim=32,
            patch_repr_dim=8,
            prop_dim=18,
            action_dim=8,
            cfg=cfg,
            action_scale_vec=[0.1] * 7 + [1.0],
            residual_actor=True,
        )
        obs = {
            "feat": torch.randn(2, 4, 8),
            "observation.state": torch.randn(2, 18),
            "observation.base_action": torch.randn(2, 8),
        }
        dist = actor.forward(obs, std=0.05)
        self.assertEqual(tuple(dist.mean.shape), (2, 8))
        self.assertEqual(float(dist.mean.abs().max().item()), 0.0)
        self.assertEqual(actor.action_scale_vec[0, 6].item(), 0.10000000149011612)
        self.assertEqual(actor.action_scale_vec[0, 7].item(), 1.0)

    def test_evaluate_phase_vec_env_act_calls_and_quantiles(self) -> None:
        resfit_root = Path(__file__).resolve().parents[1] / "third_party" / "resfit"
        if str(resfit_root) not in sys.path:
            sys.path.insert(0, str(resfit_root))
        from ps_resrl.phase_vec_env import evaluate_phase_vec_env

        class _FakeAgent:
            def __init__(self) -> None:
                self.training = True

            def train(self, mode: bool = True):
                self.training = mode
                return self

            def act(self, obs, *, eval_mode=False, stddev=0.0, cpu=False):
                assert eval_mode and not self.training
                return torch.zeros((2, 8), dtype=torch.float32)

        class _FakePhaseEnv:
            num_envs = 2

            def __init__(self) -> None:
                self._tick = 0

            def reset(self, *, seed=None):
                self._tick = 0
                return {"observation.state": torch.zeros((2, 4))}, {}

            def step(self, actions):
                self._tick += 1
                t = self._tick
                # Env 0 finishes at t=2 (1 refill while alive -> 2 ACT calls) and again at t=4 (1 episode ACT call).
                # Env 1 finishes at t=3 (0 refills while alive -> 1 ACT call).
                done = torch.tensor([t in (2, 4), t == 3], dtype=torch.bool)
                term = torch.tensor([t == 2, False], dtype=torch.bool)
                trunc = torch.tensor([t == 4, t == 3], dtype=torch.bool)
                reward = torch.tensor([1.0 if t == 2 else 0.0, 0.0], dtype=torch.float32)
                refill = torch.tensor([t in (1, 2, 4), t == 3], dtype=torch.bool)
                stall = torch.tensor([t == 1, False], dtype=torch.bool)
                nu = torch.tensor([0.5 if t <= 2 else 1.25, 1.0], dtype=torch.float32)
                res = torch.zeros((2, 7), dtype=torch.float32)
                if t == 2:
                    res[0, 0] = 0.10  # saturated step for env 0
                info = {
                    "phase_velocity": nu,
                    "stall_cap_triggered": stall,
                    "act_refill_mask": refill,
                    "proposed_residual_action": res,
                }
                return {"observation.state": torch.zeros((2, 4))}, reward, term, trunc, info

        agent = _FakeAgent()
        metrics, episodes = evaluate_phase_vec_env(
            _FakePhaseEnv(),
            agent,
            num_episodes=3,
            device=torch.device("cpu"),
            reset_seed=20262124,
            saturation_threshold=0.095,
        )
        self.assertTrue(agent.training)
        self.assertEqual(len(episodes), 3)
        self.assertEqual([ep["act_calls"] for ep in episodes], [2, 1, 1])
        self.assertAlmostEqual(metrics["eval/success_rate"], 1.0 / 3.0, places=6)
        self.assertEqual(metrics["eval/act_calls_total"], 4)
        self.assertAlmostEqual(metrics["eval/mean_act_calls"], 4.0 / 3.0, places=6)
        self.assertEqual(metrics["eval/stall_cap_triggers"], 1)
        self.assertAlmostEqual(metrics["eval/residual_saturation_rate"], 1.0 / 7.0, places=6)
        self.assertAlmostEqual(metrics["eval/mean_phase_velocity"], 6.5 / 7.0, places=6)
        self.assertAlmostEqual(metrics["eval/phase_velocity_p50"], 1.0, places=6)
        self.assertEqual(metrics["eval/environment_steps"], 8)


if __name__ == "__main__":
    unittest.main()
