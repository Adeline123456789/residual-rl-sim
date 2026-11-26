import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("numpy", "torch"))
if HAS_RUNTIME:
    import numpy as np

    from cf_resrl.closed_loop import run_episode


class FakeRunner:
    def __init__(self, *, horizon=12, chunk_length=4, delay=2, terminal_after=None):
        self.wrapper = SimpleNamespace(horizon=horizon, env=SimpleNamespace(), reset=self.reset)
        self.policy = SimpleNamespace(config=SimpleNamespace(n_action_steps=chunk_length))
        self.delay = delay
        self.gamma = 0.99
        self.terminal_after = terminal_after
        self.events = []

    def reset(self, *, seed):
        self.events.append(("reset", seed))
        return 0, {}

    def _chunk(self, observation, source_tick, generation):
        assert observation == source_tick
        self.events.append(("ACT", source_tick, generation))
        return [[0.1 + generation / 100]] * self.policy.config.n_action_steps

    def _actor_observation(self, scheduler, observation):
        assert scheduler.logical_tick == observation
        return observation

    def _propose(self, _actor_observation):
        return [0.25]

    def _step(self, scheduler, residual, refresh):
        entry = scheduler.action(residual)
        self.events.append(("action", entry.tick, entry.applied_action))
        observation = entry.tick + 1
        terminated = observation == self.terminal_after
        truncated = observation == self.wrapper.horizon
        scheduler.after_step(observation, request_refresh=refresh and not (terminated or truncated))
        pending = scheduler.pending_refresh
        return observation, {
            "tick": entry.tick,
            "generation": entry.generation,
            "chunk_source_tick": entry.chunk_source_tick,
            "chunk_index": entry.chunk_index,
            "used_fallback": entry.used_fallback,
            "applied_env_action": entry.applied_action,
            "refresh_requested_after_step": refresh,
            "pending_refresh_source_tick": pending.chunk.source_tick if pending else None,
            "pending_refresh_ready_tick": pending.ready_tick if pending else None,
            "reward": float(terminated),
            "terminated": terminated,
            "truncated": truncated,
        }


@unittest.skipUnless(HAS_RUNTIME, "closed-loop rollout needs NumPy and PyTorch")
class ClosedLoopTests(unittest.TestCase):
    def test_original_keeps_nominal_refills_and_counts_initial_inference(self):
        runner = FakeRunner()
        episode = run_episode(runner, seed=5, act_budget=3, method="original_resfit", parameter=None)
        self.assertEqual([event[1] for event in runner.events if event[0] == "ACT"], [0, 4, 8])
        self.assertEqual(episode["initial_act_calls"], 1)
        self.assertEqual(episode["nominal_act_calls"], 2)
        self.assertEqual(episode["act_calls"], 3)
        self.assertEqual(episode["refresh_requests"], 0)
        self.assertEqual(episode["environment_steps"], 12)
        self.assertFalse(episode["success"])
        self.assertTrue(episode["steps"][-1]["truncated"])
        self.assertGreaterEqual(episode["wall_seconds"], episode["policy_wall_seconds"])

    def test_reserves_budget_for_nominal_refills(self):
        runner = FakeRunner()
        episode = run_episode(runner, seed=5, act_budget=3, method="residual_norm_threshold", parameter=0.1)
        self.assertEqual(episode["act_calls"], 3)
        self.assertTrue(all(row["refresh_block"] == "act_budget" for row in episode["steps"][:3]))
        self.assertTrue(episode["steps"][3]["refresh_dispatched"])
        self.assertEqual(episode["steps"][-1]["refresh_block"], "horizon")
        with self.assertRaisesRegex(ValueError, "cover all nominal refills"):
            run_episode(runner, seed=5, act_budget=2, method="original_resfit", parameter=None)

    def test_refresh_uses_post_action_source_and_absolute_index_after_fallback(self):
        runner = FakeRunner()
        episode = run_episode(runner, seed=5, act_budget=4, method="fixed_period_refresh", parameter=4)
        rows = episode["steps"]
        self.assertTrue(rows[3]["refresh_dispatched"])
        self.assertEqual(rows[3]["pending_refresh_source_tick"], 4)
        self.assertEqual(rows[3]["pending_refresh_ready_tick"], 6)
        self.assertLess(runner.events.index(("action", 3, (0.35,))), runner.events.index(("ACT", 4, 1)))
        self.assertTrue(all(rows[tick]["used_fallback"] for tick in (4, 5)))
        self.assertTrue(all(rows[tick]["refresh_block"] == "pending" for tick in (4, 5)))
        self.assertEqual((rows[6]["chunk_source_tick"], rows[6]["chunk_index"]), (4, 2))
        self.assertTrue(rows[6]["refresh_activated"])
        self.assertEqual(episode["refresh_requests"], episode["refresh_activations"])
        self.assertEqual(episode["act_calls"], 3)

    def test_terminal_action_prevents_dispatch_and_keeps_real_return(self):
        runner = FakeRunner(terminal_after=1)
        episode = run_episode(runner, seed=5, act_budget=5, method="residual_norm_threshold", parameter=0.1)
        self.assertTrue(episode["steps"][0]["refresh_requested_after_step"])
        self.assertFalse(episode["steps"][0]["refresh_dispatched"])
        self.assertEqual(episode["refresh_requests"], 0)
        self.assertEqual(episode["act_calls"], 1)
        self.assertEqual(episode["return"], 1)
        self.assertTrue(episode["success"])
        self.assertEqual(episode["environment_steps"], 1)

    def test_terminal_while_waiting_charges_unused_request(self):
        runner = FakeRunner(terminal_after=2)
        episode = run_episode(runner, seed=5, act_budget=5, method="residual_norm_threshold", parameter=0.1)
        self.assertTrue(episode["steps"][0]["refresh_dispatched"])
        self.assertEqual(episode["refresh_requests"], 1)
        self.assertEqual(episode["refresh_activations"], 0)
        self.assertEqual(episode["act_calls"], 2)
        self.assertAlmostEqual(episode["discounted_return"], 0.99)

    def test_probes_use_pre_action_observation_and_strict_gain_decision(self):
        runner = FakeRunner(terminal_after=2)
        probe = SimpleNamespace(method="P1")
        context = np.array([0.4, 0.5], dtype=np.float32)
        residual = np.array([0.25], dtype=np.float32)
        with (
            patch("cf_resrl.closed_loop.visible_features", return_value={"observed": True}) as extract,
            patch("cf_resrl.closed_loop.pack_features", return_value=(context, residual)) as pack,
            patch("cf_resrl.closed_loop.predict_probe", return_value={"g_delta": [0.1]}) as predict,
        ):
            episode = run_episode(runner, seed=5, act_budget=5, method="P1", parameter=0.1, probe=probe)
        self.assertEqual(extract.call_args_list[0].args[2:], (0, [0.25]))
        pack.assert_called_with({"observed": True})
        self.assertIs(predict.call_args.args[0], probe)
        np.testing.assert_array_equal(predict.call_args.args[1], context[None, :])
        np.testing.assert_array_equal(predict.call_args.args[2], residual[None, :])
        self.assertEqual(episode["steps"][0]["decision_score"], 0.1)
        self.assertEqual(episode["refresh_requests"], 0)


if __name__ == "__main__":
    unittest.main()
