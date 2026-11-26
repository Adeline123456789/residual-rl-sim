import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.verify_square_closed_loop import _archived_attempt, _verify_steps, verify


def delayed_refresh_trace():
    """Request at tick 19, fall back for ticks 20/21, then start at new chunk index 2."""
    steps = []
    for tick in range(24):
        activated = tick >= 22
        block = "act_budget" if tick < 3 else None
        if tick in (20, 21):
            block = "pending"
        elif activated:
            block = "horizon"
        pending = tick in (19, 20)
        action = [0.0] * 7
        steps.append({
            "tick": tick, "control_steps": tick + 1,
            "simulation_time": 0.05 * (tick + 1),
            "chunk_source_tick": 20 if activated else 0,
            "generation": int(activated), "chunk_index": tick - 20 if activated else tick,
            "used_fallback": tick in (20, 21), "nominal_refill": False,
            "base_action": action.copy(), "proposed_residual": action.copy(),
            "applied_env_action": action.copy(), "proposed_action": action.copy(),
            "applied_action": action.copy(), "applied_normalized_action": action.copy(),
            "reward": 0, "terminated": False, "truncated": tick == 23,
            "refresh_block": block, "decision_score": float(tick == 19) if block is None else None,
            "refresh_requested_after_step": tick == 19, "refresh_dispatched": tick == 19,
            "refresh_activated": tick == 22,
            "pending_refresh_source_tick": 20 if pending else None,
            "pending_refresh_ready_tick": 22 if pending else None,
            "act_calls": 1 + int(tick >= 19), "act_calls_since_initial_chunk": int(tick >= 19),
        })
    return {
        "method": "fixed_period_refresh", "parameter": 20, "act_budget": 2, "act_calls": 2,
        "initial_act_calls": 1, "nominal_act_calls": 0, "refresh_requests": 1, "refresh_activations": 1,
        "success": False, "return": 0.0, "discounted_return": 0.0,
        "wall_seconds": 2.0, "policy_wall_seconds": 1.0, "environment_steps": 24, "steps": steps,
    }


class SquareEvidenceTest(unittest.TestCase):
    def test_two_step_refresh_and_zero_fallback(self):
        _verify_steps(delayed_refresh_trace(), horizon=24, delay=2, gamma=0.99)

    def test_rejects_corrupted_action_and_refresh_ledgers(self):
        original = delayed_refresh_trace()
        for tick, field, value, error in (
            (21, "refresh_activated", True, "activation flag"),
            (22, "chunk_index", 0, "chunk or index"),
            (20, "base_action", [0.1] + [0.0] * 6, "normalized action limit"),
            (20, "applied_env_action", [1.00000001] + [0.0] * 6, "out of bounds"),
            (20, "act_calls", 3, "ACT ledger"),
        ):
            with self.subTest(field=field):
                trace = copy.deepcopy(original)
                trace["steps"][tick][field] = value
                with self.assertRaisesRegex(ValueError, error):
                    _verify_steps(trace, horizon=24, delay=2, gamma=0.99)

    def test_incomplete_attempt_rejected_before_traces_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            (attempt / "status.json").write_text('{"status":"RUNNING"}')
            with self.assertRaisesRegex(ValueError, "Only a completed"):
                verify(attempt)

    def test_x1_actor_scale_and_clamp_are_verified(self):
        trace = delayed_refresh_trace()
        for row in trace["steps"]:
            row.update({
                "actor_original_residual": [0.08, -0.08] + [0.0] * 5,
                "actor_scaled_residual": [0.12, -0.12] + [0.0] * 5,
                "actor_scale_clip_mask": [True, True] + [False] * 5,
                "proposed_residual": [0.1, -0.1] + [0.0] * 5,
                "proposed_action": [0.1, -0.1] + [0.0] * 5,
                "applied_action": [0.1, -0.1] + [0.0] * 5,
            })
        settings = {"horizon": 24, "delay": 2, "gamma": 0.99, "residual_transform": (1.5, 0.1)}
        _verify_steps(trace, **settings)
        trace["steps"][0]["actor_scale_clip_mask"][0] = False
        with self.assertRaisesRegex(ValueError, "Wrong X1 residual scaling or clamp"):
            _verify_steps(trace, **settings)

    def test_x1_archive_resolves_the_recorded_repository_suffix(self):
        relative = Path("results/cf_resrl/expansion_square/branch_data/cell/attempts/run")
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            restored = workspace / "restored" / relative
            restored.mkdir(parents=True)
            with patch("scripts.verify_square_closed_loop.ROOT", workspace / "restored"):
                self.assertEqual(
                    _archived_attempt(workspace / "original" / relative, "expansion_square/branch_data"),
                    restored,
                )
