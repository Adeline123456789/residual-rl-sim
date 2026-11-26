import base64
from collections import Counter
import hashlib
import importlib.util
import random
from types import SimpleNamespace
import unittest
import zlib

from cf_resrl.e0_collector import (
    _encode_actor_observation,
    _should_replace,
    _validate_branches,
    visible_features,
)
from cf_resrl.scheduler import AbsoluteTickScheduler


CANDIDATE = [0.1, -0.2]
DELAY = 2


def _paired_branches(source_tick, remaining, *, terminal_after=None, delay=DELAY):
    scheduler = AbsoluteTickScheduler(
        initial_chunk=[[0.5, 0.5]] * 20,
        provider=lambda _obs, _tick, _generation: [[0.7, 0.7]] * 20,
        latency_steps=delay,
        fallback_action=[0.0, 0.0],
    )
    for _ in range(source_tick):
        scheduler.action(CANDIDATE)
        scheduler.after_step(None, request_refresh=False)
    snapshot = scheduler.snapshot()
    results = {}
    for label in ("V10", "V11"):
        scheduler.restore(snapshot)
        rows = []
        for offset in range(remaining):
            if offset:
                scheduler.refill_if_exhausted(None)
            entry = scheduler.action(CANDIDATE)
            intent = label == "V11" and offset == 0
            terminal = terminal_after == offset + 1
            scheduler.after_step(None, request_refresh=intent and not terminal)
            pending = scheduler.pending_refresh
            rows.append({
                "tick": entry.tick,
                "control_steps": scheduler.counters.control_steps,
                "generation": entry.generation,
                "chunk_source_tick": entry.chunk_source_tick,
                "chunk_index": entry.chunk_index,
                "used_fallback": entry.used_fallback,
                "applied_env_action": entry.applied_action,
                "proposed_residual": entry.proposed_residual,
                "refresh_requested_after_step": intent,
                "pending_refresh_source_tick": pending.chunk.source_tick if pending else None,
                "pending_refresh_ready_tick": pending.ready_tick if pending else None,
                "act_calls_since_initial_chunk": scheduler.counters.act_calls,
                "reward": float(terminal),
                "terminated": terminal,
                "truncated": False,
                "simulation_time": (entry.tick + 1) / 20,
                "observation_state": entry.applied_action,
                "observation_image_sha256": {},
                "integration_state_sha256": "synthetic",
            })
            if terminal:
                break
        results[label] = {
            "steps": rows,
            "environment_steps": len(rows),
            "remaining_horizon": remaining,
            "act_calls": scheduler.counters.act_calls - snapshot.counters.act_calls,
        }
    return snapshot, results


class E0CollectorTests(unittest.TestCase):
    def test_natural_reservoir_covers_every_visited_tick(self):
        counts = Counter()
        for seed in range(3000):
            rng = random.Random(seed)
            chosen = None
            for tick in range(5):
                if _should_replace("natural", tick, rng, residual_norm_sq=tick, best_norm_sq=tick + 1):
                    chosen = tick
            counts[chosen] += 1
        self.assertEqual(set(counts), set(range(5)))
        self.assertTrue(all(530 <= count <= 670 for count in counts.values()), counts)

    def test_enrichment_keeps_earliest_maximum(self):
        best = float("-inf")
        selected = None
        for tick, norm_sq in enumerate([0.01, 0.05, 0.05, 0.04]):
            if _should_replace("enriched", tick, None, norm_sq, best):
                best = norm_sq
                selected = tick
        self.assertEqual(selected, 1)

    def _check(self, snapshot, results, remaining):
        _validate_branches(
            results, scheduler_snapshot=snapshot, delay=DELAY,
            remaining=remaining, candidate=CANDIDATE, low=[-1.0, -1.0], high=[1.0, 1.0],
        )

    def test_registered_x1_delays_preserve_absolute_chunk_index(self):
        for delay in (0, 2, 4):
            with self.subTest(delay=delay):
                snapshot, results = _paired_branches(18, 7, delay=delay)
                _validate_branches(
                    results, scheduler_snapshot=snapshot, delay=delay,
                    remaining=7, candidate=CANDIDATE, low=[-1.0, -1.0], high=[1.0, 1.0],
                )
                refresh = results["V11"]
                self.assertEqual(refresh["refresh_activation_tick"], 19 + delay)
                self.assertEqual(refresh["steps"][1 + delay]["chunk_index"], delay)

    def test_terminal_before_refresh_is_not_replaced(self):
        snapshot, results = _paired_branches(19, 6, terminal_after=1)
        self._check(snapshot, results, 6)
        refresh = results["V11"]
        self.assertEqual(refresh["environment_steps"], 1)
        self.assertFalse(refresh["refresh_dispatched"])
        self.assertIsNone(refresh["refresh_activation_tick"])

    def test_terminal_while_pending_is_not_replaced(self):
        snapshot, results = _paired_branches(7, 6, terminal_after=2)
        self._check(snapshot, results, 6)
        refresh = results["V11"]
        self.assertEqual(refresh["environment_steps"], 2)
        self.assertTrue(refresh["refresh_dispatched"])
        self.assertIsNone(refresh["refresh_activation_tick"])

    def test_expired_chunk_waits_with_fallback(self):
        snapshot, results = _paired_branches(19, 6)
        self._check(snapshot, results, 6)
        self.assertEqual(results["V10"]["waiting_fallback_steps"], 0)
        self.assertEqual(results["V11"]["waiting_fallback_steps"], 2)
        self.assertEqual(results["V11"]["refresh_activation_tick"], 22)
        results["V11"]["steps"][3]["chunk_index"] = 0
        with self.assertRaisesRegex(RuntimeError, "activation or absolute index"):
            self._check(snapshot, results, 6)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "pinned PyTorch runtime required")
    def test_visible_features_and_lossless_actor_input(self):
        import numpy as np
        import torch

        scheduler = AbsoluteTickScheduler(
            initial_chunk=[[0.5, 0.5]] * 20,
            provider=lambda *_: None, latency_steps=DELAY, fallback_action=[0.0, 0.0],
        )
        for _ in range(7):
            scheduler.action(CANDIDATE)
            scheduler.after_step(None, request_refresh=False)
        runner = SimpleNamespace(
            policy=SimpleNamespace(config=SimpleNamespace(n_action_steps=20)),
            agent=SimpleNamespace(
                _encode=lambda obs, augment: obs["image"],
                actor=SimpleNamespace(compress=lambda visual: visual * 2),
            ),
        )
        actor_obs = {
            "image": torch.tensor([[[1.0, 2.0]]]),
            "observation.state": torch.tensor([[0.5, 0.1]]),
            "observation.base_action": torch.tensor([[0.5, 0.5]]),
        }
        features = visible_features(runner, scheduler, actor_obs, CANDIDATE)
        self.assertEqual(set(features), {"visual", "state", "base", "chunk_suffix", "chunk_mask",
                                         "tick", "chunk_index", "residual"})
        np.testing.assert_array_equal(features["visual"], [2.0, 4.0])
        np.testing.assert_array_equal(features["tick"], [7])
        np.testing.assert_array_equal(features["chunk_mask"], [1.0] * 13 + [0.0] * 7)
        np.testing.assert_array_equal(features["chunk_suffix"][13:], np.zeros((7, 2)))
        evidence = _encode_actor_observation(actor_obs)
        for key, value in evidence.items():
            raw = zlib.decompress(base64.b64decode(value["data"]))
            self.assertEqual(hashlib.sha256(raw).hexdigest(), value["sha256"])
            restored = np.frombuffer(raw, dtype=value["dtype"]).reshape(value["shape"])
            np.testing.assert_array_equal(restored, actor_obs[key].numpy())


if __name__ == "__main__":
    unittest.main()
