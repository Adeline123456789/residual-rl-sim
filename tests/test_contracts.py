import unittest
from pa_reswm.contracts import (
    AppliedAction, ActionLedger, ActionChunk, ChunkQueue,
    align_prediction, recenter_residual, bootstrap_target,
)


def entry(tick=0, episode="e", applied=(0.25,)):
    return AppliedAction(episode, tick, "v0", (0.1,), (0.9,), applied)


def chunk(generation=0, source=0, ready=2, episode="e", version="v0"):
    return ActionChunk(episode, version, generation, source, ready,
                       tuple((float(i),) for i in range(6)))


class ExecutionContracts(unittest.TestCase):
    def test_alignment_uses_applied_not_proposed(self):
        ledger = ActionLedger("e")
        ledger.append(entry(0))
        ledger.append(entry(1, applied=(0.5,)))
        out = align_prediction((1.0,), 0, 2, ledger,
                               lambda s, a: (s[0] + a[0],))
        self.assertEqual(out, (1.75,))

    def test_zero_delay_identity(self):
        self.assertEqual(align_prediction((2,), 0, 0, ActionLedger("e"),
                                          lambda s, a: (99,)), (2,))

    def test_no_future_action_leak(self):
        with self.assertRaises(ValueError):
            ActionLedger("e").window(0, 1)

    def test_reject_missing_history(self):
        with self.assertRaises(ValueError):
            ActionLedger("e").append(entry(1))

    def test_reject_cross_episode(self):
        with self.assertRaises(ValueError):
            ActionLedger("e").append(entry(episode="other"))

    def test_reject_duplicate_action(self):
        ledger = ActionLedger("e")
        ledger.append(entry())
        with self.assertRaises(ValueError):
            ledger.append(entry())

    def test_history_end_exclusive(self):
        ledger = ActionLedger("e")
        ledger.append(entry(0))
        ledger.append(entry(1))
        self.assertEqual([e.tick for e in ledger.window(0, 1)], [0])

    def test_drop_expired_chunk_prefix(self):
        q = ChunkQueue("e", "v0")
        self.assertEqual(q.install(chunk(), now=2), 4)
        self.assertEqual(q.action_at(2), (2.0,))

    def test_do_not_use_unfinished_inference(self):
        with self.assertRaises(ValueError):
            ChunkQueue("e", "v0").install(chunk(), now=1)

    def test_committed_prefix_preserved(self):
        q = ChunkQueue("e", "v0")
        q.install(chunk(), now=2)
        q.install(chunk(generation=1, source=2, ready=3), now=3,
                  committed_until=5)
        self.assertEqual(q.action_at(3), (3.0,))
        self.assertEqual(q.action_at(4), (4.0,))
        self.assertEqual(q.action_at(5), (3.0,))

    def test_late_old_chunk_cannot_overwrite(self):
        q = ChunkQueue("e", "v0")
        q.install(chunk(generation=2), now=2)
        self.assertEqual(q.install(chunk(generation=1), now=3), 0)
        self.assertEqual(q.action_at(3), (3.0,))

    def test_unknown_base_version_rejected(self):
        with self.assertRaises(ValueError):
            ChunkQueue("e", "v0").install(chunk(version="v1"), now=2)

    def test_new_episode_discards_old_chunk(self):
        with self.assertRaises(ValueError):
            ChunkQueue("other", "v0").install(chunk(), now=2)

    def test_expired_chunk_has_no_silent_fallback(self):
        q = ChunkQueue("e", "v0")
        self.assertEqual(q.install(chunk(), now=8), 0)
        with self.assertRaises(ValueError):
            q.action_at(8)

    def test_recenter_preserves_total_action(self):
        r = recenter_residual((0.5, -0.2), (0.4, -0.15), 0.2)
        for actual, expected in zip((0.4+r[0], -0.15+r[1]), (0.5, -0.2)):
            self.assertAlmostEqual(actual, expected)

    def test_infeasible_recenter_not_silently_clipped(self):
        with self.assertRaises(ValueError):
            recenter_residual((0.9,), (0.0,), 0.2)

    def test_nstep_bootstrap_discount(self):
        self.assertAlmostEqual(bootstrap_target((1, 2, 3), .9, 10,
            terminated=False, truncated=False, bootstrap_on_truncation=False),
            1 + .9*2 + .9**2*3 + .9**3*10)

    def test_success_terminal_never_bootstraps(self):
        self.assertEqual(bootstrap_target((0, 1), .9, 100,
            terminated=True, truncated=False, bootstrap_on_truncation=True), .9)

    def test_finite_horizon_timeout(self):
        self.assertEqual(bootstrap_target((0,), .9, 100,
            terminated=False, truncated=True, bootstrap_on_truncation=False), 0)

    def test_artificial_truncation_can_bootstrap(self):
        self.assertEqual(bootstrap_target((0,), .9, 100,
            terminated=False, truncated=True, bootstrap_on_truncation=True), 90)

    def test_nonfinite_action_rejected(self):
        with self.assertRaises(ValueError):
            entry(applied=(float("nan"),))


if __name__ == "__main__":
    unittest.main()
