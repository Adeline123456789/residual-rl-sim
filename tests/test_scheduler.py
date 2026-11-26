import unittest

from cf_resrl.scheduler import AbsoluteTickScheduler


def _chunk(start: float, length: int = 8):
    return [[start + index, -(start + index)] for index in range(length)]


class Provider:
    def __init__(self, start: float = 10.0):
        self.start = start
        self.calls = []

    def __call__(self, observation, source_tick, generation):
        self.calls.append((observation, source_tick, generation))
        return _chunk(self.start)


def _scheduler(provider, *, latency=2):
    return AbsoluteTickScheduler(
        initial_chunk=_chunk(0.0),
        provider=provider,
        latency_steps=latency,
        fallback_action=[0.0, 0.0],
        action_low=-100.0,
        action_high=100.0,
    )


def _advance(scheduler, *, refresh=False, observation=None, residual=(0.0, 0.0)):
    entry = scheduler.action(residual)
    scheduler.after_step(observation, request_refresh=refresh)
    return entry


class AbsoluteTickSemanticsTests(unittest.TestCase):
    def test_refresh_keeps_prefix_and_installs_at_absolute_index(self):
        keep = _scheduler(Provider())
        refresh_provider = Provider()
        refresh = _scheduler(refresh_provider)

        keep_rows = [_advance(keep)]
        refresh_rows = [_advance(refresh, refresh=True, observation="obs_1")]
        for _ in range(2):
            keep_rows.append(_advance(keep))
            refresh_rows.append(_advance(refresh))

        self.assertEqual(
            [row.applied_action for row in keep_rows],
            [row.applied_action for row in refresh_rows],
        )
        installed = refresh.action([0.0, 0.0])
        self.assertEqual(installed.tick, 3)
        self.assertEqual(installed.chunk_source_tick, 1)
        self.assertEqual(installed.chunk_index, 2)
        self.assertEqual(installed.applied_action, (12.0, -12.0))
        self.assertEqual(refresh_provider.calls, [("obs_1", 1, 1)])

    def test_pending_repeat_rule_is_counted_and_does_not_call_act(self):
        provider = Provider()
        scheduler = _scheduler(provider, latency=3)
        _advance(scheduler, refresh=True, observation="first")
        _advance(scheduler, refresh=True, observation="repeat")

        self.assertEqual(provider.calls, [("first", 1, 1)])
        self.assertEqual(scheduler.counters.refresh_attempts, 2)
        self.assertEqual(scheduler.counters.refresh_requests, 1)
        self.assertEqual(scheduler.counters.ignored_refresh_requests, 1)
        self.assertEqual(scheduler.counters.act_calls, 1)
        self.assertEqual(scheduler.counters.control_steps, 2)

    def test_clip_ledger_records_proposed_and_applied_residual(self):
        scheduler = AbsoluteTickScheduler(
            initial_chunk=[[0.9, -0.9]],
            provider=Provider(),
            latency_steps=0,
            fallback_action=[0.0, 0.0],
        )
        row = scheduler.action([0.4, -0.4])

        self.assertEqual(row.proposed_action, (1.3, -1.3))
        self.assertEqual(row.applied_action, (1.0, -1.0))
        self.assertAlmostEqual(row.applied_residual[0], 0.1)
        self.assertAlmostEqual(row.applied_residual[1], -0.1)

    def test_zero_residual_stays_zero_when_base_action_is_clipped(self):
        scheduler = AbsoluteTickScheduler(
            initial_chunk=[[1.2, -1.2]],
            provider=Provider(),
            latency_steps=0,
            fallback_action=[0.0, 0.0],
        )
        row = scheduler.action([0.0, 0.0])

        self.assertEqual(row.base_action, (1.2, -1.2))
        self.assertEqual(row.applied_base_action, (1.0, -1.0))
        self.assertEqual(row.applied_action, (1.0, -1.0))
        self.assertEqual(row.applied_residual, (0.0, 0.0))

    def test_ready_boundary_accepts_a_new_refresh(self):
        provider = Provider()
        scheduler = _scheduler(provider, latency=2)
        _advance(scheduler, refresh=True, observation="obs_1")
        _advance(scheduler)
        _advance(scheduler, refresh=True, observation="obs_3")

        self.assertEqual(
            provider.calls,
            [("obs_1", 1, 1), ("obs_3", 3, 2)],
        )
        self.assertEqual(scheduler.active_chunk.generation, 1)
        self.assertEqual(scheduler.pending_refresh.chunk.generation, 2)
        self.assertEqual(scheduler.counters.refresh_requests, 2)
        self.assertEqual(scheduler.counters.ignored_refresh_requests, 0)

    def test_exhausted_old_chunk_uses_registered_fallback(self):
        scheduler = AbsoluteTickScheduler(
            initial_chunk=[[1.0, -1.0]],
            provider=Provider(),
            latency_steps=2,
            fallback_action=[0.25, -0.25],
            action_low=-100.0,
            action_high=100.0,
        )
        _advance(scheduler)
        row = scheduler.action([0.0, 0.0])
        self.assertTrue(row.used_fallback)
        self.assertEqual(row.applied_action, (0.25, -0.25))

    def test_nominal_expiry_refills_without_refresh_treatment(self):
        provider = Provider()
        scheduler = _scheduler(provider)
        for _ in range(8):
            _advance(scheduler)
        self.assertTrue(scheduler.refill_if_exhausted("expiry"))
        self.assertEqual(scheduler.current_base_action(), (10.0, -10.0))
        self.assertEqual(provider.calls, [("expiry", 8, 1)])
        self.assertEqual(scheduler.counters.act_calls, 1)
        self.assertEqual(scheduler.counters.refresh_requests, 0)
        self.assertEqual(_advance(scheduler).chunk_index, 0)

    def test_pending_refresh_controls_refill_and_fallback(self):
        provider = Provider()
        scheduler = _scheduler(provider)
        for _ in range(7):
            _advance(scheduler)
        _advance(scheduler, refresh=True, observation="expiry")
        for _ in range(2):
            self.assertFalse(scheduler.refill_if_exhausted("waiting"))
            self.assertTrue(_advance(scheduler).used_fallback)
        self.assertFalse(scheduler.refill_if_exhausted("ready"))
        self.assertEqual(scheduler.current_base_action(), (12.0, -12.0))
        self.assertEqual(_advance(scheduler).chunk_index, 2)
        self.assertEqual(provider.calls, [("expiry", 8, 1)])


class SchedulerSnapshotTests(unittest.TestCase):
    def test_pending_snapshot_can_be_restored_twice_without_new_act_call(self):
        provider = Provider()
        scheduler = _scheduler(provider, latency=2)
        _advance(scheduler, refresh=True, observation="obs_1")
        snapshot = scheduler.snapshot()

        def suffix():
            rows = [_advance(scheduler), _advance(scheduler), _advance(scheduler)]
            return [
                (row.tick, row.generation, row.chunk_index, row.applied_action)
                for row in rows
            ]

        first = suffix()
        scheduler.restore(snapshot)
        second = suffix()
        scheduler.restore(snapshot)
        third = suffix()

        self.assertEqual(first, second)
        self.assertEqual(first, third)
        self.assertEqual(provider.calls, [("obs_1", 1, 1)])
        self.assertEqual(scheduler.counters.act_calls, 1)
        self.assertEqual(scheduler.counters.control_steps, 4)

    def test_snapshot_rejects_different_latency_contract(self):
        source = _scheduler(Provider(), latency=2)
        target = _scheduler(Provider(), latency=1)
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            target.restore(source.snapshot())


class OrderingTests(unittest.TestCase):
    def test_after_step_requires_a_submitted_action(self):
        scheduler = _scheduler(Provider())
        with self.assertRaisesRegex(RuntimeError, "action must be submitted"):
            scheduler.after_step(None, request_refresh=True)

    def test_each_action_requires_after_step(self):
        scheduler = _scheduler(Provider())
        scheduler.action([0.0, 0.0])
        with self.assertRaisesRegex(RuntimeError, "after_step must follow"):
            scheduler.action([0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
