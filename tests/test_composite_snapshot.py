from collections import deque
from dataclasses import dataclass, replace
import unittest
from unittest.mock import patch

import cf_resrl.composite_snapshot as composite


EVENTS = []


class FakeTensor:
    def __init__(self, values, *, dtype="float32", device="cuda:0"):
        self.values = list(values)
        self.dtype = dtype
        self.device = device
        self.shape = (len(self.values),)

    def detach(self):
        return self

    def clone(self):
        EVENTS.append("act_or_rng_clone")
        return FakeTensor(self.values, dtype=self.dtype, device=self.device)

    def __eq__(self, other):
        return (
            isinstance(other, FakeTensor)
            and self.values == other.values
            and self.dtype == other.dtype
            and self.device == other.device
        )


class FakeCuda:
    available = True
    device_count_value = 1
    states = [FakeTensor([20], device="cpu")]

    @classmethod
    def is_available(cls):
        return cls.available

    @classmethod
    def device_count(cls):
        return cls.device_count_value

    @classmethod
    def get_rng_state_all(cls):
        return cls.states

    @classmethod
    def set_rng_state_all(cls, states):
        EVENTS.append("rng_cuda")
        cls.states = states


class FakeTorch:
    cpu_state = FakeTensor([10], device="cpu")
    cuda = FakeCuda

    @classmethod
    def get_rng_state(cls):
        return cls.cpu_state

    @classmethod
    def set_rng_state(cls, state):
        EVENTS.append("rng_cpu")
        cls.cpu_state = state


@dataclass(frozen=True)
class FakeSchedulerState:
    logical_tick: int
    marker: str = "scheduler-state"


class FakeScheduler:
    def __init__(self, tick=5):
        self.state = FakeSchedulerState(tick)
        self.restore_calls = 0

    def snapshot(self):
        return self.state

    def restore(self, state):
        EVENTS.append("scheduler")
        self.state = state
        self.restore_calls += 1


class Config:
    temporal_ensemble_coeff = None
    n_action_steps = 4
    device = "cuda"


class FakePolicy:
    def __init__(self):
        self.config = Config()
        self._action_queues = [
            deque(
                [FakeTensor([1.0, 2.0]), FakeTensor([3.0, 4.0])],
                maxlen=4,
            )
        ]
        self.reset_calls = 0
        self.forward_calls = 0

    def reset(self):
        self.reset_calls += 1

    def forward(self, *_args, **_kwargs):
        self.forward_calls += 1


class CompositeSnapshotTests(unittest.TestCase):
    POLICY_IDENTITY = "a" * 64
    PROVIDER_IDENTITY = "registered-act-chunk-provider-v1"

    def setUp(self):
        EVENTS.clear()
        FakeTorch.cpu_state = FakeTensor([10], device="cpu")
        FakeCuda.available = True
        FakeCuda.device_count_value = 1
        FakeCuda.states = [FakeTensor([20], device="cpu")]
        self.environment = type(
            "FakeEnvironmentSnapshot",
            (),
            {"wrapper_state": {"_last_obs": {"state": [1.0, 2.0]}}},
        )()
        self.wrapper = object()
        self.scheduler = FakeScheduler()
        self.policy = FakePolicy()

    def capture(self):
        with patch.object(
            composite, "capture_square_snapshot", return_value=self.environment
        ):
            return composite.capture_composite_snapshot(
                self.wrapper,
                self.scheduler,
                self.policy,
                policy_identity=self.POLICY_IDENTITY,
                provider_identity=self.PROVIDER_IDENTITY,
                observation_tick=5,
                residual_history=([0.1, -0.1],),
                torch_api=FakeTorch,
            )

    def restore(self, snapshot):
        def restore_environment(_wrapper, state, *, mujoco_api=None):
            self.assertIs(state, self.environment)
            EVENTS.append("environment")

        with patch.object(
            composite, "restore_square_snapshot", side_effect=restore_environment
        ):
            return composite.restore_composite_snapshot(
                self.wrapper,
                self.scheduler,
                self.policy,
                snapshot,
                policy_identity=self.POLICY_IDENTITY,
                provider_identity=self.PROVIDER_IDENTITY,
                torch_api=FakeTorch,
            )

    def test_restores_twice_without_reset_forward_or_snapshot_mutation(self):
        snapshot = self.capture()
        saved_values = snapshot.act_policy.queues[0].items[0].value.values.copy()
        EVENTS.clear()

        self.policy._action_queues[0].popleft()
        self.scheduler.state = FakeSchedulerState(99, "changed")
        FakeTorch.cpu_state.values[0] = 99
        first = self.restore(snapshot)
        self.policy._action_queues[0][0].values[0] = 500
        first.current_observation["state"][0] = 500
        first.residual_history[0][0] = 500
        second = self.restore(snapshot)

        self.assertEqual(
            [item.values for item in self.policy._action_queues[0]],
            [[1.0, 2.0], [3.0, 4.0]],
        )
        self.assertEqual(second.current_observation, {"state": [1.0, 2.0]})
        self.assertEqual(second.residual_history, ([0.1, -0.1],))
        self.assertEqual(snapshot.act_policy.queues[0].items[0].value.values, saved_values)
        self.assertEqual(self.scheduler.restore_calls, 2)
        self.assertEqual(self.policy.reset_calls, 0)
        self.assertEqual(self.policy.forward_calls, 0)

    def test_restore_order_is_environment_act_scheduler_rng(self):
        snapshot = self.capture()
        EVENTS.clear()
        self.restore(snapshot)

        environment_index = EVENTS.index("environment")
        scheduler_index = EVENTS.index("scheduler")
        rng_index = EVENTS.index("rng_cpu")
        act_clone_index = EVENTS.index("act_or_rng_clone")
        self.assertLess(environment_index, act_clone_index)
        self.assertLess(act_clone_index, scheduler_index)
        self.assertLess(scheduler_index, rng_index)

    def test_rejects_queue_topology_and_tensor_metadata_changes(self):
        snapshot = self.capture()
        self.policy._action_queues.append(deque([], maxlen=4))
        with self.assertRaisesRegex(ValueError, "topology"):
            self.restore(snapshot)

        self.policy = FakePolicy()
        self.policy._action_queues[0][0] = FakeTensor(
            [1.0, 2.0], dtype="float64"
        )
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.restore(snapshot)

    def test_rejects_temporal_ensemble_and_misaligned_tick(self):
        self.policy.config.temporal_ensemble_coeff = 0.01
        with self.assertRaisesRegex(NotImplementedError, "action-queue mode"):
            self.capture()

        self.policy = FakePolicy()
        with patch.object(
            composite, "capture_square_snapshot", return_value=self.environment
        ):
            with self.assertRaisesRegex(ValueError, "observation tick"):
                composite.capture_composite_snapshot(
                    self.wrapper,
                    self.scheduler,
                    self.policy,
                    policy_identity=self.POLICY_IDENTITY,
                    provider_identity=self.PROVIDER_IDENTITY,
                    observation_tick=4,
                    torch_api=FakeTorch,
                )

    def test_binds_policy_and_provider_identities_before_restore(self):
        snapshot = self.capture()

        with self.assertRaisesRegex(ValueError, "policy identity"):
            composite.restore_composite_snapshot(
                self.wrapper,
                self.scheduler,
                self.policy,
                snapshot,
                policy_identity="b" * 64,
                provider_identity=self.PROVIDER_IDENTITY,
                torch_api=FakeTorch,
            )
        with self.assertRaisesRegex(ValueError, "provider identity"):
            composite.restore_composite_snapshot(
                self.wrapper,
                self.scheduler,
                self.policy,
                snapshot,
                policy_identity=self.POLICY_IDENTITY,
                provider_identity="another-provider",
                torch_api=FakeTorch,
            )

        self.assertEqual(self.scheduler.restore_calls, 0)

    def test_environment_last_observation_is_authoritative(self):
        snapshot = self.capture()

        self.assertEqual(snapshot.current_observation, {"state": [1.0, 2.0]})
        self.environment.wrapper_state["_last_obs"]["state"][0] = 700.0
        self.assertEqual(snapshot.current_observation, {"state": [1.0, 2.0]})

        with self.assertRaisesRegex(TypeError, "current_observation"):
            composite.capture_composite_snapshot(
                self.wrapper,
                self.scheduler,
                self.policy,
                policy_identity=self.POLICY_IDENTITY,
                provider_identity=self.PROVIDER_IDENTITY,
                observation_tick=5,
                current_observation={"state": [999.0, 999.0]},
                torch_api=FakeTorch,
            )

    def test_validates_cuda_device_and_rng_state_topology(self):
        snapshot = self.capture()
        self.assertEqual(snapshot.torch_rng.cuda_device_count, 1)

        FakeCuda.device_count_value = 2
        FakeCuda.states = [
            FakeTensor([20], device="cpu"),
            FakeTensor([30], device="cpu"),
        ]
        with self.assertRaisesRegex(ValueError, "device count"):
            self.restore(snapshot)

        FakeCuda.device_count_value = 1
        malformed = replace(
            snapshot,
            torch_rng=replace(snapshot.torch_rng, cuda_states=()),
        )
        with self.assertRaisesRegex(ValueError, "state count"):
            self.restore(malformed)

        FakeCuda.device_count_value = 2
        FakeCuda.states = [FakeTensor([20], device="cpu")]
        with self.assertRaisesRegex(ValueError, "state count"):
            self.capture()


if __name__ == "__main__":
    unittest.main()
