from collections import deque
from types import SimpleNamespace
import unittest

from cf_resrl.resfit_resume_env import capture_resfit_env, restore_resfit_env


class FakeTensor:
    def __init__(self, value, device="cuda:0"):
        self.value = value
        self.device = device
        self.shape = (1,)
        self.dtype = "float32"

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self.value, self.device)

    def to(self, device, *, copy):
        assert copy
        return FakeTensor(self.value, device)


class FakePolicy:
    config = SimpleNamespace(device="cuda", temporal_ensemble_coeff=None, n_action_steps=2)

    def _ensure_action_queues(self, count):
        if not hasattr(self, "_action_queues"):
            self._action_queues = [deque(maxlen=self.config.n_action_steps) for _ in range(count)]


class FakeVector:
    num_envs = 1
    autoreset_mode = SimpleNamespace(value="SameStep")

    def __init__(self, state):
        self.state = state

    def call(self, method, *args):
        if method == "capture_training_state":
            return (self.state,)
        assert method == "restore_training_state"
        (self.state,) = args
        return (None,)


class ResfitResumeEnvTests(unittest.TestCase):
    def test_worker_state_and_pending_action_restore_into_new_wrapper(self):
        source = SimpleNamespace(
            vec_env=SimpleNamespace(vec_env=FakeVector("captured-worker-state")),
            base_policy=FakePolicy(),
            _last_base_naction=FakeTensor(3),
        )
        source.base_policy._ensure_action_queues(1)
        source.base_policy._action_queues[0].append(FakeTensor(9))
        obs = {"observation.base_action": source._last_base_naction}
        snapshot = capture_resfit_env(source, obs)

        source._last_base_naction.value = -1
        source.base_policy._action_queues[0][0].value = -2
        fresh = SimpleNamespace(
            vec_env=SimpleNamespace(vec_env=FakeVector("fresh-worker-state")),
            base_policy=FakePolicy(),
        )
        restored_obs = restore_resfit_env(fresh, snapshot, "cuda:0")

        self.assertEqual(fresh.vec_env.vec_env.state, "captured-worker-state")
        self.assertEqual(fresh._last_base_naction.value, 3)
        self.assertEqual(fresh.base_policy._action_queues[0][0].value, 9)
        self.assertEqual(restored_obs["observation.base_action"].value, 3)
        self.assertEqual(restored_obs["observation.base_action"].device, "cuda:0")
        self.assertEqual(snapshot.observation["observation.base_action"].device, "cpu")
        self.assertEqual(snapshot.act_policy.queues[0].items[0].value.device, "cpu")

    def test_rejects_non_same_step_vector(self):
        vector = FakeVector(None)
        vector.autoreset_mode = SimpleNamespace(value="NextStep")
        env = SimpleNamespace(vec_env=SimpleNamespace(vec_env=vector))
        with self.assertRaisesRegex(ValueError, "SAME_STEP"):
            capture_resfit_env(env, {})
