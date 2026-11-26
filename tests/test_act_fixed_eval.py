from importlib.util import find_spec
from types import SimpleNamespace
import unittest

from scripts.run_act_fixed_eval import _checkpoint_plan, _collect, _lane_episode_ids


class ActFixedEvalTests(unittest.TestCase):
    def test_saved_step_labels_resolve_to_post_update_weights(self):
        spec = {
            "checkpoint_selection": {"checkpoint_step_label": 5},
            "training": {"save_frequency": 2},
        }
        labels = [2, 4, 5]
        during_training_steps = [2, 4, 6]
        files = {}
        for label, round_step in zip(labels, during_training_steps):
            path = f"training/policy_step_{label}/policy"
            files[f"{path}/config.json"] = {"sha256": f"config-{label}"}
            files[f"{path}/model.safetensors"] = {"sha256": f"model-{label}"}
            files[f"video-{round_step}"] = {"sha256": f"original-video-{round_step}"}
        manifest = {
            "fixed_checkpoint_labels": labels,
            "files": files,
            "evaluation_rounds": [
                {"optimization_steps_completed": step, "video": f"video-{step}"}
                for step in during_training_steps
            ],
        }
        planned = _checkpoint_plan(spec, manifest, None)
        self.assertEqual([r["checkpoint_step_label"] for r in planned], [5, 2, 4])
        self.assertEqual([r["optimization_steps_completed"] for r in planned], [6, 3, 5])
        self.assertEqual([r["model"]["sha256"] for r in planned], ["model-5", "model-2", "model-4"])
        self.assertEqual(
            [r["original_during_training_rollout"]["same_post_update_weights"] for r in planned],
            [True, False, False],
        )
        self.assertEqual(
            _checkpoint_plan(spec, manifest, [4])[0]["original_during_training_rollout"]["video_evidence"]["sha256"],
            "original-video-4",
        )
        with self.assertRaisesRegex(ValueError, "registered fixed ACT checkpoint"):
            _checkpoint_plan(spec, manifest, [6])

    def test_episode_cohort_is_a_fixed_disjoint_partition(self):
        lanes = _lane_episode_ids(100, 16)
        self.assertEqual([len(lane) for lane in lanes], [7] * 4 + [6] * 12)
        self.assertEqual(sorted(episode for lane in lanes for episode in lane), list(range(100)))

    @unittest.skipUnless(find_spec("torch") and find_spec("numpy"), "ACT runtime not installed")
    def test_slow_lanes_complete_the_same_cohort_and_extra_steps_are_counted(self):
        import numpy as np
        import torch

        class FakePolicy:
            def __init__(self):
                self.reset_calls = []

            def reset(self, env_ids=None):
                self.reset_calls.append(env_ids)

            def select_action(self, obs):
                return torch.tensor([[1.5, -1.0], [0.0, -1.5]])

        class FakeVectorEnv:
            num_envs = 2
            action_space = SimpleNamespace(low=-np.ones((2, 2)), high=np.ones((2, 2)))

            def reset(self, *, seed):
                self.reset_seed = seed
                self.steps = 0
                return None, {}

            def step(self, action):
                np.testing.assert_array_equal(action, [[1.0, -1.0], [0.0, -1.0]])
                self.steps += 1
                return (
                    None,
                    torch.tensor([1.0, 0.0]),
                    torch.tensor([True, False]),
                    torch.tensor([False, self.steps % 2 == 0]),
                    {},
                )

        env, policy = FakeVectorEnv(), FakePolicy()
        data = _collect(policy, env, seed=100, num_episodes=5, horizon=2)
        rows = data["episodes"]
        self.assertEqual(env.reset_seed, 100)
        self.assertEqual(policy.reset_calls, [None, [0], [0, 1], [0], [0, 1]])
        self.assertEqual([r["episode_seed"] for r in rows], [100, 101, 102, 103, 104])
        self.assertEqual([r["environment_steps"] for r in rows], [1, 2, 1, 2, 1])
        self.assertEqual([r["clipped_action_components"] for r in rows], [1, 2, 1, 2, 1])
        self.assertEqual(data["successes"], 3)
        self.assertEqual(data["environment_steps"], 8)
        self.assertEqual(data["scored_environment_steps"], 7)
        self.assertEqual(data["unscored_environment_steps"], 1)


if __name__ == "__main__":
    unittest.main()
