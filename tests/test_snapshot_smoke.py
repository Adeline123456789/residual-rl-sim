import importlib.util
import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "smoke_snapshot_restore", REPO_ROOT / "scripts" / "smoke_snapshot_restore.py"
)
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


def _row(offset=0.0):
    return {
        "suffix_index": 0,
        "action": [0.1, -0.1],
        "integration_state": [1.0 + offset, 2.0],
        "qpos": [3.0],
        "qvel": [4.0],
        "ctrl": [5.0],
        "observation_state": [6.0],
        "image_sha256": {"a": "abc", "b": "def"},
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "counters": {
            "episode_steps": 5,
            "timestep": 5,
            "cur_time": 0.25,
            "sim_time": 0.25,
            "done": False,
            "objects_on_pegs": [0.0],
            "gripper_current_action": [0.4],
        },
    }


class ComparisonTests(unittest.TestCase):
    def test_within_registered_tolerance(self):
        maxima = SMOKE.compare_replay([_row()], [_row(5e-13)], atol=1e-12)
        self.assertEqual(maxima["integration_state"], 5.000444502911705e-13)

    def test_numeric_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeded atol"):
            SMOKE.compare_replay([_row()], [_row(2e-12)], atol=1e-12)

    def test_image_hash_mismatch_is_rejected(self):
        observed = _row()
        observed["image_sha256"]["a"] = "changed"
        with self.assertRaisesRegex(ValueError, "image hash mismatch"):
            SMOKE.compare_replay([_row()], [observed], atol=1e-12)


class RegisteredConfigTests(unittest.TestCase):
    def test_actions_are_explicit_nonzero_square_actions(self):
        config = json.loads(
            (REPO_ROOT / "configs/cf_resrl/m0_snapshot_smoke.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertGreater(len(config["prefix_actions"]), 0)
        self.assertGreater(len(config["suffix_actions"]), 0)
        for action in config["prefix_actions"] + config["suffix_actions"]:
            self.assertEqual(len(action), 7)
            self.assertTrue(any(value != 0.0 for value in action))
            self.assertTrue(all(-1.0 <= value <= 1.0 for value in action))
        self.assertEqual(config["restored_replays"], 2)
        self.assertEqual(config["numeric_tolerance"], {"atol": 1e-12, "rtol": 0.0})


if __name__ == "__main__":
    unittest.main()
