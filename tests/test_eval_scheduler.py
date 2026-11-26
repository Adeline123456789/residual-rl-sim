import functools
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from scripts.eval_scheduler import MEASUREMENTS, ORIGINAL, _experiment, _seeds, _settings


CONFIG = {
    "act_budgets": [15, 25], "fixed_period": 10, "norm_thresholds": [0.1, 0.2],
    "learned_costs": [0, 0.1], "bootstrap_samples": 50, "bootstrap_seed": 7,
    "validation_seeds": {"first": 20, "count": 2}, "test_seeds": {"first": 40, "count": 3},
}


class SchedulerProtocolTests(unittest.TestCase):
    def test_seeds_exclude_prior_data_and_each_other(self):
        self.assertEqual(_seeds(CONFIG, {1, 2})["validation"], (20, 21))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            _seeds(CONFIG, {40})
        with self.assertRaisesRegex(ValueError, "disjoint"):
            _seeds({**CONFIG, "test_seeds": {"first": 21, "count": 1}}, set())

    def test_call_caps_cover_the_nominal_horizon(self):
        with self.assertRaisesRegex(ValueError, "full-horizon"):
            _settings({**CONFIG, "act_budgets": [14, 25]}, 300, 20)
        settings = _settings(CONFIG, 300, 20)
        self.assertEqual(settings[15][ORIGINAL], settings[25][ORIGINAL])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "requires NumPy")
    def test_validation_is_locked_before_test_and_fixed_performance_can_reuse_scores(self):
        settings = _settings(CONFIG, 300, 20)
        cohorts = _seeds(CONFIG, set())
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            actual = []

            @functools.lru_cache
            def evaluate(split, setting):
                if split == "test":
                    self.assertTrue((out / "validation_selection.json").exists())
                actual.append((split, setting))
                rows = []
                for offset, seed in enumerate(cohorts[split]):
                    cheap = setting.parameter == 0.1 and setting.method == "P1"
                    success = (offset == 0 or not cheap) if setting.method != ORIGINAL else offset == 0
                    if split == "test" and cheap:
                        success = False
                    rows.append({
                        **dict.fromkeys(MEASUREMENTS, 0), "seed": seed, "success": success,
                        "return": int(success), "act_calls": 7 if cheap else 9,
                        "episode_id": f"{split}-{setting}-{seed}",
                    })
                return rows

            summary = _experiment(out, CONFIG, settings, cohorts, evaluate)
            locked = json.loads((out / "validation_selection.json").read_text())
            self.assertEqual(locked["fixed_budget"]["15"]["P1"]["parameter"], 0)
            self.assertEqual(locked["fixed_performance"]["P1"]["parameter"], 0.1)
            self.assertFalse(summary["fixed_task_performance"]["P1"]["met_target_on_test"])
            self.assertEqual(sum(setting.method == ORIGINAL for _, setting in actual), 2)


if __name__ == "__main__":
    unittest.main()
