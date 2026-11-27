import unittest

from cf_resrl.e0_report import summarize_e0


def _row(episode_id, distribution, length, returns, *, split="train", bank="factorial"):
    branch_labels = ("V00", "V01", "V10", "V11") if bank == "factorial" else ("V10", "V11")
    return {
        "source_episode_id": episode_id,
        "distribution": distribution,
        "split": split,
        "bank": bank,
        "source_episode_steps": length,
        "source_episode_act_calls": 1,
        "source_episode_wall_seconds": 0.5,
        "branches": {
            label: {
                "environment_steps": index + 1,
                "act_calls": index,
                "wall_seconds": 0.25,
                "discounted_return": value,
            }
            for index, (label, value) in enumerate(zip(branch_labels, returns))
        },
    }


class E0ReportTest(unittest.TestCase):
    def test_distribution_weights_flip_ties_and_actual_budgets(self):
        short = _row("short", "natural", 2, (0.125, 0.375, 0.25, 0.25))
        long = _row("long", "natural", 6, (0.125, 0.125, 0.0, 0.5), split="test")
        enriched = _row("marked", "enriched", 4, (0.0, 0.5, 0.25, 0.75))
        pair = _row("extra", "natural", 5, (0.0, 0.75), bank="pair")
        report = summarize_e0(
            [short, long, enriched, pair], costs=[0.0, 0.25], bootstrap_samples=300, bootstrap_seed=12
        )

        overall = report["budgets"]["overall"]
        self.assertEqual(
            overall["source_scout"], {"episodes": 4, "environment_steps": 17, "act_calls": 4, "wall_seconds": 2.0}
        )
        self.assertEqual(overall["branches"]["rollouts"], 14)
        self.assertEqual(overall["branches"]["environment_steps"], 33)
        self.assertEqual(overall["branches"]["act_calls"], 19)
        self.assertEqual(overall["branches"]["wall_seconds"], 3.5)
        self.assertEqual(overall["branches"]["by_label"]["V11"]["environment_steps"], 14)
        self.assertEqual(report["budgets"]["by_bank"]["pair"]["branches"]["rollouts"], 2)
        self.assertEqual(report["budgets"]["by_split"]["test"]["source_scout"]["episodes"], 1)

        natural = report["distributions"]["natural"]
        uniform, weighted = natural["episode_uniform"], natural["control_tick_weighted"]
        self.assertEqual(uniform["source_episodes"], 2)
        self.assertEqual(uniform["g0"]["mean"], 0.125)
        self.assertEqual(uniform["g_delta"]["mean"], 0.25)
        self.assertEqual(uniform["interaction"]["mean"], 0.125)
        self.assertEqual(uniform["g_delta"]["mean_ci95"], [0.0, 0.5])
        self.assertEqual(uniform["decision_flips"][0]["flip_rate"], 1.0)
        self.assertEqual(uniform["decision_flips"][1]["flip_rate"], 0.5)
        self.assertEqual(uniform["decision_flips"][1]["base_refresh_rate"], 0.0)
        self.assertEqual(weighted["total_weight"], 8)
        self.assertEqual(weighted["g0"]["mean"], 0.0625)
        self.assertEqual(weighted["g_delta"]["mean"], 0.375)
        self.assertEqual(weighted["decision_flips"][1]["keep_to_refresh_rate"], 0.75)
        self.assertEqual(weighted["g_delta"]["median"], 0.5)

        enriched_report = report["distributions"]["enriched"]
        self.assertEqual(set(enriched_report), {"episode_uniform"})
        self.assertEqual(enriched_report["episode_uniform"]["decision_flips"][0]["flip_rate_ci95"], [0.0, 0.0])

    def test_bootstrap_reproducible_under_row_reordering(self):
        rows = [
            _row("1", "natural", 2, (0.0, 0.0, 0.0, 1.0)),
            _row("2", "natural", 5, (0.0, 0.25, 0.0, 0.0)),
            _row("3", "natural", 3, (0.0, 0.0, 0.0, 0.75)),
        ]
        options = {"costs": [0.1, 0.5], "bootstrap_samples": 129, "bootstrap_seed": 5}
        self.assertEqual(summarize_e0(rows, **options), summarize_e0(reversed(rows), **options))

    def test_rejects_source_episode_reused_across_banks(self):
        four = _row("same", "natural", 2, (0.0, 0.0, 0.0, 1.0))
        pair = _row("same", "enriched", 2, (0.0, 1.0), bank="pair")
        with self.assertRaisesRegex(ValueError, "duplicate source episode"):
            summarize_e0([four, pair], costs=[0], bootstrap_samples=2, bootstrap_seed=1)

    def test_pair_only_has_actual_budget_and_no_outcome_distribution(self):
        pair = _row("pair", "natural", 2, (0.0, 1.0), bank="pair")
        report = summarize_e0([pair], costs=[0], bootstrap_samples=2, bootstrap_seed=1)
        self.assertFalse(report["distributions"])
        self.assertEqual(report["budgets"]["overall"]["branches"]["rollouts"], 2)


if __name__ == "__main__":
    unittest.main()
