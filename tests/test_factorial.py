import unittest

from cf_resrl.factorial import BranchBudget, FactorialOutcome, assert_disjoint_groups


class FactorialOutcomeTests(unittest.TestCase):
    def test_factorial_identity(self) -> None:
        outcome = FactorialOutcome(v00=0.1, v01=0.4, v10=0.3, v11=0.9)
        self.assertAlmostEqual(outcome.base_refresh_gain, 0.3)
        self.assertAlmostEqual(outcome.residual_refresh_gain, 0.6)
        self.assertAlmostEqual(outcome.interaction, 0.3)
        self.assertTrue(outcome.verify_identity())

    def test_interaction_can_be_negative_while_refresh_is_still_best(self) -> None:
        outcome = FactorialOutcome(v00=0.0, v01=0.8, v10=0.2, v11=0.9)
        self.assertAlmostEqual(outcome.base_refresh_gain, 0.8)
        self.assertAlmostEqual(outcome.residual_refresh_gain, 0.7)
        self.assertAlmostEqual(outcome.interaction, -0.1)
        self.assertTrue(outcome.oracle_refresh(cost=0.2))

    def test_tie_prefers_keep(self) -> None:
        outcome = FactorialOutcome(v00=0.0, v01=0.0, v10=0.5, v11=0.6)
        self.assertFalse(outcome.oracle_refresh(cost=0.1))

    def test_oracle_has_zero_regret(self) -> None:
        outcome = FactorialOutcome(v00=0.0, v01=0.0, v10=0.2, v11=0.8)
        decision = outcome.oracle_refresh(cost=0.1)
        self.assertEqual(outcome.decision_regret(decision, cost=0.1), 0.0)
        self.assertAlmostEqual(outcome.decision_regret(False, cost=0.1), 0.5)

    def test_nonfinite_value_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FactorialOutcome(v00=float("nan"), v01=0.0, v10=0.0, v11=0.0)


class BudgetTests(unittest.TestCase):
    def test_equal_branch_budget_gives_direct_model_twice_the_contexts(self) -> None:
        budget = BranchBudget(400)
        self.assertEqual(budget.factorial_contexts, 100)
        self.assertEqual(budget.direct_two_branch_contexts, 200)
        self.assertEqual(budget.factorial_unused, 0)
        self.assertEqual(budget.direct_unused, 0)

    def test_budget_rejects_negative_values(self) -> None:
        with self.assertRaises(ValueError):
            BranchBudget(-1)


class SplitTests(unittest.TestCase):
    def test_disjoint_group_split(self) -> None:
        assert_disjoint_groups({"ep0", "ep1"}, {"ep2"}, {"ep3"})

    def test_group_leakage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_disjoint_groups({"ep0", "ep1"}, {"ep1"}, {"ep2"})


if __name__ == "__main__":
    unittest.main()
