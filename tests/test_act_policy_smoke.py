import copy
import json
from pathlib import Path
import unittest

from scripts.smoke_act_policy import EXPECTED, validate_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "cf_resrl" / "m0_act_policy_smoke.json"


class ActPolicySmokeSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_registered_spec_is_valid(self) -> None:
        validate_spec(self.config)
        self.assertEqual(self.config["upstream"]["lerobot_commit"], EXPECTED["lerobot_commit"])

    def test_wandb_equivalence_cannot_be_promoted(self) -> None:
        config = copy.deepcopy(self.config)
        config["checkpoint"]["wandb_equivalence"] = "VERIFIED"
        with self.assertRaisesRegex(ValueError, "must remain UNVERIFIED"):
            validate_spec(config)

    def test_checkpoint_hash_change_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["checkpoint"]["model_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "model_sha256"):
            validate_spec(config)

    def test_rollout_contract_change_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["task"]["rollout_steps"] = 5
        with self.assertRaisesRegex(ValueError, "task settings"):
            validate_spec(config)

    def test_runtime_scope_cannot_claim_a_full_lerobot_install(self) -> None:
        config = copy.deepcopy(self.config)
        config["environment"]["runtime_scope"] = "full_lerobot"
        with self.assertRaisesRegex(ValueError, "runtime scope"):
            validate_spec(config)


if __name__ == "__main__":
    unittest.main()
