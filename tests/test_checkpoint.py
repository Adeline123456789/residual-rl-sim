import unittest

from cf_resrl.checkpoint import (
    validate_act_config,
    validate_file_inventory,
    validate_unpublished_license,
)


def _valid_config():
    return {
        "type": "act",
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [9]},
            "observation.images.agentview": {"type": "VISUAL", "shape": [3, 84, 84]},
            "observation.images.robot0_eye_in_hand": {
                "type": "VISUAL",
                "shape": [3, 84, 84],
            },
        },
        "output_features": {"action": {"type": "ACTION", "shape": [7]}},
        "chunk_size": 20,
        "n_action_steps": 20,
        "vision_backbone": "resnet18",
        "use_vae": False,
    }


class ActConfigTests(unittest.TestCase):
    def test_square_act_contract(self) -> None:
        summary = validate_act_config(_valid_config())
        self.assertEqual(summary["state_dim"], 9)
        self.assertEqual(summary["action_dim"], 7)
        self.assertEqual(summary["camera_shape_chw"], [3, 84, 84])

    def test_wrong_camera_shape_is_rejected(self) -> None:
        config = _valid_config()
        config["input_features"]["observation.images.agentview"]["shape"] = [3, 96, 96]
        with self.assertRaises(ValueError):
            validate_act_config(config)


class InventoryTests(unittest.TestCase):
    def test_inventory_is_order_independent(self) -> None:
        expected = [
            {"path": "config.json", "bytes": 10, "lfs_sha256": None},
            {"path": "model.safetensors", "bytes": 20, "lfs_sha256": "abc"},
        ]
        validate_file_inventory(list(reversed(expected)), expected)

    def test_inventory_mismatch_is_rejected(self) -> None:
        expected = [{"path": "model.safetensors", "bytes": 20, "lfs_sha256": "abc"}]
        observed = [{"path": "model.safetensors", "bytes": 19, "lfs_sha256": "abc"}]
        with self.assertRaises(ValueError):
            validate_file_inventory(observed, expected)


class LicenseTests(unittest.TestCase):
    def test_missing_model_card_is_recorded(self) -> None:
        observation = validate_unpublished_license(
            None, ["safetensors", "region:us"], ["config.json", "model.safetensors"]
        )
        self.assertEqual(observation["license_tags"], [])
        self.assertFalse(observation["readme_present"])

    def test_license_tag_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_unpublished_license(
                None, ["license:apache-2.0"], ["config.json", "model.safetensors"]
            )


if __name__ == "__main__":
    unittest.main()
