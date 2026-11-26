import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

if importlib.util.find_spec("numpy") is not None:
    import numpy as np

    from cf_resrl.offline_data import (
        CONTEXT_FIELDS,
        FACTORIAL_BRANCHES,
        FEATURE_SHAPES,
        PAIR_BRANCHES,
        load_e0,
        pack_features,
        select_budget,
    )


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _features():
    return {
        field: np.full(shape, index + 1, dtype=np.float32)
        for index, (field, shape) in enumerate(FEATURE_SHAPES.items())
    }


def _write_index(attempt, rows):
    index = attempt / "group_index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _write_json(attempt / "status.json", {
        "status": "COMPLETED",
        "planned_groups": len(rows),
        "completed_groups": len(rows),
        "index_sha256": _sha(index),
        "summary_sha256": _sha(attempt / "summary.json"),
    })


def _fixture(attempt):
    (attempt / "features").mkdir()
    _write_json(attempt / "feature_schema.json", {
        field: {"dtype": "float32", "shape": list(shape)} for field, shape in FEATURE_SHAPES.items()
    })
    _write_json(attempt / "summary.json", {})
    protocol = {
        "task": "Square",
        "dataset": "synthetic_test_fixture",
        "refresh_delay_steps": 2,
        "selector_seed": 10,
        "factorial_cohorts": {},
        "factorial_split_counts": {"train": 2, "validation": 1, "test": 1},
        "pair_training_cohorts": {},
    }
    rows = []
    plan = []
    for distribution, seed_offset in (("natural", 100), ("enriched", 200)):
        protocol["factorial_cohorts"][distribution] = {"first_seed": seed_offset, "size": 4}
        protocol["pair_training_cohorts"][distribution] = {"first_seed": seed_offset + 4, "size": 2}
        for index, split in enumerate(("train", "train", "validation", "test", "train", "train")):
            seed = seed_offset + index
            bank = "factorial" if index < 4 else "pair"
            episode_id = f"square_e0_{seed}"
            relative = f"features/{episode_id}.npz"
            feature_path = attempt / relative
            np.savez_compressed(feature_path, **_features())
            record = {
                "source_episode_id": episode_id,
                "seed": seed,
                "distribution": distribution,
                "split": split,
                "bank": bank,
                "selector_seed": protocol["selector_seed"] + seed,
            }
            plan.append(record)
            labels = FACTORIAL_BRANCHES if bank == "factorial" else PAIR_BRANCHES
            returns = {"V00": 0.75, "V01": 0.50, "V10": 0.50, "V11": 0.75}
            rows.append({
                **record,
                "source_tick": 10,
                "source_episode_steps": 200,
                "source_episode_act_calls": 10,
                "source_episode_wall_seconds": 2.0,
                "branches": {
                    label: {
                        "label": label,
                        "environment_steps": (
                            8 if bank == "pair" else 1 if label in PAIR_BRANCHES else 3
                        ),
                        "remaining_horizon": 290,
                        "discounted_return": returns[label],
                        "act_calls": 1,
                        "wall_seconds": 0.1,
                    }
                    for label in labels
                },
                "features": relative,
                "features_sha256": _sha(feature_path),
            })
    _write_json(attempt / "run_manifest.json", {
        "protocol": protocol,
        "source_plan": plan,
        "identities": {"policy_identity": "synthetic"},
    })
    _write_index(attempt, rows)
    return rows


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None, "NumPy needed for offline probe data")
class OfflineDataTests(unittest.TestCase):
    def test_frozen_feature_layout_does_not_depend_on_mapping_order(self):
        features = _features()
        context, residual = pack_features(dict(reversed(list(features.items()))))
        np.testing.assert_array_equal(context, np.concatenate([
            features[field].reshape(-1) for field in CONTEXT_FIELDS
        ]))
        np.testing.assert_array_equal(residual, features["residual"])
        self.assertEqual(context.dtype, np.float32)
        self.assertEqual((context.shape, residual.shape), ((306,), (7,)))
        with self.assertRaisesRegex(ValueError, "visible"):
            pack_features({**features, "privileged_pose": np.zeros(3, dtype=np.float32)})

    def test_budget_prefixes_are_nested_and_joint_cap_never_skips(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            _fixture(attempt)
            dataset = load_e0(attempt)
            half = select_budget(dataset, "natural", 0.5, 17)
            full = select_budget(dataset, "natural", 1.0, 17)

        self.assertEqual(full.factorial.source_episode_ids[:1], half.factorial.source_episode_ids)
        self.assertEqual(full.b6_rollout.source_episode_ids[:2], half.b6_rollout.source_episode_ids)
        self.assertEqual(full.validation.source_episode_ids, half.validation.source_episode_ids)
        self.assertEqual(full.test.source_episode_ids, half.test.source_episode_ids)
        self.assertEqual(full.factorial.budget["branches"]["rollouts"], 8)
        self.assertEqual(full.b6_rollout.budget["branches"]["rollouts"], 8)
        self.assertEqual(full.b6_rollout.budget["branches"]["environment_steps"], 36)
        self.assertEqual(full.b6_joint.source_episode_ids, full.b6_rollout.source_episode_ids[:1])
        self.assertEqual(full.b6_joint.budget["branches"]["environment_steps"], 2)
        self.assertEqual(full.manifest["b6_joint_unused"], {
            "branch_rollouts": 6, "branch_environment_steps": 14,
        })
        self.assertEqual(full.factorial.budget["source_scout"]["environment_steps"], 400)
        np.testing.assert_array_equal(full.factorial.g0, [-0.25, -0.25])
        np.testing.assert_array_equal(full.factorial.g_delta, [0.25, 0.25])
        self.assertIsNone(full.b6_rollout.g0)
        train_ids = set(full.factorial.source_episode_ids) | set(full.b6_rollout.source_episode_ids)
        self.assertFalse(train_ids & set(full.validation.source_episode_ids + full.test.source_episode_ids))

    def test_feature_checksum_binds_actual_loaded_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            rows = _fixture(attempt)
            feature_file = attempt / rows[-1]["features"]
            with feature_file.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Feature checksum mismatch"):
                load_e0(attempt)

    def test_duplicate_episode_cannot_enter_another_split(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            rows = _fixture(attempt)
            manifest_file = attempt / "run_manifest.json"
            manifest = json.loads(manifest_file.read_text())
            duplicates = {
                key: rows[0][key] for key in ("source_episode_id", "seed", "selector_seed")
            }
            rows[2].update(duplicates)
            manifest["source_plan"][2].update(duplicates)
            _write_json(manifest_file, manifest)
            _write_index(attempt, rows)
            with self.assertRaisesRegex(ValueError, "duplicate E0 episode"):
                load_e0(attempt)
