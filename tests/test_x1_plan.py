from collections import Counter
import json
from pathlib import Path

from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from cf_resrl import x1_plan as x1
from cf_resrl.dataset import sha256_file


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return sha256_file(path)


def _model(path, name):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode())
    return sha256_file(path)


class RegistrationTests(unittest.TestCase):
    def test_registered_cells_and_disjoint_source_cohorts(self):
        config = x1.load_x1_config()
        cells = x1.plan_x1(config)
        assert len(cells) == len({cell["cell_id"] for cell in cells}) == 216
        assert config["act_call_caps"] == [15, 20, 25]
        assert Counter(cell["act_evaluation_mode"] for cell in cells) == {
            "matched_training_base": 108,
            "base_swap_without_resfit_retraining": 108,
        }

        assigned = set()
        for distribution in config["axes"]["distribution"]:
            rows = x1.source_cohort(config, distribution)
            assert len(rows) == 160
            assert rows == x1.source_cohort(config, distribution)
            assert Counter((row["bank"], row["split"]) for row in rows) == {
                ("factorial", "train"): 60,
                ("factorial", "validation"): 20,
                ("factorial", "test"): 20,
                ("pair", "train"): 60,
            }
            for row in rows:
                assert row["distribution"] == distribution
                assert row["seed"] >= 20270124
                assert row["seed"] not in assigned
                assigned.add(row["seed"])
            assert len({row["source_episode_id"] for row in rows}) == 160


def _provenance(tmp_path):
    config = x1.load_x1_config()
    act = config["act"]
    act_attempt = tmp_path / "act_attempt"
    act["attempt"] = str(act_attempt)
    artifacts = {}
    for label in (25000, 49999):
        prefix = f"training/policy_step_{label}/policy/"
        digests = {
            "config.json": _write(act_attempt / (prefix + "config.json"), {"n_action_steps": 20}),
            "model.safetensors": _model(act_attempt / (prefix + "model.safetensors"), str(label)),
        }
        act["config_sha256"] = digests["config.json"]
        act["models"][str(label)] = digests["model.safetensors"]
        for name, digest in digests.items():
            key = prefix + name
            artifacts[key] = {"sha256": digest, "bytes": (act_attempt / key).stat().st_size}
    act["evidence_manifest_sha256"] = _write(act_attempt / "evidence_manifest.json", {
        "status": "COMPLETED",
        "attempt_id": act_attempt.name,
        "upstream": config["upstream"],
        "files": artifacts,
    })
    _write(act_attempt / "status.json", {
        "status": "COMPLETED", "dry_run": False, "returncode": 0,
        "checkpoint_selection": {"checkpoint_step_label": 49999},
        "fixed_final_checkpoint": {
            name: artifacts[f"training/policy_step_49999/policy/{name}"]
            for name in ("config.json", "model.safetensors")
        },
    })

    registered = tmp_path / config["resfit"]["training_config"]
    config_sha = _write(registered, {
        "upstream": config["upstream"],
        "base_policy": {"model_sha256": act["models"]["49999"]},
    })
    seed = 20261125
    origin = tmp_path / "attempts/origin"
    attempt = tmp_path / "attempts/resumed"
    locks = {}
    for name in ("conda-explicit.txt", "environment.lock.txt"):
        digest = _model(origin / "runtime" / name, name)
        _model(attempt / "runtime" / name, name)
        locks[name] = digest
    budget = config["resfit"]["registered_budget"]
    base_receipt = {
        "dry_run": False,
        "config_path": str(registered),
        "config_sha256": config_sha,
        "training_seed": seed,
        "overlay_mode": "resumable",
        "environment_step_budget": budget,
        "resume_checkpoint_interval_environment_steps": 100000,
        "resume_checkpoint_retention": 3,
        "status": "RUNNING",
        "runtime": {"locks": {
            name: {"sha256": digest} for name, digest in locks.items()
        }},
    }
    _write(origin / "resolved_launcher.json", {
        **base_receipt, "attempt_dir": str(origin), "resume": None,
    })
    generation = origin / "training/resume/step_000200000"
    generation_sha = _write(generation / "resume.json", {"global_step": 200000})
    receipt = {
        **base_receipt, "attempt_dir": str(attempt),
        "resume": {
            "origin_attempt": str(origin),
            "producer_attempt": str(origin),
            "restored_environment_steps": 200000,
            "runtime_lock_sha256": locks,
            "generation": str(generation),
            "manifest_sha256": generation_sha,
        },
    }
    _write(attempt / "resolved_launcher.json", receipt)
    models = []
    for step in range(10000, 300001, 10000):
        producer = origin if step <= 200000 else attempt
        model = producer / "training" / "models" / f"step_{step:09d}.pt"
        models.append({
            "global_step": step, "path": str(model),
            "selection": "fixed_environment_step",
            "sha256": _model(model, str(step)),
        })
    path = attempt / "training/checkpoint_manifest.json"
    manifest = {
        "selection": "fixed_environment_step",
        "checkpoint_interval_environment_steps": 10000,
        "environment_step_budget": budget,
        "action_ledger": {"records": 300000},
        "checkpoints": models,
    }
    status = {
        **receipt, "status": "COMPLETED", "child_returncode": 0,
        "resfit_clean_after_cleanup": True, "lerobot_clean_after_cleanup": True,
        "checkpoint_manifest_sha256": _write(path, manifest),
    }
    _write(attempt / "status.json", status)
    _write(attempt / "training/resolved_training_config.json", {
        "config": {"seed": seed, "task": "Square"},
        "environment_step_budget": budget,
        "normalization": {"action_scale": 0.1},
        "base_policy_dir": str(act_attempt / "training/policy_step_49999/policy"),
        "base_policy_artifacts": {
            "config.json": act["config_sha256"],
            "model.safetensors": act["models"]["49999"],
        },
    })
    return config, seed, attempt, origin, manifest, status


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        repo_root = patch.object(x1, "REPO_ROOT", root)
        repo_root.start()
        self.addCleanup(repo_root.stop)
        self.provenance = _provenance(root)

    def test_resumed_checkpoint_uses_original_model_and_act_swap(self):
        config, seed, attempt, origin, _, _ = self.provenance
        cell = {"resfit_seed": seed, "resfit_step": 150000, "act_label": 25000}
        policy = x1.resolve_cell_policies(config, cell, {seed: attempt})
        assert policy["resfit_model"] == origin / "training/models/step_000150000.pt"
        assert policy["resfit_attempt"] == attempt
        assert policy["act_dir"].parent.name == "policy_step_25000"
        assert policy["act_artifacts"]["model.safetensors"] != policy["training_base_artifacts"]["model.safetensors"]

        final = x1.resolve_cell_policies(
            config, {**cell, "resfit_step": 300000, "act_label": 49999}, {seed: attempt},
        )
        assert final["resfit_model"] == attempt / "training/models/step_000300000.pt"
        assert final["act_artifacts"] == final["training_base_artifacts"]


    def test_rejects_checkpoint_outside_its_lineage_prefix(self):
        config, seed, attempt, _, manifest, status = self.provenance
        same_bytes = attempt / "training/models/step_000150000.pt"
        _model(same_bytes, "150000")
        manifest["checkpoints"][14]["path"] = str(same_bytes)
        status["checkpoint_manifest_sha256"] = _write(attempt / "training/checkpoint_manifest.json", manifest)
        _write(attempt / "status.json", status)
        with self.assertRaisesRegex(ValueError, "outside its recorded attempt lineage"):
            x1.resolve_cell_policies(
                config, {"resfit_seed": seed, "resfit_step": 150000, "act_label": 49999}, {seed: attempt},
            )


    def test_rejects_a_completed_attempt_for_a_different_seed(self):
        config, seed, attempt, _, _, _ = self.provenance
        wrong_seed = seed + 1
        with self.assertRaisesRegex(ValueError, "differs from its registered launch"):
            x1.resolve_cell_policies(
                config, {"resfit_seed": wrong_seed, "resfit_step": 150000, "act_label": 49999}, {wrong_seed: attempt},
            )


    def test_rejects_altered_local_act_file(self):
        config, seed, attempt, _, _, _ = self.provenance
        file = Path(config["act"]["attempt"]) / "training/policy_step_25000/policy/model.safetensors"
        _model(file, "99999")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            x1.resolve_cell_policies(
                config, {"resfit_seed": seed, "resfit_step": 150000, "act_label": 25000}, {seed: attempt},
            )
