import json
import inspect
from pathlib import Path
import tempfile
import unittest

from scripts.smoke_act_training import (
    REPO_ROOT,
    RESFIT_ROOT,
    _capture_runtime,
    _discard_heavy_smoke_artifacts,
    _evidence_checksums,
    build_training_argv,
    main,
    patch_states,
    _verify_checkpoint,
    validate_spec,
)


CONFIG_PATH = REPO_ROOT / "configs" / "cf_resrl" / "m0_act_training_smoke.json"


class ActTrainingSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_registered_smoke_is_one_real_batch256_optimizer_step(self) -> None:
        episodes = validate_spec(self.config)
        self.assertEqual(len(episodes), 240)
        training = self.config["training"]
        self.assertEqual(training["steps"], 1)
        self.assertEqual(training["batch_size"], 256)
        self.assertEqual(training["num_workers"], 0)
        self.assertFalse(training["wandb_enabled"])
        self.assertFalse(training["rollout_enabled"])
        self.assertIsNone(training["policy_kwargs"]["pretrained_backbone_weights"])
        self.assertEqual(self.config["checkpoint"]["step_label"], 0)
        self.assertEqual(self.config["checkpoint"]["reload"], "strict_local_only")

    def test_training_argv_has_no_wandb_or_rollout_and_uses_pinned_split(self) -> None:
        attempt = Path("/tmp/cf-resrl-act-training-smoke-contract")
        argv = build_training_argv(self.config, attempt)
        self.assertEqual(argv[0], "resfit.lerobot.scripts.train_bc_dexmg")
        self.assertEqual(argv[argv.index("--steps") + 1], "1")
        self.assertEqual(argv[argv.index("--batch_size") + 1], "256")
        self.assertEqual(argv[argv.index("--num_workers") + 1], "0")
        self.assertEqual(argv[argv.index("--dataset_revision") + 1], self.config["dataset"]["revision"])
        self.assertEqual(argv[argv.index("--episode_split") + 1], "train")
        self.assertEqual(argv[argv.index("--video_backend") + 1], "pyav")
        self.assertNotIn("--wandb_enable", argv)
        self.assertNotIn("--rollout_freq", argv)
        self.assertNotIn("--eval_env", argv)

    def test_registered_patches_are_cleanly_applicable(self) -> None:
        self.assertEqual(
            [row["state"] for row in patch_states(self.config)],
            ["not_applied", "not_applied"],
        )
        self.assertEqual(
            RESFIT_ROOT.joinpath(".git").exists() or RESFIT_ROOT.is_dir(),
            True,
        )

    def test_dry_run_writes_immutable_attempt_without_applying_patches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "smoke"
            result = main(["--dry-run", "--out", str(output_root)])
            self.assertEqual(result, 0)
            attempts = list((output_root / "attempts").iterdir())
            self.assertEqual(len(attempts), 1)
            status = json.loads((attempts[0] / "status.json").read_text(encoding="utf-8"))
            resolved = json.loads((attempts[0] / "resolved_config.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "PREPARED")
            self.assertTrue(status["run"]["dry_run"])
            self.assertEqual(resolved["resolved"]["episode_indices"], validate_spec(self.config))
            self.assertEqual(
                [row["state"] for row in patch_states(self.config)],
                ["not_applied", "not_applied"],
            )
            self.assertFalse((attempts[0] / "training").exists())
            self.assertFalse((attempts[0] / "runtime").exists())

    def test_batch_reduction_is_rejected(self) -> None:
        changed = json.loads(json.dumps(self.config))
        changed["training"]["batch_size"] = 2
        with self.assertRaisesRegex(ValueError, "optimizer-step contract"):
            validate_spec(changed)

    def test_strict_reload_passes_the_policy_directory_once(self) -> None:
        source = inspect.getsource(_verify_checkpoint)
        self.assertIn(
            "ACTPolicy.from_pretrained(\n        policy_dir,\n        config=policy_config,",
            source,
        )
        self.assertNotIn("policy_dir,\n        policy_dir,", source)

    def test_attempt_side_effects_begin_inside_failure_persistence_scope(self) -> None:
        source = inspect.getsource(main)
        attempt = source.index("attempt_dir.mkdir(parents=True, exist_ok=False)")
        guarded = source.index("try:", attempt)
        cache = source.index('(attempt_dir / "cache").mkdir()', attempt)
        self.assertLess(attempt, guarded)
        self.assertLess(guarded, cache)

    def test_runtime_is_bound_to_the_registered_current_conda_environment(self) -> None:
        source = inspect.getsource(_capture_runtime)
        self.assertIn('os.environ.get("CONDA_DEFAULT_ENV") != registered_environment', source)

    def test_heavy_smoke_payloads_are_hashed_then_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            heavy = (
                "training/policy_step_0/policy/model.safetensors",
                "training/latest/policy/model.safetensors",
                "training/latest/trainer_state.pt",
            )
            for index, name in enumerate(heavy):
                path = attempt / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"payload-{index}".encode())
            retained = attempt / "training/policy_step_0/policy/config.json"
            retained.write_text("{}", encoding="utf-8")

            discarded = _discard_heavy_smoke_artifacts(attempt)

            self.assertEqual(set(discarded), set(heavy))
            self.assertTrue(all(not (attempt / name).exists() for name in heavy))
            self.assertTrue(retained.is_file())
            for row in discarded.values():
                self.assertFalse(row["retained"])
                self.assertEqual(len(row["sha256"]), 64)
                self.assertIn("Git artifact limits", row["discard_reason"])

    def test_retained_evidence_manifest_excludes_heavy_payloads(self) -> None:
        source = inspect.getsource(_evidence_checksums)
        self.assertNotIn('"training/policy_step_0/policy/model.safetensors"', source)
        self.assertNotIn('"training/latest/trainer_state.pt"', source)
        self.assertIn('"training/latest/policy/config.json"', source)

    def test_full_latest_policy_is_part_of_strict_checkpoint_verification(self) -> None:
        source = inspect.getsource(_verify_checkpoint)
        self.assertIn('latest_policy_dir / "model.safetensors"', source)
        self.assertIn("latest_policy = ACTPolicy.from_pretrained(", source)
        self.assertIn('"strict_local_reload_latest": True', source)


if __name__ == "__main__":
    unittest.main()
