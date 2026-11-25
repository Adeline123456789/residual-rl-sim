import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/cf_resrl/m0_resfit_training.json"
PATCH_PATH = REPO_ROOT / "patches/resfit/residual_td3_cf_resrl_eval.patch"
LAUNCHER_PATH = REPO_ROOT / "scripts/run_resfit_training.py"
APPLY_PATCH_SCRIPT = REPO_ROOT / "scripts/apply_resfit_cf_resrl_patch.sh"
RESFIT_ROOT = REPO_ROOT / "third_party/resfit"
LEROBOT_ROOT = RESFIT_ROOT / "deps/lerobot"


class ResfitTrainingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.patch = PATCH_PATH.read_text(encoding="utf-8")
        self.added_lines = "\n".join(
            line[1:]
            for line in self.patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )

    def test_registered_dataset_split_is_pinned(self) -> None:
        dataset = self.config["dataset"]
        manifest_path = REPO_ROOT / dataset["split_manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.assertEqual(digest, dataset["split_manifest_sha256"])
        self.assertEqual(manifest["dataset_revision"], dataset["revision"])
        self.assertEqual(len(manifest["splits"][dataset["split"]]), dataset["episodes"])

    def test_registered_patch_targets_and_hashes_are_pinned(self) -> None:
        patches = self.config["upstream"]["patches"]
        self.assertEqual([row["target_root"] for row in patches], ["resfit", "resfit", "lerobot"])
        for row in patches:
            digest = hashlib.sha256((REPO_ROOT / row["path"]).read_bytes()).hexdigest()
            self.assertEqual(digest, row["sha256"])

    def test_fixed_step_checkpoint_contract_is_present(self) -> None:
        training = self.config["training"]
        self.assertEqual(training["selection"], "fixed_environment_step")
        self.assertEqual(training["total_environment_steps"], 300000)
        self.assertEqual(training["warmup_environment_steps"], 10000)
        self.assertEqual(training["main_environment_steps"], 290000)
        self.assertEqual(training["checkpoint_interval_environment_steps"], 10000)
        self.assertEqual(training["checkpoint_contract"], "evaluation_loadable_not_training_resumable")
        for token in (
            "while global_step < cfg.algo.total_timesteps",
            "_save_fixed_checkpoint(global_step)",
            '"checkpoint_contract": "evaluation_loadable_not_training_resumable"',
            'global_step = cfg.algo.learning_starts',
            '"warmup": cfg.algo.learning_starts',
            '"main": cfg.algo.total_timesteps - cfg.algo.learning_starts',
            'normalization=normalization_state',
            'base_policy_artifacts=base_policy_artifacts',
            '"torch_cpu": torch.get_rng_state()',
            "def load_checkpoint(agent: QAgent",
        ):
            self.assertIn(token, self.patch)

    def test_train_split_stats_and_applied_action_are_recorded(self) -> None:
        for token in (
            "training_stats = dataset.stats",
            'action_stats=training_stats["action"]',
            'state_stats=training_stats["observation.state"]',
            'info["proposed_scaled_action"]',
            "self.vec_env.action_space.low",
            "self.vec_env.action_space.high",
            'info["scaled_action"] = applied_combined_naction',
            'info["applied_env_action"]',
            'info["action_clip_mask"]',
            'action_ledger_path.open("x"',
            '"sha256": _sha256_file(action_ledger_path)',
        ):
            self.assertIn(token, self.added_lines)

    def test_autoreset_clears_act_queue_before_next_action(self) -> None:
        wrapper_patch = self.patch.split(
            "diff --git a/resfit/rl_finetuning/wrappers/residual_env_wrapper.py", 1
        )[1]
        done = wrapper_patch.index("done = terminated | truncated")
        reset = wrapper_patch.index("self.base_policy.reset(env_ids=reset_ids)", done)
        select = wrapper_patch.index("base_action = self.base_policy.select_action(raw_obs)", done)
        self.assertLess(reset, select)

    def test_main_step_tick_is_advanced_before_ledger_and_evaluation(self) -> None:
        training_patch = self.patch.split(
            "diff --git a/resfit/rl_finetuning/utils/checkpoint.py", 1
        )[0]
        upstream_source = (
            RESFIT_ROOT / "resfit/rl_finetuning/scripts/train_residual_td3.py"
        ).read_text(encoding="utf-8")
        upstream_step = upstream_source.index(
            "next_obs, reward, terminated, truncated, info = env.step(action)"
        )
        upstream_roll = upstream_source.index("obs = next_obs  # roll", upstream_step)
        self.assertLess(upstream_step, upstream_roll)

        main_loop = training_patch.index("while global_step < cfg.algo.total_timesteps")
        roll = training_patch.index("obs = next_obs  # roll", main_loop)
        advance = training_patch.index("global_step += cfg.num_envs", roll)
        ledger = training_patch.index('_record_applied_action(global_step, "main", info)', advance)
        checkpoint = training_patch.index("if global_step % cfg.checkpoint_interval == 0", ledger)
        self.assertLess(roll, advance)
        self.assertLess(advance, ledger)
        self.assertLess(ledger, checkpoint)
        self.assertNotIn('_record_applied_action(global_step + cfg.num_envs, "main", info)', training_patch)

    def test_checkpoint_evaluation_is_bound_to_saved_weights(self) -> None:
        save = self.added_lines.index("def _save_fixed_checkpoint(step: int)")
        checkpoint = self.added_lines.index("_sha256_file(checkpoint_path)", save)
        evaluation = self.added_lines.index("run_dexmg_evaluation(", checkpoint)
        self.assertLess(checkpoint, evaluation)
        self.assertIn("global_step=step", self.added_lines[evaluation:])
        self.assertIn("episode_records=episodes", self.added_lines[evaluation:])

    def test_launcher_dry_run_resolves_exact_hydra_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [sys.executable, str(LAUNCHER_PATH), "--dry-run", "--out", temp_dir],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            output = json.loads(result.stdout.strip().splitlines()[-1])
            attempt_dir = Path(output["attempt_dir"])
            resolved = json.loads((attempt_dir / "resolved_launcher.json").read_text(encoding="utf-8"))
            command = resolved["command"]

            self.assertEqual(resolved["status"], "DRY_RUN_VALIDATED")
            self.assertTrue((attempt_dir / "cache").is_dir())
            self.assertFalse((attempt_dir / "training").exists())
            self.assertIn("--config-name=residual_td3_square_config", command)
            self.assertIn("cf-resrl-resfit", command)
            self.assertEqual(resolved["validation"]["base_policy_source"]["checkpoint_selection"]["checkpoint_step_label"], 49999)
            expected = {
                "seed": "20261124",
                "offline_data.name": "ankile/robomimic-mh-square-image",
                "offline_data.revision": self.config["dataset"]["revision"],
                "offline_data.split": "train",
                "offline_data.video_backend": "pyav",
                "offline_data.num_episodes": "240",
                "algo.learning_starts": "10000",
                "algo.total_timesteps": "300000",
                "checkpoint_interval": "10000",
                "wandb.mode": "disabled",
                "hydra.job.chdir": "false",
            }
            overrides = dict(arg.split("=", 1) for arg in command if "=" in arg)
            for key, value in expected.items():
                self.assertEqual(overrides[key], value)
            self.assertEqual(
                overrides["offline_data.expected_split_manifest_sha256"],
                self.config["dataset"]["split_manifest_sha256"],
            )
            self.assertEqual(
                overrides["base_policy.expected_config_sha256"],
                self.config["base_policy"]["config_sha256"],
            )
            self.assertEqual(
                overrides["base_policy.expected_model_sha256"],
                self.config["base_policy"]["model_sha256"],
            )

    def test_launcher_records_validation_failure_in_immutable_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            invalid = json.loads(json.dumps(self.config))
            invalid["dataset"]["split_manifest_sha256"] = "0" * 64
            invalid_path = root / "invalid.json"
            invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(LAUNCHER_PATH),
                    "--dry-run",
                    "--config",
                    str(invalid_path),
                    "--out",
                    str(root / "output"),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            attempts = list((root / "output" / "attempts").iterdir())
            self.assertEqual(len(attempts), 1)
            failed = json.loads((attempts[0] / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "FAILED")
            self.assertEqual(failed["error_type"], "ValueError")
            self.assertFalse((attempts[0] / "training").exists())

    def test_patch_matches_pinned_resfit_checkout(self) -> None:
        result = subprocess.run(
            ["git", "-C", str(RESFIT_ROOT), "apply", "--check", str(PATCH_PATH)],
            capture_output=True,
            text=True,
        )
        reverse = subprocess.run(
            ["git", "-C", str(RESFIT_ROOT), "apply", "--reverse", "--check", str(PATCH_PATH)],
            capture_output=True,
            text=True,
        )
        self.assertTrue(result.returncode == 0 or reverse.returncode == 0)

    def test_patch_script_apply_check_reverse_round_trip_cleans_both_checkouts(self) -> None:
        self.assertEqual(
            subprocess.run(
                ["git", "-C", str(RESFIT_ROOT), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "",
        )
        self.assertEqual(
            subprocess.run(
                ["git", "-C", str(LEROBOT_ROOT), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "",
        )
        try:
            initial = subprocess.run(
                ["bash", str(APPLY_PATCH_SCRIPT), "check"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(initial.stdout.count(": not_applied"), 3)
            subprocess.run(
                ["bash", str(APPLY_PATCH_SCRIPT), "apply"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            applied = subprocess.run(
                ["bash", str(APPLY_PATCH_SCRIPT), "check"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(applied.stdout.count(": applied"), 3)
        finally:
            subprocess.run(
                ["bash", str(APPLY_PATCH_SCRIPT), "reverse"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertEqual(
            subprocess.run(
                ["git", "-C", str(LEROBOT_ROOT), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "",
        )
        self.assertEqual(
            subprocess.run(
                ["git", "-C", str(RESFIT_ROOT), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "",
        )


if __name__ == "__main__":
    unittest.main()
