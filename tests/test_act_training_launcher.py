from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_act_training import (
    LEROBOT_ROOT,
    REPO_ROOT,
    RESFIT_ROOT,
    _sha256,
    build_command,
    main,
    patch_states,
    validate_spec,
)


CONFIG_PATH = REPO_ROOT / "configs" / "cf_resrl" / "m0_act_training.json"


class ActTrainingLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def expected_command(self, attempt: Path) -> list[str]:
        dataset = self.config["dataset"]
        return [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "cf-resrl-m0",
            "env",
            "PYTHONNOUSERSITE=1",
            "PYTHONUNBUFFERED=1",
            f"PYTHONPATH={RESFIT_ROOT}{os.pathsep}{LEROBOT_ROOT}",
            f"CACHE_DIR={attempt / 'cache'}",
            "WANDB_MODE=disabled",
            "HF_HUB_DISABLE_XET=1",
            "python",
            "-m",
            "resfit.lerobot.scripts.train_bc_dexmg",
            "--dataset",
            "ankile/robomimic-mh-square-image",
            "--dataset_revision",
            "4e5cee7ea7d63b920ff76c2529b781c2474697da",
            "--dataset_root",
            str((REPO_ROOT / dataset["root"]).resolve()),
            "--episode_split_manifest",
            str((REPO_ROOT / dataset["split_manifest"]).resolve()),
            "--episode_split",
            "train",
            "--video_backend",
            "pyav",
            "--policy",
            "act",
            "--seed",
            "20261124",
            "--device",
            "cuda",
            "--steps",
            "50000",
            "--batch_size",
            "256",
            "--num_workers",
            "4",
            "--grad_clip_norm",
            "10.0",
            "--log_freq",
            "100",
            "--save_freq",
            "1000",
            "--output_dir",
            str((attempt / "training").resolve()),
            "--eval_env",
            "Square",
            "--rollout_freq",
            "1000",
            "--eval_num_envs",
            "16",
            "--eval_num_episodes",
            "100",
            "--eval_camera_size",
            "84",
            "--eval_video_key",
            "observation.images.agentview",
        ]

    def test_registered_split_and_fixed_final_checkpoint(self) -> None:
        episode_indices = validate_spec(self.config)
        self.assertEqual(len(episode_indices), 240)
        self.assertEqual(self.config["checkpoint_selection"]["checkpoint_step_label"], 49999)
        self.assertTrue(self.config["checkpoint_selection"]["best_checkpoint_is_not_selected"])

    def test_registered_runtime_locks_match_committed_files(self) -> None:
        runtime = self.config["runtime"]
        self.assertEqual(runtime["conda_environment"], "cf-resrl-m0")
        for lock_name in ("pip_freeze_lock", "conda_explicit_lock"):
            lock = runtime[lock_name]
            self.assertEqual(_sha256(REPO_ROOT / lock["path"]), lock["sha256"])

    def test_command_contains_exact_registered_budget_without_wandb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt"
            command = build_command(self.config, attempt)
        self.assertEqual(command, self.expected_command(attempt))
        self.assertIn("--steps", command)
        self.assertEqual(command[command.index("--policy") + 1], "act")
        self.assertEqual(command[command.index("--seed") + 1], "20261124")
        self.assertEqual(command[command.index("--steps") + 1], "50000")
        self.assertEqual(command[command.index("--batch_size") + 1], "256")
        self.assertEqual(command[command.index("--save_freq") + 1], "1000")
        self.assertEqual(command[command.index("--log_freq") + 1], "100")
        self.assertEqual(command[command.index("--rollout_freq") + 1], "1000")
        self.assertEqual(command[command.index("--eval_num_envs") + 1], "16")
        self.assertEqual(command[command.index("--eval_num_episodes") + 1], "100")
        self.assertEqual(command[command.index("--dataset_revision") + 1], self.config["dataset"]["revision"])
        self.assertEqual(command[command.index("--episode_split") + 1], "train")
        self.assertEqual(command[command.index("--video_backend") + 1], "pyav")
        self.assertEqual(Path(command[command.index("--output_dir") + 1]), attempt / "training")
        self.assertIn("WANDB_MODE=disabled", command)
        self.assertIn("PYTHONUNBUFFERED=1", command)
        self.assertNotIn("--wandb_enable", command)
        self.assertNotIn("--wandb_project", command)

    def test_conflicted_patch_state_is_rejected(self) -> None:
        failed = type("Result", (), {"returncode": 1})()
        with patch("scripts.run_act_training.subprocess.run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "cannot be applied or reversed"):
                patch_states(self.config)

    def test_dry_run_writes_resolved_command_without_training_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "act_training"
            patch_rows = [
                {"path": str(REPO_ROOT / relative), "sha256": "registered", "state": "not_applied"}
                for relative in self.config["upstream"]["required_patches"]
            ]
            with patch("scripts.run_act_training.patch_states", return_value=patch_rows):
                with redirect_stdout(io.StringIO()):
                    result = main(["--dry-run", "--out", str(output_root)])
            self.assertEqual(result, 0)
            attempts = list((output_root / "attempts").iterdir())
            self.assertEqual(len(attempts), 1)
            attempt = attempts[0]
            resolved = json.loads((attempt / "resolved_launcher.json").read_text(encoding="utf-8"))
            status = json.loads((attempt / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "PREPARED")
            self.assertEqual(resolved["episode_indices"], validate_spec(self.config))
            self.assertEqual(resolved["command"], self.expected_command(attempt))
            self.assertEqual(resolved["runtime_verification"]["status"], "NOT_EXECUTED_DRY_RUN")
            self.assertIsInstance(status["launcher_pid"], int)
            self.assertIsNone(status["child_pid"])
            self.assertEqual(status["log_path"], str((attempt / "stdout_stderr.log").resolve()))
            self.assertIsNone(status["returncode"])
            self.assertTrue(status["host"])
            self.assertEqual(status["gpus"], [])
            self.assertFalse((attempt / "training").exists())
            self.assertFalse((attempt / "runtime").exists())
            self.assertIn("PYTHONUNBUFFERED=1", resolved["command"])
            self.assertNotIn("--wandb_enable", resolved["command"])

    def test_preflight_failure_is_persisted_inside_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_config = root / "bad.json"
            broken = json.loads(json.dumps(self.config))
            broken["training"]["steps"] = 49999
            bad_config.write_text(json.dumps(broken), encoding="utf-8")
            output_root = root / "act_training"
            with self.assertRaisesRegex(ValueError, "settings do not match"):
                main(
                    [
                        "--dry-run",
                        "--config",
                        str(bad_config),
                        "--out",
                        str(output_root),
                    ]
                )
            attempts = list((output_root / "attempts").iterdir())
            self.assertEqual(len(attempts), 1)
            status = json.loads((attempts[0] / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "FAILED")
            self.assertIsInstance(status["launcher_pid"], int)
            self.assertIsNone(status["child_pid"])
            self.assertIsNone(status["returncode"])
            self.assertTrue(status["host"])
            self.assertEqual(status["gpus"], [])

    def test_hostname_failure_is_persisted_inside_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "act_training"
            with patch(
                "scripts.run_act_training.socket.gethostname",
                side_effect=OSError("hostname unavailable"),
            ):
                with self.assertRaisesRegex(OSError, "hostname unavailable"):
                    main(["--dry-run", "--out", str(output_root)])
            attempts = list((output_root / "attempts").iterdir())
            self.assertEqual(len(attempts), 1)
            status = json.loads((attempts[0] / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "FAILED")
            self.assertEqual(status["error_type"], "OSError")
            self.assertEqual(status["error"], "hostname unavailable")
            self.assertIsInstance(status["launcher_pid"], int)
            self.assertIsNone(status["host"])
            self.assertIsNone(status["child_pid"])
            self.assertIsNone(status["returncode"])


if __name__ == "__main__":
    unittest.main()
