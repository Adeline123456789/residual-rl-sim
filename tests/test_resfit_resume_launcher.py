import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_resfit_training as launcher


ACTUAL_CONFIG_SHA = "1" * 64
RUNTIME_SHA = {"conda-explicit.txt": "2" * 64, "environment.lock.txt": "3" * 64}
SEED = 20261125


def _x1_config():
    config = json.loads(launcher.DEFAULT_CONFIG.read_text())
    config["training"].update(
        seed=SEED,
        output_root=launcher.X1_OUTPUT_ROOT,
        resume_checkpoint_interval_environment_steps=100000,
        resume_checkpoint_retention=3,
    )
    return config


def _receipt(attempt, *, prior_resume=None):
    attempt.mkdir(parents=True)
    (attempt / "resolved_launcher.json").write_text(json.dumps({
        "dry_run": False,
        "config_sha256": ACTUAL_CONFIG_SHA,
        "training_seed": SEED,
        "overlay_mode": "resumable",
        "attempt_dir": str(attempt),
        "environment_step_budget": {"warmup": 10000, "main": 290000, "total": 300000},
        "resume_checkpoint_interval_environment_steps": 100000,
        "resume_checkpoint_retention": 3,
        "runtime": {"locks": {name: {"sha256": digest} for name, digest in RUNTIME_SHA.items()}},
        "resume": prior_resume,
    }))
    ledger = attempt / "training" / "action_ledger.jsonl"
    ledger.parent.mkdir()
    ledger.touch()
    return ledger


def _generation(origin, source, step):
    generation = origin / "training" / "resume" / f"step_{step:09d}"
    generation.mkdir(parents=True)
    (generation / "resume.json").write_text(json.dumps({
        "version": 1,
        "step": step,
        "action_ledger": str(source),
    }))
    (generation / "state.pt").touch()
    return generation


class ResfitResumeLauncherTests(unittest.TestCase):
    def test_x1_argv_only_adds_resume_overrides_for_resumable_config(self):
        legacy_config = json.loads(launcher.DEFAULT_CONFIG.read_text())
        old = launcher.build_command(legacy_config, Path("/tmp/legacy_attempt"))
        self.assertFalse(any(arg.startswith("resume_") for arg in old))
        self.assertNotIn("HF_HUB_OFFLINE=1", old)
        self.assertIn("seed=20261124", old)

        generation = Path("/tmp/prior/training/resume/step_000100000")
        argv = launcher.build_command(_x1_config(), Path("/tmp/fresh_attempt"), generation)
        self.assertIn("seed=20261125", argv)
        self.assertIn("algo.total_timesteps=300000", argv)
        self.assertIn("algo.learning_starts=10000", argv)
        self.assertIn("resume_checkpoint_interval=100000", argv)
        self.assertIn("resume_checkpoint_retention=3", argv)
        self.assertIn(f"resume_from={generation}", argv)
        self.assertIn("HF_HUB_OFFLINE=1", argv)
        pythonpath = next(arg for arg in argv if arg.startswith("PYTHONPATH="))
        self.assertIn(str(launcher.REPO_ROOT), pythonpath.split("=", 1)[1].split(os.pathsep))

    def test_x1_registration_requires_the_exact_cadence_and_legacy_patch_pins(self):
        config = _x1_config()
        x1_patches = copy.deepcopy(launcher.EXPECTED_PATCHES)
        x1_patches[1].update(path=launcher.RESUMABLE_PATCH, sha256="a" * 64)
        config["upstream"]["patches"] = x1_patches
        actual_hash = launcher._sha256

        def digest(path):
            if path == launcher.REPO_ROOT / launcher.RESUMABLE_PATCH:
                return "a" * 64
            return actual_hash(path)

        with patch.object(launcher, "_sha256", side_effect=digest):
            launcher.validate_spec(config)
            config["training"]["resume_checkpoint_retention"] = 4
            with self.assertRaisesRegex(ValueError, "registered fixed-step"):
                launcher.validate_spec(config)
            config["training"]["resume_checkpoint_retention"] = 3
            x1_patches[0]["sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "patch registration"):
                launcher.validate_spec(config)

    @patch.object(launcher, "_ensure_training_is_inactive")
    def test_accepts_latest_generation_from_a_recorded_continuation(self, inactive):
        with tempfile.TemporaryDirectory() as root:
            origin, continuation = (Path(root) / "attempts" / name for name in ("origin", "continuation"))
            old = _generation(origin, _receipt(origin), 100000)
            source = _receipt(continuation, prior_resume={"origin_attempt": str(origin)})
            latest = _generation(origin, source, 200000)
            config = _x1_config()
            result = launcher._resume_provenance(latest, config, ACTUAL_CONFIG_SHA, SEED)
            self.assertEqual(result["origin_attempt"], str(origin))
            self.assertEqual(result["producer_attempt"], str(continuation))
            self.assertEqual(result["restored_environment_steps"], 200000)
            self.assertEqual(result["runtime_lock_sha256"], RUNTIME_SHA)
            self.assertEqual([call.args[0] for call in inactive.call_args_list], [origin, continuation])
            with self.assertRaisesRegex(ValueError, "latest completed generation"):
                launcher._resume_provenance(old, config, ACTUAL_CONFIG_SHA, SEED)
            with self.assertRaisesRegex(ValueError, "config and seed"):
                launcher._resume_provenance(latest, config, ACTUAL_CONFIG_SHA, 20261126)

    @patch.object(launcher, "_ensure_training_is_inactive")
    def test_rejects_an_unbound_ledger_producer(self, inactive):
        with tempfile.TemporaryDirectory() as root:
            origin, other = (Path(root) / "attempts" / name for name in ("origin", "other"))
            _receipt(origin)
            generation = _generation(origin, _receipt(other), 100000)
            with self.assertRaisesRegex(ValueError, "original training lineage"):
                launcher._resume_provenance(generation, _x1_config(), ACTUAL_CONFIG_SHA, SEED)

    def test_exact_overlay_check_leaves_real_index_and_worktree_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            checkout = Path(root) / "checkout"
            checkout.mkdir()
            subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
            content = checkout / "model.py"
            content.write_text("first\nmiddle\nlast\n")
            git = ["git", "-C", str(checkout)]
            subprocess.run([*git, "add", "model.py"], check=True)
            subprocess.run([
                *git, "-c", "user.name=Test", "-c", "user.email=test@example.com",
                "commit", "--quiet", "-m", "original",
            ], check=True)
            content.write_text("first\nlauncher edit\nlast\n")
            overlay = Path(root) / "overlay.patch"
            overlay.write_bytes(subprocess.check_output([*git, "diff"]))
            launcher._check_exact_overlay(checkout, [overlay])
            self.assertEqual(launcher._git(checkout, "status", "--porcelain"), "M model.py")
            content.write_text("first\nlauncher edit\nlast\nuser edit\n")
            with self.assertRaisesRegex(RuntimeError, "outside the registered overlay"):
                launcher._check_exact_overlay(checkout, [overlay])
            self.assertIn("user edit", content.read_text())

    def test_checkout_lock_excludes_a_second_launcher(self):
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root) / "owner.json"
            with patch.object(launcher, "_git", return_value=str(marker)):
                fd, path = launcher._lock_overlay()
                try:
                    self.assertEqual(path, marker)
                    with self.assertRaisesRegex(RuntimeError, "holds the checkout overlay lock"):
                        launcher._lock_overlay()
                finally:
                    os.close(fd)

    def test_stale_recovery_requires_matching_owner_and_no_live_trainer(self):
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root) / "owner.json"
            prior = Path(root) / "attempts" / "prior"
            prior.mkdir(parents=True)
            launcher._record_overlay_owner(marker, {
                "mode": "resumable", "config_sha256": ACTUAL_CONFIG_SHA,
                "training_seed": SEED, "attempt_dir": str(prior), "launcher_pid": 42,
            })
            config = _x1_config()
            with (
                patch.object(launcher, "_ensure_training_is_inactive") as inactive,
                patch.object(launcher, "patch_states", return_value=[]),
                patch.object(launcher, "_check_exact_overlay") as exact,
                patch.object(launcher, "restore_patch_checkouts", return_value="reversed") as restore,
            ):
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    launcher._recover_stale_overlay(marker, config, ACTUAL_CONFIG_SHA, 20261126)
                inactive.assert_not_called()
                inactive.side_effect = RuntimeError("prior process still live")
                with self.assertRaisesRegex(RuntimeError, "prior process still live"):
                    launcher._recover_stale_overlay(marker, config, ACTUAL_CONFIG_SHA, SEED)
                self.assertTrue(marker.exists())
                restore.assert_not_called()
                inactive.side_effect = None
                result = launcher._recover_stale_overlay(marker, config, ACTUAL_CONFIG_SHA, SEED)
                self.assertEqual(result["attempt_dir"], str(prior))
                self.assertEqual(result["cleanup_stdout"], "reversed")
                restore.assert_called_once_with("resumable")
                self.assertEqual(exact.call_count, 2)
                self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
