import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cf_resrl.resfit_resume_store import MANIFEST, STATE, ledger_prefix, read_manifest, sha256_tree, write_generation


def _generation(root, step):
    return root / f"step_{step:09d}"


def _publish(root, step):
    with write_generation(root, step, 3) as dest:
        (dest / MANIFEST).write_text(json.dumps({"version": 1, "step": step}))
        (dest / STATE).write_bytes(f"trainer_state_{step}".encode())
        (dest / "replay_runtime.pt").write_bytes(f"replay_state_{step}".encode())


class ResumeCheckpointStoreTests(unittest.TestCase):
    def test_checksums_and_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "resume"
            for step in range(100_000, 500_000, 100_000):
                _publish(root, step)
            self.assertEqual(len(list(root.glob("step_*"))), 3)
            self.assertFalse(_generation(root, 100_000).exists())
            latest = _generation(root, 400_000)
            self.assertEqual(read_manifest(latest)["step"], 400_000)
            (latest / "replay_runtime.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "payload checksum mismatch"):
                read_manifest(latest)
            self.assertEqual(read_manifest(_generation(root, 300_000))["step"], 300_000)
            with self.assertRaises(ValueError):
                with write_generation(root, 500_000, 4):
                    self.fail("accepting retention greater than three")

    def test_install_failure_restores_the_previous_three_generations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "resume"
            for step in range(100_000, 400_000, 100_000):
                _publish(root, step)
            rename = Path.rename

            def fail_publication(source, target):
                if source.name.startswith(".writing-"):
                    raise OSError("simulated publication failure")
                return rename(source, target)

            with patch.object(Path, "rename", fail_publication):
                with self.assertRaisesRegex(OSError, "simulated publication failure"):
                    _publish(root, 400_000)
            self.assertEqual(sorted(p.name for p in root.glob("step_*")), [
                "step_000100000", "step_000200000", "step_000300000",
            ])
            self.assertFalse(list(root.glob(".writing-*")))
            self.assertFalse(list(root.glob(".retired-*")))
            for step in (100_000, 200_000, 300_000):
                self.assertEqual(read_manifest(_generation(root, step))["step"], step)

    def test_reads_recover_a_process_exit_before_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "resume"
            for step in range(100_000, 400_000, 100_000):
                _publish(root, step)
            oldest = _generation(root, 100_000)
            oldest.rename(root / ".retired-step_000100000-for-step_000400000")
            (root / ".writing-interrupted").mkdir()
            self.assertEqual(read_manifest(oldest)["step"], 100_000)
            self.assertFalse(list(root.glob(".retired-*")))
            _publish(root, 400_000)
            self.assertFalse(list(root.glob(".writing-*")))
            self.assertFalse(oldest.exists())

    def test_tree_rejects_symlinks_and_unsealed_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "resume"
            _publish(root, 100_000)
            generation = _generation(root, 100_000)
            link = generation / "unexpected"
            link.symlink_to(generation / STATE)
            with self.assertRaisesRegex(ValueError, "Unexpected resume payload member"):
                sha256_tree(generation)
            link.unlink()
            link.write_bytes(b"unsealed")
            with self.assertRaisesRegex(ValueError, "payload checksum mismatch"):
                read_manifest(generation)

    def test_ledger_copies_only_the_sealed_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            target = Path(tmp) / "target.jsonl"
            prefix = b'{"global_step": 1}\n{"global_step": 2}\n'
            source.write_bytes(prefix + b'{"global_step": 5}\n')
            self.assertEqual(ledger_prefix(source, 2, target), hashlib.sha256(prefix).hexdigest())
            self.assertEqual(target.read_bytes(), prefix)
            with self.assertRaisesRegex(ValueError, "Invalid action ledger at step 3"):
                ledger_prefix(source, 3)
