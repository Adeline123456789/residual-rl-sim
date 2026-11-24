from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import scripts.smoke_composite_snapshot as smoke
from scripts.smoke_composite_snapshot import (
    DEFAULT_CONFIG,
    EXACT_FIELDS,
    RESFIT_ROOT,
    VECTOR_FIELDS,
    compare_replay,
    main,
    validate_absolute_alignment,
    validate_spec,
)


class CompositeSnapshotSmokeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def test_registered_real_composite_contract(self) -> None:
        validate_spec(self.config)
        self.assertEqual(self.config["task"]["prefix_ticks"], 1)
        self.assertEqual(self.config["task"]["suffix_ticks"], 3)
        self.assertEqual(self.config["task"]["restored_replays"], 2)
        self.assertEqual(self.config["numeric_tolerance"], {"atol": 1e-12, "rtol": 0.0})
        self.assertEqual(self.config["checkpoint"]["provenance"], "author-owned candidate")
        self.assertEqual(self.config["checkpoint"]["wandb_equivalence"], "UNVERIFIED")
        self.assertEqual(self.config["checkpoint"]["license"], "NOT_PUBLISHED")
        self.assertIn("not the registered eight-group four-branch smoke", self.config["limitations"][1])

    def test_compare_replay_requires_exact_control_metadata(self) -> None:
        row = {field: 0 for field in EXACT_FIELDS}
        row.update({field: [0.0] for field in VECTOR_FIELDS})
        row.update({"reward": 0.0, "cur_time": 0.0, "sim_time": 0.0})
        maxima = compare_replay([row], [dict(row)], atol=1e-12)
        self.assertTrue(all(value == 0.0 for value in maxima.values()))

        changed = dict(row)
        changed["scheduler_tick"] = 1
        with self.assertRaisesRegex(ValueError, "scheduler_tick"):
            compare_replay([row], [changed], atol=1e-12)

    def test_compare_replay_enforces_fixed_numeric_tolerance(self) -> None:
        row = {field: 0 for field in EXACT_FIELDS}
        row.update({field: [0.0] for field in VECTOR_FIELDS})
        row.update({"reward": 0.0, "cur_time": 0.0, "sim_time": 0.0})
        changed = dict(row)
        changed["qpos"] = [2e-12]
        with self.assertRaisesRegex(ValueError, "exceeded atol"):
            compare_replay([row], [changed], atol=1e-12)

    def test_absolute_tick_source_index_and_queue_alignment_are_hard_gates(self) -> None:
        wrapper = SimpleNamespace(episode_steps=1)
        policy = SimpleNamespace(_action_queues=[[object()] * 19])
        scheduler = SimpleNamespace(logical_tick=1)
        entry = SimpleNamespace(
            tick=1,
            chunk_source_tick=0,
            chunk_index=1,
            generation=0,
            used_fallback=False,
        )
        validate_absolute_alignment(
            wrapper,
            policy,
            scheduler,
            expected_tick=1,
            expected_queue_length=19,
            entry=entry,
        )

        wrapper.episode_steps = 2
        with self.assertRaisesRegex(RuntimeError, "absolute ticks differ"):
            validate_absolute_alignment(
                wrapper,
                policy,
                scheduler,
                expected_tick=1,
                expected_queue_length=19,
                entry=entry,
            )
        wrapper.episode_steps = 1
        entry.chunk_index = 0
        with self.assertRaisesRegex(RuntimeError, "source-tick aligned"):
            validate_absolute_alignment(
                wrapper,
                policy,
                scheduler,
                expected_tick=1,
                expected_queue_length=19,
                entry=entry,
            )
        entry.chunk_index = 1
        policy._action_queues[0].pop()
        with self.assertRaisesRegex(RuntimeError, "queue remaining"):
            validate_absolute_alignment(
                wrapper,
                policy,
                scheduler,
                expected_tick=1,
                expected_queue_length=19,
                entry=entry,
            )

    def test_dry_run_creates_prepared_attempt_without_runtime_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with redirect_stdout(io.StringIO()):
                result = main(["--dry-run", "--out", str(root)])
            self.assertEqual(result, 0)
            attempts = list((root / "attempts").iterdir())
            self.assertEqual(len(attempts), 1)
            attempt = attempts[0]
            status = json.loads((attempt / "status.json").read_text(encoding="utf-8"))
            resolved = json.loads(
                (attempt / "resolved_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "PREPARED")
            self.assertTrue(status["dry_run"])
            self.assertEqual(resolved["checkpoint"]["model_sha256"], self.config["checkpoint"]["model_sha256"])
            self.assertFalse((attempt / "tick_records.jsonl").exists())
            self.assertFalse((attempt / "smoke_report.json").exists())

    def test_hostname_failure_after_attempt_creation_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                smoke.socket,
                "gethostname",
                side_effect=RuntimeError("hostname unavailable"),
            ):
                with self.assertRaisesRegex(RuntimeError, "hostname unavailable"):
                    main(["--dry-run", "--out", str(root)])
            attempt = next((root / "attempts").iterdir())
            status = json.loads((attempt / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "FAILED")
            self.assertEqual(status["error"], "hostname unavailable")
            self.assertTrue((attempt / "smoke_report.json").is_file())

    def test_submodule_probe_failure_after_attempt_creation_is_persisted(self) -> None:
        original_git = smoke._git

        def failing_git(path, *args):
            if path == RESFIT_ROOT and args == ("rev-parse", "HEAD"):
                raise RuntimeError("submodule probe unavailable")
            return original_git(path, *args)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(smoke, "_git", side_effect=failing_git):
                with self.assertRaisesRegex(RuntimeError, "submodule probe unavailable"):
                    main(["--dry-run", "--out", str(root)])
            attempt = next((root / "attempts").iterdir())
            status = json.loads((attempt / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "FAILED")
            self.assertEqual(status["error"], "submodule probe unavailable")


if __name__ == "__main__":
    unittest.main()
