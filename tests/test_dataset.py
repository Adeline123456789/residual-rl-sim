import hashlib
from pathlib import Path
import tempfile
import unittest

from cf_resrl.dataset import make_episode_splits, sha256_file


class EpisodeSplitTests(unittest.TestCase):
    def test_preregistered_split_is_deterministic_and_complete(self) -> None:
        first = make_episode_splits(300, 240, 30, 30, 20261124)
        second = make_episode_splits(300, 240, 30, 30, 20261124)

        self.assertEqual(first, second)
        self.assertEqual(
            {name: len(ids) for name, ids in first.items()},
            {"train": 240, "validation": 30, "test": 30},
        )
        self.assertFalse(set(first["train"]) & set(first["validation"]))
        self.assertFalse(set(first["train"]) & set(first["test"]))
        self.assertFalse(set(first["validation"]) & set(first["test"]))
        self.assertEqual(
            set(first["train"]) | set(first["validation"]) | set(first["test"]),
            set(range(300)),
        )


class HashTests(unittest.TestCase):
    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"cf-resrl\n")
            self.assertEqual(
                sha256_file(path), hashlib.sha256(b"cf-resrl\n").hexdigest()
            )


if __name__ == "__main__":
    unittest.main()
