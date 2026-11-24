"""Small, dependency-free dataset contracts for CF-ResRL."""

from __future__ import annotations

import hashlib
from pathlib import Path
import random
from typing import Mapping, Sequence

from .factorial import assert_disjoint_groups


def make_episode_splits(
    total_episodes: int,
    train_count: int,
    validation_count: int,
    test_count: int,
    seed: int,
) -> dict[str, list[int]]:
    """Create one deterministic, episode-grouped train/validation/test split."""
    if train_count + validation_count + test_count != total_episodes:
        raise ValueError("split counts must cover every episode exactly once")

    episode_ids = list(range(total_episodes))
    random.Random(seed).shuffle(episode_ids)
    train_end = train_count
    validation_end = train_end + validation_count
    splits = {
        "train": sorted(episode_ids[:train_end]),
        "validation": sorted(episode_ids[train_end:validation_end]),
        "test": sorted(episode_ids[validation_end:]),
    }
    assert_episode_partition(splits, total_episodes)
    return splits


def assert_episode_partition(
    splits: Mapping[str, list[int]], total_episodes: int
) -> None:
    """Require disjoint splits whose union is exactly ``range(total_episodes)``."""
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    assert_disjoint_groups(train, validation, test)
    assigned = set(train) | set(validation) | set(test)
    expected = set(range(total_episodes))
    if assigned != expected:
        raise ValueError(
            f"episode split does not cover the dataset: "
            f"missing={sorted(expected - assigned)}, extra={sorted(assigned - expected)}"
        )


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def terminal_true_suffix(values: Sequence[bool]) -> tuple[int, int]:
    """Return the first terminal frame and suffix length for repeated done labels."""
    done = [bool(value) for value in values]
    if not done or not done[-1]:
        raise ValueError("done labels must end with true")
    first = done.index(True)
    if not all(done[first:]):
        raise ValueError("done labels must form one contiguous true suffix")
    return first, len(done) - first
