"""Publish and verify local ResFiT training-state generations."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from cf_resrl.dataset import sha256_file


MANIFEST = "resume.json"
STATE = "state.pt"
_INTEGRITY = "payload.sha256.json"
_STEP = re.compile(r"step_\d{9}")
_RETIRED = re.compile(r"\.retired-(step_\d{9})-for-(step_\d{9})")


def _sync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _tree_files(root: Path):
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Resume payload must be a local directory: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"Unexpected resume payload member: {path}")
        if path.is_file():
            yield path


def sha256_tree(root: Path) -> dict[str, str]:
    """Hash a regular-file tree with bounded memory."""
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in _tree_files(root)}


def _step_name(step: int) -> str:
    if type(step) is not int or not 0 < step <= 999_999_999:
        raise ValueError("Resume checkpoint step must be a positive nine-digit integer")
    return f"step_{step:09d}"


def _validate_manifest(path: Path, step_name: str) -> dict:
    data = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict) or type(data.get("version")) is not int
        or data["version"] != 1 or _step_name(data.get("step")) != step_name
    ):
        raise ValueError(f"Unsupported ResFiT resume checkpoint: {path}")
    return data


def _completed(root: Path) -> list[Path]:
    return sorted(
        path for path in root.iterdir() if _STEP.fullmatch(path.name)
        and not path.is_symlink() and (path / _INTEGRITY).is_file()
    )


def _recover_retirements(root: Path) -> None:
    for path in sorted(root.iterdir()):
        match = _RETIRED.fullmatch(path.name)
        if match is None:
            continue
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"Unexpected retired generation: {path}")
        original, publication = (root / name for name in match.groups())
        if not publication.is_symlink() and (publication / _INTEGRITY).is_file():
            shutil.rmtree(path)
        else:
            path.rename(original)
        _sync(root)


@contextmanager
def _locked(root: Path):
    with (root / ".resume.lock").open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        _recover_retirements(root)
        yield


@contextmanager
def write_generation(root: Path, step: int, retention: int):
    """Publish a complete checkpoint atomically and keep at most three generations."""
    if type(retention) is not int or not 2 <= retention <= 3:
        raise ValueError("ResFiT resume retention must be between two and three")
    name = _step_name(step)
    root.mkdir(parents=True, exist_ok=True)
    _sync(root.parent)
    destination = root / name
    with _locked(root):
        complete = _completed(root)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        if complete and complete[-1] >= destination:
            raise ValueError("Resume checkpoints must progress beyond the newest saved step")
        for interrupted in root.glob(".writing-*"):
            if interrupted.is_dir() and not interrupted.is_symlink():
                shutil.rmtree(interrupted)
        staging = Path(tempfile.mkdtemp(prefix=".writing-", dir=root))
        try:
            yield staging
            checksums = sha256_tree(staging)
            if not {MANIFEST, STATE} <= checksums.keys() or _INTEGRITY in checksums:
                raise ValueError("Resume checkpoint requires its trainer state and an unsealed payload")
            _validate_manifest(staging, name)
            (staging / _INTEGRITY).write_text(
                json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            for directory, _, files in os.walk(staging, topdown=False):
                for file in files:
                    _sync(Path(directory) / file)
                _sync(Path(directory))
            # Hide expired names only after the replacement payload is durable.
            # Both directory names in the journal allow recovery after a process kill.
            to_retire = complete[:max(0, len(complete) + 1 - retention)]
            for expired in to_retire:
                expired.rename(root / f".retired-{expired.name}-for-{name}")
            if to_retire:
                _sync(root)
            staging.rename(destination)
            _sync(root)
        except BaseException:
            if destination.exists() and not staging.exists():
                destination.rename(staging)
            _recover_retirements(root)
            raise
        else:
            _recover_retirements(root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def read_manifest(checkpoint: Path) -> dict:
    """Verify the entire local payload before any training-state deserialization."""
    with _locked(checkpoint.parent):
        actual = {
            path.relative_to(checkpoint).as_posix(): sha256_file(path)
            for path in _tree_files(checkpoint) if path != checkpoint / _INTEGRITY
        }
        expected = json.loads((checkpoint / _INTEGRITY).read_text(encoding="utf-8"))
        if expected != actual or STATE not in actual or MANIFEST not in actual:
            raise ValueError(f"ResFiT resume payload checksum mismatch: {checkpoint}")
        return _validate_manifest(checkpoint, checkpoint.name)


def ledger_prefix(source: Path, count: int, destination: Path | None = None) -> str:
    """Hash a contiguous ledger prefix and optionally copy it to a new attempt."""
    if type(count) is not int or count < 0:
        raise ValueError("Action ledger prefix length must be nonnegative")
    digest = hashlib.sha256()
    output = None if destination is None else destination.open("xb")
    try:
        with source.open("rb") as stream:
            for step in range(1, count + 1):
                line = stream.readline()
                if not line.endswith(b"\n") or json.loads(line)["global_step"] != step:
                    raise ValueError(f"Invalid action ledger at step {step}: {source}")
                digest.update(line)
                if output is not None:
                    output.write(line)
    finally:
        if output is not None:
            output.close()
    return digest.hexdigest()
