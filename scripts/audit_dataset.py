#!/usr/bin/env python3
"""Download and fully audit the pinned Square-MH LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cf_resrl.dataset import make_episode_splits, sha256_file  # noqa: E402


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _required_packages():
    try:
        import numpy as np
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.hf_api import RepoFile
    except ImportError as exc:
        raise RuntimeError(
            "dataset audit requires numpy, pyarrow, and huggingface-hub"
        ) from exc
    return np, pq, HfApi, RepoFile, snapshot_download


def _validate_remote(api, RepoFile, dataset: dict[str, Any]) -> tuple[list[Any], Any]:
    repo_id = dataset["repo_id"]
    revision = dataset["revision"]
    repo_type = dataset["repo_type"]
    expected = dataset["expected"]

    info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError(f"resolved revision {info.sha} does not match {revision}")
    actual_license = str(info.card_data.license).lower()
    if actual_license != dataset["license"]:
        raise ValueError(
            f"dataset license is {actual_license}, expected {dataset['license']}"
        )

    files = sorted(
        (
            entry
            for entry in api.list_repo_tree(
                repo_id,
                recursive=True,
                expand=True,
                revision=revision,
                repo_type=repo_type,
            )
            if isinstance(entry, RepoFile)
        ),
        key=lambda entry: entry.path,
    )
    total_bytes = sum(entry.size for entry in files)
    lfs_files = sum(entry.lfs is not None for entry in files)
    observed = (len(files), total_bytes, lfs_files)
    wanted = (expected["files"], expected["bytes"], expected["lfs_files"])
    if observed != wanted:
        raise ValueError(
            "remote file inventory mismatch: "
            f"observed files/bytes/lfs={observed}, expected={wanted}"
        )
    return files, info


def _verify_download(files: list[Any], local_dir: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for entry in files:
        local_path = local_dir / entry.path
        if not local_path.is_file():
            raise FileNotFoundError(f"download is missing {entry.path}")
        if local_path.stat().st_size != entry.size:
            raise ValueError(f"size mismatch for {entry.path}")
        digest = sha256_file(local_path)
        expected_sha = entry.lfs.sha256 if entry.lfs is not None else None
        if expected_sha is not None and digest != expected_sha:
            raise ValueError(f"LFS SHA256 mismatch for {entry.path}")
        manifest.append(
            {
                "path": entry.path,
                "bytes": entry.size,
                "sha256": digest,
                "lfs_sha256": expected_sha,
            }
        )
    return manifest


def _read_metadata(
    local_dir: Path, dataset: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info = json.loads((local_dir / "meta/info.json").read_text(encoding="utf-8"))
    episodes = [
        json.loads(line)
        for line in (local_dir / "meta/episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    expected = dataset["expected"]
    observed = {
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "videos": info["total_videos"],
        "fps": info["fps"],
    }
    wanted = {key: expected[key] for key in observed}
    if observed != wanted:
        raise ValueError(f"meta/info.json mismatch: observed={observed}, expected={wanted}")
    if info["codebase_version"] != "v2.1" or info["splits"] != {"train": "0:300"}:
        raise ValueError("expected a LeRobot v2.1 train-only dataset")
    if len(episodes) != expected["episodes"]:
        raise ValueError("meta/episodes.jsonl has the wrong episode count")
    if [row["episode_index"] for row in episodes] != list(range(expected["episodes"])):
        raise ValueError("episode indexes are not contiguous 0..299")
    if sum(row["length"] for row in episodes) != expected["frames"]:
        raise ValueError("episode lengths do not sum to total_frames")
    if any(row["tasks"] != ["NutAssemblySquare"] for row in episodes):
        raise ValueError("episode task metadata is not NutAssemblySquare")

    features = info["features"]
    if features["action"]["shape"] != [expected["action_dim"]]:
        raise ValueError("action shape does not match the registered contract")
    if features["observation.state"]["shape"] != [expected["state_dim"]]:
        raise ValueError("state shape does not match the registered contract")
    for key in dataset["video_keys"]:
        feature = features[key]
        video = feature["info"]
        if feature["shape"] != [expected["video_height"], expected["video_width"], 3]:
            raise ValueError(f"video feature shape mismatch for {key}")
        if video["video.codec"] != expected["video_codec"] or video["video.fps"] != expected["fps"]:
            raise ValueError(f"video metadata mismatch for {key}")
    return info, episodes


def _column(table, name: str, np):
    return np.asarray(table[name].combine_chunks().to_pylist())


def _numeric_summary(values, np) -> dict[str, Any]:
    data = np.concatenate(values, axis=0).astype(np.float64, copy=False)
    return {
        "count": int(data.shape[0]),
        "min": data.min(axis=0).tolist(),
        "max": data.max(axis=0).tolist(),
        "mean": data.mean(axis=0).tolist(),
        "std": data.std(axis=0).tolist(),
    }


def _audit_parquet(
    local_dir: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    dataset: dict[str, Any],
    np,
    pq,
) -> dict[str, Any]:
    expected = dataset["expected"]
    action_values = []
    state_values = []
    total_rows = 0
    done_true = 0

    for episode in episodes:
        episode_index = episode["episode_index"]
        length = episode["length"]
        episode_chunk = episode_index // info["chunks_size"]
        path = local_dir / info["data_path"].format(
            episode_chunk=episode_chunk, episode_index=episode_index
        )
        table = pq.read_table(path)
        required = {
            "action",
            "next.done",
            "observation.state",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        }
        if not required.issubset(table.column_names):
            raise ValueError(f"missing parquet columns in episode {episode_index}")
        if table.num_rows != length:
            raise ValueError(f"row count mismatch in episode {episode_index}")

        action = _column(table, "action", np)
        state = _column(table, "observation.state", np)
        timestamp = _column(table, "timestamp", np)
        frame_index = _column(table, "frame_index", np)
        episode_column = _column(table, "episode_index", np)
        global_index = _column(table, "index", np)
        task_index = _column(table, "task_index", np)
        done = _column(table, "next.done", np).astype(bool)

        if action.shape != (length, expected["action_dim"]):
            raise ValueError(f"action shape mismatch in episode {episode_index}")
        if state.shape != (length, expected["state_dim"]):
            raise ValueError(f"state shape mismatch in episode {episode_index}")
        if not np.isfinite(action).all() or not np.isfinite(state).all():
            raise ValueError(f"NaN/Inf found in episode {episode_index}")
        if not np.array_equal(frame_index, np.arange(length)):
            raise ValueError(f"frame indexes are not contiguous in episode {episode_index}")
        if not np.all(episode_column == episode_index) or not np.all(task_index == 0):
            raise ValueError(f"episode/task index mismatch in episode {episode_index}")
        if not np.array_equal(global_index, np.arange(total_rows, total_rows + length)):
            raise ValueError(f"global indexes are not contiguous in episode {episode_index}")
        expected_time = np.arange(length, dtype=np.float64) / expected["fps"]
        if not np.allclose(timestamp, expected_time, atol=1e-5, rtol=0.0):
            raise ValueError(f"timestamp mismatch in episode {episode_index}")
        if not bool(done[-1]) or bool(done[:-1].any()):
            raise ValueError(f"done alignment mismatch in episode {episode_index}")

        action_values.append(action)
        state_values.append(state)
        done_true += int(done.sum())
        total_rows += length

    if total_rows != expected["frames"]:
        raise ValueError("parquet row count does not equal total_frames")
    return {
        "episodes": len(episodes),
        "rows": total_rows,
        "done_true": done_true,
        "action": _numeric_summary(action_values, np),
        "state": _numeric_summary(state_values, np),
    }


def _probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(completed.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError(f"expected one video stream in {path}")
    return streams[0]


def _audit_videos(
    local_dir: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    dataset: dict[str, Any],
) -> dict[str, Any]:
    expected = dataset["expected"]
    checked = 0
    for episode in episodes:
        episode_index = episode["episode_index"]
        episode_chunk = episode_index // info["chunks_size"]
        for video_key in dataset["video_keys"]:
            path = local_dir / info["video_path"].format(
                episode_chunk=episode_chunk,
                video_key=video_key,
                episode_index=episode_index,
            )
            stream = _probe_video(path)
            fps = float(Fraction(stream["avg_frame_rate"]))
            observed = (
                stream["codec_name"],
                int(stream["width"]),
                int(stream["height"]),
                fps,
                int(stream["nb_read_frames"]),
            )
            wanted = (
                expected["video_codec"],
                expected["video_width"],
                expected["video_height"],
                float(expected["fps"]),
                episode["length"],
            )
            if observed != wanted:
                raise ValueError(
                    f"video metadata/frame mismatch for {path}: "
                    f"observed={observed}, expected={wanted}"
                )
            checked += 1
    if checked != expected["videos"]:
        raise ValueError(f"checked {checked} videos, expected {expected['videos']}")
    return {
        "videos": checked,
        "codec": expected["video_codec"],
        "width": expected["video_width"],
        "height": expected["video_height"],
        "fps": expected["fps"],
        "all_frame_counts_match_episode_lengths": True,
    }


def run_audit(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    np, pq, HfApi, RepoFile, snapshot_download = _required_packages()
    dataset = config["dataset"]
    local_dir = REPO_ROOT / dataset["local_dir"]
    output_dir = REPO_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    run = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }

    resolved = {
        **config,
        "run": run,
        "resolved": {
            "config_path": str(config_path.resolve()),
            "local_dir": str(local_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
        },
    }
    resolved_path = output_dir / "resolved_config.json"
    _write_json(resolved_path, resolved)

    api = HfApi()
    remote_files, repo_info = _validate_remote(api, RepoFile, dataset)
    snapshot_download(
        repo_id=dataset["repo_id"],
        repo_type=dataset["repo_type"],
        revision=dataset["revision"],
        local_dir=local_dir,
    )
    file_manifest = _verify_download(remote_files, local_dir)
    file_manifest_path = output_dir / "dataset_files.jsonl"
    _write_jsonl(file_manifest_path, file_manifest)

    info, episodes = _read_metadata(local_dir, dataset)
    split = config["split"]
    splits = make_episode_splits(
        dataset["expected"]["episodes"],
        split["train"],
        split["validation"],
        split["test"],
        split["seed"],
    )
    split_manifest = {
        "status": split["status"],
        "dataset_revision": dataset["revision"],
        "group": split["group"],
        "seed": split["seed"],
        "splits": splits,
    }
    split_manifest_path = output_dir / "split_manifest.json"
    _write_json(split_manifest_path, split_manifest)

    parquet = _audit_parquet(local_dir, info, episodes, dataset, np, pq)
    videos = _audit_videos(local_dir, info, episodes, dataset)
    report = {
        "status": "COMPLETED",
        "experiment_kind": "dataset_audit",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "dataset": {
            "repo_id": dataset["repo_id"],
            "revision": dataset["revision"],
            "resolved_revision": repo_info.sha,
            "license": dataset["license"],
            "files": len(remote_files),
            "bytes": sum(entry.size for entry in remote_files),
        },
        "episode_split": {
            name: len(ids) for name, ids in splits.items()
        },
        "parquet": parquet,
        "videos": videos,
        "no_simulator_state": True,
        "snapshot_replay_supported": False,
        "limitations": [
            "This LeRobot artifact has no MuJoCo/Robosuite simulator state and cannot validate snapshot replay.",
            "It exposes next.done but not separate terminated/truncated labels."
        ],
        "artifacts": {
            "resolved_config.json": sha256_file(resolved_path),
            "dataset_files.jsonl": sha256_file(file_manifest_path),
            "split_manifest.json": sha256_file(split_manifest_path),
        },
    }
    _write_json(output_dir / "audit_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/cf_resrl/m0_square_dataset.json",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = _load_config(config_path)
    output_dir = REPO_ROOT / config["output_dir"]
    try:
        report = run_audit(config, config_path)
    except Exception as exc:
        _write_json(
            output_dir / "audit_report.json",
            {
                "status": "FAILED",
                "experiment_kind": "dataset_audit",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
