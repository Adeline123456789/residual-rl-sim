#!/usr/bin/env python3
"""Download and audit the pinned author-owned Square ACT checkpoint candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cf_resrl.checkpoint import (  # noqa: E402
    validate_act_config,
    validate_file_inventory,
    validate_unpublished_license,
)
from cf_resrl.dataset import sha256_file  # noqa: E402


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _required_packages():
    try:
        import huggingface_hub
        import mujoco
        import robosuite
        import torch
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("checkpoint audit requires huggingface-hub") from exc
    return HfApi, snapshot_download, {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "huggingface_hub": huggingface_hub.__version__,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "robosuite": robosuite.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _remote_inventory(info: Any) -> list[dict[str, Any]]:
    return [
        {
            "path": sibling.rfilename,
            "bytes": sibling.size,
            "lfs_sha256": sibling.lfs.sha256 if sibling.lfs is not None else None,
        }
        for sibling in info.siblings
    ]


def run_audit(
    config: dict[str, Any], config_path: Path, output_dir: Path, run: dict[str, Any]
) -> dict[str, Any]:
    checkpoint = config["checkpoint"]
    if checkpoint["disable_xet"]:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    HfApi, snapshot_download, environment = _required_packages()
    run["environment"] = environment
    resolved = {
        **config,
        "run": run,
        "resolved": {
            "config_path": str(config_path.resolve()),
            "local_dir": str((REPO_ROOT / checkpoint["local_dir"]).resolve()),
            "output_dir": str(output_dir.resolve()),
        },
    }
    resolved_path = output_dir / "resolved_config.json"
    _write_json(resolved_path, resolved)

    info = HfApi().model_info(
        checkpoint["repo_id"],
        revision=checkpoint["revision"],
        files_metadata=True,
    )
    if info.sha != checkpoint["revision"]:
        raise ValueError(f"resolved revision {info.sha} does not match the registered revision")
    if info.private or info.gated or info.disabled:
        raise ValueError("checkpoint repository is private, gated, or disabled")

    remote_files = _remote_inventory(info)
    validate_file_inventory(remote_files, checkpoint["files"])
    license_observation = validate_unpublished_license(
        info.card_data,
        info.tags,
        [entry["path"] for entry in remote_files],
    )
    tensor_elements = info.safetensors.total if info.safetensors is not None else None
    if tensor_elements != checkpoint["expected_safetensors_tensor_elements"]:
        raise ValueError(
            f"safetensors tensor element count is {tensor_elements}, "
            f"expected {checkpoint['expected_safetensors_tensor_elements']}"
        )
    if info.safetensors.parameters != {"F32": tensor_elements}:
        raise ValueError("checkpoint parameters are not entirely float32")

    local_dir = REPO_ROOT / checkpoint["local_dir"]
    snapshot_download(
        repo_id=checkpoint["repo_id"],
        revision=checkpoint["revision"],
        local_dir=local_dir,
    )

    local_files = []
    for expected in checkpoint["files"]:
        path = local_dir / expected["path"]
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint download is missing {expected['path']}")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"checkpoint size mismatch for {expected['path']}")
        digest = sha256_file(path)
        if expected["lfs_sha256"] is not None and digest != expected["lfs_sha256"]:
            raise ValueError(f"checkpoint SHA256 mismatch for {expected['path']}")
        local_files.append(
            {"path": expected["path"], "bytes": expected["bytes"], "sha256": digest}
        )

    model_config = json.loads((local_dir / "config.json").read_text(encoding="utf-8"))
    architecture = validate_act_config(model_config)
    report = {
        "status": "COMPLETED",
        "experiment_kind": "checkpoint_audit",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "checkpoint": {
            "repo_id": checkpoint["repo_id"],
            "revision": checkpoint["revision"],
            "resolved_revision": info.sha,
            "url": checkpoint["url"],
            "local_dir": str(local_dir.resolve()),
            "public": True,
            "gated": False,
            "provenance": checkpoint["provenance"],
            "wandb_reference": checkpoint["wandb_reference"],
            "wandb_equivalence": checkpoint["wandb_equivalence"],
            "license": checkpoint["license"],
            "license_observation": license_observation,
            "safetensors_tensor_elements": tensor_elements,
            "files": local_files,
            "recovery_command": checkpoint["recovery_command"],
        },
        "architecture": architecture,
        "limitations": [
            "The publisher did not provide a model card, license, training step, dataset revision, or rollout results.",
            "This audit does not establish equivalence to the unrecoverable W&B dzbkdpwp_best artifact.",
            "Policy loading and independent Square rollout remain required before using this candidate."
        ],
        "artifacts": {
            "resolved_config.json": sha256_file(resolved_path),
        },
    }
    _write_json(output_dir / "audit_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/cf_resrl/m0_act_checkpoint.json",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = _load_config(config_path)
    checkpoint = config["checkpoint"]
    repo_head = _git("rev-parse", "HEAD")
    started_at = datetime.now(timezone.utc)
    attempt_id = f"{started_at.strftime('%Y%m%dT%H%M%S%fZ')}_{repo_head[:12]}"
    output_dir = REPO_ROOT / config["output_dir"] / "attempts" / attempt_id
    output_dir.mkdir(parents=True, exist_ok=False)
    run = {
        "attempt_id": attempt_id,
        "started_at_utc": started_at.isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "git_commit": repo_head,
        "git_dirty": bool(_git("status", "--porcelain")),
        "evidence_dir": str(output_dir.resolve()),
    }
    try:
        report = run_audit(config, config_path, output_dir, run)
    except Exception as exc:
        failed = {
            "status": "FAILED",
            "experiment_kind": "checkpoint_audit",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "run": run,
            "checkpoint": {
                "repo_id": checkpoint["repo_id"],
                "revision": checkpoint["revision"],
                "provenance": checkpoint["provenance"],
                "wandb_equivalence": checkpoint["wandb_equivalence"],
                "license": checkpoint["license"],
            },
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(output_dir / "audit_report.json", failed)
        print(f"failed evidence: {output_dir / 'audit_report.json'}", file=sys.stderr)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
