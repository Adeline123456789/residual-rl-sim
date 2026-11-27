"""Run PS-0 fifth-branch causal diagnostic on exact 400 E0 factorial source episodes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from collect_factorial import load_square_runner
from ps_resrl.fifth_branch_runner import (
    PHASE_BRANCHES,
    evaluate_e0_source_episode,
    summarize_ps0,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _format_markdown(summary: dict[str, Any], out: Path) -> str:
    lines = [
        "# PS-0 Fifth-Branch Causal Diagnostic (Square)",
        "",
        f"Evaluated {summary['total_episodes']} factorial source episodes from `cf_square_factorial_v1` (zero extra ACT calls for phase branches).",
        "",
    ]
    for dist, block in summary["distributions"].items():
        v10 = block["baselines"]["V10"]
        v11 = block["baselines"]["V11"]
        lines.extend([
            f"## Distribution: `{dist}` (n={block['episodes']})",
            "",
            f"- `V10` (Keep): success={v10['success_rate']['mean']*100:.1f}%, return={v10['discounted_return']['mean']:.4f}, calls={v10['act_calls']['mean']:.2f}",
            f"- `V11` (Refresh): success={v11['success_rate']['mean']*100:.1f}%, return={v11['discounted_return']['mean']:.4f}, calls={v11['act_calls']['mean']:.2f}",
            f"- Refresh rescued (`V10=0 & V11=1`): {block['v10_fail_v11_success']} / {block['v10_failures']} V10 failures",
            "",
            "| Branch | Success % | ΔSucc vs V10 (95% CI) | ΔRet vs V10 (95% CI) | Mean ACT Calls | Rescued / Regressed | Phase Recovery Rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for mode in PHASE_BRANCHES:
            b = block["phase_branches"][mode]
            sr = b["success_rate"]["mean"] * 100.0
            ds = b["delta_success_vs_V10"]
            dr = b["delta_discounted_return_vs_V10"]
            calls = b["act_calls"]["mean"]
            rec = (
                f"{b['phase_recovery_rate']*100:.1f}% ({b['recovered_of_refresh_rescued_count']}/{b['refresh_rescued_count']})"
                if b["phase_recovery_rate"] is not None
                else "n/a"
            )
            lines.append(
                f"| `{mode}` | {sr:.1f}% | {ds['mean']*100:+.1f} [{ds['ci95'][0]*100:+.1f}, {ds['ci95'][1]*100:+.1f}] | "
                f"{dr['mean']:+.4f} [{dr['ci95'][0]:+.4f}, {dr['ci95'][1]:+.4f}] | {calls:.2f} | "
                f"+{b['rescued_from_V10_failure']} / -{b['regressed_from_V10_success']} | {rec} |"
            )
        lines.append("")
    lines.append(f"Attempt directory: `{out}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/ps_resrl/ps0_square_fifth_branch.json",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.out or (REPO_ROOT / config["output_root"] / "attempts" / stamp)
    out.mkdir(parents=True, exist_ok=False)

    e0_index = REPO_ROOT / config["e0_index_path"]
    all_rows = [
        json.loads(line)
        for line in e0_index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    factorial_rows = [r for r in all_rows if r["bank"] == config["bank"]]
    if args.limit > 0:
        # Take balanced natural/enriched slice for smoke runs.
        nat = [r for r in factorial_rows if r["distribution"] == "natural"][: max(1, args.limit // 2)]
        enr = [r for r in factorial_rows if r["distribution"] == "enriched"][: args.limit - len(nat)]
        factorial_rows = nat + enr

    gate_config = json.loads((REPO_ROOT / config["gate_config"]).read_text(encoding="utf-8"))
    runner, provenance = load_square_runner(gate_config)
    _write_json(out / "resolved_config.json", {**config, "limit": args.limit, "provenance": provenance})
    _write_json(out / "status.json", {"status": "RUNNING", "planned_episodes": len(factorial_rows)})

    records: list[dict[str, Any]] = []
    jsonl_path = out / "episode_records.jsonl"
    try:
        with jsonl_path.open("w", encoding="utf-8") as stream:
            for idx, row in enumerate(factorial_rows, start=1):
                verify_v10 = idx <= int(config["verify_v10_first_n"]) or args.limit > 0
                rec = evaluate_e0_source_episode(
                    runner,
                    row,
                    hold_steps=int(config["hold_steps"]),
                    kappa=float(config["kappa"]),
                    nu_min=float(config["nu_min"]),
                    saturation_threshold=float(config["saturation_threshold"]),
                    saturation_window=int(config["saturation_window"]),
                    verify_v10=verify_v10,
                )
                records.append(rec)
                stream.write(json.dumps(rec, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                if idx % 20 == 0 or idx == len(factorial_rows):
                    print(
                        f"PS-0 progress: {idx}/{len(factorial_rows)} "
                        f"({row['source_episode_id']}, {row['distribution']}, parity={verify_v10})",
                        flush=True,
                    )
                    _write_json(
                        out / "status.json",
                        {
                            "status": "RUNNING",
                            "completed_episodes": idx,
                            "planned_episodes": len(factorial_rows),
                        },
                    )
    except Exception as exc:
        _write_json(
            out / "status.json",
            {
                "status": "FAILED",
                "completed_episodes": len(records),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        runner.wrapper.close()

    summary = summarize_ps0(
        records,
        bootstrap_samples=int(config["bootstrap_samples"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
    )
    _write_json(out / "summary.json", summary)
    (out / "COMPLETED_RUN.md").write_text(_format_markdown(summary, out), encoding="utf-8")

    sha_lines = []
    for name in ("resolved_config.json", "episode_records.jsonl", "summary.json", "COMPLETED_RUN.md"):
        sha_lines.append(f"{_sha256(out / name)}  {name}")
    (out / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    _write_json(
        out / "status.json",
        {
            "status": "COMPLETED",
            "completed_episodes": len(records),
            "planned_episodes": len(factorial_rows),
            "summary_sha256": _sha256(out / "summary.json"),
        },
    )
    print(f"PS-0 completed: {out}", flush=True)


if __name__ == "__main__":
    main()
