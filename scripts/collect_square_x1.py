"""Collect one prespecified Square X1 checkpoint, delay, amplitude, and data cell."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from collect_e0 import DEFAULT_CONFIG as E0_CONFIG, _status, collect_branch_bank
from collect_factorial import REPO_ROOT, _json, _sha256, load_square_runner


MVP = REPO_ROOT / "results/cf_resrl/mvp_square/closed_loop/attempts/20261126T195036106877Z_d549bf669ea3"
sys.path.insert(0, str(REPO_ROOT))
from cf_resrl.x1_plan import DEFAULT_CONFIG, load_x1_config, plan_x1, resolve_cell_policies, source_cohort


def _attempts(specifications):
    attempts = {}
    for entry in specifications:
        seed, separator, path = entry.partition("=")
        if not separator or not seed.isdecimal() or not path or int(seed) in attempts:
            raise ValueError("Specify each completed ResFiT attempt once as SEED=PATH")
        attempts[int(seed)] = Path(path).expanduser().resolve()
    return attempts


def _mvp_provenance():
    status = _json(MVP / "status.json")
    if status["status"] != "COMPLETED" or status["summary_sha256"] != _sha256(MVP / "summary.json"):
        raise ValueError("Square MVP must complete before X1 collection")
    return {"attempt": str(MVP.relative_to(REPO_ROOT)), "summary_sha256": status["summary_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cell")
    parser.add_argument("--resfit-attempt", action="append", default=[], metavar="SEED=PATH")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--list-cells", action="store_true")
    args = parser.parse_args()
    x1 = load_x1_config(args.config)
    cells = plan_x1(x1)
    if args.list_cells:
        print(json.dumps(cells, indent=2))
        return
    matches = [cell for cell in cells if cell["cell_id"] == args.cell]
    if len(matches) != 1:
        raise ValueError("Select one registered X1 cell from --list-cells")
    cell = matches[0]
    mvp = _mvp_provenance()
    policies = resolve_cell_policies(x1, cell, _attempts(args.resfit_attempt))
    output_root = Path("results/cf_resrl/expansion_square/branch_data") / cell["cell_id"]
    protocol = {
        **_json(E0_CONFIG),
        "task": "Square",
        "dataset": "cf_square_x1_factorial_v1",
        "status": "REGISTERED_X1_CELL",
        "x1_config_sha256": _sha256(args.config),
        "x1_cell": cell,
        "split_seed": x1["split_seed"],
        "selector_seed": x1["selector_seed"],
        "factorial_split_counts": x1["factorial_split"],
        "factorial_cohorts": x1["factorial_cohorts"],
        "pair_training_cohorts": x1["pair_training_cohorts"],
        "simulation_horizon_steps": 300,
        "refresh_delay_steps": cell["delay"],
        "residual_scale": cell["residual_scale"],
        "residual_component_limit": x1["residual_component_limit"],
        "output_root": output_root.as_posix(),
        "mvp": mvp,
    }
    del protocol["gate_attempt"], protocol["gate_summary_sha256"]
    stem = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.out or REPO_ROOT / output_root / "attempts" / stem
    out.mkdir(parents=True, exist_ok=False)
    print(f"Square X1 cell {args.cell}: {out}", flush=True)
    runner = None
    try:
        runner, provenance = load_square_runner(protocol, x1_selection=policies)
        collect_branch_bank(protocol, source_cohort(x1, cell["distribution"]), out, runner, provenance)
    except Exception as error:
        status = out / "status.json"
        progress = _json(status).get("completed_groups", 0) if status.exists() else 0
        _status(out, "FAILED", completed_groups=progress, error=str(error), type=type(error).__name__)
        raise
    finally:
        if runner is not None:
            runner.wrapper.close()


if __name__ == "__main__":
    main()
