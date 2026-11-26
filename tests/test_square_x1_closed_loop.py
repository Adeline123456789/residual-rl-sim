import json
from pathlib import Path
import tempfile
import unittest

from cf_resrl.x1_plan import DEFAULT_CONFIG, load_x1_config, plan_x1, source_cohort
from scripts.collect_factorial import _sha256, _write
from scripts.eval_scheduler import (
    PROBE_ROWS,
    X1Run,
    _seeds,
    _settings,
    _x1_closed_loop_protocol,
    _x1_probe_inputs,
)


class X1ClosedLoopTests(unittest.TestCase):
    def setUp(self):
        self.registration = load_x1_config()
        self.config_sha256 = _sha256(DEFAULT_CONFIG)
        self.cell = next(
            cell for cell in plan_x1(self.registration)
            if cell["delay"] == 4 and cell["residual_scale"] == 0.5 and cell["distribution"] == "enriched"
        )
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.source = Path(workspace.name) / "data"
        self.offline = Path(workspace.name) / "offline"
        (self.offline / "models").mkdir(parents=True)
        self.source.mkdir()
        self.run = X1Run(self.registration, self.config_sha256, self.cell, self.offline, None)

    def _write_evidence(self):
        plan = source_cohort(self.registration, self.cell["distribution"])
        identities = {"policy_identity": "act25k", "residual_policy_identity": "resfit150k"}
        _write(self.source / "run_manifest.json", {
            "protocol": {
                "x1_cell": self.cell, "x1_config_sha256": self.config_sha256, "task": "Square",
                "simulation_horizon_steps": 300, "refresh_delay_steps": 4, "residual_scale": 0.5,
                "residual_component_limit": 0.1,
            },
            "source_plan": plan, "identities": identities,
        })
        (self.source / "group_index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in plan))
        _write(self.source / "summary.json", {})
        _write(self.source / "feature_schema.json", {})
        _write(self.source / "status.json", {
            "status": "COMPLETED", "completed_groups": len(plan),
            "index_sha256": _sha256(self.source / "group_index.jsonl"),
            "summary_sha256": _sha256(self.source / "summary.json"),
        })
        rows = []
        for method in PROBE_ROWS.values():
            model = self.offline / "models" / f"{method}.pt"
            model.write_text(method)
            rows.append({
                "method": method, "distribution": self.cell["distribution"], "fraction": 1.0,
                "status": "COMPLETED", "model": model.relative_to(self.offline).as_posix(),
                "model_sha256": _sha256(model), "training_budget": {},
            })
        _write(self.offline / "results.json", rows)
        _write(self.offline / "run_manifest.json", {"data": {
            "attempt": str(self.source), "x1_cell": self.cell, "x1_config_sha256": self.config_sha256,
            "identities": identities,
            **{f"{key}_sha256": _sha256(self.source / name) for key, name in (
                ("status", "status.json"), ("index", "group_index.jsonl"), ("summary", "summary.json"),
                ("run_manifest", "run_manifest.json"), ("feature_schema", "feature_schema.json"),
            )},
        }})
        _write(self.offline / "status.json", {
            "status": "COMPLETED", "run_manifest_sha256": _sha256(self.offline / "run_manifest.json"),
            "results_sha256": _sha256(self.offline / "results.json"),
        })

    def test_distribution_models_and_delay_are_bound_to_real_source_receipt(self):
        self._write_evidence()
        _, seeds, probes, protocol, provenance = _x1_probe_inputs(self.run)
        self.assertEqual({model["distribution"] for model in probes.values()}, {"enriched"})
        self.assertEqual(protocol["refresh_delay_steps"], 4)
        self.assertEqual(protocol["residual_scale"], 0.5)
        self.assertEqual(provenance["source_index_sha256"], _sha256(self.source / "group_index.jsonl"))
        self.assertEqual(len(seeds), 160)
        with (self.source / "group_index.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "source evidence differs"):
            _x1_probe_inputs(self.run)

    def test_cross_cell_model_reuse_is_rejected(self):
        self._write_evidence()
        different_cell = next(cell for cell in plan_x1(self.registration) if cell["delay"] == 0)
        different = X1Run(self.registration, self.config_sha256, different_cell, self.offline, None)
        with self.assertRaisesRegex(ValueError, "selected cell and registration"):
            _x1_probe_inputs(different)

    def test_closed_loop_caps_shared_seeds_and_effective_norm_grid(self):
        half = _x1_closed_loop_protocol(self.run)
        nominal_run = X1Run(self.registration, self.config_sha256, {
            **self.cell, "residual_scale": 1.0, "distribution": "natural",
        }, self.offline, None)
        nominal = _x1_closed_loop_protocol(nominal_run)
        self.assertEqual(half["act_budgets"], [15, 20, 25])
        self.assertEqual(half["norm_thresholds"], [0.075, 0.115])
        self.assertEqual(half["norm_thresholds_nominal"], nominal["norm_thresholds"])
        self.assertEqual(_seeds(half, set()), _seeds(nominal, set()))
        for cap, methods in _settings(half, 300, 20).items():
            for method in ("B5", "B6", "P1"):
                self.assertEqual([setting.act_budget for setting in methods[method]], [cap, cap])


if __name__ == "__main__":
    unittest.main()
