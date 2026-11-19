# Execution instructions for coding agents

Read `CODEX_TASK.md` before editing. This is a research project, not a request to create a decorative scaffold.

## Current status

Only the dependency-free execution contracts and their unit tests exist. No ResFiT/ACT/world-model training or robot evaluation has run. `results/bootstrap` is a unit-test result, NOT evidence of robot success.

## Required order

M0 data/environment audit -> verified base and residual baseline -> M1 matched prediction experiment -> M2 short-horizon candidate checks -> M3 behavior-preserving recentering. Do not enable every extension at once.

Use the pinned upstream source, preserve licenses, and retain negative results. Validate data BEFORE long BC training. All methods share data, base checkpoint, timing semantics, hard action limits, evaluation scenes, and environment-step accounting.

A passing unit test is not a completed experiment. Do not manufacture CSVs, learning curves, confidence intervals, or success rates. If GPU, renderer, data access, or authentication prevents execution, write a specific `BLOCKED.md`, commit all genuine work, and report exactly which stage is not run.

Make changes on `research/pa-reswm-execution-consistency` or a child branch. Push code, resolved configurations, per-episode results, reproducibility manifests, and reports to `Adeline123456789/residual-rl-sim`; open/update a PR. Never force-push or silently overwrite user work. Keep credentials and restricted datasets out of Git. Keep large weights out of ordinary Git; publish licensed artifacts with checksum and recovery instructions.

Only mark a stage COMPLETED after its acceptance criteria and artifacts exist. Plan thresholds are not measured results. A valid negative result is an acceptable research outcome.
