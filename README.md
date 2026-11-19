# residual-rl-sim

Execution-consistent predictive residual RL and behavior-preserving base-policy updates, based on ResFiT.

## Start here

Read [CODEX_TASK.md](CODEX_TASK.md) for the complete Chinese implementation plan, literature boundary, minimal experiment, full extensions, acceptance criteria, and publication requirements. Coding agents must also follow [AGENTS.md](AGENTS.md).

## What exists

- A staged research specification, **not** a completed RL implementation.
- Dependency-free execution-contract reference code in `pa_reswm/contracts.py`.
- 21 locally passing unit tests and their raw report in `results/bootstrap`.
- A machine-readable MVP specification in `configs/mvp_spec.json`; this is **not** a runnable training configuration.

**No ACT, residual RL, or world model has been trained here. No MuJoCo integration test or manipulation success-rate evaluation has been run.** Do not treat unit tests as research performance evidence.

## Run the available tests

```bash
python -m unittest discover -s tests -v
```

## Planned progression

M0 data/environment validation and baseline -> M1 timestamped prediction alignment -> M2 short-horizon residual candidate evaluation -> M3 behavior-preserving recentering. Synthetic imagination is optional and disabled by default.

## Upstream

Official ResFiT: https://github.com/amazon-far/residual-offpolicy-rl

Inspected upstream commit: `66262b06a0caf159fb001504949b7b48ccda5f64`.

Paper: https://arxiv.org/abs/2509.19301

Preserve upstream copyright, licenses, and THIRD-PARTY notices when incorporating code. Data and checkpoint redistribution require their own license checks. This initialization does not vendor the upstream implementation.

## Results policy

Push actual source code, resolved configs, per-episode data, logs, manifests, reports, and negative results. Do not fabricate runs when resources are unavailable; record `BLOCKED` or `NOT_RUN`. Keep credentials, datasets, and large model files out of ordinary Git.
