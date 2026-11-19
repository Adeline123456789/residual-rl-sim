# residual-rl-sim

Research workspace for execution-consistent predictive residual RL and behavior-preserving base-policy updates, based on ResFiT.

## Status

This repository is being initialized with a Codex implementation specification and small, dependency-free execution-contract tests. **No robot-learning training run or manipulation success-rate result has been completed by this initialization.** Unit tests are not RL experiments.

The implementation must proceed through data/environment validation, a matched ResFiT baseline, a minimal prediction experiment, and gated extensions. Failed and negative experiments must be retained.

## Upstream

- ResFiT: https://github.com/amazon-far/residual-offpolicy-rl
- Inspected upstream commit: `66262b06a0caf159fb001504949b7b48ccda5f64`
- Paper: https://arxiv.org/abs/2509.19301

Preserve upstream copyright, licenses, and THIRD-PARTY notices when incorporating code. Do not assume that upstream code or datasets permit unrestricted redistribution or commercial use.

See the research branch / pull request for the detailed implementation handoff. All training commands listed in that handoff are interface requirements until their implementation and execution are recorded.
