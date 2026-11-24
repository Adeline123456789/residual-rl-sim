# residual-rl-sim

This branch studies a narrowly defined question on top of ResFiT:

> Can 2x2 factorial counterfactual supervision improve the interaction efficiency of residual-aware replanning for action-chunked robot policies?

## Start here

- [CF_RESRL_CODEX_TASK.md](CF_RESRL_CODEX_TASK.md): frozen research question, dataset choice, minimal validation, full expansion, validity rules, and upload protocol.
- [AGENTS.md](AGENTS.md): mandatory execution discipline for Codex and other coding agents.
- [docs/CF_RESRL_DECISION_LOG.md](docs/CF_RESRL_DECISION_LOG.md): decisions that must not be silently moved during the project.

The older PA-ResWM `CODEX_TASK.md`, contracts, tests, and branch are retained as research history.

## Chosen data

- Primary demonstrations: `ankile/robomimic-mh-square-image`.
- First cross-task expansion: `ankile/robomimic-mh-can-image`.
- Research labels: a derived simulator branch dataset containing matched V00/V01/V10/V11 outcomes. Demonstrations alone do not contain the required counterfactual labels.

## What exists now

- A detailed implementation and experiment specification.
- Dependency-free 2x2 factorial algebra, local decision-regret, leakage, and equal-budget accounting contracts in `cf_resrl/factorial.py`.
- Unit tests for those contracts.
- Previous PA-ResWM execution contracts and their tests.

**No ACT/ResFiT training, MuJoCo branch collection, offline probe, closed-loop scheduler evaluation, or online CF-ResRL result has been run on this branch yet.** Planning files and unit tests are not robot-learning evidence.

## Run available tests

```bash
python -m unittest discover -s tests -v
```

## Progression policy

Valid negative results are first-class results. Performance thresholds do not gate later registered experiments. Only experiment-invalidating failures—such as snapshot errors, privileged input, action-limit violations, or data leakage—pause the run for repair.

## Upstream

Official ResFiT: `amazon-far/residual-offpolicy-rl`, pinned commit `66262b06a0caf159fb001504949b7b48ccda5f64`. Preserve upstream licenses and THIRD-PARTY notices.
