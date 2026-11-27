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

## Experiment results

- [Square branch data](results/cf_resrl/mvp_square/e0_data/attempts/20261126T184751184660Z_90a4f137d61c/COMPLETED_RUN.md) records 2,240 real simulator branches from 720 source episodes.
- [Square offline probes](results/cf_resrl/mvp_square/offline_probe/attempts/20261126T194609984038Z_bfdd16f86003/COMPLETED_RUN.md) give the same-data B5/P1 comparison and the interaction-budget-matched B6 comparison.
- [Square closed-loop MVP](results/cf_resrl/mvp_square/closed_loop/attempts/20261126T195036106877Z_d549bf669ea3/COMPLETED_RUN.md) verifies 2,020 physical rollouts. All six paired P1 versus B5/B6 success intervals include zero.
- [X1 closeout](results/cf_resrl/expansion_square/resfit_training/attempts/20261126T205747128632Z_c5f43e8aa271/INTERRUPTED_RUN.md) records the user-requested stop at 64,700 training steps for the first additional Square seed. The X1 matrix, Can transfer (X2), and online comparison (X3) were not run.

Experiment code and small evidence files are tracked on this branch. The run reports link to the verified archives for larger artifacts. Checkpoint weights remain local.

## Run available tests

```bash
python -m unittest discover -s tests -v
```

## Progression policy

Valid negative results are first-class results. Performance thresholds do not gate later registered experiments. Only experiment-invalidating failures—such as snapshot errors, privileged input, action-limit violations, or data leakage—pause the run for repair.

## Upstream

Official ResFiT: `amazon-far/residual-offpolicy-rl`, pinned commit `66262b06a0caf159fb001504949b7b48ccda5f64`. Preserve upstream licenses and THIRD-PARTY notices.
