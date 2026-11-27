# Execution instructions for coding agents

Read `CF_RESRL_CODEX_TASK.md` before editing. The old `CODEX_TASK.md` and PA-ResWM files are retained as research history; do not delete them.

## Frozen question

Every task must begin by answering: **How does this experiment directly advance the final question: whether 2x2 factorial counterfactual supervision improves the interaction efficiency of residual-aware replanning?**

Do not move the endpoint. The mandatory strong comparisons are P1 vs B5 on the same four-branch data and P1 vs B6 under the same real branch-rollout / environment-step budget.

## Progression rule

Progression depends on valid implementation and complete evidence, not on a positive effect size. Once the Square MVP is valid and fully run, execute the registered expansion matrix even if P1 is worse than B5/B6. Negative results change the supported claim; they do not erase completed work or block expansion.

Only experiment-invalidating failures are hard stops: a non-executable nominal task, invalid snapshot/branch semantics, privileged-information leakage, broken action/safety limits, train/validation/test contamination, or missing/corrupted evidence. Fix these before interpreting results.

World models, Agent, Memory, Event Observer, Direction Classifier, selective repair, base distillation, and similar auxiliary modules are not prerequisites and may not block the core experiments.

## Evidence policy

A passing unit test is not a manipulation result. Do not fabricate CSVs, success rates, confidence intervals, screenshots, or checkpoints. Retain null and negative results. If resources or authentication block a run, add a precise `BLOCKED.md` without downgrading already completed stages.

Push code, resolved configurations, raw per-branch and per-episode results, manifests, logs, checksums, and reports to `Adeline123456789/residual-rl-sim` on `research/cf-resrl-factorial-supervision`; update the PR. Never force-push or overwrite user work. Keep credentials and restricted data out of Git; use licensed external storage or Git LFS for large artifacts.

ACT checkpoint binaries, optimizer state, and archives containing them stay local. Publish only their provenance and checksums alongside experiment records.
