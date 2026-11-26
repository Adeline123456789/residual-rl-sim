# Square E0-C offline probe completed

The probes use the frozen, episode-disjoint Square E0 feature bank. Each stratum has 160 factorial training contexts, 20 validation contexts, and 20 locked test contexts. Five nested fractions were evaluated separately in each stratum. B5 and P1 used the exact same episodes, deployed features, parameter count, optimizer updates, and batch sampling. B6 used only V10 and V11 labels from a fixed candidate order. The rollout-matched B6 uses the same number of real branches. The joint-cap B6 uses the longest leading prefix of whole pairs that fits both registered caps. Source scouting is reported separately.

A probe requests refresh when the predicted gain exceeds the cost. Mean regret gives equal weight to the five registered costs and to each held-out source episode. Intervals resample the held-out episodes with their paired predictions. Positive differences in the table favor P1.

| Cohort, full data | Comparison | Baseline − P1 mean regret | 95% paired interval |
| --- | --- | ---: | --- |
| Natural | B5 | 0.00168 | [-0.00427, 0.00819] |
| Natural | B6-rollouts | 0.00319 | [-0.01008, 0.02302] |
| Natural | B6-joint-cap | 0.00279 | [-0.01050, 0.02275] |
| Enriched | B5 | -0.00488 | [-0.01120, 0.00184] |
| Enriched | B6-rollouts | -0.00756 | [-0.01584, 0.00148] |
| Enriched | B6-joint-cap | -0.00650 | [-0.01520, 0.00265] |

The full-data natural point estimates favor P1 against the direct models. The enriched point estimates favor the direct models, and all six full-data intervals include zero. The complete learning curves also retain Norm and Linear-full. This result advances the registered comparison through closed-loop validation.

`selection_audit.json` records realized branch counts, control steps, scouting and unused budget for every training fraction. `predictions.jsonl` records both held-out splits by episode, and `results.json` includes gain MAE, interaction MAE where available, calibration, and cost-specific critical decision errors. All 60 saved probes reproduced their original predictions exactly on reload.
