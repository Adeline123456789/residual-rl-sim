# CF-ResRL frozen decisions

1. The research question is fixed: whether 2x2 factorial counterfactual supervision improves the interaction efficiency of residual-aware replanning decisions.
2. The primary task is Robomimic / Robosuite Square; Can is the first cross-task expansion.
3. The base demonstration data and the derived branch data are different artifacts. Demonstrations train ACT / ResFiT. Counterfactual labels come only from matched simulator branches.
4. B5 (direct full-residual prediction, same four-branch data) and B6 (direct two-branch prediction, equal branch-rollout budget) are mandatory strong baselines.
5. Performance is not a stage gate. Once the MVP is valid and fully executed, the registered expansion matrix is run even when P1 is worse than B5/B6. Results change the supported claim, not the experimental endpoint.
6. New auxiliary modules or newly discovered papers update the related-work and limitations sections; they do not become prerequisites unless they expose an actual validity failure.
7. World models, Agent, Memory, Event Observer, Direction Classifier, selective action repair, and base-policy distillation are outside the core experiment and cannot block it.
8. The only hard stops are experiment-invalidating failures: non-executable nominal task, invalid snapshot/branch semantics, privileged information leakage, action/safety-limit violation, train/test contamination, or missing/corrupted evidence.
