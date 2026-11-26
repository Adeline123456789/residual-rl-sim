# Independent ACT evaluation on Square

All 50 registered local ACT checkpoints were evaluated on the same 100 seeded Square episodes. This establishes the fixed ACT baseline for the paired counterfactual branches. The evaluator ran on the RTX 3090 with 16 parallel environment lanes and completed at 02:32:15 on 2026-11-27 (Asia/Shanghai), after 1 h 12 min 37 s.

The final checkpoint at label 49,999, saved after 50,000 optimizer updates, succeeded in **55 of 100** independently seeded episodes. The original during-training evaluation of the same weights recorded **68 of 100** on its own unseeded episode cohort. Both results and the original video reference are preserved. Intermediate training evaluations correspond to their original reported steps. This independent evaluation records the saved intermediate checkpoint after the zero-based update labeled by its checkpoint name.

| Evaluation interactions | Environment control steps |
| --- | ---: |
| Scored episode steps | 1,223,573 |
| Additional vector-lane steps | 269,163 |
| Total | 1,492,736 |

The 5,000 scored records cover episode seeds 20262124–20262223 at every fixed checkpoint. Each record retains its lane and lane-local episode index, return, terminal flags, control steps, and action-clipping counts. The protocol clips proposed finite actions to the Square action space. Additional vector-lane interactions are included in the actual evaluation budget, which is separate from ACT training.

The audit reconciled every per-checkpoint JSON digest, success count, seed and lane assignment, control-step total, and original video reference against the plan, source manifest, and logs. The evaluator hashed each local model, configuration, and original video immediately before rollout. The audit also rehashed the final checkpoint and corresponding video, the source training manifest and log, the run log, and both dependency locks.

`resolved_config.json` records the exact command and source manifests, `runtime.json` records the simulator and CUDA environment, and `verification.json` links the per-checkpoint digests and counts. Run `sha256sum -c SHA256SUMS` in this directory to verify the published evidence. ACT weights and original video files are stored locally.
