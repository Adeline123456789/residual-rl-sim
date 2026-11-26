# Square E0 branch collection completed

The `cf_square_factorial_v1` collection finished on the local RTX 3090 on November 27, 2026 at 03:35 Asia/Shanghai. It supplies the fixed policy-paired outcomes used to test whether factorial supervision improves the interaction efficiency of residual-aware replanning.

## Cohorts and realized budget

The frozen final ACT and 300k ResFiT policies used two control ticks of refresh latency. The natural cohort selects a uniformly sampled decision tick within each nominal source rollout. The enriched cohort selects the earliest tick at which the proposed residual has maximum L2 norm. Both use the originally proposed residual.

| Distribution | Four-branch train | Validation | Test | Two-branch train only |
| --- | ---: | ---: | ---: | ---: |
| Natural | 160 | 20 | 20 | 160 |
| Enriched | 160 | 20 | 20 | 160 |

All 720 source episodes have distinct seeds. The 2,240 intervention branches took 182,808 environment control steps, 9,565 ACT calls and 1,344.54 cumulative seconds. Nominal source scouting took a separate 126,066 control steps, 6,624 ACT calls and 1,209.97 cumulative seconds.

## Matched decision outcomes

| Estimate | Zero-cost decision flip rate, 95% interval | Mean interaction, 95% interval |
| --- | ---: | ---: |
| Natural, episode-uniform | 19.0% [13.5%, 24.5%] | 0.00053 [−0.01788, 0.02037] |
| Natural, weighted by source episode length | 19.0% [13.5%, 25.3%] | 0.00446 [−0.02041, 0.03080] |
| Enriched, episode-uniform | 20.0% [14.5%, 25.5%] | 0.00309 [−0.01149, 0.01881] |

The intervals use 1,000 source-episode bootstrap samples within each distribution. The enriched estimate describes the residual-norm selection rule. The mean interaction intervals include zero. The matched decision outcomes supply labels for the offline comparisons.

## Verification and recovery

`verification.json` records the preregistered configuration, simulator gate, dependency lock, index and evidence checksums. The verification read all 720 gzip group files and 720 NPZ files, reproduced the registered selection and bootstrap summary, checked the split and real control step accounting, and verified that each restored V10 trace exactly matched its nominal scouting suffix.

The archive `cf-square-factorial-v1-e0-20261127.tar` contains the 1,440 raw group and feature files. It is 298,803,200 bytes with SHA-256 `0a5da328332c38a4c782bbd70c91edf50859e0ba0db16162c33fe54ad2736cd1`. Extract its two directories into this attempt and run `python <attempt>/verify_evidence.py --archive <archive>` from the repository root. The verifier checks every extracted file and archive member against the original collection index.

The source demonstration dataset is `ankile/robomimic-mh-square-image`, revision `4e5cee7ea7d63b920ff76c2529b781c2474697da`. Its dataset audit records Apache-2.0. The pinned ResFiT upstream code is CC BY-NC 4.0. This archive contains simulator trajectories and observation-derived features. ACT and ResFiT model files and optimizer state remain local.
