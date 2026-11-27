# Square closed-loop MVP

Completed on the local RTX 3090 on 2026-11-27 at 04:53:16 Beijing time. The run froze ACT `49999`, ResFiT seed `20261124` at 300,000 environment steps, and a two-control-step refresh delay. B5, B6 and P1 used the full natural training set for their respective offline probe budgets. All configuration choices were made on 20 validation seeds before evaluating the same 50 held-out seeds across methods.

## Held-out performance

Entries give successes out of 50 and mean ACT calls per episode.

| Scheduler | 15-call cap | 20-call cap | 25-call cap |
| --- | ---: | ---: | ---: |
| Original ResFiT | 42 / 9.32 | 42 / 9.32 | 42 / 9.32 |
| Fixed-period refresh | 43 / 8.96 | 46 / 13.68 | 42 / 16.02 |
| Residual-norm threshold | 42 / 9.32 | 42 / 9.32 | 42 / 9.32 |
| B5 | 42 / 8.78 | 44 / 13.98 | 45 / 18.26 |
| B6 | 43 / 8.68 | 43 / 13.66 | 44 / 15.46 |
| P1 | 43 / 8.86 | 43 / 14.04 | 42 / 19.16 |

| ACT cap | P1 minus B5, success percentage points (95% interval) | P1 minus B6, success percentage points (95% interval) |
| ---: | ---: | ---: |
| 15 | +2 [−4, +8] | 0 [−8.05, +10] |
| 20 | −2 [−12, +8] | 0 [−10, +10] |
| 25 | −6 [−14, +2] | −4 [−16, +8] |

Intervals use paired resampling of the 50 held-out episode seeds with 1,000 draws and seed `20261201`. All include zero. At the 25-call cap, P1 had lower observed success and higher mean calls than B5 and B6. The closed-loop results leave the factorial interaction-efficiency advantage unestablished for this Square setting.

The fixed-performance target was 19/20 successes on validation. Fixed-period refresh had no eligible validation setting. On test the selected B5, B6, P1, original and norm settings each finished at 42 or 43 successes out of 50, below the 95% target. The underlying call, return and timing measurements remain in the budget curve and summary. Those measurements apply to the actual achieved performance.

## Verified interaction accounting

| Split | Physical episodes | Control steps | ACT calls |
| --- | ---: | ---: | ---: |
| Validation | 1,220 | 212,680 | 16,697 |
| Held-out test | 800 | 135,917 | 9,841 |
| Total | 2,020 | 348,597 | 26,538 |

The fixed-performance view required zero extra physical rollouts. The verifier reconciled 350 reused logical test entries with the physical episode index. It checked all 2,020 trace SHA-256 values, per-episode ACT caps, applied actions, chunk timing, validation selection and reported metrics. See [verification.json](verification.json) and the [raw trace recovery instructions](ARTIFACTS.md).

The Square MVP is complete. The registered X1 matrix follows under the same experiment progression rules.
