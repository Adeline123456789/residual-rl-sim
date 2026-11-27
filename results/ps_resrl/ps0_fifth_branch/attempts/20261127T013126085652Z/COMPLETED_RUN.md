# PS-0 Fifth-Branch Causal Diagnostic (Square)

Evaluated 400 factorial source episodes from `cf_square_factorial_v1` (zero extra ACT calls for phase branches).

## Distribution: `natural` (n=200)

- `V10` (Keep): success=81.5%, return=0.4447, calls=4.16
- `V11` (Refresh): success=82.5%, return=0.4462, calls=4.75
- Refresh rescued (`V10=0 & V11=1`): 7 / 37 V10 failures

| Branch | Success % | ΔSucc vs V10 (95% CI) | ΔRet vs V10 (95% CI) | Mean ACT Calls | Rescued / Regressed | Phase Recovery Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `V10_PhaseFreeze1` | 81.5% | +0.0 [-4.0, +4.0] | +0.0042 [-0.0092, +0.0172] | 4.09 | +8 / -8 | 85.7% (6/7) |
| `V10_PhaseHold4` | 83.0% | +1.5 [-1.0, +4.0] | +0.0035 [-0.0046, +0.0122] | 4.08 | +5 / -2 | 57.1% (4/7) |
| `V10_PhaseRuleSuffix` | 80.0% | -1.5 [-5.0, +2.0] | -0.0359 [-0.0497, -0.0224] | 3.64 | +5 / -8 | 57.1% (4/7) |

## Distribution: `enriched` (n=200)

- `V10` (Keep): success=84.0%, return=0.5510, calls=3.60
- `V11` (Refresh): success=85.0%, return=0.5551, calls=4.11
- Refresh rescued (`V10=0 & V11=1`): 6 / 32 V10 failures

| Branch | Success % | ΔSucc vs V10 (95% CI) | ΔRet vs V10 (95% CI) | Mean ACT Calls | Rescued / Regressed | Phase Recovery Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `V10_PhaseFreeze1` | 85.0% | +1.0 [-1.5, +3.5] | -0.0029 [-0.0112, +0.0051] | 3.61 | +4 / -2 | 33.3% (2/6) |
| `V10_PhaseHold4` | 83.5% | -0.5 [-3.0, +2.0] | -0.0053 [-0.0153, +0.0041] | 3.69 | +3 / -4 | 33.3% (2/6) |
| `V10_PhaseRuleSuffix` | 85.0% | +1.0 [-2.0, +4.0] | -0.0295 [-0.0437, -0.0157] | 3.19 | +6 / -4 | 66.7% (4/6) |

Attempt directory: `/media/midea/work/CF-ResRL/residual-rl-sim/results/ps_resrl/ps0_fifth_branch/attempts/20261127T013126085652Z`
