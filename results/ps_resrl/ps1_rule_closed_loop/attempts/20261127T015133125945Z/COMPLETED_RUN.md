# PS-1 Closed-Loop Evaluation: PS-Rule (B2) vs Original ResFiT & CF-ResRL MVP

Validation-selected setting: `kappa=0.5`, `nu_min=0.25`, `nu_max=1.0`.

## Held-Out Test (50 seeds `20269124..20269173`) Across Refill Delays

| Refill Delay τ | Method | Success | Mean ACT Calls | Mean Steps | Sat Rate | Action Jerk L2 | ΔSuccess vs Orig (95% CI) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Original ResFiT | 42/50 (84.0%) | 9.32 | 177.6 | 0.984 | 0.1191 | — |
| 0 | PS-Rule (B2) | 43/50 (86.0%) | 8.36 | 172.6 | 0.991 | 0.1131 | +2.0 [-10.0, +14.0] |
| 2 | Original ResFiT | 43/50 (86.0%) | 8.96 | 168.9 | 0.993 | 0.1847 | — |
| 2 | PS-Rule (B2) | 42/50 (84.0%) | 8.50 | 177.1 | 0.996 | 0.1744 | -2.0 [-12.0, +8.0] |
| 4 | Original ResFiT | 36/50 (72.0%) | 10.46 | 201.4 | 0.988 | 0.1785 | — |
| 4 | PS-Rule (B2) | 32/50 (64.0%) | 10.18 | 213.0 | 0.996 | 0.1656 | -8.0 [-24.0, +8.0] |

Attempt directory: `results/ps_resrl/ps1_rule_closed_loop/attempts/20261127T015133125945Z`
