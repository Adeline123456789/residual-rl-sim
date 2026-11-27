# Square ResFiT 300k completed

This run establishes the fixed residual policy for the paired counterfactual branches. The final model is selected by the registered 300,000-step budget. The intermediate evaluations remain available for analysis.

The launcher exited with code 0 on **2026-11-27 01:08:40 Asia/Shanghai** after 15.98 hours. The final fixed checkpoint completed **41 of 50 episodes successfully (82%)**. Its mean episode return was 0.82. The evaluation took 64.98 seconds and 8,920 simulator control steps.

## Training and verification

- The ledger covers 10,000 warm-up steps plus 290,000 main training steps. The training seed was 20261124, and the registered training split contained 240 episodes of Square-MH at revision `4e5cee7ea7d63b920ff76c2529b781c2474697da`.
- The fixed ACT base was the local policy at step label 49,999 after 50,000 optimization steps. Its configuration and model hashes were verified against the configured pins. ACT weights and optimizer state stay local.
- All 30 model SHA-256 digests matched the original checkpoint manifest. Each checkpoint has a separate 50-episode evaluation on eight simulator lanes with deterministic initial lane seeds from 20262124 through 20262131. The 1,500 completed records total 323,912 evaluation control steps, including steps in lanes still running when the evaluator reached 50 completed episodes. These control steps are recorded separately from the 300,000 training steps.
- The raw ledger has 300,000 consecutive records. Every applied environment action was finite and within `[-1, 1]`. Recomputing the environment clamp, applied normalization, and implied base action from the ledger gave maximum absolute discrepancies of 1.8e-07, 1.3e-07, and 6e-08, respectively, against a 5e-07 audit tolerance. The exact recorded clip masks and dimension counts also reconciled.
- Launcher configuration, the three pinned overlay hashes, both dependency locks, and original log digest were verified. ResFiT and LeRobot were clean after the launcher reversed the registered overlays. The launch commit, full invocation, upstream pins, and lock digests are recorded in the status and verification files.

## Fixed checkpoint evaluations

| Training step | Successful episodes | Success rate | Evaluation control steps | Evaluation seconds |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 31/50 | 62% | 12,808 | 86.5 |
| 20,000 | 16/50 | 32% | 14,400 | 81.7 |
| 30,000 | 23/50 | 46% | 12,280 | 88.6 |
| 40,000 | 23/50 | 46% | 13,200 | 93.7 |
| 50,000 | 11/50 | 22% | 14,856 | 95.3 |
| 60,000 | 3/50 | 6% | 15,400 | 85.1 |
| 70,000 | 28/50 | 56% | 12,048 | 87.5 |
| 80,000 | 33/50 | 66% | 10,768 | 70.9 |
| 90,000 | 29/50 | 58% | 11,728 | 89.3 |
| 100,000 | 33/50 | 66% | 10,832 | 72.3 |
| 110,000 | 32/50 | 64% | 10,944 | 71.3 |
| 120,000 | 37/50 | 74% | 10,224 | 71.2 |
| 130,000 | 35/50 | 70% | 10,432 | 79.4 |
| 140,000 | 38/50 | 76% | 9,952 | 75.1 |
| 150,000 | 36/50 | 72% | 10,168 | 76.5 |
| 160,000 | 41/50 | 82% | 9,208 | 65.2 |
| 170,000 | 41/50 | 82% | 9,800 | 67.2 |
| 180,000 | 34/50 | 68% | 10,288 | 66.5 |
| 190,000 | 37/50 | 74% | 10,312 | 70.8 |
| 200,000 | 43/50 | 86% | 9,280 | 65.9 |
| 210,000 | 33/50 | 66% | 10,768 | 71.7 |
| 220,000 | 39/50 | 78% | 10,048 | 70.0 |
| 230,000 | 44/50 | 88% | 9,288 | 64.2 |
| 240,000 | 44/50 | 88% | 9,296 | 64.1 |
| 250,000 | 44/50 | 88% | 9,216 | 67.9 |
| 260,000 | 40/50 | 80% | 9,944 | 65.5 |
| 270,000 | 41/50 | 82% | 9,432 | 64.8 |
| 280,000 | 41/50 | 82% | 8,952 | 62.2 |
| 290,000 | 43/50 | 86% | 9,120 | 66.1 |
| 300,000 | 41/50 | 82% | 8,920 | 65.0 |

## Artifact verification and recovery

From this attempt directory, run the following to verify the published artifacts and check that the compressed action ledger decodes to the original SHA-256:

```sh
sha256sum -c SHA256SUMS
zstd -dc training/action_ledger.jsonl.zst | sha256sum
```

The decompressed digest is `e9ecbfe69a97ce8bfdff0f7f97e94ae7e5b7f68ee3393c789c92699c146caaec`. Restore a missing raw ledger with `zstd -d training/action_ledger.jsonl.zst -o training/action_ledger.jsonl`. The archive was decompressed and compared byte for byte against the original 306,178,466-byte ledger.

`LOCAL_SHA256SUMS` covers the original ledger, the 30 ResFiT model archives, and the retained evaluation videos and plots. Their original paths are under `training/` on the run host. With those files and the local ACT model present, run `sha256sum -c LOCAL_SHA256SUMS` and `python verify_evidence.py` to recheck the run. ResFiT model archives are evaluation-loadable and omit online replay buffers and live simulator state.
