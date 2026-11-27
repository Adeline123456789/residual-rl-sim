# Square X1 closeout

Training seed `20261125` stopped at the user's request on 2026-11-27 at 08:55:04 Beijing time. The launcher records `FAILED` and child exit code `143` because it received the trainer's exit after `SIGTERM`. The three overlays were reversed and the ResFiT and LeRobot checkouts were clean. The recorded launch commit is `c5f43e8aa271224f7c804e219fd01d3b7c020d1d`.

The contiguous training action ledger contains 64,700 control steps out of the planned 300,000, including 10,000 warm-up and 54,700 main-training steps. The latest evaluation checkpoint is at step 60,000. The run stopped before the first complete resume generation at step 100,000. No second X1 training seed, X1 matrix rollout, X2 Can transfer, or X3 online comparison was run.

## Fixed evaluations

| Training step | Successes out of 50 | Physical evaluation control steps |
| ---: | ---: | ---: |
| 10,000 | 34 | 12,536 |
| 20,000 | 9 | 14,400 |
| 30,000 | 20 | 13,192 |
| 40,000 | 22 | 12,968 |
| 50,000 | 24 | 12,840 |
| 60,000 | 30 | 11,896 |

The table covers 300 completed episode records and 77,832 physical evaluation control steps. The physical counter includes all simulator lanes, including partial episodes at each evaluation's end. The paired scheduler comparison remains the completed [Square closed-loop MVP](../../../../mvp_square/closed_loop/attempts/20261126T195036106877Z_d549bf669ea3/COMPLETED_RUN.md).

## Evidence and recovery

The original launcher, configuration, evaluation, and log files are retained in this directory. The SHA-256 in `status.json` matches the first 609,953 bytes of the final log. A multiprocessing shutdown warning added the final 261 bytes. [`closeout.json`](closeout.json) and `SHA256SUMS` record the final log hash. All six local model hashes, six evaluation hashes, twelve media hashes, and both dependency locks were verified. Model weights and replay caches remain local.

The [Square X1 evidence release](https://github.com/Adeline123456789/residual-rl-sim/releases/tag/cf-resrl-square-x1-stop-20261127) contains the 64,700-row action ledger and six evaluation videos and plots. The compressed archive is 69,838,693 bytes. Its SHA-256 is `6612d3a90951e7f62b7ab0a6924c9c166b346002ea2ef8f39a2457b006f15981`. The receipt contains hashes for each archived file.

```bash
attempt=results/cf_resrl/expansion_square/resfit_training/attempts/20261126T205747128632Z_c5f43e8aa271
scratch=$(mktemp -d)
asset=square-x1-stopped-training-evidence.tar.gz
gh release download cf-resrl-square-x1-stop-20261127 --repo Adeline123456789/residual-rl-sim --pattern "$asset" --dir "$scratch"
(cd "$scratch" && echo "6612d3a90951e7f62b7ab0a6924c9c166b346002ea2ef8f39a2457b006f15981  $asset" | sha256sum -c -)
tar -xzf "$scratch/$asset" -C "$scratch"
(cd "$attempt" && sha256sum -c SHA256SUMS)
```
