# Square minimal branch gate passed

On 2026-11-27 Asia/Shanghai, the fixed local ACT and ResFiT 300k policies executed eight source episodes and all 32 registered real Square branches on the RTX 3090. The gate checks whether an observed four-way interaction can be attributed to the two registered interventions and measured on paired, restored simulator states.

- Each KEEP/REFRESH pair submitted exactly the same physical action at the intervention tick. MuJoCo integration state and both camera hashes matched through the current action and both waiting steps.
- Refresh used the post-action observation at `t+1`. The new chunk first supplied action index 2 at `t+3`.
- All 5,175 branch control steps were consecutive and every executed action stayed within the seven-dimensional environment action bounds. The branches used 265 ACT calls after the source snapshot.
- An extra restored V11 rollout on the first source matched its original trace exactly. Its 293 control steps and 15 ACT calls are recorded separately, along with the source prefixes.

Successes by treatment were `V00` 6/8, `V01` 6/8, `V10` 6/8, `V11` 6/8. These eight early source snapshots validate execution and timing. The natural and enriched data collections use their own prespecified source cohorts.

`run_manifest.json` records the policy hashes, seed/tick schedule, dependency and source commits, and exact command. `summary.json` lists counters and SHA-256 values for every raw group trace. Both sets of checksums were verified.
