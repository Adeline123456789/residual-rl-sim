# ACT 50k binary artifacts

ACT checkpoints and optimizer state are retained on the training host.
The previously uploaded final model asset was withdrawn. Git contains the run
configuration, metrics, logs, and checksums.

On the training host, verify the final model used by ResFiT:

```sh
sha256sum /media/midea/work/CF-ResRL/residual-rl-sim/results/cf_resrl/m0/act_training/attempts/20261125T005956570423Z_3dfd8208847a/training/policy_step_49999/policy/model.safetensors
# e5f5b3c7fa7512a2ae202d5912922289a52348f8a3bb5a9da33f606d671cb859
```

`release_assets.sha256` records the checksums of the local export bundles.
`evidence_manifest.json` lists the files retained in the original attempt.
