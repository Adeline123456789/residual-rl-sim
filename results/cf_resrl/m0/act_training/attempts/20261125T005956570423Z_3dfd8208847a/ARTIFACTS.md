# ACT 50k binary artifacts

The fixed final model is available from the [ACT experiment release](https://github.com/Adeline123456789/residual-rl-sim/releases/tag/cf-resrl-square-act-50k-20261125).
GitHub reported the same final model SHA256 as the run's checkpoint manifest.
The release contains generated research artifacts. ResFiT is CC BY-NC 4.0 and
the Square demonstration dataset is Apache-2.0. See the repository's
`THIRD_PARTY.md` and the dataset and training evidence for provenance.

From the repository root, recover the final model used by ResFiT:

```sh
attempt=results/cf_resrl/m0/act_training/attempts/20261125T005956570423Z_3dfd8208847a
final_policy="$attempt/training/policy_step_49999/policy"
gh release download cf-resrl-square-act-50k-20261125 \
    --repo Adeline123456789/residual-rl-sim \
    --pattern model.safetensors --dir "$final_policy"
sha256sum "$final_policy/model.safetensors"
# e5f5b3c7fa7512a2ae202d5912922289a52348f8a3bb5a9da33f606d671cb859
```

`release_assets.sha256` lists the checksums of the final model and the
seven archive files prepared locally. The archives upload separately from the
final model. Once each archive is available in the release, verify it with
`sha256sum` against that list and extract it with `tar -xf ARCHIVE -C "$attempt/training"`.
The per-file checksums in `evidence_manifest.json` cover the extracted payloads.
