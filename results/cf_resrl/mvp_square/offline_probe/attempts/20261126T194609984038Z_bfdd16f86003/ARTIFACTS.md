# Square offline probe checkpoints

The asset contains the 60 fitted scheduler probes and their SHA-256 list. These are small offline decision models trained on the published E0 features. [Download the probe checkpoints](https://github.com/Adeline123456789/residual-rl-sim/releases/download/cf-resrl-square-e0-20261127/cf-square-e0-probe-models-20261127.tar)

| File | Bytes | Verified SHA-256 |
| --- | ---: | --- |
| `cf-square-e0-probe-models-20261127.tar` | 7,690,240 | `af6196a6922695e0f2cf34e2efccd0c312a264a6eda88587df7c603f634870f3` |

From the repository root, download and verify the asset. Extract the model files into the registered attempt directory.

```sh
run=results/cf_resrl/mvp_square/offline_probe/attempts/20261126T194609984038Z_bfdd16f86003
asset=cf-square-e0-probe-models-20261127.tar
download_dir=$(mktemp -d)
gh release download cf-resrl-square-e0-20261127 \
  --repo Adeline123456789/residual-rl-sim --pattern "$asset" --dir "$download_dir"
printf '%s  %s\n' 'af6196a6922695e0f2cf34e2efccd0c312a264a6eda88587df7c603f634870f3' "$download_dir/$asset" | sha256sum -c -
tar --keep-old-files -xf "$download_dir/$asset" -C "$run" models
(cd "$run" && sha256sum -c LOCAL_MODELS_SHA256SUMS)
```
