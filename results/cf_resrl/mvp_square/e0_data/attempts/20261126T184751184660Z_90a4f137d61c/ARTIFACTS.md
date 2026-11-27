# Square E0 archive

The GitHub release asset contains the 720 raw simulator branch records and the 720 deployment-visible feature files. The Git tree retains the collection index, per-file SHA-256 values, configuration, log and reports. [Download the release asset](https://github.com/Adeline123456789/residual-rl-sim/releases/download/cf-resrl-square-e0-20261127/cf-square-factorial-v1-e0-20261127.tar)

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `cf-square-factorial-v1-e0-20261127.tar` | 298,803,200 | `0a5da328332c38a4c782bbd70c91edf50859e0ba0db16162c33fe54ad2736cd1` |

Restore from the repository root with GitHub CLI, GNU tar and a Python environment with NumPy.

```sh
run=results/cf_resrl/mvp_square/e0_data/attempts/20261126T184751184660Z_90a4f137d61c
asset=cf-square-factorial-v1-e0-20261127.tar
download_dir=$(mktemp -d)
gh release download cf-resrl-square-e0-20261127 \
  --repo Adeline123456789/residual-rl-sim --pattern "$asset" --dir "$download_dir"
printf '%s  %s\n' '0a5da328332c38a4c782bbd70c91edf50859e0ba0db16162c33fe54ad2736cd1' "$download_dir/$asset" | sha256sum -c -
(cd "$run" && sha256sum -c SHA256SUMS)
tar --keep-old-files -xf "$download_dir/$asset" -C "$run"
python "$run/verify_evidence.py" --archive "$download_dir/$asset"
```

The verifier checks the contents of both the archive and the restored files against `group_index.jsonl`. Run it directly if the attempt already has both directories.

The source demonstration dataset is `ankile/robomimic-mh-square-image` at revision `4e5cee7ea7d63b920ff76c2529b781c2474697da`, recorded as Apache-2.0 by the dataset audit. The pinned ResFiT upstream code is CC BY-NC 4.0. Model and optimizer files remain at their local paths.
