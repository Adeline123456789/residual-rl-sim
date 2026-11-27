# Square closed-loop rollout archive

The [Square MVP evidence release](https://github.com/Adeline123456789/residual-rl-sim/releases/tag/cf-resrl-square-mvp-20261127) contains `square-mvp-closed-loop-traces.tar`. The 264,058,880-byte archive holds the 2,020 gzip JSON traces indexed in `episode_index.jsonl`. SHA-256 `deeb0ec7c68179af0413a3473a537186a8f83cb5e8cedf2f7c89318bdbeed33e`.

From the checked-out repository, restore to the matching attempt directory and run the verifier.

```bash
attempt=results/cf_resrl/mvp_square/closed_loop/attempts/20261126T195036106877Z_d549bf669ea3
scratch=$(mktemp -d)
asset=square-mvp-closed-loop-traces.tar
gh release download cf-resrl-square-mvp-20261127 --repo Adeline123456789/residual-rl-sim --pattern "$asset" --dir "$scratch"
(cd "$scratch" && echo "deeb0ec7c68179af0413a3473a537186a8f83cb5e8cedf2f7c89318bdbeed33e  $asset" | sha256sum -c -)
tar --extract --file "$scratch/$asset" --directory "$attempt"
(cd "$attempt" && sha256sum -c SHA256SUMS)
python scripts/verify_square_closed_loop.py --attempt "$attempt" --out "$scratch/reverification.json"
```

The verifier checks per-trace hashes from the committed index and recomputes paired cohort accounting and timing rules.
