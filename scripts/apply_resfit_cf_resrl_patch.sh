#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
resfit_root="$repo_root/third_party/resfit"
patch_file="$repo_root/patches/resfit/train_bc_dexmg_cf_resrl.patch"

test "$(git -C "$resfit_root" rev-parse HEAD)" = "66262b06a0caf159fb001504949b7b48ccda5f64"

if git -C "$resfit_root" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "CF-ResRL ResFiT patch is already applied"
else
    git -C "$resfit_root" apply --check "$patch_file"
    git -C "$resfit_root" apply "$patch_file"
    echo "Applied CF-ResRL ResFiT patch"
fi
