#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
resfit_root="$repo_root/third_party/resfit"
lerobot_root="$resfit_root/deps/lerobot"
mode="${1:-apply}"

case "$mode" in
    apply|reverse|check) ;;
    *)
        echo "Usage: $0 [apply|reverse|check]" >&2
        exit 2
        ;;
esac

test "$(git -C "$resfit_root" rev-parse HEAD)" = "66262b06a0caf159fb001504949b7b48ccda5f64"
test "$(git -C "$lerobot_root" rev-parse HEAD)" = "69901b9b6a2300914ca3de0ea14b6fa6e0203bd4"

target_roots=(
    "$resfit_root"
    "$resfit_root"
    "$lerobot_root"
)
patch_files=(
    "$repo_root/patches/resfit/train_bc_dexmg_cf_resrl.patch"
    "$repo_root/patches/resfit/residual_td3_cf_resrl.patch"
    "$repo_root/patches/lerobot/noncontiguous_episode_subset.patch"
)

patch_state() {
    local target_root="$1"
    local patch_file="$2"
    if git -C "$target_root" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
        echo applied
    elif git -C "$target_root" apply --check "$patch_file" >/dev/null 2>&1; then
        echo not_applied
    else
        echo conflicted
    fi
}

apply_one() {
    local target_root="$1"
    local patch_file="$2"
    local patch_name state
    patch_name="$(basename "$patch_file")"
    state="$(patch_state "$target_root" "$patch_file")"
    case "$state" in
        applied)
            echo "$patch_name is already applied"
            ;;
        not_applied)
            git -C "$target_root" apply "$patch_file"
            echo "Applied $patch_name"
            ;;
        *)
            echo "$patch_name cannot be applied or reversed cleanly" >&2
            return 1
            ;;
    esac
}

reverse_one() {
    local target_root="$1"
    local patch_file="$2"
    local patch_name state
    patch_name="$(basename "$patch_file")"
    state="$(patch_state "$target_root" "$patch_file")"
    case "$state" in
        applied)
            git -C "$target_root" apply --reverse "$patch_file"
            echo "Reversed $patch_name"
            ;;
        not_applied)
            echo "$patch_name is already unapplied"
            ;;
        *)
            echo "$patch_name cannot be applied or reversed cleanly" >&2
            return 1
            ;;
    esac
}

check_one() {
    local target_root="$1"
    local patch_file="$2"
    local patch_name state
    patch_name="$(basename "$patch_file")"
    state="$(patch_state "$target_root" "$patch_file")"
    if [[ "$state" = conflicted ]]; then
        echo "$patch_name: $state" >&2
        return 1
    fi
    echo "$patch_name: $state"
}

if [[ "$mode" = reverse ]]; then
    for ((index=${#patch_files[@]} - 1; index >= 0; index--)); do
        reverse_one "${target_roots[$index]}" "${patch_files[$index]}"
    done
else
    for ((index=0; index<${#patch_files[@]}; index++)); do
        "${mode}_one" "${target_roots[$index]}" "${patch_files[$index]}"
    done
fi
