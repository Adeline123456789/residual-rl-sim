#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="cf-resrl-m0"
LEROBOT_REPOSITORY="https://github.com/huggingface/lerobot.git"
LEROBOT_COMMIT="69901b9b6a2300914ca3de0ea14b6fa6e0203bd4"
LEROBOT_DIR="$REPO_ROOT/third_party/resfit/deps/lerobot"

run_env() {
    conda run --no-capture-output -n "$ENV_NAME" \
        env PYTHONNOUSERSITE=1 "$@"
}

pip_install() {
    run_env PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
        python -m pip install --index-url https://pypi.org/simple "$@"
}

cd "$REPO_ROOT"

test "$(git -C third_party/resfit rev-parse HEAD)" = "66262b06a0caf159fb001504949b7b48ccda5f64"
test "$(git -C third_party/robosuite rev-parse HEAD)" = "77a4751233c29456a5381209e30dd0dbf39a6557"

if [[ ! -d "$LEROBOT_DIR/.git" ]]; then
    git clone --filter=blob:none --no-checkout "$LEROBOT_REPOSITORY" "$LEROBOT_DIR"
    git -C "$LEROBOT_DIR" checkout --detach "$LEROBOT_COMMIT"
fi
test "$(git -C "$LEROBOT_DIR" remote get-url origin)" = "$LEROBOT_REPOSITORY"
test "$(git -C "$LEROBOT_DIR" rev-parse HEAD)" = "$LEROBOT_COMMIT"
test -z "$(git -C "$LEROBOT_DIR" status --porcelain)"

pip_install \
    datasets==3.6.0 \
    deepdiff==8.5.0 \
    draccus==0.10.0 \
    einops==0.8.1 \
    'imageio[ffmpeg]==2.37.0' \
    jsonlines==4.0.0 \
    av==14.2.0 \
    packaging==24.2 \
    pyserial==3.5 \
    safetensors==0.5.3 \
    termcolor==2.5.0 \
    wandb==0.19.11

run_env PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
    python -m pip install --no-deps -e "$LEROBOT_DIR"
test -z "$(git -C "$LEROBOT_DIR" status --porcelain)"

mkdir -p "$REPO_ROOT/results/cf_resrl/m0/act_runtime"
run_env python -m pip freeze --all \
    > "$REPO_ROOT/results/cf_resrl/m0/act_runtime/environment.lock.txt"
conda list -n "$ENV_NAME" --explicit \
    > "$REPO_ROOT/results/cf_resrl/m0/act_runtime/conda-explicit.txt"
conda env export -n "$ENV_NAME" --no-builds \
    > "$REPO_ROOT/results/cf_resrl/m0/act_runtime/environment.resolved.yml"

run_env PYTHONPATH="$REPO_ROOT/third_party/resfit:$LEROBOT_DIR" python - <<'PY'
from importlib.metadata import version

import av
import torch
from resfit.lerobot.policies.act.modeling_act import ACTPolicy

assert torch.__version__ == "2.7.1+cu128"
assert version("gymnasium") == "1.1.1"
assert version("lerobot") == "0.1.0"
assert av.__version__ == "14.2.0"
print("ACT_RUNTIME_IMPORTS_OK", ACTPolicy.__name__, torch.__version__, "pyav", av.__version__)
print("RUNTIME_SCOPE minimal ACT Square load, rollout, and registered ACT training path")
print("INTENTIONAL_PIP_CHECK_DEVIATIONS are registered in configs/cf_resrl/m0_act_policy_smoke.json")
PY
