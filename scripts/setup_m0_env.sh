#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="cf-resrl-m0"

cd "$REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required" >&2
    exit 1
fi

if conda env list | awk -v name="$ENV_NAME" '$1 == name { found=1 } END { exit !found }'; then
    echo "conda environment already exists: $ENV_NAME" >&2
    exit 1
fi

test "$(git -C third_party/resfit rev-parse HEAD)" = "66262b06a0caf159fb001504949b7b48ccda5f64"
test "$(git -C third_party/robosuite rev-parse HEAD)" = "77a4751233c29456a5381209e30dd0dbf39a6557"

conda create -n "$ENV_NAME" --solver libmamba --override-channels \
    -c https://conda.anaconda.org/conda-forge \
    python=3.10 pip "ffmpeg>=7,<8" -y

conda run -n "$ENV_NAME" python -m pip install \
    torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

conda run -n "$ENV_NAME" python -m pip install \
    gymnasium==1.1.1 \
    mujoco==3.3.2 \
    mink==0.0.7 \
    huggingface-hub==0.36.0 \
    pyarrow==19.0.1

conda run -n "$ENV_NAME" python -m pip install -e "$REPO_ROOT/third_party/robosuite"

mkdir -p "$REPO_ROOT/results/cf_resrl/m0"
conda run -n "$ENV_NAME" python -m pip freeze --all \
    > "$REPO_ROOT/results/cf_resrl/m0/environment.lock.txt"
conda list -n "$ENV_NAME" --explicit \
    > "$REPO_ROOT/results/cf_resrl/m0/conda-explicit.txt"
conda env export -n "$ENV_NAME" --no-builds \
    > "$REPO_ROOT/results/cf_resrl/m0/environment.resolved.yml"

conda run -n "$ENV_NAME" env PYTHONPATH="$REPO_ROOT/third_party/resfit" \
    python -c 'import gymnasium, huggingface_hub, mujoco, pyarrow, robosuite, torch; from resfit.dexmg.environments.dexmg import RobosuiteGymWrapper; print("IMPORTS_OK", torch.__version__, mujoco.__version__, gymnasium.__version__)'

conda run -n "$ENV_NAME" python -c \
    'import torch; torch.manual_seed(0); x=torch.randn(8,8,device="cuda"); y=x@x; print("WITNESS", tuple(y.shape), torch.cuda.get_device_name(), float(y.sum()))'
