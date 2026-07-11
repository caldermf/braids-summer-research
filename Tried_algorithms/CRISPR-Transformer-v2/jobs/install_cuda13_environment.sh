#!/usr/bin/env bash
# Create a separate PyTorch environment with Blackwell sm_120 support.

set -euo pipefail

BASE_PYTHON="${BASE_PYTHON:-/home/as4843/braids-torch/bin/python}"
ENV_DIR="${ENV_DIR:-/home/as4843/braids-torch-cu130}"

if [[ ! -x "$BASE_PYTHON" ]]; then
  echo "Base Python is not executable: $BASE_PYTHON" >&2
  exit 2
fi
if [[ -e "$ENV_DIR" ]]; then
  echo "Refusing to overwrite existing path: $ENV_DIR" >&2
  echo "Choose another ENV_DIR or remove it yourself after inspecting it." >&2
  exit 2
fi

"$BASE_PYTHON" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV_DIR/bin/python" -m pip install \
  torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130

"$ENV_DIR/bin/python" - <<'PY'
import torch

print("Python environment installed successfully")
print("PyTorch:", torch.__version__)
print("Bundled CUDA:", torch.version.cuda)
print("Compiled architectures:", torch._C._cuda_getArchFlags())
PY

echo "Use this for future submissions:"
echo "PYTHON_PATH=$ENV_DIR/bin/python"
