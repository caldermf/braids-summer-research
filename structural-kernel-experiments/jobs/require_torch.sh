#!/usr/bin/env bash

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python is not executable: $PYTHON_PATH" >&2
  exit 2
fi

echo "Python executable: $PYTHON_PATH"
if ! "$PYTHON_PATH" -c \
  'import sys, torch; print("sys.executable:", sys.executable); print("PyTorch:", torch.__version__); print("Bundled CUDA:", torch.version.cuda)' \
  ; then
  echo "PyTorch is not installed in $PYTHON_PATH" >&2
  echo "Run structural-kernel-experiments/jobs/install_cuda13_environment.sh on a login node." >&2
  exit 2
fi
