#!/usr/bin/env bash
# Validate that the selected Python/Torch environment can actually run CUDA kernels
# on Bouchet scavenge_gpu nodes.

#SBATCH --job-name=lake-gpucheck
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"

cd "$REPO_ROOT"
mkdir -p slurm_logs

"$PYTHON_PATH" - <<'PY'
import json
import torch

payload = {
    "torch": torch.__version__,
    "cuda_version": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
}
if not torch.cuda.is_available():
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit("CUDA is not available")

payload["device_name"] = torch.cuda.get_device_name(0)
payload["device_capability"] = torch.cuda.get_device_capability(0)
x = torch.ones((1024, 1024), device="cuda", dtype=torch.float32)
y = x @ x
torch.cuda.synchronize()
payload["matmul_entry"] = float(y[0, 0].item())
print(json.dumps(payload, indent=2, sort_keys=True))
PY
