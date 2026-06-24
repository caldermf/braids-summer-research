#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/Braid-GPT}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/braid_gpt}"

CPU_PYTHON_PATH="${CPU_PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
GPU_PYTHON_PATH="${GPU_PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"

P="${P:-7}"
SEED="${SEED:-1}"

SEQUENCE_COUNT="${SEQUENCE_COUNT:-1000000}"
PRETRAIN_MIN_LENGTH="${PRETRAIN_MIN_LENGTH:-8}"
PRETRAIN_MAX_LENGTH="${PRETRAIN_MAX_LENGTH:-96}"
MAX_FACTORS="${MAX_FACTORS:-96}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-256}"
D_MODEL="${D_MODEL:-256}"
NUM_LAYERS="${NUM_LAYERS:-8}"
NHEAD="${NHEAD:-8}"

STATE_COUNT="${STATE_COUNT:-100000}"
POLICY_MIN_LENGTH="${POLICY_MIN_LENGTH:-12}"
POLICY_MAX_LENGTH="${POLICY_MAX_LENGTH:-72}"
LOOKAHEAD="${LOOKAHEAD:-2}"
ROLLOUTS_PER_ACTION="${ROLLOUTS_PER_ACTION:-4}"

FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-128}"

STEPS="${STEPS:-96}"
BEAM_SIZE="${BEAM_SIZE:-512}"
ACTIONS_PER_STATE="${ACTIONS_PER_STATE:-4}"
RANDOM_ROOTS="${RANDOM_ROOTS:-128}"
MAX_LENGTH="${MAX_LENGTH:-96}"

if [[ ! -f "$PROJECT_ROOT/braid_gpt.py" ]]; then
  echo "Cannot find Braid GPT script at: $PROJECT_ROOT/braid_gpt.py" >&2
  echo "Run this from the repo root or set REPO_ROOT=/path/to/braids-summer-research." >&2
  exit 2
fi

normalize_job_id() {
  local raw="$1"
  echo "${raw%%;*}"
}

submit_job() {
  local raw
  raw="$("$@")"
  normalize_job_id "$raw"
}

cd "$REPO_ROOT"
mkdir -p slurm_logs "$RESULTS_ROOT"

PRETRAIN_DATA_OUTDIR="${PRETRAIN_DATA_OUTDIR:-$RESULTS_ROOT/p${P}_pretrain_data_seed${SEED}}"
PRETRAIN_OUTDIR="${PRETRAIN_OUTDIR:-$RESULTS_ROOT/p${P}_pretrained_seed${SEED}}"
POLICY_DATA_OUTDIR="${POLICY_DATA_OUTDIR:-$RESULTS_ROOT/p${P}_policy_data_seed${SEED}}"
FINETUNE_OUTDIR="${FINETUNE_OUTDIR:-$RESULTS_ROOT/p${P}_finetuned_seed${SEED}}"
GENERATE_OUTDIR="${GENERATE_OUTDIR:-$RESULTS_ROOT/p${P}_generate_seed${SEED}}"

mkdir -p \
  "$PRETRAIN_DATA_OUTDIR" \
  "$PRETRAIN_OUTDIR" \
  "$POLICY_DATA_OUTDIR" \
  "$FINETUNE_OUTDIR" \
  "$GENERATE_OUTDIR"

echo "Submitting Braid-GPT pipeline from $REPO_ROOT"
echo "P=$P SEED=$SEED"

pretrain_data_job="$(
  PYTHON_PATH="$CPU_PYTHON_PATH" \
  P="$P" SEED="$SEED" \
  SEQUENCE_COUNT="$SEQUENCE_COUNT" \
  MIN_LENGTH="$PRETRAIN_MIN_LENGTH" \
  MAX_LENGTH="$PRETRAIN_MAX_LENGTH" \
  MAX_FACTORS="$MAX_FACTORS" \
  OUTPUT_DIR="$PRETRAIN_DATA_OUTDIR" \
  sbatch --parsable \
    --output="$PRETRAIN_DATA_OUTDIR/output.out" \
    --error="$PRETRAIN_DATA_OUTDIR/output.err" \
    "$PROJECT_ROOT/jobs/01_generate_pretrain_data_cpu.sh"
)"
pretrain_data_job="$(normalize_job_id "$pretrain_data_job")"
echo "pretrain-data job: $pretrain_data_job"

policy_data_job="$(
  PYTHON_PATH="$CPU_PYTHON_PATH" \
  P="$P" SEED="$SEED" \
  STATE_COUNT="$STATE_COUNT" \
  MIN_LENGTH="$POLICY_MIN_LENGTH" \
  MAX_LENGTH="$POLICY_MAX_LENGTH" \
  MAX_FACTORS="$MAX_FACTORS" \
  LOOKAHEAD="$LOOKAHEAD" \
  ROLLOUTS_PER_ACTION="$ROLLOUTS_PER_ACTION" \
  OUTPUT_DIR="$POLICY_DATA_OUTDIR" \
  sbatch --parsable \
    --output="$POLICY_DATA_OUTDIR/output.out" \
    --error="$POLICY_DATA_OUTDIR/output.err" \
    "$PROJECT_ROOT/jobs/03_generate_policy_data_cpu.sh"
)"
policy_data_job="$(normalize_job_id "$policy_data_job")"
echo "policy-data job: $policy_data_job"

pretrain_job="$(
  PYTHON_PATH="$GPU_PYTHON_PATH" \
  P="$P" SEED="$SEED" \
  DATASET="$PRETRAIN_DATA_OUTDIR/pretrain_dataset.npz" \
  EPOCHS="$PRETRAIN_EPOCHS" \
  BATCH_SIZE="$PRETRAIN_BATCH_SIZE" \
  D_MODEL="$D_MODEL" \
  NUM_LAYERS="$NUM_LAYERS" \
  NHEAD="$NHEAD" \
  OUTPUT_DIR="$PRETRAIN_OUTDIR" \
  sbatch --parsable \
    --dependency="afterok:$pretrain_data_job" \
    --output="$PRETRAIN_OUTDIR/output.out" \
    --error="$PRETRAIN_OUTDIR/output.err" \
    "$PROJECT_ROOT/jobs/02_pretrain_gpt_gpu.sh"
)"
pretrain_job="$(normalize_job_id "$pretrain_job")"
echo "pretrain job: $pretrain_job afterok:$pretrain_data_job"

finetune_job="$(
  PYTHON_PATH="$GPU_PYTHON_PATH" \
  P="$P" SEED="$SEED" \
  DATASET="$POLICY_DATA_OUTDIR/policy_dataset.npz" \
  INIT_CHECKPOINT="$PRETRAIN_OUTDIR/braid_gpt_pretrained.pt" \
  EPOCHS="$FINETUNE_EPOCHS" \
  BATCH_SIZE="$FINETUNE_BATCH_SIZE" \
  OUTPUT_DIR="$FINETUNE_OUTDIR" \
  sbatch --parsable \
    --dependency="afterok:$pretrain_job:$policy_data_job" \
    --output="$FINETUNE_OUTDIR/output.out" \
    --error="$FINETUNE_OUTDIR/output.err" \
    "$PROJECT_ROOT/jobs/04_finetune_gpt_gpu.sh"
)"
finetune_job="$(normalize_job_id "$finetune_job")"
echo "finetune job: $finetune_job afterok:$pretrain_job:$policy_data_job"

generate_job="$(
  PYTHON_PATH="$GPU_PYTHON_PATH" \
  P="$P" SEED="$SEED" \
  CHECKPOINT="$FINETUNE_OUTDIR/braid_gpt_finetuned.pt" \
  STEPS="$STEPS" \
  BEAM_SIZE="$BEAM_SIZE" \
  ACTIONS_PER_STATE="$ACTIONS_PER_STATE" \
  RANDOM_ROOTS="$RANDOM_ROOTS" \
  MAX_LENGTH="$MAX_LENGTH" \
  OUTPUT_DIR="$GENERATE_OUTDIR" \
  sbatch --parsable \
    --dependency="afterok:$finetune_job" \
    --output="$GENERATE_OUTDIR/output.out" \
    --error="$GENERATE_OUTDIR/output.err" \
    "$PROJECT_ROOT/jobs/05_generate_search_gpu.sh"
)"
generate_job="$(normalize_job_id "$generate_job")"
echo "generate job: $generate_job afterok:$finetune_job"

cat <<EOF

Submitted full Braid-GPT pipeline.

Monitor:
  squeue -u as4843
  sacct -j $pretrain_data_job,$policy_data_job,$pretrain_job,$finetune_job,$generate_job --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS

Outputs:
  $PRETRAIN_DATA_OUTDIR
  $POLICY_DATA_OUTDIR
  $PRETRAIN_OUTDIR
  $FINETUNE_OUTDIR
  $GENERATE_OUTDIR
EOF
