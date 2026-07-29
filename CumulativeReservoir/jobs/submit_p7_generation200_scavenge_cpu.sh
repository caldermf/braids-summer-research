#!/usr/bin/env bash
# Submit one cumulative p=7 generation from the full length-8 DB frontier.
#
# This launches several bounded-population workers. Each worker reads the
# global cross-prime projlen DB, writes a local_run.sqlite, and can auto-merge
# back into the global DB at the end.
#
# Submit from the braids-summer-research directory:
#   bash CumulativeReservoir/jobs/submit_p7_generation200_scavenge_cpu.sh

set -euo pipefail

mkdir -p slurm_logs results/CumulativeReservoir

TARGET_LENGTH="${TARGET_LENGTH:-200}"
BUCKET_SIZE="${BUCKET_SIZE:-10000}"
USE_BEST="${USE_BEST:-75000}"
AUTO_MERGE="${AUTO_MERGE:-1}"
MEM="${MEM:-16G}"
TIME="${TIME:-1-00:00:00}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
BASE_SEED="${BASE_SEED:-92000}"

submit_worker() {
  local mode="$1"
  local group="$2"
  local seed_min="$3"
  local seed_max="$4"
  local seed_order="$5"
  local shards="$6"
  local base_seed="$7"
  local power="${8:-0}"

  echo "Submitting $group mode=$mode seed_projlen=[$seed_min,$seed_max] order=$seed_order shards=$shards"
  MODE="$mode" \
  P=7 N=4 R=1 POWER="$power" \
  SEED_LENGTH=8 \
  SEED_MIN_PROJLEN="$seed_min" \
  SEED_MAX_PROJLEN="$seed_max" \
  SEED_ORDER="$seed_order" \
  SEED_SHARD_COUNT="$shards" \
  TARGET_LENGTH="$TARGET_LENGTH" \
  BUCKET_SIZE="$BUCKET_SIZE" \
  USE_BEST="$USE_BEST" \
  AUTO_MERGE="$AUTO_MERGE" \
  BASE_SEED="$base_seed" \
  RUN_GROUP="$group" \
  sbatch --mem="$MEM" --time="$TIME" --array="1-${shards}%${MAX_CONCURRENT}" \
    CumulativeReservoir/jobs/run_array_scavenge_cpu.sh
}

submit_worker paper \
  p7_gen200_paper_elite_proj14_16 \
  14 16 projlen 16 "$BASE_SEED" 0

submit_worker paper \
  p7_gen200_paper_mid_proj17_19 \
  17 19 projlen 32 "$((BASE_SEED + 1000))" 0

submit_worker paper \
  p7_gen200_paper_broad_random_proj14_25 \
  14 25 random 32 "$((BASE_SEED + 2000))" 0

submit_worker power_v2 \
  p7_gen200_power_elite_proj14_16 \
  14 16 projlen 16 "$((BASE_SEED + 3000))" 7

submit_worker power_v2 \
  p7_gen200_power_mid_proj17_19 \
  17 19 projlen 32 "$((BASE_SEED + 4000))" 7

submit_worker power_v2 \
  p7_gen200_power_broad_random_proj14_25 \
  14 25 random 32 "$((BASE_SEED + 5000))" 7

echo "Submitted cumulative p=7 generation workers."
