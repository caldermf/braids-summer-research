#!/usr/bin/env bash
# Submit a conservative p=7 BraidLake generation suite.
# This submits several CumulativeReservoir families that export to BraidLake.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
cd "$REPO_ROOT"
mkdir -p slurm_logs results/BraidLake

TARGET_LENGTH="${TARGET_LENGTH:-200}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
MEM_PAPER="${MEM_PAPER:-8G}"
MEM_POWER="${MEM_POWER:-16G}"
BUCKET_SIZE_PAPER="${BUCKET_SIZE_PAPER:-10000}"
USE_BEST_PAPER="${USE_BEST_PAPER:-75000}"
BUCKET_SIZE_POWER="${BUCKET_SIZE_POWER:-3000}"
USE_BEST_POWER="${USE_BEST_POWER:-15000}"

echo "Submitting BraidLake p=7 generation suite"
echo "TARGET_LENGTH=$TARGET_LENGTH MAX_CONCURRENT=$MAX_CONCURRENT"

MODE=paper P=7 N=4 R=1 \
BASE_SEED=92000 \
SEED_LENGTH=8 SEED_MIN_PROJLEN=14 SEED_MAX_PROJLEN=16 SEED_ORDER=random \
TARGET_LENGTH="$TARGET_LENGTH" BUCKET_SIZE="$BUCKET_SIZE_PAPER" USE_BEST="$USE_BEST_PAPER" \
RUN_GROUP=p7_lake_paper_low_random_proj14_16_len${TARGET_LENGTH} \
  sbatch --array=1-16%"$MAX_CONCURRENT" --mem="$MEM_PAPER" \
  BraidLake/jobs/run_cumulative_to_lake_array_scavenge_cpu.sh

MODE=paper P=7 N=4 R=1 \
BASE_SEED=93000 \
SEED_LENGTH=8 SEED_MIN_PROJLEN=17 SEED_MAX_PROJLEN=19 SEED_ORDER=random \
TARGET_LENGTH="$TARGET_LENGTH" BUCKET_SIZE="$BUCKET_SIZE_PAPER" USE_BEST="$USE_BEST_PAPER" \
RUN_GROUP=p7_lake_paper_mid_random_proj17_19_len${TARGET_LENGTH} \
  sbatch --array=1-32%"$MAX_CONCURRENT" --mem="$MEM_PAPER" \
  BraidLake/jobs/run_cumulative_to_lake_array_scavenge_cpu.sh

MODE=paper P=7 N=4 R=1 \
BASE_SEED=94000 \
SEED_LENGTH=8 SEED_MIN_PROJLEN=14 SEED_MAX_PROJLEN=25 SEED_ORDER=random \
TARGET_LENGTH="$TARGET_LENGTH" BUCKET_SIZE="$BUCKET_SIZE_PAPER" USE_BEST="$USE_BEST_PAPER" \
RUN_GROUP=p7_lake_paper_broad_random_proj14_25_len${TARGET_LENGTH} \
  sbatch --array=1-32%"$MAX_CONCURRENT" --mem="$MEM_PAPER" \
  BraidLake/jobs/run_cumulative_to_lake_array_scavenge_cpu.sh

MODE=power_v2 P=7 N=4 R=1 POWER=7 \
BASE_SEED=95000 \
SEED_LENGTH=8 SEED_MIN_PROJLEN=14 SEED_MAX_PROJLEN=25 SEED_ORDER=random \
TARGET_LENGTH=80 BUCKET_SIZE="$BUCKET_SIZE_POWER" USE_BEST="$USE_BEST_POWER" \
RUN_GROUP=p7_lake_power_v2_broad_random_proj14_25_len80 \
  sbatch --array=1-8%1 --mem="$MEM_POWER" \
  BraidLake/jobs/run_cumulative_to_lake_array_scavenge_cpu.sh
