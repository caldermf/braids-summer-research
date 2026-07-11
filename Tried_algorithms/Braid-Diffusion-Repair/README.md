# Braid-Diffusion-Repair

This experiment treats kernel search as discrete denoising.

Instead of asking a transformer to generate a rare kernel braid from scratch, we train it to repair corrupted kernel braids:

```text
clean kernel K
corrupt K -> C
input:  C, p, noise level, projectivized Burau/Jones matrix, residual-to-scalar matrix
output: one same-length local repair move
```

The repair move is deliberately same-length:

```text
replace positions i..i+w-1 with a predicted Garside block of the same width
```

There is no deletion action. That keeps the model from learning the trivial identity/kernel shortcut.

## Noise Curriculum

The dataset supports six noise levels:

```text
1: replace one factor
2: replace two scattered factors
3: replace one local window of length 2-4
4: replace a prefix or suffix block
5: replace multiple local windows
6: replace many same-length windows, almost random but still legal GNF
```

The trainer advances gradually:

```text
stage 1 -> train on noise <= 1
stage 2 -> train on noise <= 2
...
stage 6 -> train on noise <= 6
```

Each stage resumes from the previous checkpoint.

## Architecture

The model has two streams:

```text
Garside stream:
  BOS, g_1, ..., g_L
  + p embedding
  + noise-level embedding

Matrix stream:
  degree-slice tokens from peyl projectivized matrix
  channel 0 = raw projectivized matrix
  channel 1 = residual-to-scalar matrix
```

The braid stream cross-attends to the matrix stream. The model predicts:

```text
repair position
repair width
replacement Garside factors for that window
```

During search, if the model chooses a plausible position/window but its top replacement is illegal in GNF, the search can sample a small number of legal bridge replacements at that same window. This is controlled by `BRIDGE_SAMPLES_PER_EDIT`. The model still chooses where to repair; the fallback just prevents illegal proposals from freezing the beam early.

## p=5 Benchmark Commands

Run from the cluster repo root:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research
```

### 1. Generate Corrupted-Kernel Repair Data

```bash
OUTDIR=$PWD/results/braid_diffusion_repair/p5_data_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
P=5 SEED=1 \
EXAMPLE_COUNT=150000 \
MAX_FACTORS=128 MAX_REPAIR_WINDOW=4 MATRIX_MAX_DEGREE=256 \
NOISE_LEVEL_WEIGHTS="1:2,2:2,3:2,4:1.5,5:1,6:0.5" \
KERNEL_SOURCES="$PWD/results/crispr_transformer/p5/guided_repair_seed1/kernel_hits.json;$PWD/results/MCTS_results/surprise_beam_results/surprise_beam_p5_n4_depth59_seed1/kernel_hits.json;$PWD/results/MCTS_results/trajectory_surprise_results/trajectory_surprise_p5_n4_seed1/kernel_hits.json;$PWD/results/crispr_results/crispr_p5_n4_seed1 (corrupted)/kernel_hits.json" \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  Braid-Diffusion-Repair/jobs/01_generate_repair_data_cpu.sh
```

### 2. Curriculum Train

```bash
OUTDIR=$PWD/results/braid_diffusion_repair/p5_curriculum_model_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
P=5 SEED=1 \
DATASET=$PWD/results/braid_diffusion_repair/p5_data_seed1/diffusion_repair_dataset.npz \
EPOCHS_PER_STAGE=5 BATCH_SIZE=128 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  Braid-Diffusion-Repair/jobs/02_train_curriculum_gpu.sh
```

Final checkpoint:

```text
results/braid_diffusion_repair/p5_curriculum_model_seed1/braid_diffusion_repair.pt
```

### 3. Repair Search on Corrupted p=5 Kernels

```bash
OUTDIR=$PWD/results/braid_diffusion_repair/p5_repair_search_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
P=5 SEED=1 \
CHECKPOINT=$PWD/results/braid_diffusion_repair/p5_curriculum_model_seed1/braid_diffusion_repair.pt \
KERNEL_SOURCES="$PWD/results/crispr_transformer/p5/guided_repair_seed1/kernel_hits.json;$PWD/results/MCTS_results/surprise_beam_results/surprise_beam_p5_n4_depth59_seed1/kernel_hits.json;$PWD/results/MCTS_results/trajectory_surprise_results/trajectory_surprise_p5_n4_seed1/kernel_hits.json;$PWD/results/crispr_results/crispr_p5_n4_seed1 (corrupted)/kernel_hits.json" \
START_MODE=corrupted-kernels CORRUPTED_ROOTS=1024 \
ROOT_MIN_NOISE_LEVEL=1 ROOT_MAX_NOISE_LEVEL=5 \
STEPS=40 BEAM_SIZE=1024 KEEP_BEST=300 \
POSITIONS_PER_STATE=10 WIDTHS_PER_POSITION=2 FACTOR_CHOICES_PER_SLOT=2 EDITS_PER_STATE=20 \
BRIDGE_SAMPLES_PER_EDIT=1 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  Braid-Diffusion-Repair/jobs/03_repair_search_gpu.sh
```

The first serious pass/fail test is whether this recovers p=5 kernels from held-out corruptions. If it cannot do that, it is not ready for p=7.

## Next Step Toward p=7

Once p=5 repair works, do not blindly run it zero-shot on p=7. Use it to propose p=7 edits, then keep only edits that improve the exact p=7 residual/projlen and fine-tune on those p=7 improvement examples.

## Multi-Prime Pretraining From Reservoir Kernels

The model has a `p` embedding, so we can pretrain on verified kernels from several primes. Generate each prime separately so exact verification uses the correct modulus, then merge the datasets.

### Generate p=2 Data

```bash
OUTDIR=$PWD/results/braid_diffusion_repair/p2_reservoir_data_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
P=2 SEED=1 EXAMPLE_COUNT=80000 \
MIN_KERNEL_LENGTH=4 MAX_KERNEL_LENGTH=128 \
MAX_FACTORS=128 MAX_REPAIR_WINDOW=4 MATRIX_MAX_DEGREE=256 \
NOISE_LEVEL_WEIGHTS="1:2,2:2,3:1.5,4:1,5:0.5,6:0.25" \
KERNEL_SOURCES="$PWD/results/resovoir_style_evaluation/check_p2_n4/kernel_hits.json" \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  Braid-Diffusion-Repair/jobs/01_generate_repair_data_cpu.sh
```

### Generate p=3 Data

```bash
OUTDIR=$PWD/results/braid_diffusion_repair/p3_reservoir_data_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
P=3 SEED=1 EXAMPLE_COUNT=100000 \
MIN_KERNEL_LENGTH=8 MAX_KERNEL_LENGTH=128 \
MAX_FACTORS=128 MAX_REPAIR_WINDOW=4 MATRIX_MAX_DEGREE=256 \
NOISE_LEVEL_WEIGHTS="1:2,2:2,3:2,4:1.5,5:1,6:0.5" \
KERNEL_SOURCES="$PWD/results/resovoir_style_evaluation/check_p3_n4/kernel_hit.json;$PWD/results/MCTS_results/surprise_beam_results/surprise_beam_p3_n4_seed1/kernel_hits.json;$PWD/results/crispr_results/crispr_p3_n4_seed1/kernel_hits.json;$PWD/results/crispr_results/crispr_p3_n4_seed2/kernel_hits.json;$PWD/results/crispr_results/crispr_p3_n4_seed3/kernel_hits.json" \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  Braid-Diffusion-Repair/jobs/01_generate_repair_data_cpu.sh
```

### Generate p=5 Data

```bash
OUTDIR=$PWD/results/braid_diffusion_repair/p5_reservoir_data_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
P=5 SEED=1 EXAMPLE_COUNT=150000 \
MIN_KERNEL_LENGTH=12 MAX_KERNEL_LENGTH=128 \
MAX_FACTORS=128 MAX_REPAIR_WINDOW=4 MATRIX_MAX_DEGREE=256 \
NOISE_LEVEL_WEIGHTS="1:2,2:2,3:2,4:1.5,5:1,6:0.5" \
KERNEL_SOURCES="$PWD/results/resovoir_style_evaluation/check_p5_n4/mcts_reservoir_20260528_014132_seed1/kernel_hits.json;$PWD/results/crispr_results/crispr_p5_n4_seed1 (corrupted)/kernel_hits.json;$PWD/results/crispr_transformer/p5/guided_repair_seed1/kernel_hits.json" \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  Braid-Diffusion-Repair/jobs/01_generate_repair_data_cpu.sh
```

### Merge p=2, p=3, p=5

```bash
OUTDIR=$PWD/results/braid_diffusion_repair/p235_merged_data_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
SEED=1 \
DATASETS="$PWD/results/braid_diffusion_repair/p2_reservoir_data_seed1/diffusion_repair_dataset.npz;$PWD/results/braid_diffusion_repair/p3_reservoir_data_seed1/diffusion_repair_dataset.npz;$PWD/results/braid_diffusion_repair/p5_reservoir_data_seed1/diffusion_repair_dataset.npz" \
MAX_EXAMPLES_PER_P_NOISE=40000 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  Braid-Diffusion-Repair/jobs/04_merge_repair_data_cpu.sh
```

Train on the merged dataset by setting:

```bash
DATASET=$PWD/results/braid_diffusion_repair/p235_merged_data_seed1/diffusion_repair_dataset.npz
```
