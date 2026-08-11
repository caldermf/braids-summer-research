# ModSynthesis

ModSynthesis searches for modular congruence collapses by recombining already
interesting braids instead of growing one word from scratch.

The main target is the two-rowed `(3,1)` representation of `B_4` over
`F_7`.  The controls are `p=3` and `p=5`.

## Algorithm

1. Load parent braids from one or more sources:
   - `candidates.jsonl` files from earlier searches;
   - `results/BraidExperienceDB/cross_prime_projlen.sqlite`;
   - `results/BraidLake`;
   - optional exhaustive/random GNF bootstrap.

2. Exact-evaluate every parent with the same `peyl`/BraidZero verifier.

3. Synthesize new braids by algebraic closure operations:
   - powers: `x^p`, `x^(2p)`, `x^(3p)`;
   - image-collision quotients: if `rho(a)` and `rho(b)` have the same
     projective digest, test `a b^{-1}` and `b a^{-1}`;
   - quotients: `a b^{-1}`, `b a^{-1}` for promising pairs;
   - commutators: `[a,b] = a b a^{-1} b^{-1}`;
   - short commutators: `[a,g]` for short simple factors `g`;
   - residual-sign probes: if the non-scalar residual signature of `a` is the
     negative of that of `b`, test products and quotients.

4. Every synthesized braid is exact-verified for projective identity and
   projective Delta matches.  Candidate rows are written to `candidates.jsonl`.

This is not another blind reservoir.  It asks whether the already-seen p=7
library generates a modular scalar relation under powers, quotients,
commutators, and residual cancellation.

## Quick Control: p=5 From Known PowerReservoir Candidates

Run on Bouchet from the repository root:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research
mkdir -p slurm_logs results/ModSynthesis

P=5 \
PARENT_JSONL='results/PowerReservoirV2/B4_r1_p5_power_reservoir_v2_boot7_len60/*/candidates.jsonl' \
MAX_PARENTS=500 \
OPERATIONS=powers \
POWER_EXPONENTS=5 \
RUN_GROUP=p5_power_control_from_candidates \
sbatch --output=slurm_logs/mod-synth-p5-%A_%a.out --error=slurm_logs/mod-synth-p5-%A_%a.err \
  --array=1-1 ModSynthesis/jobs/run_mod_synthesis_scavenge_cpu.sh
```

Expected signal: `candidates.jsonl` should become nonempty quickly.  This
checks that the exact p-power synthesis path is working.

If `results/PowerReservoirV2` was deleted from Bouchet during storage cleanup,
make a tiny known-parent file first:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research
mkdir -p results/ModSynthesis/known_controls

cat > results/ModSynthesis/known_controls/p5_precursor.jsonl <<'JSONL'
{"factor_ids":[13,8,13,21,11,13,11,13,10,2,13,8],"identity_defect":68,"projlen":20,"source":"known_p5_power_precursor"}
JSONL

P=5 \
PARENT_JSONL='results/ModSynthesis/known_controls/p5_precursor.jsonl' \
MAX_PARENTS=20 \
OPERATIONS=powers \
POWER_EXPONENTS=5 \
RUN_GROUP=p5_power_control_known_precursor \
sbatch --output=slurm_logs/mod-synth-p5-known-%A_%a.out --error=slurm_logs/mod-synth-p5-known-%A_%a.err \
  --array=1-1 ModSynthesis/jobs/run_mod_synthesis_scavenge_cpu.sh
```

## Quick Control: p=3 From Known PowerReservoir Candidates

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research
mkdir -p slurm_logs results/ModSynthesis

P=3 \
PARENT_JSONL='results/PowerReservoirV2/B4_r1_p3_power_reservoir_v2_boot6_len40/*/candidates.jsonl' \
MAX_PARENTS=1000 \
OPERATIONS=powers \
POWER_EXPONENTS=3 \
RUN_GROUP=p3_power_control_from_candidates \
sbatch --output=slurm_logs/mod-synth-p3-%A_%a.out --error=slurm_logs/mod-synth-p3-%A_%a.err \
  --array=1-1 ModSynthesis/jobs/run_mod_synthesis_scavenge_cpu.sh
```

Expected signal: direct `x^3` candidates should be recovered.

If the old p=3 result folder is missing:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research
mkdir -p results/ModSynthesis/known_controls

cat > results/ModSynthesis/known_controls/p3_precursor.jsonl <<'JSONL'
{"factor_ids":[11],"identity_defect":6,"projlen":4,"source":"known_p3_power_precursor"}
JSONL

P=3 \
PARENT_JSONL='results/ModSynthesis/known_controls/p3_precursor.jsonl' \
MAX_PARENTS=20 \
OPERATIONS=powers \
POWER_EXPONENTS=3 \
RUN_GROUP=p3_power_control_known_precursor \
sbatch --output=slurm_logs/mod-synth-p3-known-%A_%a.out --error=slurm_logs/mod-synth-p3-known-%A_%a.err \
  --array=1-1 ModSynthesis/jobs/run_mod_synthesis_scavenge_cpu.sh
```

## p=7 Main Run From Global Database

This samples low-projlen and medium-length p=7 parents from the global DB if it
exists, otherwise BraidLake if it exists.  It then applies powers,
collisions, quotients, commutators, short commutators, and residual probes.

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research
mkdir -p slurm_logs results/ModSynthesis

P=7 \
MIN_LENGTH=8 MAX_LENGTH=80 \
MIN_PROJLEN=14 MAX_PROJLEN=35 \
MAX_PARENTS=8000 \
PARENT_ORDER=projlen \
OPERATIONS=powers,collisions,quotients,commutators,residual,short_commutators \
POWER_EXPONENTS=7,14,21 \
UNARY_POOL_SIZE=8000 \
PAIR_POOL_SIZE=400 \
MAX_PAIRS=50000 \
MAX_SYNTHESIZED_PER_PHASE=80000 \
POWER_COMMUTATORS=1 \
RUN_GROUP=p7_mod_synthesis_low_mid_projlen_len8_80 \
sbatch --output=slurm_logs/mod-synth-p7-%A_%a.out --error=slurm_logs/mod-synth-p7-%A_%a.err \
  --array=1-8 ModSynthesis/jobs/run_mod_synthesis_scavenge_cpu.sh
```

## p=7 Random-Diverse Run

This deliberately avoids only using the best low-projlen parents.

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research
mkdir -p slurm_logs results/ModSynthesis

P=7 \
MIN_LENGTH=8 MAX_LENGTH=120 \
MIN_PROJLEN=14 MAX_PROJLEN=80 \
MAX_PARENTS=8000 \
PARENT_ORDER=random \
OPERATIONS=powers,collisions,quotients,commutators,residual,short_commutators \
POWER_EXPONENTS=7,14,21 \
UNARY_POOL_SIZE=8000 \
PAIR_POOL_SIZE=400 \
MAX_PAIRS=50000 \
MAX_SYNTHESIZED_PER_PHASE=80000 \
POWER_COMMUTATORS=1 \
RUN_GROUP=p7_mod_synthesis_random_len8_120 \
sbatch --output=slurm_logs/mod-synth-p7-random-%A_%a.out --error=slurm_logs/mod-synth-p7-random-%A_%a.err \
  --array=1-8 ModSynthesis/jobs/run_mod_synthesis_scavenge_cpu.sh
```

## Check Results

```bash
RUN=results/ModSynthesis/p7_mod_synthesis_low_mid_projlen_len8_80

find "$RUN" -name candidates.jsonl -size +0 -print
grep -R '"verified_candidates": [1-9]' "$RUN"/*/summary.json

for f in "$RUN"/*/summary.json; do
  echo "==== $f ===="
  python - <<PY
import json
r=json.load(open("$f"))
print({k:r.get(k) for k in ["parents","synthesized","exact_evaluations","verified_candidates","best_projlen","best_identity_defect","best_delta_defect","elapsed_seconds"]})
PY
done
```

If `/usr/bin/python` is unavailable on Bouchet, replace `python` with:

```bash
/home/as4843/braids-torch/bin/python
```
