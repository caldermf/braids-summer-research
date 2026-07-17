# B6 (4,2) commutator search

This is the dimension-independent port of the successful B4 commutator
reservoir. It searches `[sigma_i, g^-1]` for all `i=1,...,5` using the
precomputed `tables_B6_r2_p3.pt` matrices and exact `peyl` verification.

Run from `braids-summer-research` on Bouchet. First construct the five exact
twisted-matrix caches on CPU, then submit the GPU array after they all succeed:

```bash
TABLE=/nfs/roberts/project/pi_com36/as4843/burau-experiments/beta/precomputed_tables/tables_B6_r2_p3.pt
PREP_JOB=$(TABLE="$TABLE" sbatch --parsable B6-Commutator-Search/prepare_twisted_all_scavenge_cpu.sh)
TABLE="$TABLE" PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  sbatch --dependency=afterok:"$PREP_JOB" B6-Commutator-Search/run_all_generators_scavenge_gpu.sh
```

Results go to
`results/B6-Commutator-Search/B6_r2_p3_commutator_all_generators/`.
