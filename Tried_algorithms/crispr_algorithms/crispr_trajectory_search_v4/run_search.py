#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crispr_trajectory_search_v4.config import SearchConfig
from crispr_trajectory_search_v4.search import VariableLengthIslandSearch


def comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def comma_separated_floats(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one number is required")
    return parsed


def parse_args() -> SearchConfig:
    parser = argparse.ArgumentParser(
        description="Run CRISPR V4 variable-length four-island search."
    )
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--min-horizon", type=int, default=36)
    parser.add_argument("--initial-max-horizon", type=int, default=72)
    parser.add_argument("--hard-max-horizon", type=int, default=120)
    parser.add_argument("--horizon-expand-step", type=int, default=8)
    parser.add_argument("--horizon-expand-interval", type=int, default=5)
    parser.add_argument("--horizon-boundary-margin", type=int, default=4)
    parser.add_argument("--horizon-boundary-elite-fraction", type=float, default=0.15)
    parser.add_argument("--length-niche-width", type=int, default=4)
    parser.add_argument("--population-size", type=int, default=50_000)
    parser.add_argument("--min-generations", type=int, default=60)
    parser.add_argument("--max-generations", type=int, default=200)
    parser.add_argument("--global-stagnation-generations", type=int, default=20)
    parser.add_argument("--island-fractions", type=comma_separated_floats, default=(0.25, 0.25, 0.25, 0.25))
    parser.add_argument("--elite-fraction", type=float, default=0.05)
    parser.add_argument("--carry-fraction", type=float, default=0.08)
    parser.add_argument("--random-fraction", type=float, default=0.12)
    parser.add_argument("--crossover-fraction", type=float, default=0.08)
    parser.add_argument("--offspring-per-parent", type=int, default=6)
    parser.add_argument("--endpoint-block-sizes", type=comma_separated_ints, default=(1, 3, 5, 8, 12, 16))
    parser.add_argument("--envelope-block-sizes", type=comma_separated_ints, default=(3, 5, 8, 12, 16, 24))
    parser.add_argument("--collapse-block-sizes", type=comma_separated_ints, default=(5, 8, 12, 16, 24, 32))
    parser.add_argument("--suffix-block-sizes", type=comma_separated_ints, default=(8, 12, 16, 24, 32, 40))
    parser.add_argument("--stagnation-block-sizes", type=comma_separated_ints, default=(24, 32, 40, 48))
    parser.add_argument("--length-edit-sizes", type=comma_separated_ints, default=(1, 2, 3, 5, 8))
    parser.add_argument("--structural-mutation-fraction", type=float, default=0.30)
    parser.add_argument("--post-turn-rewrite-fraction", type=float, default=0.25)
    parser.add_argument("--migration-interval", type=int, default=10)
    parser.add_argument("--migration-fraction", type=float, default=0.005)
    parser.add_argument("--stagnation-generations", type=int, default=10)
    parser.add_argument("--stagnation-min-improvement", type=float, default=0.005)
    parser.add_argument("--finishing-queue-size-per-island", type=int, default=512)
    parser.add_argument("--evaluation-cache-size", type=int, default=250_000)
    parser.add_argument("--mcts-interval", type=int, default=5)
    parser.add_argument("--mcts-seed-count", type=int, default=96)
    parser.add_argument("--mcts-simulations-per-seed", type=int, default=64)
    parser.add_argument("--mcts-max-depth", type=int, default=12)
    parser.add_argument("--mcts-branching-factor", type=int, default=4)
    parser.add_argument("--mcts-block-sizes", type=comma_separated_ints, default=(8, 12, 16, 24, 32, 40))
    parser.add_argument("--mcts-length-edit-fraction", type=float, default=0.25)
    parser.add_argument("--disable-mcts", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "torch"), default="cpu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--required-cuda-partition", default="scavenge_gpu")
    parser.add_argument("--eval-batch-size", type=int, default=10_000)
    parser.add_argument("--seed-trajectory-json")
    parser.add_argument("--seed-known-example", choices=("p5_length54",))
    parser.add_argument("--seed-population-fraction", type=float, default=0.0)
    parser.add_argument("--seed-corruption-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="results/crispr_trajectory_search_v4")
    parser.add_argument("--no-stop-at-kernel", action="store_true")
    parser.add_argument("--max-kernel-hits", type=int, default=20)
    args = parser.parse_args()

    return SearchConfig(
        p=args.p,
        n=args.n,
        min_horizon=args.min_horizon,
        initial_max_horizon=args.initial_max_horizon,
        hard_max_horizon=args.hard_max_horizon,
        horizon_expand_step=args.horizon_expand_step,
        horizon_expand_interval=args.horizon_expand_interval,
        horizon_boundary_margin=args.horizon_boundary_margin,
        horizon_boundary_elite_fraction=args.horizon_boundary_elite_fraction,
        length_niche_width=args.length_niche_width,
        population_size=args.population_size,
        min_generations=args.min_generations,
        max_generations=args.max_generations,
        global_stagnation_generations=args.global_stagnation_generations,
        island_fractions=args.island_fractions,
        elite_fraction=args.elite_fraction,
        carry_fraction=args.carry_fraction,
        random_fraction=args.random_fraction,
        crossover_fraction=args.crossover_fraction,
        offspring_per_parent=args.offspring_per_parent,
        endpoint_block_sizes=args.endpoint_block_sizes,
        envelope_block_sizes=args.envelope_block_sizes,
        collapse_block_sizes=args.collapse_block_sizes,
        suffix_block_sizes=args.suffix_block_sizes,
        stagnation_block_sizes=args.stagnation_block_sizes,
        length_edit_sizes=args.length_edit_sizes,
        structural_mutation_fraction=args.structural_mutation_fraction,
        post_turn_rewrite_fraction=args.post_turn_rewrite_fraction,
        migration_interval=args.migration_interval,
        migration_fraction=args.migration_fraction,
        stagnation_generations=args.stagnation_generations,
        stagnation_min_improvement=args.stagnation_min_improvement,
        finishing_queue_size_per_island=args.finishing_queue_size_per_island,
        evaluation_cache_size=args.evaluation_cache_size,
        mcts_enabled=not args.disable_mcts,
        mcts_interval=args.mcts_interval,
        mcts_seed_count=args.mcts_seed_count,
        mcts_simulations_per_seed=args.mcts_simulations_per_seed,
        mcts_max_depth=args.mcts_max_depth,
        mcts_branching_factor=args.mcts_branching_factor,
        mcts_block_sizes=args.mcts_block_sizes,
        mcts_length_edit_fraction=args.mcts_length_edit_fraction,
        backend=args.backend,
        device=args.device,
        required_cuda_partition=args.required_cuda_partition,
        eval_batch_size=args.eval_batch_size,
        seed_trajectory_json=args.seed_trajectory_json,
        seed_known_example=args.seed_known_example,
        seed_population_fraction=args.seed_population_fraction,
        seed_corruption_fraction=args.seed_corruption_fraction,
        seed=args.seed,
        output_dir=args.output_dir,
        stop_at_kernel=not args.no_stop_at_kernel,
        max_kernel_hits=args.max_kernel_hits,
    )


def main() -> None:
    search = VariableLengthIslandSearch(parse_args())
    print(json.dumps(search.run(), indent=2), flush=True)


if __name__ == "__main__":
    main()
