#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crispr_trajectory_search_v2.config import SearchConfig
from crispr_trajectory_search_v2.search import EvolutionaryTrajectorySearch


def comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def parse_args() -> SearchConfig:
    parser = argparse.ArgumentParser(
        description="Run CRISPR v2 quality-diversity search over complete GNF trajectories."
    )
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--horizons", type=comma_separated_ints, default=(30, 35, 40))
    parser.add_argument("--population-size", type=int, default=5000)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--elite-fraction", type=float, default=0.05)
    parser.add_argument("--archive-fraction", type=float, default=0.10)
    parser.add_argument("--local-mutation-fraction", type=float, default=0.50)
    parser.add_argument("--escape-mutation-fraction", type=float, default=0.20)
    parser.add_argument("--crossover-fraction", type=float, default=0.10)
    parser.add_argument("--random-sample-fraction", type=float, default=0.10)
    parser.add_argument(
        "--local-mutation-block-sizes",
        type=comma_separated_ints,
        default=(1, 3),
    )
    parser.add_argument(
        "--escape-mutation-block-sizes",
        type=comma_separated_ints,
        default=(5, 8, 12, 16),
    )
    parser.add_argument("--archive-size", type=int, default=5000)
    parser.add_argument("--mutation-statistics-decay", type=float, default=0.95)
    parser.add_argument("--stagnation-generations", type=int, default=10)
    parser.add_argument("--stagnation-escape-boost", type=float, default=0.15)
    parser.add_argument("--stagnation-random-boost", type=float, default=0.05)
    parser.add_argument("--late-start-fraction", type=float, default=0.55)
    parser.add_argument("--final-advantage-weight", type=float, default=6.0)
    parser.add_argument("--terminal-collapse-weight", type=float, default=8.0)
    parser.add_argument("--terminal-slope-weight", type=float, default=20.0)
    parser.add_argument("--terminal-descent-weight", type=float, default=0.5)
    parser.add_argument("--rebound-penalty-weight", type=float, default=8.0)
    parser.add_argument("--periodic-distance-weight", type=float, default=1.0)
    parser.add_argument("--periodic-distance", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "torch"), default="cpu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--required-cuda-partition", default="scavenge_gpu")
    parser.add_argument("--eval-batch-size", type=int, default=5000)
    parser.add_argument("--seed-trajectory-json")
    parser.add_argument("--seed-known-example", choices=("p5_length54",))
    parser.add_argument("--seed-population-fraction", type=float, default=0.0)
    parser.add_argument("--seed-corruption-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="results/crispr_trajectory_search_v2")
    parser.add_argument("--no-stop-at-kernel", action="store_true")
    parser.add_argument("--max-kernel-hits", type=int, default=20)
    args = parser.parse_args()

    return SearchConfig(
        p=args.p,
        n=args.n,
        horizons=args.horizons,
        population_size=args.population_size,
        generations=args.generations,
        elite_fraction=args.elite_fraction,
        archive_fraction=args.archive_fraction,
        local_mutation_fraction=args.local_mutation_fraction,
        escape_mutation_fraction=args.escape_mutation_fraction,
        crossover_fraction=args.crossover_fraction,
        random_sample_fraction=args.random_sample_fraction,
        local_mutation_block_sizes=args.local_mutation_block_sizes,
        escape_mutation_block_sizes=args.escape_mutation_block_sizes,
        archive_size=args.archive_size,
        mutation_statistics_decay=args.mutation_statistics_decay,
        stagnation_generations=args.stagnation_generations,
        stagnation_escape_boost=args.stagnation_escape_boost,
        stagnation_random_boost=args.stagnation_random_boost,
        late_start_fraction=args.late_start_fraction,
        final_advantage_weight=args.final_advantage_weight,
        terminal_collapse_weight=args.terminal_collapse_weight,
        terminal_slope_weight=args.terminal_slope_weight,
        terminal_descent_weight=args.terminal_descent_weight,
        rebound_penalty_weight=args.rebound_penalty_weight,
        periodic_distance_weight=args.periodic_distance_weight,
        periodic_distance=args.periodic_distance,
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
    search = EvolutionaryTrajectorySearch(parse_args())
    print(json.dumps(search.run(), indent=2), flush=True)


if __name__ == "__main__":
    main()
