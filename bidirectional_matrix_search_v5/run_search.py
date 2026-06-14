#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bidirectional_matrix_search_v5.config import SearchConfig
from bidirectional_matrix_search_v5.search import BidirectionalMatrixSearch


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
        description="Run V5 bidirectional Burau matrix-state search."
    )
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--prefix-count", type=int, default=12_000)
    parser.add_argument("--suffix-count", type=int, default=60_000)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--prefix-length-min", type=int, default=18)
    parser.add_argument("--prefix-length-max", type=int, default=48)
    parser.add_argument("--suffix-length-min", type=int, default=10)
    parser.add_argument("--suffix-length-max", type=int, default=36)
    parser.add_argument("--field-points", type=int, default=8)
    parser.add_argument("--lsh-tables", type=int, default=16)
    parser.add_argument("--lsh-key-components", type=int, default=4)
    parser.add_argument("--max-lsh-candidates", type=int, default=1_024)
    parser.add_argument("--join-candidates-per-prefix", type=int, default=4)
    parser.add_argument("--elite-pairs", type=int, default=1_000)
    parser.add_argument("--algebra-elite-fraction", type=float, default=0.50)
    parser.add_argument("--length-niche-width", type=int, default=4)
    parser.add_argument("--carry-fraction", type=float, default=0.10)
    parser.add_argument("--random-fraction", type=float, default=0.15)
    parser.add_argument("--refinement-pairs", type=int, default=128)
    parser.add_argument("--refinement-trials", type=int, default=16)
    parser.add_argument("--mutation-block-sizes", type=comma_separated_ints, default=(1, 3, 5, 8, 12, 16))
    parser.add_argument("--length-edit-sizes", type=comma_separated_ints, default=(1, 2, 3, 5))
    parser.add_argument("--signature-batch-size", type=int, default=20_000)
    parser.add_argument("--exact-batch-size", type=int, default=10_000)
    parser.add_argument("--backend", choices=("cpu", "torch"), default="torch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--required-cuda-partition", default="scavenge_gpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="results/bidirectional_v5")
    parser.add_argument("--no-stop-at-kernel", action="store_true")
    parser.add_argument("--no-resume-latest", action="store_true")
    args = parser.parse_args()
    return SearchConfig(
        p=args.p,
        n=args.n,
        prefix_count=args.prefix_count,
        suffix_count=args.suffix_count,
        generations=args.generations,
        prefix_length_min=args.prefix_length_min,
        prefix_length_max=args.prefix_length_max,
        suffix_length_min=args.suffix_length_min,
        suffix_length_max=args.suffix_length_max,
        field_points=args.field_points,
        lsh_tables=args.lsh_tables,
        lsh_key_components=args.lsh_key_components,
        max_lsh_candidates=args.max_lsh_candidates,
        join_candidates_per_prefix=args.join_candidates_per_prefix,
        elite_pairs=args.elite_pairs,
        algebra_elite_fraction=args.algebra_elite_fraction,
        length_niche_width=args.length_niche_width,
        carry_fraction=args.carry_fraction,
        random_fraction=args.random_fraction,
        refinement_pairs=args.refinement_pairs,
        refinement_trials=args.refinement_trials,
        mutation_block_sizes=args.mutation_block_sizes,
        length_edit_sizes=args.length_edit_sizes,
        signature_batch_size=args.signature_batch_size,
        exact_batch_size=args.exact_batch_size,
        backend=args.backend,
        device=args.device,
        required_cuda_partition=args.required_cuda_partition,
        seed=args.seed,
        output_dir=args.output_dir,
        stop_at_kernel=not args.no_stop_at_kernel,
        resume_latest=not args.no_resume_latest,
    )


def main() -> None:
    search = BidirectionalMatrixSearch(parse_args())
    print(json.dumps(search.run(), indent=2), flush=True)


if __name__ == "__main__":
    main()
