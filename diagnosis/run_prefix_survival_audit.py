#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnosis.audit import PrefixSurvivalAudit
from diagnosis.known_examples import EMBEDDED_CASES, load_kernel_cases
from diagnosis.models import AuditConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Force known kernel prefixes through a paper-style reservoir search "
            "and measure where each search policy would discard them."
        )
    )
    parser.add_argument(
        "--known-example",
        action="append",
        default=[],
        choices=sorted(EMBEDDED_CASES),
        help="Embedded known trajectory; may be repeated.",
    )
    parser.add_argument(
        "--kernel-json",
        action="append",
        default=[],
        type=Path,
        help="JSON containing factor_ids, gnf_factors, or a kernel_hits list.",
    )
    parser.add_argument(
        "--all-json-kernels",
        action="store_true",
        help="Audit every kernel_hits entry instead of only the first.",
    )
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--bootstrap-depth", type=int, default=5)
    parser.add_argument("--bucket-size", type=int, default=15_000)
    parser.add_argument("--use-best", type=int, default=30_000)
    parser.add_argument("--baseline-samples", type=int, default=512)
    parser.add_argument("--periodic-bucket-size", type=int, default=3_000)
    parser.add_argument("--periodic-use-best", type=int, default=50_000)
    parser.add_argument("--mcts-beam-width", type=int, default=64)
    parser.add_argument("--crispr-sample-size", type=int, default=5_000)
    parser.add_argument("--crispr-population-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results/diagnosis"))
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_kernel_cases(
        names=args.known_example,
        json_paths=args.kernel_json,
        n=args.n,
        all_json_kernels=args.all_json_kernels,
    )
    config = AuditConfig(
        p=args.p,
        n=args.n,
        max_depth=args.max_depth,
        bootstrap_depth=args.bootstrap_depth,
        bucket_size=args.bucket_size,
        use_best=args.use_best,
        baseline_samples=args.baseline_samples,
        periodic_bucket_size=args.periodic_bucket_size,
        periodic_use_best=args.periodic_use_best,
        mcts_beam_width=args.mcts_beam_width,
        crispr_sample_size=args.crispr_sample_size,
        crispr_population_size=args.crispr_population_size,
        seed=args.seed,
        output_dir=args.output_dir,
        render_plots=not args.no_plots,
    )
    summaries = []
    for index, case in enumerate(cases):
        case_config = AuditConfig(**{**config.__dict__, "seed": config.seed + index})
        summaries.append(PrefixSurvivalAudit(case_config).run(case))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
