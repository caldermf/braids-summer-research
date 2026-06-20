from __future__ import annotations

import argparse
import json

from structural_experiments.audit import run_p5_prefix_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Structural Burau kernel experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("datta-audit")
    audit.add_argument("--output-dir", required=True)
    audit.add_argument("--random-trajectories", type=int, default=128)
    audit.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.command == "datta-audit":
        result = run_p5_prefix_audit(
            args.output_dir,
            random_trajectories=args.random_trajectories,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "summary": f"{args.output_dir}/summary.json",
                    "known_kernel_garside_length": result[
                        "known_kernel_garside_length"
                    ],
                    "known_first_exceptional_depth": result[
                        "known_first_exceptional_depth"
                    ],
                    "decision": result["decision"],
                    "elapsed_seconds": result["elapsed_seconds"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
