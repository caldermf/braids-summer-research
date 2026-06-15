from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from crispr_algorithms.crispr_trajectory_search_v4.config import SearchConfig
from crispr_algorithms.crispr_trajectory_search_v4.evaluators import make_evaluator
from crispr_algorithms.crispr_trajectory_search_v4.gnf import GNFAutomaton
from crispr_algorithms.crispr_trajectory_search_v4.models import Trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", required=True)
    parser.add_argument("--p", type=int, default=5)
    args = parser.parse_args()

    if os.environ.get("SLURM_JOB_PARTITION") != "scavenge_gpu":
        raise RuntimeError("GPU validation must run on scavenge_gpu")

    automaton = GNFAutomaton(4)
    rng = random.Random(20260615)
    trajectories = [
        Trajectory(
            factor_ids=automaton.sample_prefix(length, rng),
            trajectory_id=f"validation-{length}-{index}",
        )
        for length in (4, 7, 11)
        for index in range(8)
    ]
    common = dict(
        p=args.p,
        n=4,
        min_horizon=1,
        initial_max_horizon=11,
        hard_max_horizon=11,
        min_generations=1,
        max_generations=1,
        mcts_enabled=False,
    )
    cpu = make_evaluator(SearchConfig(**common, backend="cpu", device="cpu"))
    gpu = make_evaluator(SearchConfig(**common, backend="torch", device="cuda"))
    cpu_rows = cpu.evaluate(trajectories)
    gpu_rows = gpu.evaluate(trajectories)
    for cpu_row, gpu_row in zip(cpu_rows, gpu_rows):
        if cpu_row.projlen_history != gpu_row.projlen_history:
            raise AssertionError(
                f"CPU/GPU mismatch for {cpu_row.trajectory.factor_ids}: "
                f"{cpu_row.projlen_history} != {gpu_row.projlen_history}"
            )

    marker = Path(args.marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "status": "validated",
                "partition": os.environ["SLURM_JOB_PARTITION"],
                "p": args.p,
                "trajectories": len(trajectories),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(marker)


if __name__ == "__main__":
    main()
