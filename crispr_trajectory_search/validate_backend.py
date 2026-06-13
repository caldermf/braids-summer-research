#!/usr/bin/env python3
"""Validate the CUDA evaluator against the exact CPU implementation."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch

from .config import SearchConfig
from .evaluators import CPUTrajectoryEvaluator, TorchTrajectoryEvaluator
from .gnf import GNFAutomaton
from .known_examples import KNOWN_P5_LENGTH54_FACTOR_IDS
from .models import Trajectory
from .mutation import MutationPlanner
from .transition_model import TransitionModel


def compare_cpu_and_cuda(p: int, trajectories: list[Trajectory]) -> None:
    common = {
        "p": p,
        "n": 4,
        "horizons": (trajectories[0].horizon,),
        "population_size": len(trajectories),
        "generations": 1,
        "eval_batch_size": 17,
    }
    cpu = CPUTrajectoryEvaluator(SearchConfig(**common)).evaluate(trajectories)
    cuda = TorchTrajectoryEvaluator(
        SearchConfig(**common, backend="torch", device="cuda")
    ).evaluate(trajectories)

    for cpu_item, cuda_item in zip(cpu, cuda):
        if cpu_item.projlen_history != cuda_item.projlen_history:
            raise AssertionError(
                f"CPU/CUDA mismatch for p={p}, trajectory "
                f"{cpu_item.trajectory.trajectory_id}"
            )


def validate_mutations() -> None:
    config = SearchConfig(
        p=3,
        horizons=(20,),
        population_size=32,
        generations=1,
        seed=20260613,
    )
    automaton = GNFAutomaton(n=4)
    rng = random.Random(config.seed)
    model = TransitionModel(config, automaton)
    planner = MutationPlanner(config, automaton, model, rng)
    parent = Trajectory(
        automaton.sample_uniform(20, rng),
        trajectory_id="validation-parent",
    )
    evaluation = CPUTrajectoryEvaluator(config).evaluate_one(parent)
    for _ in range(250):
        child = planner.make_child(evaluation, two_mutations=rng.random() < 0.25)
        if not automaton.is_legal(child.factor_ids):
            raise AssertionError("mutation produced an illegal GNF trajectory")


def validate_known_p5_kernel() -> None:
    trajectory = Trajectory(
        KNOWN_P5_LENGTH54_FACTOR_IDS,
        trajectory_id="known-p5-length54",
    )
    config = SearchConfig(
        p=5,
        horizons=(54,),
        population_size=1,
        generations=1,
        backend="torch",
        device="cuda",
        eval_batch_size=1,
    )
    evaluation = TorchTrajectoryEvaluator(config).evaluate([trajectory])[0]
    if not evaluation.has_kernel or evaluation.kernel_depths != (54,):
        raise AssertionError("CUDA evaluator did not recover the known p=5 kernel")


def main() -> None:
    if os.environ.get("SLURM_JOB_PARTITION") != "scavenge_gpu":
        raise SystemExit("Validation must run in the scavenge_gpu partition.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this scavenge_gpu job.")

    automaton = GNFAutomaton(n=4)
    rng = random.Random(20260613)
    trajectories = [
        Trajectory(
            automaton.sample_uniform(18, rng),
            trajectory_id=f"random-{index:03d}",
        )
        for index in range(64)
    ]

    validate_mutations()
    for p in (3, 5, 7):
        compare_cpu_and_cuda(p, trajectories)
    validate_known_p5_kernel()

    marker_path = Path(
        os.environ.get(
            "CRISPR_VALIDATION_MARKER",
            "results/crispr_validation/scavenge_gpu_validated.json",
        )
    )
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "passed",
        "partition": os.environ["SLURM_JOB_PARTITION"],
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "random_trajectories_compared_per_prime": len(trajectories),
        "primes_compared": [3, 5, 7],
        "legal_mutations_checked": 250,
        "known_p5_kernel_verified": True,
    }
    marker_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Validation marker: {marker_path}", flush=True)


if __name__ == "__main__":
    main()
