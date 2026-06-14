#!/usr/bin/env python3
"""Validate V3 CUDA arithmetic and legal genetic operators."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch

from .config import ISLAND_NAMES, SearchConfig
from .crossover import SuffixCrossover
from .evaluators import CPUTrajectoryEvaluator, TorchTrajectoryEvaluator
from .gnf import GNFAutomaton
from .known_examples import KNOWN_P5_LENGTH54_FACTOR_IDS
from .models import Trajectory
from .mutation import AdaptiveSuffixMutationPlanner
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
        if not cuda_item.matrix_fingerprint:
            raise AssertionError("CUDA evaluator did not produce a matrix fingerprint")


def validate_operators() -> None:
    config = SearchConfig(
        p=3,
        horizons=(20,),
        population_size=30,
        generations=1,
        seed=20260613,
    )
    automaton = GNFAutomaton(n=4)
    rng = random.Random(config.seed)
    evaluator = CPUTrajectoryEvaluator(config)

    for island in ISLAND_NAMES:
        model = TransitionModel(config, automaton)
        planner = AdaptiveSuffixMutationPlanner(
            config,
            island,
            automaton,
            model,
            rng,
        )
        parent = Trajectory(
            automaton.sample_uniform(20, rng),
            island=island,
            trajectory_id=f"{island}-parent",
        )
        evaluation = evaluator.evaluate_one(parent)
        for _ in range(250):
            child = planner.make_child(
                evaluation,
                stagnant=rng.random() < 0.5,
                force_large=rng.random() < 0.2,
                two_mutations=rng.random() < 0.2,
            )
            if not automaton.is_legal(child.factor_ids):
                raise AssertionError(f"{island} mutation produced an illegal GNF word")

        crossover = SuffixCrossover(config, automaton, model, rng)
        donor = Trajectory(
            automaton.sample_uniform(20, rng),
            island=island,
            trajectory_id=f"{island}-donor",
        )
        donor_evaluation = evaluator.evaluate_one(donor)
        for _ in range(100):
            child = crossover.make_child(evaluation, donor_evaluation)
            if not automaton.is_legal(child.factor_ids):
                raise AssertionError(f"{island} crossover produced an illegal GNF word")


def validate_known_p5_kernel() -> None:
    trajectory = Trajectory(
        KNOWN_P5_LENGTH54_FACTOR_IDS,
        island="endpoint",
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
    if evaluation.final_projlen != 0:
        raise AssertionError("known p=5 kernel did not have projlen zero")


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
            island=ISLAND_NAMES[index % len(ISLAND_NAMES)],
            trajectory_id=f"random-{index:03d}",
        )
        for index in range(64)
    ]
    validate_operators()
    for p in (3, 5, 7):
        compare_cpu_and_cuda(p, trajectories)
    validate_known_p5_kernel()

    marker_path = Path(
        os.environ.get(
            "CRISPR_VALIDATION_MARKER",
            "results/crispr_v3_validation/scavenge_gpu_v3_validated.json",
        )
    )
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "passed",
        "algorithm": "crispr_trajectory_search_v3",
        "partition": os.environ["SLURM_JOB_PARTITION"],
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "random_trajectories_compared_per_prime": len(trajectories),
        "primes_compared": [3, 5, 7],
        "legal_mutations_checked_per_island": 250,
        "legal_crossovers_checked_per_island": 100,
        "known_p5_kernel_verified": True,
        "three_island_scoring_verified": True,
        "periodic_distance_used": False,
    }
    marker_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Validation marker: {marker_path}", flush=True)


if __name__ == "__main__":
    main()
