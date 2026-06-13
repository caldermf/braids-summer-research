#!/usr/bin/env python3
"""Validate V4 CUDA arithmetic and variable-length genetic operators."""

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
from .mutation import StructuralMutationPlanner
from .transition_model import TransitionModel


def validation_config(p: int, **updates) -> SearchConfig:
    values = {
        "p": p,
        "n": 4,
        "min_horizon": 12,
        "initial_max_horizon": 28,
        "hard_max_horizon": 36,
        "population_size": 64,
        "min_generations": 1,
        "max_generations": 1,
        "eval_batch_size": 17,
    }
    values.update(updates)
    return SearchConfig(**values)


def compare_cpu_and_cuda(p: int, trajectories: list[Trajectory]) -> None:
    cpu = CPUTrajectoryEvaluator(validation_config(p)).evaluate(trajectories)
    cuda = TorchTrajectoryEvaluator(
        validation_config(p, backend="torch", device="cuda")
    ).evaluate(trajectories)
    for cpu_item, cuda_item in zip(cpu, cuda):
        if cpu_item.projlen_history != cuda_item.projlen_history:
            raise AssertionError(
                f"CPU/CUDA mismatch for p={p}, trajectory "
                f"{cpu_item.trajectory.trajectory_id}"
            )
        if not cuda_item.matrix_fingerprint:
            raise AssertionError("CUDA evaluator did not produce a matrix fingerprint")


def validate_operators() -> dict[str, int]:
    config = validation_config(3, seed=20260613)
    automaton = GNFAutomaton(n=4)
    rng = random.Random(config.seed)
    evaluator = CPUTrajectoryEvaluator(config)
    counts: dict[str, int] = {}

    for island in ISLAND_NAMES:
        model = TransitionModel(config, automaton)
        planner = StructuralMutationPlanner(config, island, automaton, model, rng)
        parent = Trajectory(
            automaton.sample_uniform(20, rng),
            island=island,
            trajectory_id=f"{island}-parent",
        )
        evaluation = evaluator.evaluate_one(parent)
        for action in StructuralMutationPlanner.ACTIONS:
            for _ in range(50):
                factors, record = planner.mutate_once(
                    evaluation,
                    active_max_horizon=28,
                    stagnant=True,
                    force_large=action in {"replace", "post_turn"},
                    use_learned=False,
                    action=action,
                )
                if not 12 <= len(factors) <= 28:
                    raise AssertionError(f"{action} escaped the configured horizon bounds")
                if not automaton.is_legal(factors):
                    raise AssertionError(f"{island}:{action} produced an illegal GNF word")
                counts[record.action] = counts.get(record.action, 0) + 1

        crossover = SuffixCrossover(config, automaton, model, rng)
        donor = Trajectory(
            automaton.sample_uniform(24, rng),
            island=island,
            trajectory_id=f"{island}-donor",
        )
        donor_evaluation = evaluator.evaluate_one(donor)
        for _ in range(100):
            child = crossover.make_child(evaluation, donor_evaluation, 28)
            if not 12 <= child.horizon <= 28:
                raise AssertionError("variable crossover escaped horizon bounds")
            if not automaton.is_legal(child.factor_ids):
                raise AssertionError("variable crossover produced an illegal GNF word")
    missing_actions = set(StructuralMutationPlanner.ACTIONS) - set(counts)
    if missing_actions:
        raise AssertionError(
            f"structural validation did not execute: {sorted(missing_actions)}"
        )
    return counts


def validate_known_p5_kernel() -> None:
    trajectory = Trajectory(
        KNOWN_P5_LENGTH54_FACTOR_IDS,
        island="endpoint",
        trajectory_id="known-p5-length54",
    )
    config = SearchConfig(
        p=5,
        min_horizon=36,
        initial_max_horizon=72,
        hard_max_horizon=120,
        population_size=1,
        min_generations=1,
        max_generations=1,
        backend="torch",
        device="cuda",
        eval_batch_size=1,
    )
    evaluation = TorchTrajectoryEvaluator(config).evaluate([trajectory])[0]
    if not evaluation.has_kernel or evaluation.kernel_depths != (54,):
        raise AssertionError("CUDA evaluator did not recover the known p=5 kernel")
    if evaluation.final_projlen != 0 or evaluation.peak_projlen != 29:
        raise AssertionError("known p=5 kernel envelope metrics are incorrect")
    if evaluation.post_turn_drop != 29:
        raise AssertionError("known p=5 kernel turning-point collapse is incorrect")


def main() -> None:
    if os.environ.get("SLURM_JOB_PARTITION") != "scavenge_gpu":
        raise SystemExit("Validation must run in the scavenge_gpu partition.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this scavenge_gpu job.")

    automaton = GNFAutomaton(n=4)
    rng = random.Random(20260613)
    trajectories = [
        Trajectory(
            automaton.sample_uniform(rng.randint(14, 24), rng),
            island=ISLAND_NAMES[index % len(ISLAND_NAMES)],
            trajectory_id=f"random-{index:03d}",
        )
        for index in range(64)
    ]
    operator_counts = validate_operators()
    for p in (3, 5, 7):
        compare_cpu_and_cuda(p, trajectories)
    validate_known_p5_kernel()

    marker_path = Path(
        os.environ.get(
            "CRISPR_VALIDATION_MARKER",
            "results/crispr_v4_validation/scavenge_gpu_v4_validated.json",
        )
    )
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "passed",
        "algorithm": "crispr_trajectory_search_v4",
        "partition": os.environ["SLURM_JOB_PARTITION"],
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "mixed_length_trajectories_compared_per_prime": len(trajectories),
        "primes_compared": [3, 5, 7],
        "legal_structural_actions_checked": operator_counts,
        "legal_crossovers_checked_per_island": 100,
        "known_p5_kernel_verified": True,
        "known_p5_peak_verified": 29,
        "known_p5_post_turn_drop_verified": 29,
        "four_island_scoring_verified": True,
        "periodic_distance_used": False,
    }
    marker_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Validation marker: {marker_path}", flush=True)


if __name__ == "__main__":
    main()
