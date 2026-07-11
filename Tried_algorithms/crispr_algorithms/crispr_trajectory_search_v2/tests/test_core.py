from __future__ import annotations

import random
import unittest
from unittest.mock import patch

from crispr_trajectory_search_v2.archive import QualityDiversityArchive
from crispr_trajectory_search_v2.config import SearchConfig
from crispr_trajectory_search_v2.crossover import SuffixCrossover
from crispr_trajectory_search_v2.evaluators import (
    CPUTrajectoryEvaluator,
    TorchTrajectoryEvaluator,
)
from crispr_trajectory_search_v2.fitness import build_evaluation
from crispr_trajectory_search_v2.gnf import GNFAutomaton
from crispr_trajectory_search_v2.known_examples import KNOWN_P5_LENGTH54_FACTOR_IDS
from crispr_trajectory_search_v2.models import Trajectory
from crispr_trajectory_search_v2.mutation import MutationPlanner
from crispr_trajectory_search_v2.transition_model import TransitionModel


class TrajectorySearchV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SearchConfig(
            p=3,
            horizons=(12,),
            population_size=20,
            generations=2,
            seed=11,
        )
        self.automaton = GNFAutomaton(n=4)
        self.rng = random.Random(11)

    def test_random_generation_is_legal(self) -> None:
        for _ in range(100):
            factors = self.automaton.sample_uniform(20, self.rng)
            self.assertTrue(self.automaton.is_legal(factors))

    def test_both_mutation_lanes_remain_legal(self) -> None:
        model = TransitionModel(self.config, self.automaton)
        planner = MutationPlanner(
            self.config,
            self.automaton,
            model,
            self.rng,
        )
        parent = Trajectory(
            self.automaton.sample_uniform(12, self.rng),
            trajectory_id="parent",
        )
        evaluation = CPUTrajectoryEvaluator(self.config).evaluate_one(parent)
        for lane in ("local", "escape"):
            for _ in range(100):
                child = planner.make_child(
                    evaluation,
                    lane=lane,
                    two_mutations=lane == "escape" and self.rng.random() < 0.25,
                )
                self.assertEqual(len(child.factor_ids), len(parent.factor_ids))
                self.assertTrue(self.automaton.is_legal(child.factor_ids))

    def test_suffix_crossover_remains_legal(self) -> None:
        model = TransitionModel(self.config, self.automaton)
        crossover = SuffixCrossover(
            self.config,
            self.automaton,
            model,
            self.rng,
        )
        parents = [
            CPUTrajectoryEvaluator(self.config).evaluate_one(
                Trajectory(
                    self.automaton.sample_uniform(12, self.rng),
                    trajectory_id=f"parent-{index}",
                )
            )
            for index in range(2)
        ]
        for _ in range(100):
            child = crossover.make_child(*parents)
            self.assertEqual(child.horizon, 12)
            self.assertTrue(self.automaton.is_legal(child.factor_ids))

    def test_terminal_collapse_beats_boundary_low_with_rebound(self) -> None:
        factors = self.automaton.sample_uniform(10, self.rng)
        collapse = build_evaluation(
            Trajectory(factors, trajectory_id="collapse"),
            projlen_history=(2, 4, 6, 8, 10, 12, 10, 8, 4, 0),
            config=self.config,
        )
        rebound = build_evaluation(
            Trajectory(factors, trajectory_id="rebound"),
            projlen_history=(2, 4, 6, 8, 10, 2, 4, 8, 10, 12),
            config=self.config,
        )
        self.assertGreater(collapse.terminal_collapse, rebound.terminal_collapse)
        self.assertEqual(collapse.rebound, 0)
        self.assertGreater(rebound.rebound, 0)
        self.assertGreater(collapse.score, rebound.score)

    def test_archive_preserves_distinct_objective_champions(self) -> None:
        factors = self.automaton.sample_uniform(10, self.rng)
        low_final = build_evaluation(
            Trajectory(factors, trajectory_id="low-final"),
            projlen_history=(2, 4, 6, 8, 10, 12, 10, 8, 4, 1),
            config=self.config,
        )
        alternate = build_evaluation(
            Trajectory(
                self.automaton.sample_uniform(10, self.rng),
                trajectory_id="alternate",
            ),
            projlen_history=(2, 4, 6, 8, 10, 11, 9, 8, 7, 6),
            config=self.config,
        )
        archive = QualityDiversityArchive(self.config)
        archive.update((low_final, alternate))
        self.assertIs(archive.champions["lowest_final"], low_final)
        self.assertGreaterEqual(len(archive.members()), 2)

    def test_known_p5_kernel_is_detected(self) -> None:
        factors = KNOWN_P5_LENGTH54_FACTOR_IDS
        config = SearchConfig(
            p=5,
            horizons=(len(factors),),
            population_size=1,
            generations=1,
            periodic_distance=True,
        )
        evaluation = CPUTrajectoryEvaluator(config).evaluate_one(
            Trajectory(factors, trajectory_id="known-p5")
        )
        self.assertTrue(evaluation.has_kernel)
        self.assertEqual(evaluation.kernel_depths, (54,))
        self.assertEqual(evaluation.final_projlen, 0)
        self.assertEqual(evaluation.final_periodic_distance, 0.0)

    def test_torch_matches_cpu(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")

        trajectories = [
            Trajectory(
                self.automaton.sample_uniform(12, self.rng),
                trajectory_id=f"trajectory-{index}",
            )
            for index in range(16)
        ]
        cpu = CPUTrajectoryEvaluator(self.config).evaluate(trajectories)
        tensor_config = SearchConfig(
            **{
                **self.config.__dict__,
                "backend": "torch",
                "device": "cpu",
                "eval_batch_size": 7,
            }
        )
        tensor = TorchTrajectoryEvaluator(tensor_config).evaluate(trajectories)
        self.assertEqual(
            [item.projlen_history for item in cpu],
            [item.projlen_history for item in tensor],
        )

    def test_torch_periodic_frontier_detects_known_kernel(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")

        config = SearchConfig(
            p=5,
            horizons=(54,),
            population_size=1,
            generations=1,
            backend="torch",
            device="cpu",
            periodic_distance=True,
        )
        evaluation = TorchTrajectoryEvaluator(config).evaluate(
            [
                Trajectory(
                    KNOWN_P5_LENGTH54_FACTOR_IDS,
                    trajectory_id="known-p5-torch",
                )
            ]
        )[0]
        self.assertTrue(evaluation.has_kernel)
        self.assertEqual(evaluation.final_periodic_distance, 0.0)

    def test_cuda_rejects_non_scavenge_gpu_partition(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")

        tensor_config = SearchConfig(
            **{
                **self.config.__dict__,
                "backend": "torch",
                "device": "cuda",
            }
        )
        with patch.dict(
            "os.environ",
            {"SLURM_JOB_PARTITION": "some_other_gpu"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "restricted to"):
                TorchTrajectoryEvaluator(tensor_config)


if __name__ == "__main__":
    unittest.main()
