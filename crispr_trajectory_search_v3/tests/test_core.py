from __future__ import annotations

import random
import tempfile
import unittest
from unittest.mock import patch

from crispr_trajectory_search_v3.caches import (
    EvaluationCache,
    MatrixNoveltyArchive,
    SeenWordCache,
)
from crispr_trajectory_search_v3.config import ISLAND_NAMES, SearchConfig
from crispr_trajectory_search_v3.crossover import SuffixCrossover
from crispr_trajectory_search_v3.evaluators import (
    CPUTrajectoryEvaluator,
    TorchTrajectoryEvaluator,
)
from crispr_trajectory_search_v3.fitness import build_evaluation
from crispr_trajectory_search_v3.gnf import GNFAutomaton
from crispr_trajectory_search_v3.islands import island_rank
from crispr_trajectory_search_v3.known_examples import KNOWN_P5_LENGTH54_FACTOR_IDS
from crispr_trajectory_search_v3.models import Trajectory
from crispr_trajectory_search_v3.mutation import AdaptiveSuffixMutationPlanner
from crispr_trajectory_search_v3.search import IslandTrajectorySearch
from crispr_trajectory_search_v3.transition_model import TransitionModel


class TrajectorySearchV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SearchConfig(
            p=3,
            horizons=(12,),
            population_size=20,
            generations=2,
            endpoint_block_sizes=(1, 3),
            collapse_block_sizes=(3, 5),
            suffix_block_sizes=(5, 8),
            stagnation_block_sizes=(8, 10),
            seed=11,
        )
        self.automaton = GNFAutomaton(n=4)
        self.rng = random.Random(11)

    def test_island_sizes_sum_exactly(self) -> None:
        self.assertEqual(self.config.island_sizes, {"endpoint": 8, "collapse": 6, "suffix": 6})
        self.assertEqual(sum(self.config.island_sizes.values()), self.config.population_size)

    def test_random_generation_is_legal(self) -> None:
        for _ in range(100):
            self.assertTrue(
                self.automaton.is_legal(self.automaton.sample_uniform(20, self.rng))
            )

    def test_all_island_mutations_remain_legal(self) -> None:
        evaluator = CPUTrajectoryEvaluator(self.config)
        for island in ISLAND_NAMES:
            model = TransitionModel(self.config, self.automaton)
            planner = AdaptiveSuffixMutationPlanner(
                self.config,
                island,
                self.automaton,
                model,
                self.rng,
            )
            parent = Trajectory(
                self.automaton.sample_uniform(12, self.rng),
                island=island,
                trajectory_id=f"{island}-parent",
            )
            evaluation = evaluator.evaluate_one(parent)
            for _ in range(100):
                child = planner.make_child(
                    evaluation,
                    stagnant=True,
                    force_large=self.rng.random() < 0.25,
                    two_mutations=self.rng.random() < 0.25,
                )
                self.assertEqual(child.horizon, 12)
                self.assertTrue(self.automaton.is_legal(child.factor_ids))

    def test_suffix_crossover_remains_legal(self) -> None:
        model = TransitionModel(self.config, self.automaton)
        crossover = SuffixCrossover(self.config, self.automaton, model, self.rng)
        evaluator = CPUTrajectoryEvaluator(self.config)
        parents = [
            evaluator.evaluate_one(
                Trajectory(
                    self.automaton.sample_uniform(12, self.rng),
                    island="suffix",
                    trajectory_id=f"parent-{index}",
                )
            )
            for index in range(2)
        ]
        for _ in range(100):
            child = crossover.make_child(*parents)
            self.assertTrue(self.automaton.is_legal(child.factor_ids))

    def test_islands_prefer_different_trajectory_shapes(self) -> None:
        factors = self.automaton.sample_uniform(10, self.rng)
        endpoint = build_evaluation(
            Trajectory(factors, island="endpoint", trajectory_id="endpoint"),
            (2, 4, 6, 8, 10, 9, 8, 7, 6, 5),
            self.config,
        )
        collapse = build_evaluation(
            Trajectory(factors, island="collapse", trajectory_id="collapse"),
            (2, 4, 6, 8, 10, 14, 13, 10, 7, 6),
            self.config,
        )
        suffix = build_evaluation(
            Trajectory(factors, island="suffix", trajectory_id="suffix"),
            (2, 4, 6, 8, 9, 8, 7, 6, 6, 6),
            self.config,
        )
        self.assertGreater(island_rank(endpoint, "endpoint"), island_rank(collapse, "endpoint"))
        self.assertGreater(island_rank(collapse, "collapse"), island_rank(endpoint, "collapse"))
        self.assertGreater(island_rank(suffix, "suffix"), island_rank(collapse, "suffix"))

    def test_evaluation_cache_reuses_math_but_rebinds_island(self) -> None:
        factors = self.automaton.sample_uniform(12, self.rng)
        evaluator = CPUTrajectoryEvaluator(self.config)
        cache = EvaluationCache()
        first = cache.evaluate(
            evaluator,
            [Trajectory(factors, island="endpoint", trajectory_id="a")],
        )[0]
        second = cache.evaluate(
            evaluator,
            [Trajectory(factors, island="suffix", trajectory_id="b")],
        )[0]
        self.assertEqual(first.projlen_history, second.projlen_history)
        self.assertEqual(second.trajectory.trajectory_id, "b")
        self.assertEqual(second.trajectory.island, "suffix")
        self.assertEqual(cache.stats()["hits"], 1)

    def test_seen_word_cache_is_collision_free(self) -> None:
        cache = SeenWordCache()
        factors = self.automaton.sample_uniform(12, self.rng)
        self.assertTrue(cache.add(factors))
        self.assertFalse(cache.add(factors))
        self.assertEqual(cache.stats()["duplicate_rejections"], 1)

    def test_matrix_novelty_rewards_unseen_state(self) -> None:
        factors = self.automaton.sample_uniform(10, self.rng)
        evaluations = [
            build_evaluation(
                Trajectory(factors, island="endpoint", trajectory_id=str(index)),
                (2, 4, 6, 8, 10, 12, 11, 10, 9, 8),
                self.config,
                matrix_fingerprint="same",
            )
            for index in range(2)
        ]
        archive = MatrixNoveltyArchive(100)
        archive.assign(evaluations)
        self.assertEqual(evaluations[0].novelty, 2.5)
        archive.assign(evaluations)
        self.assertEqual(evaluations[0].novelty, 0.5)

    def test_known_p5_kernel_is_detected_without_periodic_distance(self) -> None:
        config = SearchConfig(
            p=5,
            horizons=(54,),
            population_size=1,
            generations=1,
        )
        evaluation = CPUTrajectoryEvaluator(config).evaluate_one(
            Trajectory(
                KNOWN_P5_LENGTH54_FACTOR_IDS,
                island="endpoint",
                trajectory_id="known-p5",
            )
        )
        self.assertTrue(evaluation.has_kernel)
        self.assertEqual(evaluation.kernel_depths, (54,))
        self.assertEqual(evaluation.final_projlen, 0)

    def test_small_search_runs_with_mcts(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = SearchConfig(
                p=3,
                horizons=(7,),
                population_size=30,
                generations=2,
                offspring_per_parent=2,
                endpoint_block_sizes=(1, 2),
                collapse_block_sizes=(2, 3),
                suffix_block_sizes=(2, 3),
                stagnation_block_sizes=(3, 4),
                mcts_interval=1,
                mcts_seed_count=6,
                mcts_simulations_per_seed=2,
                mcts_max_depth=2,
                mcts_branching_factor=2,
                mcts_block_sizes=(1, 2, 3),
                output_dir=output_dir,
                seed=29,
            )
            summary = IslandTrajectorySearch(config).run()
            self.assertEqual(summary["completed_generations"], 2)
            self.assertGreater(summary["mcts"]["simulations"], 0)

    def test_torch_matches_cpu(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        trajectories = [
            Trajectory(
                self.automaton.sample_uniform(12, self.rng),
                island=ISLAND_NAMES[index % 3],
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

    def test_cuda_rejects_non_scavenge_gpu_partition(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        tensor_config = SearchConfig(
            **{**self.config.__dict__, "backend": "torch", "device": "cuda"}
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
