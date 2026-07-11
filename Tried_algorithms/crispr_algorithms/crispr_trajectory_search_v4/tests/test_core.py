from __future__ import annotations

import random
import tempfile
import unittest
from unittest.mock import patch

from crispr_trajectory_search_v4.caches import (
    AdaptiveFinishingQueues,
    EvaluationCache,
    SeenWordCache,
)
from crispr_trajectory_search_v4.config import ISLAND_NAMES, SearchConfig
from crispr_trajectory_search_v4.crossover import SuffixCrossover
from crispr_trajectory_search_v4.evaluators import (
    CPUTrajectoryEvaluator,
    TorchTrajectoryEvaluator,
)
from crispr_trajectory_search_v4.fitness import build_evaluation
from crispr_trajectory_search_v4.gnf import GNFAutomaton
from crispr_trajectory_search_v4.islands import island_rank, select_island_elites
from crispr_trajectory_search_v4.known_examples import KNOWN_P5_LENGTH54_FACTOR_IDS
from crispr_trajectory_search_v4.models import Trajectory
from crispr_trajectory_search_v4.mutation import StructuralMutationPlanner
from crispr_trajectory_search_v4.search import VariableLengthIslandSearch
from crispr_trajectory_search_v4.transition_model import TransitionModel


class TrajectorySearchV4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SearchConfig(
            p=3,
            min_horizon=8,
            initial_max_horizon=16,
            hard_max_horizon=24,
            population_size=40,
            min_generations=2,
            max_generations=3,
            endpoint_block_sizes=(1, 2, 3),
            envelope_block_sizes=(2, 3, 4),
            collapse_block_sizes=(2, 4, 6),
            suffix_block_sizes=(3, 5, 7),
            stagnation_block_sizes=(5, 7, 9),
            length_edit_sizes=(1, 2, 3),
            seed=11,
        )
        self.automaton = GNFAutomaton(n=4)
        self.rng = random.Random(11)

    def test_four_island_sizes_sum_exactly(self) -> None:
        self.assertEqual(
            self.config.island_sizes,
            {"endpoint": 10, "envelope": 10, "collapse": 10, "suffix": 10},
        )

    def test_random_generation_is_legal_across_lengths(self) -> None:
        for horizon in range(8, 17):
            factors = self.automaton.sample_uniform(horizon, self.rng)
            self.assertEqual(len(factors), horizon)
            self.assertTrue(self.automaton.is_legal(factors))

    def test_every_structural_action_remains_legal_and_bounded(self) -> None:
        evaluator = CPUTrajectoryEvaluator(self.config)
        for island in ISLAND_NAMES:
            model = TransitionModel(self.config, self.automaton)
            planner = StructuralMutationPlanner(
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
            for action in StructuralMutationPlanner.ACTIONS:
                for _ in range(30):
                    factors, _ = planner.mutate_once(
                        evaluation,
                        active_max_horizon=20,
                        stagnant=True,
                        force_large=True,
                        use_learned=False,
                        action=action,
                    )
                    self.assertGreaterEqual(len(factors), self.config.min_horizon)
                    self.assertLessEqual(len(factors), 20)
                    self.assertTrue(self.automaton.is_legal(factors))

    def test_variable_length_crossover_is_legal(self) -> None:
        model = TransitionModel(self.config, self.automaton)
        crossover = SuffixCrossover(self.config, self.automaton, model, self.rng)
        evaluator = CPUTrajectoryEvaluator(self.config)
        recipient = evaluator.evaluate_one(
            Trajectory(
                self.automaton.sample_uniform(10, self.rng),
                island="suffix",
                trajectory_id="recipient",
            )
        )
        donor = evaluator.evaluate_one(
            Trajectory(
                self.automaton.sample_uniform(15, self.rng),
                island="suffix",
                trajectory_id="donor",
            )
        )
        for _ in range(100):
            child = crossover.make_child(recipient, donor, 20)
            self.assertGreaterEqual(child.horizon, 8)
            self.assertLessEqual(child.horizon, 20)
            self.assertTrue(self.automaton.is_legal(child.factor_ids))

    def test_known_kernel_envelope_and_turning_point(self) -> None:
        config = SearchConfig(
            p=5,
            min_horizon=36,
            initial_max_horizon=72,
            hard_max_horizon=120,
            population_size=1,
            min_generations=1,
            max_generations=1,
        )
        evaluation = CPUTrajectoryEvaluator(config).evaluate_one(
            Trajectory(
                KNOWN_P5_LENGTH54_FACTOR_IDS,
                island="collapse",
                trajectory_id="known",
            )
        )
        self.assertTrue(evaluation.has_kernel)
        self.assertEqual(evaluation.kernel_depths, (54,))
        self.assertEqual(evaluation.peak_projlen, 29)
        self.assertEqual(evaluation.post_turn_drop, 29)
        self.assertEqual(evaluation.final_projlen, 0)

    def test_kernel_prefix_is_detected_inside_longer_braid(self) -> None:
        config = SearchConfig(
            p=5,
            min_horizon=36,
            initial_max_horizon=72,
            hard_max_horizon=120,
            population_size=1,
            min_generations=1,
            max_generations=1,
        )
        factors = list(KNOWN_P5_LENGTH54_FACTOR_IDS)
        while len(factors) < 60:
            factors.append(self.automaton.successors[factors[-1]][0])
        evaluation = CPUTrajectoryEvaluator(config).evaluate_one(
            Trajectory(
                tuple(factors),
                island="endpoint",
                trajectory_id="known-prefix",
            )
        )
        self.assertIn(54, evaluation.kernel_depths)

    def test_length_normalization_does_not_automatically_prefer_shorter(self) -> None:
        short = build_evaluation(
            Trajectory(tuple(range(8)), island="endpoint", trajectory_id="short"),
            (2, 4, 6, 8, 10, 12, 14, 16),
            self.config,
        )
        long = build_evaluation(
            Trajectory(tuple(range(16)), island="endpoint", trajectory_id="long"),
            tuple(range(2, 34, 2)),
            self.config,
        )
        self.assertAlmostEqual(short.endpoint_advantage, long.endpoint_advantage)

    def test_islands_prefer_distinct_shapes(self) -> None:
        factors = self.automaton.sample_uniform(12, self.rng)
        endpoint = build_evaluation(
            Trajectory(factors, island="endpoint", trajectory_id="endpoint"),
            (2, 4, 6, 8, 10, 12, 13, 12, 10, 8, 6, 4),
            self.config,
        )
        envelope = build_evaluation(
            Trajectory(factors, island="envelope", trajectory_id="envelope"),
            (2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 8, 7),
            self.config,
        )
        collapse = build_evaluation(
            Trajectory(factors, island="collapse", trajectory_id="collapse"),
            (2, 4, 6, 8, 10, 14, 13, 11, 8, 5, 2, 0),
            self.config,
        )
        self.assertGreater(island_rank(endpoint, "endpoint"), island_rank(envelope, "endpoint"))
        self.assertGreater(island_rank(envelope, "envelope"), island_rank(endpoint, "envelope"))
        self.assertGreater(island_rank(collapse, "collapse"), island_rank(envelope, "collapse"))

    def test_length_niches_preserve_multiple_horizons(self) -> None:
        evaluations = []
        evaluator = CPUTrajectoryEvaluator(self.config)
        for horizon in (8, 12, 16):
            for index in range(5):
                evaluations.append(
                    evaluator.evaluate_one(
                        Trajectory(
                            self.automaton.sample_uniform(horizon, self.rng),
                            island="endpoint",
                            trajectory_id=f"{horizon}-{index}",
                        )
                    )
                )
        selected = select_island_elites(evaluations, "endpoint", 6, niche_width=4)
        self.assertGreaterEqual(len({item.trajectory.horizon for item in selected}), 3)

    def test_adaptive_queues_are_never_empty_after_update(self) -> None:
        evaluator = CPUTrajectoryEvaluator(self.config)
        evaluations = evaluator.evaluate(
            [
                Trajectory(
                    self.automaton.sample_uniform(12, self.rng),
                    island="endpoint",
                    trajectory_id=str(index),
                )
                for index in range(10)
            ]
        )
        queues = AdaptiveFinishingQueues(4)
        queues.update(evaluations)
        self.assertEqual(
            queues.stats()["sizes"],
            {island: 4 for island in ISLAND_NAMES},
        )

    def test_evaluation_cache_rebinds_island(self) -> None:
        factors = self.automaton.sample_uniform(12, self.rng)
        evaluator = CPUTrajectoryEvaluator(self.config)
        cache = EvaluationCache()
        cache.evaluate(evaluator, [Trajectory(factors, island="endpoint")])
        result = cache.evaluate(
            evaluator,
            [Trajectory(factors, island="envelope", trajectory_id="rebound")],
        )[0]
        self.assertEqual(result.trajectory.island, "envelope")
        self.assertEqual(result.trajectory.trajectory_id, "rebound")
        self.assertEqual(cache.stats()["hits"], 1)

    def test_seen_word_cache_rejects_exact_duplicate(self) -> None:
        cache = SeenWordCache()
        factors = self.automaton.sample_uniform(12, self.rng)
        self.assertTrue(cache.add(factors))
        self.assertFalse(cache.add(factors))

    def test_small_variable_search_runs_with_mcts_and_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = SearchConfig(
                p=3,
                min_horizon=6,
                initial_max_horizon=10,
                hard_max_horizon=14,
                horizon_expand_step=2,
                horizon_expand_interval=1,
                horizon_boundary_margin=2,
                horizon_boundary_elite_fraction=0.0,
                length_niche_width=2,
                population_size=40,
                min_generations=2,
                max_generations=3,
                global_stagnation_generations=2,
                offspring_per_parent=2,
                mutation_attempts=12,
                endpoint_block_sizes=(1, 2, 3),
                envelope_block_sizes=(1, 2, 3),
                collapse_block_sizes=(2, 3, 4),
                suffix_block_sizes=(2, 3, 4),
                stagnation_block_sizes=(3, 4, 5),
                length_edit_sizes=(1, 2),
                migration_interval=1,
                stagnation_generations=1,
                stagnation_min_improvement=10.0,
                finishing_queue_size_per_island=8,
                mcts_interval=1,
                mcts_seed_count=8,
                mcts_simulations_per_seed=2,
                mcts_max_depth=2,
                mcts_branching_factor=2,
                mcts_block_sizes=(1, 2, 3),
                output_dir=output_dir,
                seed=29,
            )
            summary = VariableLengthIslandSearch(config).run()
            self.assertGreaterEqual(summary["completed_generations"], 2)
            self.assertGreater(summary["mcts"]["simulations"], 0)
            self.assertGreater(summary["active_max_horizon"], 10)

    def test_torch_matches_cpu_for_mixed_lengths(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        trajectories = [
            Trajectory(
                self.automaton.sample_uniform(self.rng.randint(8, 16), self.rng),
                island=ISLAND_NAMES[index % 4],
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

    def test_true_restart_resets_transition_model(self) -> None:
        model = TransitionModel(self.config, self.automaton)
        original = {
            key: dict(value)
            for key, value in model.probabilities.items()
        }
        key = next(iter(model.probabilities))
        first_factor = next(iter(model.probabilities[key]))
        model.probabilities[key][first_factor] = 0.0
        model.reset_uniform()
        self.assertEqual(model.probabilities, original)

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
