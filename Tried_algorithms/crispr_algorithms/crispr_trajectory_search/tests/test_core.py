from __future__ import annotations

import random
import unittest
from unittest.mock import patch

from crispr_trajectory_search.config import SearchConfig
from crispr_trajectory_search.evaluators import (
    CPUTrajectoryEvaluator,
    TorchTrajectoryEvaluator,
)
from crispr_trajectory_search.gnf import GNFAutomaton
from crispr_trajectory_search.known_examples import KNOWN_P5_LENGTH54_FACTOR_IDS
from crispr_trajectory_search.models import Trajectory
from crispr_trajectory_search.mutation import MutationPlanner
from crispr_trajectory_search.transition_model import TransitionModel


class TrajectorySearchTests(unittest.TestCase):
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

    def test_block_mutations_remain_legal(self) -> None:
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
        for _ in range(100):
            child = planner.make_child(
                evaluation,
                two_mutations=self.rng.random() < 0.25,
            )
            self.assertEqual(len(child.factor_ids), len(parent.factor_ids))
            self.assertTrue(self.automaton.is_legal(child.factor_ids))

    def test_known_p5_kernel_is_detected(self) -> None:
        factors = KNOWN_P5_LENGTH54_FACTOR_IDS
        config = SearchConfig(
            p=5,
            horizons=(len(factors),),
            population_size=1,
            generations=1,
        )
        evaluation = CPUTrajectoryEvaluator(config).evaluate_one(
            Trajectory(factors, trajectory_id="known-p5")
        )
        self.assertTrue(evaluation.has_kernel)
        self.assertEqual(evaluation.kernel_depths, (54,))
        self.assertEqual(evaluation.final_projlen, 0)

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
