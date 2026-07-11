from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for path in (REPO_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

from crispr_transformer.checkpoint import load_checkpoint
from crispr_transformer.braid_data import simple_factor_id_maps
from crispr_transformer.dataset import generate_mutation_dataset
from crispr_transformer.edits import apply_geometry, valid_geometries
from crispr_transformer.exact import CPUExactEvaluator
from crispr_transformer.gnf import GNFAutomaton
from crispr_transformer.io_utils import write_json
from crispr_transformer.model import GeometryTransformer, ModelConfig
from crispr_transformer.percentiles import LengthPercentiles
from crispr_transformer.repair import run_guided_repair
from crispr_transformer.training import train_geometry_model


KNOWN_P5_LENGTH54 = (
    7, 7, 10, 13, 4, 13, 4, 2, 13, 20, 13, 20, 13, 10, 2, 13, 4, 13,
    4, 13, 7, 21, 20, 13, 20, 13, 10, 16, 16, 2, 13, 4, 13, 4, 13, 21,
    20, 13, 20, 13, 21, 10, 13, 4, 13, 4, 2, 16, 13, 11, 13, 11, 13, 21,
)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(17)
        self.automaton = GNFAutomaton(4)

    def _checkpoint(self, path: Path, words, p: int = 3):
        _, id_to_perm = simple_factor_id_maps(4)
        records = []
        evaluator = CPUExactEvaluator(p=p, n=4)
        for word in words:
            evaluation = evaluator.evaluate_one(word)
            records.append(
                {
                    "depth": len(word),
                    "power": 0,
                    "factor_permutations": [list(id_to_perm[value]) for value in word],
                    "author_projlen": evaluation.final_projlen + 1,
                    "matrix_fingerprint": str(hash(word)),
                }
            )
        payload = {
            "format": "paper-tracker-reservoir-run-v1",
            "metadata": {"p": p, "n": 4, "r": 1, "actual_depth": len(words[0])},
            "candidates": records,
            "kernel_candidates": [],
        }
        return write_json(path, payload)

    def test_every_integer_edit_size_is_reachable_and_legal(self):
        parent = self.automaton.sample_uniform(12, self.rng)
        geometries = valid_geometries(
            len(parent),
            min_length=9,
            max_length=15,
            max_delete=6,
            max_insert=6,
            max_net_delta=3,
        )
        self.assertEqual({item.delete_length for item in geometries}, set(range(1, 7)))
        self.assertEqual({item.insert_length for item in geometries}, set(range(1, 7)))
        checked = 0
        for geometry in self.rng.sample(geometries, min(500, len(geometries))):
            try:
                child = apply_geometry(parent, geometry, self.automaton, self.rng)
            except RuntimeError:
                continue
            self.assertTrue(self.automaton.is_legal(child))
            self.assertEqual(len(child), len(parent) + geometry.length_delta)
            checked += 1
            if checked == 100:
                break
        self.assertEqual(checked, 100)

    def test_percentile_reward_does_not_reward_shortening_by_itself(self):
        baseline = LengthPercentiles.from_samples(
            p=5,
            n=4,
            samples={10: [8, 10, 12, 14], 9: [7, 9, 11, 13]},
        )
        self.assertAlmostEqual(baseline.reward(10, 10, 9, 9), 0.0)
        self.assertGreater(baseline.reward(10, 10, 9, 7), 0.0)

    def test_known_p5_kernel_is_exactly_verified(self):
        evaluation = CPUExactEvaluator(p=5, n=4).evaluate_one(KNOWN_P5_LENGTH54)
        self.assertTrue(evaluation.has_kernel)
        self.assertEqual(evaluation.final_projlen, 0)

    def test_model_scores_variable_geometries(self):
        config = ModelConfig(
            p=3,
            max_length=16,
            max_delete=6,
            max_insert=6,
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            action_dim=8,
        )
        model = GeometryTransformer(config).eval()
        parent = self.automaton.sample_uniform(12, self.rng)
        history = CPUExactEvaluator(p=3, n=4).evaluate_one(parent).projlen_history
        tokens = torch.tensor([[value + 1 for value in parent]])
        histories = torch.tensor([history], dtype=torch.float32)
        lengths = torch.tensor([len(parent)])
        actions = torch.tensor([[0, 1, 2], [4, 3, 1], [6, 2, 2]])
        scores = model(
            tokens,
            histories,
            lengths,
            torch.zeros(3, dtype=torch.long),
            actions,
        )
        self.assertEqual(tuple(scores.shape), (3,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_tiny_dataset_train_and_repair_pipeline(self):
        words = [self.automaton.sample_uniform(8, self.rng) for _ in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self._checkpoint(root / "frontier.json.gz", words)
            metadata, loaded = load_checkpoint(checkpoint)
            self.assertEqual(metadata["p"], 3)
            self.assertEqual(len(loaded), 8)
            dataset_dir = root / "dataset"
            summary = generate_mutation_dataset(
                checkpoints=[checkpoint],
                output_dir=dataset_dir,
                parents_limit=8,
                actions_per_parent=4,
                replacements_per_action=2,
                max_delete=4,
                max_insert=4,
                max_net_delta=2,
                baseline_samples_per_length=4,
                backend="cpu",
                device="cpu",
                seed=5,
            )
            self.assertGreaterEqual(summary["parent_groups_written"], 4)
            training = train_geometry_model(
                dataset_path=dataset_dir / "mutation_groups.jsonl.gz",
                dataset_summary_path=dataset_dir / "dataset_summary.json",
                output_dir=root / "model",
                epochs=1,
                batch_size=4,
                d_model=32,
                nhead=4,
                num_layers=1,
                dim_feedforward=64,
                device="cpu",
                seed=5,
            )
            result = run_guided_repair(
                checkpoints=[checkpoint],
                baseline_path=dataset_dir / "length_percentiles.json",
                model_path=training["model"],
                mode="guided",
                output_dir=root / "repair",
                population_size=4,
                generations=1,
                actions_per_parent=2,
                replacements_per_action=1,
                backend="cpu",
                device="cpu",
                seed=5,
            )
            self.assertGreater(result["unique_evaluations"], 4)
            self.assertTrue((root / "repair" / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
