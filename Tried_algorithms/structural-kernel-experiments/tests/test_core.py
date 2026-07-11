from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for path in (REPO_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crispr_transformer.checkpoint import load_checkpoint, load_checkpoints
from crispr_transformer.edits import apply_geometry, valid_geometries
from crispr_transformer.exact import CPUExactEvaluator
from crispr_transformer.gnf import GNFAutomaton
from crispr_transformer.io_utils import write_json
from structural_experiments.audit import known_p5_factor_ids
from structural_experiments.commutator_exact import (
    CPUCommutatorEvaluator,
    TorchCommutatorEvaluator,
    commutator_artin_word,
    commutator_is_nontrivial,
)
from structural_experiments.datta import analyze_factor_ids
from structural_experiments.minimal_form import gnf_from_positive_artin_word


class StructuralCoreTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(17)
        self.automaton = GNFAutomaton(4)

    def test_known_p5_kernel_and_datta_descriptor(self):
        factors = known_p5_factor_ids()
        evaluation = CPUExactEvaluator(p=5, n=4).evaluate_one(factors)
        analysis = analyze_factor_ids(factors)
        self.assertEqual(len(factors), 54)
        self.assertTrue(evaluation.has_kernel)
        self.assertEqual(evaluation.final_projlen, 0)
        self.assertEqual(len(analysis.defects), 8)

    def test_minimal_form_respects_braid_relation(self):
        left = gnf_from_positive_artin_word((1, 2, 1))
        right = gnf_from_positive_artin_word((2, 1, 2))
        self.assertEqual(left, right)

    def test_commutator_word_and_trivial_commutator_filter(self):
        # Factor 6 is sigma_1 in the lexicographic permutation table.
        factors = (6,)
        self.assertEqual(commutator_artin_word(factors, 1), (1, -1, -1, 1))
        self.assertFalse(commutator_is_nontrivial(factors, 1))
        evaluation = CPUCommutatorEvaluator(
            p=5, n=4, generator_index=1
        ).evaluate_one(factors)
        self.assertEqual(evaluation.final_projlen, 8)
        self.assertFalse(evaluation.has_kernel)

    def test_commutator_evaluator_tracks_all_prefixes(self):
        factors = self.automaton.sample_uniform(4, self.rng)
        evaluation = CPUCommutatorEvaluator(
            p=5, n=4, generator_index=2
        ).evaluate_one(factors)
        self.assertEqual(len(evaluation.projlen_history), len(factors))
        self.assertTrue(all(value >= 0 for value in evaluation.projlen_history))

    def test_batched_commutator_evaluator_matches_exact_reference(self):
        words = [
            self.automaton.sample_uniform(self.rng.randint(2, 5), self.rng)
            for _ in range(8)
        ]
        exact = CPUCommutatorEvaluator(
            p=5, n=4, generator_index=1
        ).evaluate(words)
        batched = TorchCommutatorEvaluator(
            p=5,
            n=4,
            generator_index=1,
            device="cpu",
            batch_size=8,
            max_length=8,
        ).evaluate(words)
        self.assertEqual(
            [item.projlen_history for item in exact],
            [item.projlen_history for item in batched],
        )

    def test_variable_edits_preserve_legal_gnf(self):
        parent = self.automaton.sample_uniform(12, self.rng)
        checked = 0
        for geometry in valid_geometries(
            12,
            min_length=9,
            max_length=15,
            max_delete=6,
            max_insert=6,
            max_net_delta=3,
        ):
            try:
                child = apply_geometry(parent, geometry, self.automaton, self.rng)
            except (ValueError, RuntimeError):
                continue
            self.assertTrue(self.automaton.is_legal(child))
            self.assertEqual(len(child), 12 + geometry.length_delta)
            checked += 1
            if checked == 100:
                break
        self.assertEqual(checked, 100)

    def test_structural_checkpoint_preserves_objective(self):
        word = self.automaton.sample_uniform(4, self.rng)
        payload = {
            "format": "commutator-reservoir-checkpoint-v2",
            "metadata": {
                "p": 5,
                "n": 4,
                "r": 1,
                "actual_depth": 4,
                "objective": "commutator_projlen",
                "generator_index": 2,
            },
            "candidates": [
                {
                    "depth": 4,
                    "power": 0,
                    "factor_ids": list(word),
                    "author_projlen": 9,
                    "matrix_fingerprint": "test",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = write_json(Path(directory) / "checkpoint.json.gz", payload)
            metadata, candidates = load_checkpoint(path)
            common, merged = load_checkpoints([path])
        self.assertEqual(metadata["objective"], "commutator_projlen")
        self.assertEqual(common["generator_index"], 2)
        self.assertEqual(candidates[0].factor_ids, word)
        self.assertEqual(merged[0].factor_ids, word)


if __name__ == "__main__":
    unittest.main()
