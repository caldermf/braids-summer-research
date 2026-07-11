from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bidirectional_matrix_search_v5.config import SearchConfig
from bidirectional_matrix_search_v5.exact_evaluator import CPUExactEvaluator, TorchExactEvaluator
from bidirectional_matrix_search_v5.field_sketch import ExtensionFieldSketch
from bidirectional_matrix_search_v5.gnf import GNFAutomaton
from bidirectional_matrix_search_v5.known_examples import (
    KNOWN_P3_LENGTH24_FACTOR_IDS,
    KNOWN_P5_LENGTH54_FACTOR_IDS,
)
from bidirectional_matrix_search_v5.models import (
    JoinCandidate,
    JoinEvaluation,
    Segment,
    WordEvaluation,
)
from bidirectional_matrix_search_v5.operators import SegmentMutator
from bidirectional_matrix_search_v5.search import BidirectionalMatrixSearch
from bidirectional_matrix_search_v5.suffix_index import SuffixLSHIndex


def test_config(**updates) -> SearchConfig:
    values = {
        "p": 5,
        "n": 4,
        "prefix_count": 20,
        "suffix_count": 80,
        "generations": 1,
        "prefix_length_min": 8,
        "prefix_length_max": 12,
        "suffix_length_min": 6,
        "suffix_length_max": 10,
        "field_points": 4,
        "lsh_tables": 6,
        "lsh_key_components": 2,
        "max_lsh_candidates": 64,
        "join_candidates_per_prefix": 3,
        "elite_pairs": 12,
        "refinement_pairs": 4,
        "refinement_trials": 3,
        "signature_batch_size": 64,
        "exact_batch_size": 64,
        "backend": "cpu",
        "device": "cpu",
        "resume_latest": False,
    }
    values.update(updates)
    return SearchConfig(**values)


class BidirectionalV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = test_config(field_points=8)
        cls.automaton = GNFAutomaton(4)
        cls.sketch = ExtensionFieldSketch(cls.config)

    def test_known_kernel_suffix_matches_inverse_delta_at_many_splits(self) -> None:
        for split in (20, 27, 33, 40):
            prefix = Segment(
                KNOWN_P5_LENGTH54_FACTOR_IDS[:split],
                "prefix",
                f"p{split}",
            )
            suffix = Segment(
                KNOWN_P5_LENGTH54_FACTOR_IDS[split:],
                "suffix",
                f"s{split}",
            )
            targets = self.sketch.prefix_target_signatures([prefix])
            signature = self.sketch.suffix_signatures([suffix])
            self.assertEqual(
                int(self.sketch.distance(targets["delta"], signature)[0]),
                0,
            )
            self.assertGreater(
                int(self.sketch.distance(targets["identity"], signature)[0]),
                0,
            )

    def test_known_p3_suffix_matches_inverse_identity(self) -> None:
        config = test_config(p=3, field_points=6)
        sketch = ExtensionFieldSketch(config)
        for split in (8, 12, 16):
            prefix = Segment(
                KNOWN_P3_LENGTH24_FACTOR_IDS[:split],
                "prefix",
                f"p3-p{split}",
            )
            suffix = Segment(
                KNOWN_P3_LENGTH24_FACTOR_IDS[split:],
                "suffix",
                f"p3-s{split}",
            )
            targets = sketch.prefix_target_signatures([prefix])
            signature = sketch.suffix_signatures([suffix])
            self.assertEqual(
                int(sketch.distance(targets["identity"], signature)[0]),
                0,
            )
            self.assertGreater(
                int(sketch.distance(targets["delta"], signature)[0]),
                0,
            )

    def test_lsh_recovers_exact_suffix_with_legal_boundary(self) -> None:
        rng = random.Random(9)
        split = 27
        prefix = Segment(KNOWN_P5_LENGTH54_FACTOR_IDS[:split], "prefix", "known-p")
        known_suffix = Segment(
            KNOWN_P5_LENGTH54_FACTOR_IDS[split:],
            "suffix",
            "known-s",
        )
        suffixes = [
            Segment(
                self.automaton.sample_suffix(rng.randint(10, 28), rng),
                "suffix",
                f"s{index}",
            )
            for index in range(300)
        ] + [known_suffix]
        signatures = self.sketch.suffix_signatures(suffixes)
        target = self.sketch.prefix_target_signatures([prefix])["delta"][0]
        index = SuffixLSHIndex(self.config, suffixes, signatures, rng)
        matches = index.query(
            target,
            self.automaton.successors[prefix.factor_ids[-1]],
            4,
        )
        self.assertEqual(matches[0], (len(suffixes) - 1, 0))

    def test_exact_cpu_and_tensor_evaluators_agree(self) -> None:
        cpu = CPUExactEvaluator(self.config).evaluate_one(
            KNOWN_P5_LENGTH54_FACTOR_IDS
        )
        tensor = TorchExactEvaluator(self.config).evaluate(
            [KNOWN_P5_LENGTH54_FACTOR_IDS]
        )[0]
        self.assertEqual(cpu.projlen_history, tensor.projlen_history)
        self.assertEqual(cpu.final_projlen, 0)
        self.assertTrue(cpu.has_kernel)
        self.assertTrue(tensor.has_kernel)

    def test_mutations_preserve_role_legality_and_bounds(self) -> None:
        rng = random.Random(14)
        mutator = SegmentMutator(self.config, self.automaton, rng)
        for role in ("prefix", "suffix"):
            parent = Segment(mutator.random_factors(role), role, f"{role}-parent")
            for index in range(200):
                child = mutator.mutate(parent, f"{role}-{index}")
                if role == "prefix":
                    self.assertTrue(self.automaton.is_legal_prefix(child.factor_ids))
                    bounds = (
                        self.config.prefix_length_min,
                        self.config.prefix_length_max,
                    )
                else:
                    self.assertTrue(
                        self.automaton.is_internally_legal(child.factor_ids)
                    )
                    bounds = (
                        self.config.suffix_length_min,
                        self.config.suffix_length_max,
                    )
                self.assertLessEqual(bounds[0], child.length)
                self.assertGreaterEqual(bounds[1], child.length)
            neighbors = mutator.local_neighbors(
                parent,
                30,
                lambda value: f"{value}-neighbor-{rng.random()}",
            )
            self.assertTrue(neighbors)
            for neighbor in neighbors:
                if role == "prefix":
                    self.assertTrue(
                        self.automaton.is_legal_prefix(neighbor.factor_ids)
                    )
                else:
                    self.assertTrue(
                        self.automaton.is_internally_legal(neighbor.factor_ids)
                    )

    def test_absolute_projlen_dominates_join_rank(self) -> None:
        prefix = Segment((7,), "prefix", "p")
        suffix = Segment((7,), "suffix", "s")
        lower = JoinEvaluation(
            JoinCandidate(0, 0, "delta", 100),
            prefix,
            suffix,
            WordEvaluation((7, 7), (3, 10), 10, 3, 10, 0),
        )
        higher = JoinEvaluation(
            JoinCandidate(0, 0, "delta", 0),
            prefix,
            suffix,
            WordEvaluation((7, 7), (3, 11), 11, 3, 11, 0),
        )
        self.assertGreater(lower.rank(), higher.rank())
        self.assertGreater(higher.algebra_rank(), lower.algebra_rank())

    def test_elite_selection_preserves_length_niches_and_both_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            search = BidirectionalMatrixSearch(
                test_config(
                    output_dir=temporary,
                    prefix_count=4,
                    suffix_count=8,
                    elite_pairs=4,
                    length_niche_width=4,
                    resume_latest=False,
                )
            )
            evaluations = []
            for index, horizon in enumerate((16, 17, 24, 25, 32, 33)):
                prefix_length = horizon // 2
                prefix = Segment(
                    self.automaton.sample_prefix(prefix_length, random.Random(index)),
                    "prefix",
                    f"p{index}",
                )
                suffix = Segment(
                    self.automaton.sample_suffix(
                        horizon - prefix_length,
                        random.Random(100 + index),
                    ),
                    "suffix",
                    f"s{index}",
                )
                evaluations.append(
                    JoinEvaluation(
                        JoinCandidate(0, 0, "delta", 60 - 8 * index),
                        prefix,
                        suffix,
                        WordEvaluation(
                            prefix.factor_ids + suffix.factor_ids,
                            tuple(range(1, horizon + 1)),
                            40 - index,
                            1,
                            horizon,
                            0,
                        ),
                    )
                )
            evaluations.sort(key=JoinEvaluation.rank, reverse=True)
            combined, algebra = search._select_elites(evaluations)
            niches = {
                len(item.factor_ids) // search.config.length_niche_width
                for item in combined
            }
            self.assertGreaterEqual(len(niches), 3)
            self.assertTrue(algebra)

    def test_small_search_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first_config = test_config(
                output_dir=temporary,
                generations=1,
                resume_latest=False,
                seed=21,
            )
            first = BidirectionalMatrixSearch(first_config).run()
            self.assertEqual(first["completed_generations"], 1)
            self.assertTrue((Path(first["run_dir"]) / "checkpoint.pkl.gz").exists())

            second_config = test_config(
                output_dir=temporary,
                generations=2,
                resume_latest=True,
                seed=21,
            )
            second = BidirectionalMatrixSearch(second_config).run()
            self.assertEqual(second["completed_generations"], 2)
            self.assertEqual(first["run_dir"], second["run_dir"])

    def test_end_to_end_join_finds_known_kernel_when_halves_are_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = test_config(
                output_dir=temporary,
                prefix_count=4,
                suffix_count=20,
                generations=1,
                prefix_length_min=20,
                prefix_length_max=30,
                suffix_length_min=20,
                suffix_length_max=30,
                resume_latest=False,
                seed=44,
            )
            search = BidirectionalMatrixSearch(config)
            split = 27
            known_prefix = Segment(
                KNOWN_P5_LENGTH54_FACTOR_IDS[:split],
                "prefix",
                "known-prefix",
            )
            known_suffix = Segment(
                KNOWN_P5_LENGTH54_FACTOR_IDS[split:],
                "suffix",
                "known-suffix",
            )
            search.prefixes[0] = known_prefix
            search.suffixes[0] = known_suffix
            summary = search.run()
            self.assertEqual(summary["stop_reason"], "kernel_found")
            self.assertEqual(summary["num_kernel_hits"], 1)
            self.assertEqual(summary["best"]["factor_ids"], list(KNOWN_P5_LENGTH54_FACTOR_IDS))

    def test_signature_distance_is_zero_only_for_equal_rows(self) -> None:
        left = np.array([[0, 1, 2, 3]], dtype=np.uint8)
        right = np.array([[0, 1, 2, 3], [0, 1, 4, 3]], dtype=np.uint8)
        distance = self.sketch.distance(left, right)
        self.assertEqual(distance.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
