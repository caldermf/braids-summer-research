from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from crispr_algorithms.bidirectional_matrix_search_v5.gnf import GNFAutomaton
except ModuleNotFoundError:
    from bidirectional_matrix_search_v5.gnf import GNFAutomaton
from peyl.braid_data import simple_factor_id_maps

from hybrid_of_reservoir_crispr_mcts_suffix.candidates import (
    Candidate,
    load_author_checkpoint,
    select_diverse_candidates,
)
from hybrid_of_reservoir_crispr_mcts_suffix.config import profile_config
from hybrid_of_reservoir_crispr_mcts_suffix.exact import ExactEngine
from hybrid_of_reservoir_crispr_mcts_suffix.io_utils import write_json
from hybrid_of_reservoir_crispr_mcts_suffix.run_hybrid import _config, build_parser
from hybrid_of_reservoir_crispr_mcts_suffix.suffix_lookup_branch import (
    _generate_suffixes,
)


class HybridCoreTests(unittest.TestCase):
    def test_smoke_profile_has_short_horizon(self):
        config = profile_config("smoke")
        self.assertEqual(config.backbone.target_depth, 3)
        self.assertEqual(config.max_depth, 5)

    def test_paper_peyl_dependency_is_vendored(self):
        config = profile_config("smoke")
        self.assertTrue((config.author_repo / "peyl" / "braidsearch.py").is_file())
        self.assertTrue((config.author_repo / "LICENSE").is_file())

    def test_cli_can_override_prime_and_depths(self):
        args = build_parser().parse_args(
            [
                "prepare",
                "--profile",
                "cluster",
                "--p",
                "7",
                "--backbone-depth",
                "40",
                "--max-depth",
                "55",
            ]
        )
        config = _config(args)
        self.assertEqual(config.backbone.p, 7)
        self.assertEqual(config.backbone.target_depth, 40)
        self.assertEqual(config.max_depth, 55)

    def test_author_checkpoint_maps_factor_permutations(self):
        perm_to_id, id_to_perm = simple_factor_id_maps(4)
        factor_ids = (next(iter(perm_to_id.values())),)
        payload = {
            "format": "paper-tracker-frontier-v1",
            "metadata": {"n": 4, "p": 5, "target_depth": 1},
            "candidates": [
                {
                    "power": 0,
                    "factor_permutations": [list(id_to_perm[factor_ids[0]])],
                    "author_projlen": 2,
                    "matrix_fingerprint": "abc",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = write_json(Path(directory) / "checkpoint.json.gz", payload)
            _, candidates = load_author_checkpoint(path)
        self.assertEqual(candidates[0].factor_ids, factor_ids)

    def test_diversity_prefers_distinct_matrix_states(self):
        candidates = [
            Candidate((1, 2), 3, "same", 0),
            Candidate((1, 3), 3, "same", 1),
            Candidate((1, 4), 3, "different", 2),
        ]
        selected = select_diverse_candidates(candidates, 2, seed=1)
        self.assertEqual(len({item.matrix_fingerprint for item in selected}), 2)

    def test_diversity_exhausts_lower_projlen_first(self):
        candidates = [
            Candidate((1, 2), 3, "low-a", 0),
            Candidate((1, 3), 3, "low-b", 1),
            Candidate((1, 4), 4, "high", 2),
        ]
        selected = select_diverse_candidates(candidates, 2, seed=1)
        self.assertEqual([item.author_projlen for item in selected], [3, 3])

    def test_exact_engine_increment_matches_full_evaluation(self):
        automaton = GNFAutomaton(4)
        word = automaton.sample_prefix(4, __import__("random").Random(1))
        engine = ExactEngine(5, 4)
        full = engine.evaluate(word)
        state = engine.identity()
        for factor_id in word:
            state = engine.extend(state, factor_id)
        self.assertEqual(state.projlen_history, full.projlen_history)
        self.assertEqual(state.matrix_fingerprint, full.matrix_fingerprint)

    def test_generated_suffixes_are_internal_gnf_words(self):
        automaton = GNFAutomaton(4)
        suffixes = _generate_suffixes(
            automaton,
            length=3,
            count=100,
            rng=__import__("random").Random(2),
        )
        self.assertEqual(len(suffixes), len(set(suffixes)))
        self.assertTrue(all(automaton.is_internally_legal(word) for word in suffixes))


if __name__ == "__main__":
    unittest.main()
