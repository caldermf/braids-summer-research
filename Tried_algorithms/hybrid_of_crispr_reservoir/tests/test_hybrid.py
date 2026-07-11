from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from peyl.braid_data import simple_factor_id_maps

from hybrid_of_crispr_reservoir.checkpoint import (
    load_checkpoint,
    verify_author_kernel_candidates,
)
from hybrid_of_crispr_reservoir.config import profile_config
from hybrid_of_crispr_reservoir.crispr_repair import repair_tail_metrics
from hybrid_of_crispr_reservoir.io_utils import write_json
from hybrid_of_crispr_reservoir.run_hybrid import (
    _config,
    _run_conditional_crispr,
    build_parser,
)


KNOWN_P5_LENGTH54 = (
    7, 7, 10, 13, 4, 13, 4, 2, 13, 20, 13, 20, 13, 10, 2, 13, 4, 13,
    4, 13, 7, 21, 20, 13, 20, 13, 10, 16, 16, 2, 13, 4, 13, 4, 13, 21,
    20, 13, 20, 13, 21, 10, 13, 4, 13, 4, 2, 16, 13, 11, 13, 11, 13, 21,
)


class HybridTests(unittest.TestCase):
    def test_cluster_defaults_to_requested_depth_sixty(self):
        config = profile_config("cluster")
        self.assertEqual(config.reservoir.target_depth, 60)
        self.assertEqual(config.crispr_max_depth, 80)

    def test_paper_source_is_available(self):
        config = profile_config("smoke")
        self.assertTrue((config.author_repo / "peyl" / "braidsearch.py").is_file())

    def test_cli_can_select_paper_control_depth_sixty_five(self):
        args = build_parser().parse_args(
            [
                "all",
                "--profile",
                "cluster",
                "--reservoir-depth",
                "65",
                "--crispr-max-depth",
                "80",
            ]
        )
        config = _config(args)
        self.assertEqual(config.reservoir.target_depth, 65)

    def test_known_kernel_checkpoint_is_exactly_verified(self):
        _, id_to_perm = simple_factor_id_maps(4)
        record = {
            "depth": len(KNOWN_P5_LENGTH54),
            "power": 0,
            "factor_permutations": [
                list(id_to_perm[factor_id]) for factor_id in KNOWN_P5_LENGTH54
            ],
            "author_projlen": 1,
            "matrix_fingerprint": "paper-state",
        }
        payload = {
            "format": "paper-tracker-reservoir-run-v1",
            "metadata": {
                "n": 4,
                "r": 1,
                "p": 5,
                "actual_depth": len(KNOWN_P5_LENGTH54),
            },
            "kernel_candidates": [record],
            "candidates": [record],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = write_json(Path(directory) / "checkpoint.json.gz", payload)
            metadata, frontier, suspected = load_checkpoint(path)
            verification = verify_author_kernel_candidates(
                metadata,
                frontier,
                suspected,
            )
        self.assertEqual(len(verification["kernel_hits"]), 1)

    def test_repair_metrics_reward_late_collapse(self):
        flat = repair_tail_metrics((20, 20, 20, 20, 20), base_depth=1)
        collapse = repair_tail_metrics((20, 23, 19, 12, 5), base_depth=1)
        self.assertGreater(collapse["max_drawdown"], flat["max_drawdown"])
        self.assertGreater(collapse["terminal_slope"], flat["terminal_slope"])
        self.assertLess(
            collapse["terminal_weighted_area"],
            flat["terminal_weighted_area"],
        )

    def test_crispr_is_skipped_without_claiming_crispr_hits(self):
        verification = {"kernel_hits": [{"depth": 65}]}
        with tempfile.TemporaryDirectory() as directory:
            status, result = _run_conditional_crispr(
                SimpleNamespace(force_crispr=False),
                profile_config("cluster"),
                {"actual_depth": 65, "p": 5, "n": 4},
                [],
                verification,
                Path(directory),
            )
        self.assertEqual(status, "reservoir_kernel_found")
        self.assertNotIn("kernel_hits", result)
        self.assertEqual(result["reservoir_kernel_hits"], [{"depth": 65}])


if __name__ == "__main__":
    unittest.main()
