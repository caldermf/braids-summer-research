from __future__ import annotations

import unittest

from peyl.braid_data import simple_factor_id_maps

from annealed_reservoir_search.verification import verify_author_candidates


KNOWN_P5_LENGTH65 = (
    7, 7, 10, 13, 4, 13, 4, 13, 21, 20, 13, 20, 13, 10, 2, 13, 4,
    13, 4, 13, 21, 20, 13, 20, 13, 10, 2, 13, 4, 13, 4, 13, 21, 20,
    13, 20, 13, 10, 2, 13, 4, 13, 4, 13, 21, 20, 13, 20, 13, 10, 2,
    13, 4, 13, 4, 13, 21, 20, 13, 20, 13, 10, 16, 16, 16,
)


class VerificationTests(unittest.TestCase):
    def test_known_p5_candidate_is_exactly_verified(self):
        _, id_to_permutation = simple_factor_id_maps(4)
        record = {
            "depth": len(KNOWN_P5_LENGTH65),
            "power": 0,
            "factor_permutations": [
                list(id_to_permutation[factor_id])
                for factor_id in KNOWN_P5_LENGTH65
            ],
            "author_projlen": 1,
            "matrix_fingerprint": "author-state",
        }
        result = verify_author_candidates(
            {"p": 5, "n": 4},
            [record],
        )
        self.assertEqual(result["unique_candidates_verified"], 1)
        self.assertEqual(len(result["kernel_hits"]), 1)
        self.assertEqual(result["kernel_hits"][0]["final_projlen"], 0)


if __name__ == "__main__":
    unittest.main()
