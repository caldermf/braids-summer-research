#!/usr/bin/env python3
"""
Verify saved MCTS kernel hits against the Burau matrix calculation.

The search records projective hits: Burau(beta) is a monomial scalar times
either I or Burau(Delta). For B_4, Delta^2 maps to v^8 I, so a projective hit is
an actual kernel element after multiplying by the indicated Delta power exactly
when the scalar is v^(8m) with coefficient 1.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peyl.braid_data import (
    burau_mod_p_polynomial_matrix,
    projective_kernel_match,
)


def central_correction(match):
    scalar = match.get("scalar")
    if not scalar or len(scalar) != 1:
        return None
    exp, coeff = next(iter(scalar.items()))
    exp = int(exp)
    coeff = int(coeff)
    if coeff != 1 or exp % 8 != 0:
        return None

    half_twists_from_scalar = exp // 4
    if match.get("kernel_type") == "identity":
        return -half_twists_from_scalar
    if match.get("kernel_type") == "delta":
        return -(half_twists_from_scalar + 1)
    return None


def verify_hit(hit, p, n):
    word = hit.get("artin_word")
    if word is None:
        raise ValueError("Hit does not contain artin_word")

    matrix = burau_mod_p_polynomial_matrix(word, p=p, n=n)
    match = projective_kernel_match(matrix, p=p, n=n)
    correction = central_correction(match)
    return {
        "garside_length": hit.get("garside_length"),
        "artin_length": len(word),
        "matches_projectively": bool(match.get("matches")),
        "kernel_type": match.get("kernel_type"),
        "scalar": match.get("scalar"),
        "central_delta_correction": correction,
        "is_actual_kernel_after_correction": correction is not None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Verify MCTS kernel_hits.json")
    parser.add_argument("kernel_hits_json", help="Path to kernel_hits.json")
    parser.add_argument("--p", type=int, required=True, help="Modulus p")
    parser.add_argument("--n", type=int, default=4, help="Number of strands")
    parser.add_argument("--show-first", type=int, default=3, help="Number of verified hits to print")
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.kernel_hits_json)
    hits = json.loads(path.read_text())
    verified = [verify_hit(hit, p=args.p, n=args.n) for hit in hits]
    actual = [item for item in verified if item["is_actual_kernel_after_correction"]]

    summary = {
        "path": str(path),
        "p": args.p,
        "n": args.n,
        "num_hits": len(hits),
        "num_projective_matches_reverified": sum(item["matches_projectively"] for item in verified),
        "num_actual_kernel_after_central_correction": len(actual),
        "min_garside_length_actual": min((item["garside_length"] for item in actual), default=None),
        "first_actual_hits": actual[: args.show_first],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
