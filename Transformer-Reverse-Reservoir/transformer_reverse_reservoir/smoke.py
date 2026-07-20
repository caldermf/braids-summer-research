from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from braidzero.core import BraidEnvironment

from .algebra import ReverseAlgebra
from .model import LastFactorOracle


def main() -> None:
    p = argparse.ArgumentParser(description="Reverse arithmetic and model smoke test")
    p.add_argument("--author-repo", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--length", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    env = BraidEnvironment(author_repo=args.author_repo, n=4, r=1, p=5, t_values=(1, 2, 3, 4))
    oracle = LastFactorOracle(args.checkpoint, args.calibration, env, device)
    algebra = ReverseAlgebra(env, oracle.proper_factor_ids, target_power=0)
    factors = env.sample_normal_form(args.length, random.Random(args.seed))
    residual = env.exact_evaluate(factors)
    logits = oracle.logits([residual], 1)[0]
    legal = algebra.legal_predecessors(None)
    _, ranks, _ = oracle.legal_distribution(logits, legal)
    initial_true_rank = ranks[factors[-1]]

    for prefix_length in range(len(factors), 0, -1):
        residual = algebra.remove(residual, factors[prefix_length - 1])
        expected = env.exact_evaluate(factors[: prefix_length - 1])
        if not algebra.same_projective_matrix(residual, expected):
            raise RuntimeError(f"reverse residual mismatch at prefix length {prefix_length}")
    if not algebra.is_identity(residual):
        raise RuntimeError("known-factor reverse decoding did not reach identity")

    print(json.dumps({
        "status": "clean",
        "device": str(device),
        "cuda": torch.cuda.is_available(),
        "length": args.length,
        "factors": list(factors),
        "initial_true_rank": initial_true_rank,
        "inverse_table": "passed",
        "known_path_roundtrip": "passed",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

