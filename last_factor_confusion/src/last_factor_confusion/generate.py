from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .factors import FactorTable
from .representation import JonesAdapter, JonesSpec, install_peyl


def descent_bits(perm) -> list[int]:
    word = tuple(perm.word)
    right = [int(word[i] > word[i + 1]) for i in range(len(word) - 1)]
    inverse = [0] * len(word)
    for i, value in enumerate(word):
        inverse[value] = i
    left = [int(inverse[i] > inverse[i + 1]) for i in range(len(word) - 1)]
    return left + right


def generate(args) -> dict:
    install_peyl(args.author_repo)
    from peyl.braid import GNF

    rng = random.Random(args.seed)
    spec = JonesSpec(n=args.n, r=args.r, p=args.p)
    adapter = JonesAdapter(args.author_repo, spec)
    table = FactorTable.from_peyl(args.n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = exact_evaluations = 0
    with args.output.open("w", encoding="utf-8") as out:
        for trajectory_id in range(args.trajectories):
            length = rng.randint(args.length_min, args.length_max)
            braid = GNF.sample(args.n, length, rand=rng)
            _, factors = braid.canonical_decomposition()
            images = adapter.evaluate_prefixes([braid])
            for prefix_length in range(args.prefix_min, length + 1, args.prefix_stride):
                factor = factors[prefix_length - 1]
                image = images[prefix_length][0]
                record = {
                    "schema_version": 1,
                    "trajectory_id": f"seed{args.seed}-trajectory{trajectory_id}",
                    "prefix_length": prefix_length,
                    "infimum": int(braid.inf()),
                    "prime": args.p,
                    "representation": spec.name,
                    "sampler": "peyl.GNF.sample",
                    "matrix": adapter.degree_major(image),
                    "projlen": adapter.projlen(image),
                    "target_class": table.class_id(factor),
                    "num_target_classes": len(table.permutations),
                    "target_permutation": list(factor.word),
                    "target_descents": descent_bits(factor),
                    "factor_table_checksum": table.checksum(),
                    "status": "clean",
                }
                out.write(json.dumps(record, separators=(",", ":")) + "\n")
                count += 1
                exact_evaluations += 1
    factor_path = args.output.with_suffix(args.output.suffix + ".factors.json")
    factor_path.write_text(json.dumps(table.as_dict(), indent=2), encoding="utf-8")
    return {"records": count, "trajectories": args.trajectories, "exact_evaluations": exact_evaluations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--trajectories", type=int, required=True)
    parser.add_argument("--length-min", type=int, required=True)
    parser.add_argument("--length-max", type=int, required=True)
    parser.add_argument("--prefix-min", type=int, default=1)
    parser.add_argument("--prefix-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if not (1 <= args.prefix_min <= args.length_min <= args.length_max):
        parser.error("require 1 <= prefix-min <= length-min <= length-max")
    print(json.dumps(generate(args), indent=2))


if __name__ == "__main__":
    main()
