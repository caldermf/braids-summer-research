from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from .factors import FactorTable
from .generate import descent_bits
from .representation import JonesAdapter, JonesSpec, install_peyl
from .shards import atomic_json, write_shard


def stratified_lengths(low: int, high: int, count: int, rng: random.Random) -> list[int]:
    if count > high - low + 1:
        raise ValueError("prefixes_per_trajectory exceeds the number of available lengths")
    edges = [low + ((high - low + 1) * i) // count for i in range(count + 1)]
    values = [rng.randint(edges[i], edges[i + 1] - 1) for i in range(count)]
    return sorted(values)


def main():
    parser = argparse.ArgumentParser(description="Generate atomic, resumable Jones prefix shards")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    split_config = config["splits"][args.split]
    shard_size = int(config["shard_trajectories"])
    total_trajectories = int(split_config["trajectories"])
    num_shards = math.ceil(total_trajectories / shard_size)
    if not 0 <= args.shard_index < num_shards:
        parser.error(f"shard-index must be in [0,{num_shards})")
    install_peyl(args.author_repo)
    from peyl.braid import GNF
    split_order = list(config["splits"]).index(args.split)
    seed = int(config["base_seed"]) + split_order * 1_000_000 + args.shard_index
    rng = random.Random(seed)
    spec = JonesSpec(n=config["n"], r=config["r"], p=config["prime"])
    adapter = JonesAdapter(args.author_repo, spec)
    table = FactorTable.from_peyl(config["n"])
    start = args.shard_index * shard_size
    stop = min(total_trajectories, start + shard_size)
    records = []
    for trajectory_index in range(start, stop):
        prefix_lengths = stratified_lengths(split_config["length_min"], split_config["length_max"],
                                             split_config["prefixes_per_trajectory"], rng)
        braid = GNF.sample(config["n"], max(prefix_lengths), rand=rng)
        _, factors = braid.canonical_decomposition()
        images = adapter.evaluate_prefixes([braid])
        trajectory_id = f"{args.split}-seed{seed}-trajectory{trajectory_index}"
        for length in prefix_lengths:
            factor, image = factors[length - 1], images[length][0]
            records.append({"trajectory_id": trajectory_id, "prefix_length": length,
                            "infimum": int(braid.inf()), "matrix": adapter.degree_major(image),
                            "projlen": adapter.projlen(image), "target_class": table.class_id(factor),
                            "target_descents": descent_bits(factor)})
    relative = Path("shards") / args.split / f"shard-{args.shard_index:05d}.npz"
    entry = write_shard(args.output_dir / relative, records)
    entry.update({"path": str(relative), "shard_index": args.shard_index, "trajectories": stop - start,
                  "seed": seed, "length_min": split_config["length_min"], "length_max": split_config["length_max"]})
    sidecar = args.output_dir / "shards" / args.split / f"shard-{args.shard_index:05d}.json"
    atomic_json(sidecar, entry)
    print(json.dumps(entry, indent=2))


if __name__ == "__main__": main()

