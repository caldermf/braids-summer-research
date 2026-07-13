from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Sampler

from .metadata import sha256_file


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def write_shard(path: Path, records: list[dict]) -> dict:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    matrices = []
    for record in records:
        matrix = np.asarray(record["matrix"], dtype=np.uint8)
        matrices.append(matrix)
        offsets.append(offsets[-1] + len(matrix))
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            coefficients=np.concatenate(matrices, axis=0),
            offsets=np.asarray(offsets, dtype=np.int64),
            trajectory_id=np.asarray([r["trajectory_id"] for r in records]),
            prefix_length=np.asarray([r["prefix_length"] for r in records], dtype=np.int16),
            infimum=np.asarray([r["infimum"] for r in records], dtype=np.int16),
            projlen=np.asarray([r["projlen"] for r in records], dtype=np.int16),
            target_class=np.asarray([r["target_class"] for r in records], dtype=np.uint8),
            target_descents=np.asarray([r["target_descents"] for r in records], dtype=np.uint8),
        )
    os.replace(temporary, path)
    return {"path": str(path), "records": len(records), "sha256": sha256_file(path), "status": "clean"}


class ShardedPrefixDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str):
        self.dataset_dir = Path(dataset_dir)
        self.manifest = json.loads((self.dataset_dir / "manifest.json").read_text())
        entries = self.manifest["splits"][split]["shards"]
        self.paths = [self.dataset_dir / entry["path"] for entry in entries]
        self.cumulative = np.cumsum([0] + [entry["records"] for entry in entries])
        self._shard_index = None
        self._shard = None

    def __len__(self):
        return int(self.cumulative[-1])

    def _load(self, shard_index):
        if self._shard_index != shard_index:
            if self._shard is not None:
                self._shard.close()
            self._shard = np.load(self.paths[shard_index], allow_pickle=False)
            self._shard_index = shard_index
        return self._shard

    def __getitem__(self, index):
        shard_index = int(np.searchsorted(self.cumulative, index, side="right") - 1)
        local = int(index - self.cumulative[shard_index])
        shard = self._load(shard_index)
        start, end = shard["offsets"][local:local + 2]
        return {
            "matrix": shard["coefficients"][start:end].tolist(),
            "target_class": int(shard["target_class"][local]),
            "target_descents": shard["target_descents"][local].tolist(),
            "trajectory_id": str(shard["trajectory_id"][local]),
            "prefix_length": int(shard["prefix_length"][local]),
            "projlen": int(shard["projlen"][local]),
            "status": "clean",
        }


class ShardShuffleSampler(Sampler):
    """Shuffle shards and records within shards while retaining sequential I/O locality."""
    def __init__(self, dataset: ShardedPrefixDataset, seed: int):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def __len__(self): return len(self.dataset)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch); self.epoch += 1
        shard_order = rng.permutation(len(self.dataset.paths))
        for shard in shard_order:
            start, stop = self.dataset.cumulative[shard:shard + 2]
            yield from (int(start + x) for x in rng.permutation(stop - start))
