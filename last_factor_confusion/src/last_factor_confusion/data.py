from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class PrefixDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def collate_prefixes(records: list[dict], sparse: bool = False):
    matrices, degrees = [], []
    for record in records:
        x = torch.tensor(record["matrix"], dtype=torch.long)
        occupied = x.ne(0).any(dim=(-1, -2))
        if sparse:
            idx = torch.where(occupied)[0]
            x = x[idx]
        else:
            idx = torch.arange(x.shape[0])
        matrices.append(x)
        degrees.append(idx)
    max_depth = max(x.shape[0] for x in matrices)
    m = matrices[0].shape[-1]
    values = torch.zeros(len(records), max_depth, m, m, dtype=torch.long)
    positions = torch.zeros(len(records), max_depth, dtype=torch.long)
    mask = torch.zeros(len(records), max_depth, dtype=torch.bool)
    for i, (x, pos) in enumerate(zip(matrices, degrees)):
        values[i, : len(x)] = x
        positions[i, : len(pos)] = pos
        mask[i, : len(x)] = True
    labels = torch.tensor([r["target_class"] for r in records], dtype=torch.long)
    descents = torch.tensor([r["target_descents"] for r in records], dtype=torch.float32)
    return values, mask, positions, labels, descents, records

