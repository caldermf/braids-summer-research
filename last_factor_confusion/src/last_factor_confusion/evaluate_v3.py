from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import collate_prefixes
from .model_v3 import LastFactorTransformerV3, ModelV3Config
from .shards import ShardedPrefixDataset
from .train_v3 import run_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != LastFactorTransformerV3.architecture:
        raise ValueError("checkpoint is not an exact-degree v3 checkpoint")
    model = LastFactorTransformerV3(ModelV3Config(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    results = {}
    for split in ("validation", "test", "extrapolation_test"):
        dataset = ShardedPrefixDataset(args.dataset, split)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=partial(collate_prefixes, sparse=True), num_workers=0, pin_memory=True)
        results[split] = run_epoch(model, loader, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
