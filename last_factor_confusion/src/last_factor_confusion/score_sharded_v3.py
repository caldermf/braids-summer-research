from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import collate_prefixes
from .metrics import confusion_metrics
from .model_v3 import LastFactorTransformerV3, ModelV3Config
from .shards import ShardedPrefixDataset


def main():
    parser = argparse.ArgumentParser(description="Score a sharded split with exact-degree v3")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != LastFactorTransformerV3.architecture:
        raise ValueError("checkpoint is not an exact-degree v3 checkpoint")
    model = LastFactorTransformerV3(ModelV3Config(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    dataset = ShardedPrefixDataset(args.dataset, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=partial(collate_prefixes, sparse=True), num_workers=0, pin_memory=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    started = time.monotonic()
    processed = next_report = 0
    with temporary.open("w", encoding="utf-8") as output, torch.no_grad():
        for x, mask, degrees, targets, _, records in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, _ = model(x.to(device), mask.to(device), degrees.to(device))
            logits = logits.float()
            metrics = confusion_metrics(logits, targets.to(device))
            probabilities = logits.softmax(-1).cpu()
            for index, record in enumerate(records):
                row = {key: value[index].item() for key, value in metrics.items()}
                row.update({"trajectory_id": record["trajectory_id"],
                            "prefix_length": record["prefix_length"], "projlen": record["projlen"],
                            "target_class": record["target_class"],
                            "probabilities": probabilities[index].tolist(), "status": "clean"})
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
            processed += len(records)
            if processed >= next_report or processed == len(dataset):
                elapsed = time.monotonic() - started
                rate = processed / max(elapsed, 1e-9)
                remaining = (len(dataset) - processed) / max(rate, 1e-9)
                print(json.dumps({"split": args.split, "processed": processed, "total": len(dataset),
                                  "records_per_second": rate, "estimated_seconds_remaining": remaining}), flush=True)
                next_report = processed + args.progress_every
    temporary.replace(args.output)
    print(json.dumps({"status": "clean", "split": args.split, "records": processed,
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
