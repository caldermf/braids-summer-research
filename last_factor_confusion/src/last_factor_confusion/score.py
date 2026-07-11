from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import PrefixDataset, collate_prefixes, read_jsonl
from .metrics import confusion_metrics
from .model import LastFactorTransformer, ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = LastFactorTransformer(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    records = read_jsonl(args.data)
    loader = DataLoader(PrefixDataset(records), batch_size=args.batch_size, shuffle=False,
        collate_fn=partial(collate_prefixes, sparse=checkpoint["sparse"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out, torch.no_grad():
        for x, mask, degrees, targets, _, batch_records in loader:
            logits, _ = model(x.to(device), mask.to(device), degrees.to(device))
            metrics = confusion_metrics(logits, targets.to(device))
            probs = logits.softmax(-1).cpu()
            for i, record in enumerate(batch_records):
                result = {key: value[i].item() for key, value in metrics.items()}
                result.update({"trajectory_id": record["trajectory_id"],
                               "prefix_length": record["prefix_length"], "projlen": record["projlen"],
                               "target_class": record["target_class"],
                               "probabilities": probs[i].tolist(), "status": record["status"]})
                out.write(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()

