from __future__ import annotations

import argparse
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import PrefixDataset, collate_prefixes, read_jsonl
from .metrics import brier_score, confusion_metrics
from .metadata import sha256_file, validate_manifest
from .model import LastFactorTransformer, ModelConfig


def grouped_split(records, seed, train_fraction=.8, validation_fraction=.1):
    ids = sorted({r["trajectory_id"] for r in records})
    random.Random(seed).shuffle(ids)
    a = round(len(ids) * train_fraction)
    b = round(len(ids) * (train_fraction + validation_fraction))
    groups = {"train": set(ids[:a]), "validation": set(ids[a:b]), "test": set(ids[b:])}
    if any(not value for value in groups.values()):
        raise ValueError("Need enough trajectories for nonempty train/validation/test splits")
    return {name: [r for r in records if r["trajectory_id"] in selected] for name, selected in groups.items()}, groups


def run_epoch(model, loader, device, optimizer=None, desc_weight=.1):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0., "cross_entropy": 0., "brier": 0., "correct": 0., "count": 0}
    for x, mask, degrees, targets, descents, _ in loader:
        x, mask, degrees = x.to(device), mask.to(device), degrees.to(device)
        targets, descents = targets.to(device), descents.to(device)
        with torch.set_grad_enabled(training):
            logits, desc_logits = model(x, mask, degrees)
            factor_loss = F.cross_entropy(logits, targets)
            loss = factor_loss
            if desc_logits is not None:
                loss = loss + desc_weight * F.binary_cross_entropy_with_logits(desc_logits, descents)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
        metrics = confusion_metrics(logits.detach(), targets)
        n = len(targets)
        totals["loss"] += loss.item() * n
        totals["cross_entropy"] += metrics["cross_entropy"].sum().item()
        totals["brier"] += brier_score(logits.detach(), targets).sum().item()
        totals["correct"] += metrics["correct"].sum().item()
        totals["count"] += n
    n = totals.pop("count")
    return {key: value / n for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--global-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=.08)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--no-descents", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    records = read_jsonl(args.data)
    splits, groups = grouped_split(records, args.seed)
    p = int(records[0]["prime"])
    classes = int(records[0].get("num_target_classes", 22))
    config = ModelConfig(p=p, num_classes=classes, d_model=args.d_model, heads=args.heads,
                         local_layers=args.local_layers, global_layers=args.global_layers,
                         dropout=args.dropout, auxiliary_descents=not args.no_descents)
    device = torch.device(args.device)
    model = LastFactorTransformer(config).to(device)
    collate = partial(collate_prefixes, sparse=args.sparse)
    loaders = {name: DataLoader(PrefixDataset(data), batch_size=args.batch_size,
               shuffle=name == "train", collate_fn=collate, num_workers=0)
               for name, data in splits.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        row = {"epoch": epoch, "train": run_epoch(model, loaders["train"], device, optimizer),
               "validation": run_epoch(model, loaders["validation"], device)}
        history.append(row); print(json.dumps(row), flush=True)
        if row["validation"]["cross_entropy"] < best:
            best = row["validation"]["cross_entropy"]
            torch.save({"state_dict": model.state_dict(), "model_config": config.as_dict(),
                        "sparse": args.sparse, "seed": args.seed,
                        "split_trajectory_ids": {k: sorted(v) for k, v in groups.items()}},
                       args.out_dir / "best_model.pt")
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = torch.load(args.out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test = run_epoch(model, loaders["test"], device)
    (args.out_dir / "test_metrics.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
    artifact = args.out_dir / "best_model.pt"
    manifest = {
        "schema_version": 1,
        "prime": p,
        "representation": records[0]["representation"],
        "seed": args.seed,
        "method": "last_factor_transformer_sparse" if args.sparse else "last_factor_transformer_dense",
        "length_range": [min(r["prefix_length"] for r in records), max(r["prefix_length"] for r in records)],
        "split": {name: {"trajectory_ids": sorted(ids), "records": len(splits[name])} for name, ids in groups.items()},
        "model_config": config.as_dict(),
        "exact_evaluations": len(records),
        "best_projlen": min(r["projlen"] for r in records),
        "confusion_summary": {"best_validation_cross_entropy": best, "test": test},
        "artifact_path": str(artifact.resolve()),
        "artifact_checksum": sha256_file(artifact),
        "verifier_version": "peyl.JonesSummand adapter / last-factor-confusion 0.1.0",
        "status": "clean",
    }
    validate_manifest(manifest)
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"best_validation_cross_entropy": best, "test": test}, indent=2))


if __name__ == "__main__":
    main()
