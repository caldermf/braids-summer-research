from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import collate_prefixes
from .metadata import sha256_file
from .metrics import brier_score, confusion_metrics
from .model_v3 import LastFactorTransformerV3, ModelV3Config
from .shards import ShardedPrefixDataset, ShardBucketBatchSampler


class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.state[key].lerp_(value.detach(), 1.0 - self.decay)
            else:
                self.state[key].copy_(value)


def autocast_context(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def run_epoch(model, loader, device, optimizer=None, scheduler=None, ema=None, descent_weight=.1):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0., "cross_entropy": 0., "brier": 0., "correct": 0., "count": 0}
    for x, mask, degrees, targets, descents, _ in loader:
        x, mask, degrees = x.to(device), mask.to(device), degrees.to(device)
        targets, descents = targets.to(device), descents.to(device)
        with torch.set_grad_enabled(training), autocast_context(device):
            logits, descent_logits = model(x, mask, degrees)
            factor_loss = F.cross_entropy(logits.float(), targets)
            loss = factor_loss
            if descent_logits is not None:
                loss = loss + descent_weight * F.binary_cross_entropy_with_logits(descent_logits.float(), descents)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
        metrics = confusion_metrics(logits.detach().float(), targets)
        n = len(targets)
        totals["loss"] += loss.item() * n
        totals["cross_entropy"] += metrics["cross_entropy"].sum().item()
        totals["brier"] += brier_score(logits.detach().float(), targets).sum().item()
        totals["correct"] += metrics["correct"].sum().item()
        totals["count"] += n
    count = totals.pop("count")
    return {key: value / count for key, value in totals.items()}


def cosine_schedule(optimizer, total_steps: int, warmup_steps: int):
    def multiplier(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def atomic_torch_save(payload, path: Path):
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Train the exact-degree v3 last-factor transformer")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=.05)
    parser.add_argument("--warmup-fraction", type=float, default=.04)
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--global-layers", type=int, default=8)
    parser.add_argument("--ffn-hidden", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=.06)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    dataset_config = manifest["config"]
    datasets = {split: ShardedPrefixDataset(args.dataset, split) for split in ("train", "validation")}
    sampler = ShardBucketBatchSampler(datasets["train"], args.batch_size, args.seed)
    collate = partial(collate_prefixes, sparse=True)
    loaders = {
        "train": DataLoader(datasets["train"], batch_sampler=sampler,
                            collate_fn=collate, num_workers=0, pin_memory=True),
        "validation": DataLoader(datasets["validation"], batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate, num_workers=0, pin_memory=True),
    }
    config = ModelV3Config(
        p=dataset_config["prime"], d_model=args.d_model, heads=args.heads,
        local_layers=args.local_layers, global_layers=args.global_layers,
        ffn_hidden=args.ffn_hidden, dropout=args.dropout,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = LastFactorTransformerV3(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(.9, .95), fused=device.type == "cuda")
    total_steps = args.epochs * len(loaders["train"])
    scheduler = cosine_schedule(optimizer, total_steps, round(total_steps * args.warmup_fraction))
    ema = EMA(model, args.ema_decay)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    history, best, start_epoch = [], float("inf"), 1
    split_metadata = {
        name: {"records": manifest["splits"][name]["records"],
               "trajectories": manifest["splits"][name]["trajectories"]}
        for name in ("train", "validation", "test", "extrapolation_test")
    }
    status_path = args.out_dir / "training_status.json"
    status_path.write_text(json.dumps({
        "schema_version": 3, "status": "truncated", "reason": "training has not completed",
        "seed": args.seed, "prime": dataset_config["prime"], "method": "last_factor_transformer_exact_degree_v3",
        "length_range": [dataset_config["splits"]["train"]["length_min"],
                         dataset_config["splits"]["train"]["length_max"]],
        "split": split_metadata, "model_config": config.as_dict(), "completed_epoch": 0,
    }, indent=2))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("architecture") != LastFactorTransformerV3.architecture:
            raise ValueError("resume checkpoint is not an exact-degree v3 checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        ema.state = checkpoint["ema_state"]
        history = checkpoint["history"]
        best = checkpoint["best_validation_cross_entropy"]
        start_epoch = checkpoint["completed_epoch"] + 1
        sampler.epoch = checkpoint["sampler_epoch"]
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        if device.type == "cuda": torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
        print(json.dumps({"resumed_from": str(args.resume), "start_epoch": start_epoch}), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], device, optimizer, scheduler, ema)
        live_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(ema.state)
        validation_metrics = run_epoch(model, loaders["validation"], device)
        model.load_state_dict(live_state)
        row = {"epoch": epoch, "learning_rate": scheduler.get_last_lr()[0],
               "train": train_metrics, "validation_ema": validation_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        common = {"architecture": model.architecture, "model_config": config.as_dict(),
                  "sparse": True, "seed": args.seed,
                  "dataset_manifest": str((args.dataset / "manifest.json").resolve())}
        if validation_metrics["cross_entropy"] < best:
            best = validation_metrics["cross_entropy"]
            atomic_torch_save({**common, "state_dict": ema.state, "completed_epoch": epoch},
                              args.out_dir / "best_model.pt")
        atomic_torch_save({
            **common, "state_dict": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "ema_state": ema.state,
            "completed_epoch": epoch, "sampler_epoch": sampler.epoch,
            "best_validation_cross_entropy": best, "history": history,
            "python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }, args.out_dir / "last_checkpoint.pt")
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))
        status_path.write_text(json.dumps({
            "schema_version": 3, "status": "truncated", "reason": "training has not completed",
            "seed": args.seed, "prime": dataset_config["prime"], "method": "last_factor_transformer_exact_degree_v3",
            "length_range": [dataset_config["splits"]["train"]["length_min"],
                             dataset_config["splits"]["train"]["length_max"]],
            "split": split_metadata, "model_config": config.as_dict(), "completed_epoch": epoch,
            "confusion_metric_summaries": {"best_validation_cross_entropy": best,
                                            "latest_validation": validation_metrics},
            "artifact_path": str((args.out_dir / "last_checkpoint.pt").resolve()),
            "verifier_version": "last-factor-confusion exact-degree-v3",
        }, indent=2))

    artifact = args.out_dir / "best_model.pt"
    run_manifest = {
        "schema_version": 3, "status": "clean", "prime": dataset_config["prime"],
        "representation": "JonesSummand(n=4,r=1)", "seed": args.seed,
        "method": "last_factor_transformer_exact_degree_v3", "model_config": config.as_dict(),
        "dataset": str(args.dataset.resolve()), "train_validation_split": "fixed trajectory-disjoint shards",
        "split": split_metadata,
        "length_range": [dataset_config["splits"]["train"]["length_min"],
                         dataset_config["splits"]["train"]["length_max"]],
        "exact_evaluations_used": split_metadata["train"]["records"] + split_metadata["validation"]["records"],
        "best_projlen": None, "best_projlen_reason": "predictor training is not a braid search",
        "confusion_metric_summaries": {"best_validation_cross_entropy": best,
                                       "latest_validation": history[-1]["validation_ema"]},
        "artifact_path": str(artifact.resolve()), "artifact_checksum": sha256_file(artifact),
        "verifier_version": "last-factor-confusion exact-degree-v3",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))
    status_path.write_text(json.dumps(run_manifest, indent=2))
    print(json.dumps(run_manifest, indent=2))


if __name__ == "__main__":
    main()
