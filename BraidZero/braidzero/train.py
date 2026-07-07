from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from .model import (
    BraidZeroModelConfig,
    BraidZeroTransformer,
    factors_to_tokens,
    save_checkpoint,
)


class TelemetryDataset(Dataset):
    def __init__(self, path: Path, *, max_len: int, limit: int = 0):
        rows = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "parent_factors" not in row or "action" not in row:
                    continue
                rows.append(row)
                if limit > 0 and len(rows) >= limit:
                    break
        if not rows:
            raise ValueError(f"no usable training rows found in {path}")
        self.rows = rows
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        parent = [int(x) for x in row["parent_factors"]]
        scalar_hits = int(row.get("scalar_suffix_hits", 0))
        collision_hits = int(row.get("collision_hits", 0))
        parent_projlen = int(row.get("parent_projlen", 1))
        parent_defect = int(row.get("parent_identity_defect", 0))
        bank_length = int(row.get("bank_length", 0))
        features = torch.tensor(
            [
                len(parent) / 512.0,
                parent_projlen / 256.0,
                parent_defect / 512.0,
                math.log1p(scalar_hits) / 10.0,
                math.log1p(collision_hits) / 10.0,
                bank_length / 256.0,
            ],
            dtype=torch.float32,
        )
        target_values = torch.tensor(
            [
                math.log1p(scalar_hits),
                math.log1p(collision_hits),
                1.0 if scalar_hits > 0 else 0.0,
                1.0 if collision_hits > 0 else 0.0,
            ],
            dtype=torch.float32,
        )
        weight = 0.25 + math.log1p(scalar_hits + collision_hits)
        if row.get("exact_scalar_identity"):
            weight += 5.0
        if row.get("exact_collision"):
            weight += 5.0
        return {
            "tokens": torch.tensor(factors_to_tokens(parent, max_len=self.max_len), dtype=torch.long),
            "features": features,
            "action": torch.tensor(int(row["action"]), dtype=torch.long),
            "targets": target_values,
            "weight": torch.tensor(weight, dtype=torch.float32),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "tokens": torch.stack([item["tokens"] for item in batch]),
        "features": torch.stack([item["features"] for item in batch]),
        "action": torch.stack([item["action"] for item in batch]),
        "targets": torch.stack([item["targets"] for item in batch]),
        "weight": torch.stack([item["weight"] for item in batch]),
    }


def run_epoch(model, loader, optimizer, device: torch.device, train: bool) -> dict:
    model.train(train)
    totals = {"loss": 0.0, "action_loss": 0.0, "value_loss": 0.0, "count": 0}
    for batch in loader:
        tokens = batch["tokens"].to(device)
        features = batch["features"].to(device)
        actions = batch["action"].to(device)
        targets = batch["targets"].to(device)
        weights = batch["weight"].to(device)

        with torch.set_grad_enabled(train):
            logits, values = model(tokens, features)
            action_loss_raw = F.cross_entropy(logits, actions, reduction="none")
            action_loss = (action_loss_raw * weights).sum() / weights.sum().clamp_min(1.0)
            reg_loss = F.mse_loss(values[:, :2], targets[:, :2])
            cls_loss = F.binary_cross_entropy_with_logits(values[:, 2:], targets[:, 2:])
            value_loss = reg_loss + cls_loss
            loss = action_loss + 0.5 * value_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        batch_count = int(tokens.shape[0])
        totals["loss"] += float(loss.detach().cpu()) * batch_count
        totals["action_loss"] += float(action_loss.detach().cpu()) * batch_count
        totals["value_loss"] += float(value_loss.detach().cpu()) * batch_count
        totals["count"] += batch_count
    return {key: value / max(1, totals["count"]) for key, value in totals.items() if key != "count"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the BraidZero policy/value transformer.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = TelemetryDataset(Path(args.data), max_len=args.max_len, limit=args.limit)
    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = max(1, len(dataset) - val_size)
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    config = BraidZeroModelConfig(
        max_len=args.max_len,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
    )
    model = BraidZeroTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    log_path = out_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(out_dir / "best.pt", model, extra={"epoch": epoch, "best_val_loss": best_val})

    save_checkpoint(out_dir / "last.pt", model, extra={"epoch": args.epochs, "best_val_loss": best_val})


if __name__ == "__main__":
    main()
