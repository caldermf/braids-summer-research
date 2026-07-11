from __future__ import annotations

import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .exact import require_compatible_cuda
from .io_utils import read_json, read_jsonl, write_json
from .model import GeometryTransformer, ModelConfig


class MutationGroupDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_groups(rows: list[dict]):
    width = max(len(row["factor_ids"]) for row in rows)
    tokens = torch.zeros(len(rows), width, dtype=torch.long)
    histories = torch.zeros(len(rows), width, dtype=torch.float32)
    lengths = torch.empty(len(rows), dtype=torch.long)
    action_parents = []
    actions = []
    targets = []
    offsets = [0]
    for parent_index, row in enumerate(rows):
        length = len(row["factor_ids"])
        lengths[parent_index] = length
        tokens[parent_index, :length] = torch.tensor(row["factor_ids"], dtype=torch.long) + 1
        histories[parent_index, :length] = torch.tensor(
            row["projlen_history"], dtype=torch.float32
        )
        for action in row["actions"]:
            action_parents.append(parent_index)
            actions.append(
                (
                    int(action["start"]),
                    int(action["delete_length"]),
                    int(action["insert_length"]),
                )
            )
            targets.append(float(action["target_reward"]))
        offsets.append(len(actions))
    return {
        "tokens": tokens,
        "histories": histories,
        "lengths": lengths,
        "action_parents": torch.tensor(action_parents, dtype=torch.long),
        "actions": torch.tensor(actions, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.float32),
        "offsets": offsets,
    }


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _loss(scores, targets, offsets, target_temperature: float):
    listwise = []
    regression = []
    for left, right in zip(offsets, offsets[1:]):
        if right - left < 2:
            continue
        group_targets = targets[left:right]
        target_range = torch.max(group_targets) - torch.min(group_targets)
        normalized = (group_targets - torch.mean(group_targets)) / torch.clamp(
            target_range,
            min=1e-8,
        )
        target_distribution = F.softmax(normalized / target_temperature, dim=0)
        listwise.append(
            -(target_distribution * F.log_softmax(scores[left:right], dim=0)).sum()
        )
        regression.append(
            F.smooth_l1_loss(torch.tanh(scores[left:right]), normalized)
        )
    ranking_loss = torch.stack(listwise).mean()
    regression_loss = torch.stack(regression).mean()
    return ranking_loss + 0.25 * regression_loss, ranking_loss, regression_loss


@torch.no_grad()
def _metrics(model, loader, device, target_temperature):
    model.eval()
    losses = []
    chosen = []
    oracle = []
    random_rewards = []
    rng = random.Random(1447)
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        scores = model(
            batch["tokens"],
            batch["histories"],
            batch["lengths"],
            batch["action_parents"],
            batch["actions"],
        )
        loss, _, _ = _loss(
            scores, batch["targets"], batch["offsets"], target_temperature
        )
        losses.append(float(loss))
        for left, right in zip(batch["offsets"], batch["offsets"][1:]):
            group_scores = scores[left:right]
            group_targets = batch["targets"][left:right]
            chosen.append(float(group_targets[int(torch.argmax(group_scores))]))
            oracle.append(float(torch.max(group_targets)))
            random_rewards.append(float(group_targets[rng.randrange(right - left)]))
    mean = lambda values: sum(values) / max(1, len(values))
    return {
        "loss": mean(losses),
        "chosen_reward": mean(chosen),
        "oracle_reward": mean(oracle),
        "random_reward": mean(random_rewards),
        "regret": mean([best - value for best, value in zip(oracle, chosen)]),
        "beneficial_choice_rate": mean([value > 0 for value in chosen]),
        "groups": len(chosen),
    }


def train_geometry_model(
    *,
    dataset_path: str | Path,
    dataset_summary_path: str | Path,
    output_dir: str | Path,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    validation_fraction: float = 0.15,
    target_temperature: float = 0.20,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 512,
    dropout: float = 0.10,
    device: str = "cuda",
    seed: int = 1,
) -> dict:
    started = time.perf_counter()
    if device.startswith("cuda"):
        require_compatible_cuda(torch)
    torch.manual_seed(seed)
    random.seed(seed)
    target_device = torch.device(device)
    rows = list(read_jsonl(dataset_path))
    summary = read_json(dataset_summary_path)
    if len(rows) < 4:
        raise ValueError("training requires at least four mutation groups")
    rng = random.Random(seed)
    rng.shuffle(rows)
    validation_count = max(1, round(len(rows) * validation_fraction))
    validation_rows = rows[:validation_count]
    training_rows = rows[validation_count:]

    bounds = summary["bounds"]
    config = ModelConfig(
        p=int(summary["p"]),
        n=int(summary["n"]),
        factor_vocab_size=math.factorial(int(summary["n"])),
        max_length=max(int(bounds["max_length"]), max(len(row["factor_ids"]) for row in rows)),
        max_delete=int(bounds["max_delete"]),
        max_insert=int(bounds["max_insert"]),
        max_net_delta=int(bounds["max_net_delta"]),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    model = GeometryTransformer(config).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    training_loader = DataLoader(
        MutationGroupDataset(training_rows),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_groups,
    )
    validation_loader = DataLoader(
        MutationGroupDataset(validation_rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_groups,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_regret = float("inf")
    best_path = output / f"geometry_transformer_p{config.p}.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for raw_batch in training_loader:
            batch = _move(raw_batch, target_device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(
                batch["tokens"],
                batch["histories"],
                batch["lengths"],
                batch["action_parents"],
                batch["actions"],
            )
            loss, _, _ = _loss(
                scores,
                batch["targets"],
                batch["offsets"],
                target_temperature,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        validation = _metrics(
            model, validation_loader, target_device, target_temperature
        )
        row = {
            "epoch": epoch,
            "training_loss": sum(epoch_losses) / max(1, len(epoch_losses)),
            "validation": validation,
        }
        history.append(row)
        print(row, flush=True)
        if validation["regret"] < best_regret:
            best_regret = validation["regret"]
            torch.save(
                {
                    "format": "crispr-transformer-geometry-model-v1",
                    "model_config": config.to_dict(),
                    "model_state": model.state_dict(),
                    "dataset_summary": str(Path(dataset_summary_path).resolve()),
                    "best_epoch": epoch,
                    "validation": validation,
                },
                best_path,
            )
    result = {
        "format": "crispr-transformer-training-run-v1",
        "p": config.p,
        "n": config.n,
        "model": str(best_path),
        "model_config": config.to_dict(),
        "training_groups": len(training_rows),
        "validation_groups": len(validation_rows),
        "epochs": epochs,
        "best_regret": best_regret,
        "history": history,
        "device": device,
        "seed": seed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output / "training_summary.json", result)
    return result
