#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


FACTOR_VOCAB_SIZE = 24
IGNORE_INDEX = -100


def load_braid_gpt_module(braid_gpt_root: Path):
    module_path = braid_gpt_root / "braid_gpt.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find Braid-GPT script at {module_path}")
    spec = importlib.util.spec_from_file_location("braid_gpt_runtime", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Braid-GPT from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def metric_projlen(metrics: dict) -> float:
    return float(metrics.get("projlen", metrics.get("projective_width", 0.0)))


def clean_metrics(metrics: dict) -> dict:
    output = dict(metrics)
    if "projlen" not in output:
        output["projlen"] = output.get("projective_width", 0)
    output.pop("projective_width", None)
    return output


def projlen_density(metrics: dict, length: int) -> float:
    return metric_projlen(metrics) / max(1, int(length))


def min_context_from_metrics(metrics: dict | None, length: int) -> np.ndarray:
    if not metrics:
        return np.zeros((1,), dtype=np.float32)
    return np.array([math.log1p(max(0.0, projlen_density(metrics, length)))], dtype=np.float32)


def objective_from_metrics(
    metrics: dict,
    length: int,
    *,
    projlen_density_weight: float,
    identity_density_weight: float,
) -> float:
    length = max(1, int(length))
    return (
        projlen_density_weight * metric_projlen(metrics) / length
        + identity_density_weight * float(metrics["identity_defect"]) / length
    )


def random_suffixes_for_action(automaton, *, first: int, lookahead: int, rollouts: int, rng: random.Random):
    if lookahead <= 1:
        return [(first,)]
    suffixes: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(max(rollouts * 4, 16)):
        factors = [first]
        while len(factors) < lookahead:
            factors.append(rng.choice(automaton.successors[factors[-1]]))
        suffix = tuple(factors)
        if suffix not in seen:
            seen.add(suffix)
            suffixes.append(suffix)
        if len(suffixes) >= rollouts:
            break
    return suffixes or [(first,)]


def soft_target(scores: np.ndarray, legal_mask: np.ndarray, temperature: float) -> np.ndarray:
    target = np.zeros((FACTOR_VOCAB_SIZE,), dtype=np.float32)
    legal_scores = scores[legal_mask]
    shifted = -(legal_scores - np.min(legal_scores)) / max(temperature, 1e-6)
    shifted -= np.max(shifted)
    probs = np.exp(shifted)
    probs /= np.sum(probs)
    target[legal_mask] = probs.astype(np.float32)
    return target


def generate_pretrain_data(args: argparse.Namespace) -> None:
    bgpt = load_braid_gpt_module(Path(args.braid_gpt_root))
    start = time.time()
    rng = random.Random(args.seed)
    automaton = bgpt.GNFAutomaton(args.n)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens = np.zeros((args.sequence_count, args.max_factors + 1), dtype=np.int16)
    labels = np.full((args.sequence_count, args.max_factors + 1), IGNORE_INDEX, dtype=np.int16)
    context = np.zeros((args.sequence_count, 1), dtype=np.float32)
    lengths = np.zeros((args.sequence_count,), dtype=np.int16)
    powers = np.zeros((args.sequence_count,), dtype=np.int16)

    for index in range(args.sequence_count):
        length = rng.randint(args.min_length, args.max_length)
        factors = automaton.sample_uniform(length, rng)
        power = rng.choice((0, 1))
        tokens[index], _ = bgpt.encode_prefix(factors, args.max_factors)
        labels[index] = bgpt.pretrain_labels(factors, args.max_factors)
        lengths[index] = length
        powers[index] = power
        if (index + 1) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "phase": "mincontext_pretrain_data",
                        "generated": index + 1,
                        "elapsed_seconds": round(time.time() - start, 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    np.savez_compressed(
        output_dir / "pretrain_dataset.npz",
        tokens=tokens,
        labels=labels,
        context=context,
        lengths=lengths,
        powers=powers,
        allowed_next=bgpt.make_allowed_next_matrix(automaton),
    )
    metadata = {
        "format": "braid-gpt-mincontext-pretrain-dataset-v1",
        "context": "log1p(projlen / max(1, Garside length)); zero during grammar pretraining",
        "n": args.n,
        "p": args.p,
        "sequence_count": args.sequence_count,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "max_factors": args.max_factors,
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    write_json(output_dir / "metadata.json", metadata)
    print(json.dumps({"phase": "done", **metadata}, sort_keys=True), flush=True)


def train_pretrain(args: argparse.Namespace) -> None:
    bgpt = load_braid_gpt_module(Path(args.braid_gpt_root))

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)
    payload = np.load(args.dataset)
    metadata_path = Path(args.dataset).parent / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    device = bgpt.resolve_device(torch, args.device)
    config = bgpt.BraidGPTConfig(
        p=int(metadata.get("p", args.p)),
        max_factors=int(payload["tokens"].shape[1] - 1),
        context_dim=int(payload["context"].shape[1]),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    model = bgpt.build_braid_gpt(torch, nn, config).to(device)
    allowed_next = torch.tensor(payload["allowed_next"], dtype=torch.bool, device=device)
    indices = np.random.default_rng(args.seed).permutation(payload["tokens"].shape[0])
    split = max(1, int(len(indices) * (1.0 - args.validation_fraction)))
    train_indices = indices[:split]
    val_indices = indices[split:] if split < len(indices) else indices[: min(1024, len(indices))]

    def loader(selected, shuffle):
        return DataLoader(
            TensorDataset(
                torch.tensor(payload["tokens"][selected], dtype=torch.long),
                torch.tensor(payload["labels"][selected], dtype=torch.long),
                torch.tensor(payload["context"][selected], dtype=torch.float32),
            ),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    train_loader = loader(train_indices, True)
    val_loader = loader(val_indices, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def run_epoch(data_loader, train: bool):
        model.train(train)
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        for batch_tokens, batch_labels, batch_context in data_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            batch_context = batch_context.to(device)
            with torch.set_grad_enabled(train):
                logits, _ = model(batch_tokens, batch_context)
                legal = allowed_next[batch_tokens]
                logits = logits.masked_fill(~legal, -1e9)
                loss = F.cross_entropy(
                    logits.reshape(-1, FACTOR_VOCAB_SIZE),
                    batch_labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                )
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
            valid = batch_labels.ne(IGNORE_INDEX)
            predictions = logits.argmax(dim=-1)
            total_correct += int((predictions.eq(batch_labels) & valid).sum().detach().cpu())
            total_tokens += int(valid.sum().detach().cpu())
            total_loss += float(loss.detach().cpu()) * int(valid.sum().detach().cpu())
        return {
            "loss": total_loss / max(1, total_tokens),
            "accuracy": total_correct / max(1, total_tokens),
            "tokens": total_tokens,
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(train_loader, True)
        val_stats = run_epoch(val_loader, False)
        row = {"epoch": epoch, "train": train_stats, "validation": val_stats}
        history.append(row)
        print(json.dumps({"phase": "mincontext_pretrain", **row}, sort_keys=True), flush=True)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            bgpt.save_checkpoint(
                torch,
                output_dir / "braid_gpt_mincontext_pretrained.pt",
                model=model,
                config=config,
                history=history,
                extra={"dataset_metadata": metadata, "stage": "mincontext_pretrain"},
            )
    write_json(output_dir / "pretrain_summary.json", {"history": history, "best_validation_loss": best_val})


def generate_policy_data(args: argparse.Namespace) -> None:
    bgpt = load_braid_gpt_module(Path(args.braid_gpt_root))
    start = time.time()
    rng = random.Random(args.seed)
    automaton = bgpt.GNFAutomaton(args.n)
    evaluator = bgpt.ExactEvaluator(
        author_repo=Path(args.author_repo),
        p=args.p,
        n=args.n,
        r=args.r,
        max_degree=args.matrix_max_degree,
        width_weight=0.0,
        min_meaningful_length=0,
        degeneracy_weight=0.0,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "examples.jsonl"
    examples_path.write_text("", encoding="utf-8")

    tokens = np.zeros((args.state_count, args.max_factors + 1), dtype=np.int16)
    action_positions = np.zeros((args.state_count,), dtype=np.int16)
    context = np.zeros((args.state_count, 1), dtype=np.float32)
    legal_masks = np.zeros((args.state_count, FACTOR_VOCAB_SIZE), dtype=bool)
    targets = np.zeros((args.state_count, FACTOR_VOCAB_SIZE), dtype=np.float32)
    labels = np.zeros((args.state_count,), dtype=np.int16)
    value_targets = np.zeros((args.state_count,), dtype=np.float32)
    parent_objectives = np.zeros((args.state_count,), dtype=np.float32)
    best_objectives = np.zeros((args.state_count,), dtype=np.float32)
    lengths = np.zeros((args.state_count,), dtype=np.int16)
    powers = np.zeros((args.state_count,), dtype=np.int16)

    log_rows: list[dict] = []
    for index in range(args.state_count):
        length = rng.randint(args.min_length, args.max_length)
        factors = automaton.sample_uniform(length, rng)
        power = rng.choice((0, 1))
        parent = evaluator.evaluate_batch([(power, factors)], batch_size=1)[0]
        parent_metrics = clean_metrics(parent.metrics)
        legal = tuple(automaton.successors[factors[-1]])
        candidate_words: list[tuple[int, tuple[int, ...]]] = []
        for action in legal:
            for suffix in random_suffixes_for_action(
                automaton,
                first=action,
                lookahead=args.lookahead,
                rollouts=args.rollouts_per_action,
                rng=rng,
            ):
                candidate_words.append((action, factors + suffix))
        evaluated = evaluator.evaluate_batch(
            [(power, candidate) for _, candidate in candidate_words],
            batch_size=args.eval_batch_size,
        )
        action_scores = np.full((FACTOR_VOCAB_SIZE,), np.inf, dtype=np.float32)
        for (action, child_factors), child in zip(candidate_words, evaluated):
            metrics = clean_metrics(child.metrics)
            value = objective_from_metrics(
                metrics,
                len(child_factors),
                projlen_density_weight=args.projlen_density_weight,
                identity_density_weight=args.identity_density_weight,
            )
            action_scores[action] = min(action_scores[action], np.float32(value))
        legal_mask = np.isfinite(action_scores)
        best_action = int(np.argmin(action_scores))
        parent_objective = objective_from_metrics(
            parent_metrics,
            len(factors),
            projlen_density_weight=args.projlen_density_weight,
            identity_density_weight=args.identity_density_weight,
        )
        token_row, action_position = bgpt.encode_prefix(factors, args.max_factors)
        tokens[index] = token_row
        action_positions[index] = action_position
        context[index] = min_context_from_metrics(parent_metrics, len(factors))
        legal_masks[index] = legal_mask
        targets[index] = soft_target(action_scores, legal_mask, args.target_temperature)
        labels[index] = best_action
        value_targets[index] = math.log1p(max(0.0, float(action_scores[best_action])))
        parent_objectives[index] = np.float32(parent_objective)
        best_objectives[index] = action_scores[best_action]
        lengths[index] = length
        powers[index] = power
        if index < args.log_examples:
            log_rows.append(
                {
                    "example_id": index,
                    "power": power,
                    "factor_ids": list(factors),
                    "length": length,
                    "context_scalar": float(context[index, 0]),
                    "parent_metrics": parent_metrics,
                    "parent_projlen_density": projlen_density(parent_metrics, len(factors)),
                    "parent_objective": float(parent_objective),
                    "best_action": best_action,
                    "best_objective": float(action_scores[best_action]),
                    "legal_actions": [int(value) for value in np.flatnonzero(legal_mask)],
                }
            )
        if (index + 1) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "phase": "mincontext_policy_data",
                        "generated": index + 1,
                        "best_objective_min": float(np.min(best_objectives[: index + 1])),
                        "elapsed_seconds": round(time.time() - start, 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    append_jsonl(examples_path, log_rows)
    np.savez_compressed(
        output_dir / "policy_dataset.npz",
        tokens=tokens,
        action_positions=action_positions,
        context=context,
        legal_masks=legal_masks,
        targets=targets,
        labels=labels,
        value_targets=value_targets,
        parent_objectives=parent_objectives,
        best_objectives=best_objectives,
        lengths=lengths,
        powers=powers,
        allowed_next=bgpt.make_allowed_next_matrix(automaton),
    )
    metadata = {
        "format": "braid-gpt-mincontext-policy-dataset-v1",
        "context": "log1p(projlen / max(1, Garside length))",
        "objective": "projlen_density_weight * projlen/length + identity_density_weight * identity_defect/length",
        "n": args.n,
        "r": args.r,
        "p": args.p,
        "state_count": args.state_count,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "max_factors": args.max_factors,
        "lookahead": args.lookahead,
        "rollouts_per_action": args.rollouts_per_action,
        "projlen_density_weight": args.projlen_density_weight,
        "identity_density_weight": args.identity_density_weight,
        "target_temperature": args.target_temperature,
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - start, 2),
        "best_objective_min": float(np.min(best_objectives)),
        "best_objective_median": float(np.median(best_objectives)),
        "label_histogram": dict(Counter(int(value) for value in labels)),
    }
    write_json(output_dir / "metadata.json", metadata)
    print(json.dumps({"phase": "done", **metadata}, sort_keys=True), flush=True)


def train_finetune(args: argparse.Namespace) -> None:
    bgpt = load_braid_gpt_module(Path(args.braid_gpt_root))

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)
    payload = np.load(args.dataset)
    metadata_path = Path(args.dataset).parent / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    device = bgpt.resolve_device(torch, args.device)
    if args.init_checkpoint:
        model, config, checkpoint = bgpt.load_checkpoint(torch, nn, Path(args.init_checkpoint), device)
        history = list(checkpoint.get("history", []))
    else:
        config = bgpt.BraidGPTConfig(
            p=int(metadata.get("p", args.p)),
            max_factors=int(payload["tokens"].shape[1] - 1),
            context_dim=int(payload["context"].shape[1]),
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        )
        model = bgpt.build_braid_gpt(torch, nn, config).to(device)
        history = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    indices = np.random.default_rng(args.seed).permutation(payload["tokens"].shape[0])
    split = max(1, int(len(indices) * (1.0 - args.validation_fraction)))
    train_indices = indices[:split]
    val_indices = indices[split:] if split < len(indices) else indices[: min(1024, len(indices))]

    def loader(selected, shuffle):
        return DataLoader(
            TensorDataset(
                torch.tensor(payload["tokens"][selected], dtype=torch.long),
                torch.tensor(payload["context"][selected], dtype=torch.float32),
                torch.tensor(payload["action_positions"][selected], dtype=torch.long),
                torch.tensor(payload["legal_masks"][selected], dtype=torch.bool),
                torch.tensor(payload["targets"][selected], dtype=torch.float32),
                torch.tensor(payload["labels"][selected], dtype=torch.long),
                torch.tensor(payload["value_targets"][selected], dtype=torch.float32),
            ),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    train_loader = loader(train_indices, True)
    val_loader = loader(val_indices, False)

    def run_epoch(data_loader, train: bool):
        model.train(train)
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_top1 = 0
        total = 0
        for batch_tokens, batch_context, positions, legal, targets, labels, values in data_loader:
            batch_tokens = batch_tokens.to(device)
            batch_context = batch_context.to(device)
            positions = positions.to(device)
            legal = legal.to(device)
            targets = targets.to(device)
            labels = labels.to(device)
            values = values.to(device)
            batch_index = torch.arange(batch_tokens.shape[0], device=device)
            with torch.set_grad_enabled(train):
                logits_all, values_all = model(batch_tokens, batch_context)
                logits = logits_all[batch_index, positions].masked_fill(~legal, -1e9)
                predicted_values = values_all[batch_index, positions]
                log_probs = F.log_softmax(logits, dim=-1)
                policy_loss = -(targets * log_probs).sum(dim=-1).mean()
                value_loss = F.mse_loss(predicted_values, values)
                loss = policy_loss + args.value_loss_weight * value_loss
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
            total_loss += float(loss.detach().cpu()) * batch_tokens.shape[0]
            total_policy += float(policy_loss.detach().cpu()) * batch_tokens.shape[0]
            total_value += float(value_loss.detach().cpu()) * batch_tokens.shape[0]
            total_top1 += int((torch.argmax(logits, dim=-1) == labels).sum().detach().cpu())
            total += batch_tokens.shape[0]
        return {
            "loss": total_loss / max(1, total),
            "policy_loss": total_policy / max(1, total),
            "value_loss": total_value / max(1, total),
            "top1": total_top1 / max(1, total),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(train_loader, True)
        val_stats = run_epoch(val_loader, False)
        row = {"epoch": epoch, "train": train_stats, "validation": val_stats}
        history.append(row)
        print(json.dumps({"phase": "mincontext_finetune", **row}, sort_keys=True), flush=True)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            bgpt.save_checkpoint(
                torch,
                output_dir / "braid_gpt_mincontext_finetuned.pt",
                model=model,
                config=config,
                history=history,
                extra={"dataset_metadata": metadata, "stage": "mincontext_finetune"},
            )
    write_json(output_dir / "finetune_summary.json", {"history": history, "best_validation_loss": best_val})


@dataclass(frozen=True)
class BeamState:
    power: int
    factors: tuple[int, ...]
    metrics: dict
    objective: float


def state_context(state: BeamState) -> np.ndarray:
    return min_context_from_metrics(state.metrics, len(state.factors))


def rank_key(state: BeamState) -> tuple:
    if not state.metrics:
        return (1, float("inf"), len(state.factors))
    return (
        0 if state.metrics.get("scalar_identity") else 1,
        state.objective,
        int(state.metrics.get("identity_defect", 10**9)),
        metric_projlen(state.metrics),
        -len(state.factors),
    )


def unique_ranked(states: Sequence[BeamState], limit: int) -> list[BeamState]:
    unique: dict[tuple[int, tuple[int, ...]], BeamState] = {}
    for state in states:
        key = (state.power % 2, state.factors)
        previous = unique.get(key)
        if previous is None or rank_key(state) < rank_key(previous):
            unique[key] = state
    return sorted(unique.values(), key=rank_key)[:limit]


def search_generate(args: argparse.Namespace) -> None:
    bgpt = load_braid_gpt_module(Path(args.braid_gpt_root))

    import torch
    import torch.nn as nn

    rng = random.Random(args.seed)
    device = bgpt.resolve_device(torch, args.device)
    model, config, checkpoint = bgpt.load_checkpoint(torch, nn, Path(args.checkpoint), device)
    automaton = bgpt.GNFAutomaton(args.n)
    evaluator = bgpt.ExactEvaluator(
        author_repo=Path(args.author_repo),
        p=args.p,
        n=args.n,
        r=args.r,
        max_degree=args.matrix_max_degree,
        width_weight=0.0,
        min_meaningful_length=0,
        degeneracy_weight=0.0,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    candidates_path = output_dir / "candidates.jsonl"
    progress_path.write_text("", encoding="utf-8")
    candidates_path.write_text("", encoding="utf-8")

    def evaluated_state(power: int, factors: Sequence[int]) -> BeamState:
        item = evaluator.evaluate_batch([(power, tuple(factors))], batch_size=1)[0]
        metrics = clean_metrics(item.metrics)
        objective = objective_from_metrics(
            metrics,
            len(factors),
            projlen_density_weight=args.projlen_density_weight,
            identity_density_weight=args.identity_density_weight,
        )
        return BeamState(power=int(power), factors=tuple(factors), metrics=metrics, objective=float(objective))

    frontier: list[BeamState] = []
    for value in args.seed_word:
        power, factors = bgpt.parse_seed_word(value)
        frontier.append(evaluated_state(power, factors))
    if args.start_mode in {"empty", "both"}:
        frontier.extend(BeamState(power, (), {}, float("inf")) for power in (0, 1))
    if args.start_mode in {"random", "both"}:
        roots: list[tuple[int, tuple[int, ...]]] = []
        while len(roots) < args.random_roots:
            length = rng.randint(args.root_min_length, args.root_max_length)
            roots.append((rng.choice((0, 1)), automaton.sample_uniform(length, rng)))
        evaluated = evaluator.evaluate_batch(roots, batch_size=args.eval_batch_size)
        for (power, factors), item in zip(roots, evaluated):
            metrics = clean_metrics(item.metrics)
            objective = objective_from_metrics(
                metrics,
                len(factors),
                projlen_density_weight=args.projlen_density_weight,
                identity_density_weight=args.identity_density_weight,
            )
            frontier.append(BeamState(power, factors, metrics, objective))
    frontier = unique_ranked(frontier, args.beam_size)
    best = unique_ranked(frontier, args.keep_best)
    seen = {(state.power % 2, state.factors) for state in frontier}
    kernel_hits: list[BeamState] = []

    for step in range(1, args.steps + 1):
        expandable = [
            state
            for state in frontier
            if len(state.factors) < min(args.max_length, config.max_factors)
        ]
        if not expandable:
            break
        tokens = np.stack([bgpt.encode_prefix(state.factors, config.max_factors)[0] for state in expandable])
        positions = np.array([len(state.factors) for state in expandable], dtype=np.int64)
        contexts = np.stack([state_context(state) for state in expandable])
        legal_masks = np.zeros((len(expandable), FACTOR_VOCAB_SIZE), dtype=bool)
        for index, state in enumerate(expandable):
            if not state.factors:
                legal_masks[index, list(automaton.first_ids)] = True
            else:
                legal_masks[index, list(automaton.successors[state.factors[-1]])] = True
        with torch.no_grad():
            logits_all, _ = model(
                torch.tensor(tokens, dtype=torch.long, device=device),
                torch.tensor(contexts, dtype=torch.float32, device=device),
            )
            batch_index = torch.arange(len(expandable), device=device)
            logits = logits_all[batch_index, torch.tensor(positions, dtype=torch.long, device=device)]
            legal_tensor = torch.tensor(legal_masks, dtype=torch.bool, device=device)
            logits = logits.masked_fill(~legal_tensor, -1e9)
            probs = torch.softmax(logits / max(args.temperature, 1e-6), dim=-1).detach().cpu().numpy()
        children: list[tuple[int, tuple[int, ...]]] = []
        for state, row in zip(expandable, probs):
            legal = np.flatnonzero(row > 0)
            ranked = sorted(legal, key=lambda action: row[action], reverse=True)[: args.actions_per_state]
            for action in ranked:
                factors = state.factors + (int(action),)
                key = (state.power % 2, factors)
                if key in seen:
                    continue
                seen.add(key)
                children.append((state.power, factors))
        evaluated = evaluator.evaluate_batch(children, batch_size=args.eval_batch_size)
        child_states: list[BeamState] = []
        for (power, factors), item in zip(children, evaluated):
            metrics = clean_metrics(item.metrics)
            objective = objective_from_metrics(
                metrics,
                len(factors),
                projlen_density_weight=args.projlen_density_weight,
                identity_density_weight=args.identity_density_weight,
            )
            child_states.append(BeamState(power, factors, metrics, float(objective)))
        append_jsonl(
            candidates_path,
            [
                {
                    "step": step,
                    "power": state.power,
                    "factor_ids": list(state.factors),
                    "length": len(state.factors),
                    "metrics": state.metrics,
                    "projlen_density": projlen_density(state.metrics, len(state.factors)),
                    "objective": state.objective,
                    "context_scalar": float(state_context(state)[0]),
                }
                for state in child_states
            ],
        )
        kernel_hits.extend(state for state in child_states if state.metrics.get("scalar_identity"))
        best = unique_ranked(best + child_states, args.keep_best)
        frontier = unique_ranked(child_states + best, args.beam_size)
        row = {
            "phase": "mincontext_generate",
            "step": step,
            "expanded": len(children),
            "frontier_size": len(frontier),
            "best_objective": best[0].objective if best else None,
            "best_length": len(best[0].factors) if best else None,
            "best_metrics": best[0].metrics if best else {},
            "best_projlen_density": projlen_density(best[0].metrics, len(best[0].factors)) if best else None,
            "kernel_hits": len(kernel_hits),
        }
        append_jsonl(progress_path, [row])
        print(json.dumps(row, sort_keys=True), flush=True)
        if kernel_hits and args.stop_at_kernel:
            break

    summary = {
        "format": "braid-gpt-mincontext-generation-summary-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_stage": checkpoint.get("stage"),
        "context": "log1p(projlen / max(1, Garside length))",
        "objective": "projlen_density_weight * projlen/length + identity_density_weight * identity_defect/length",
        "projlen_density_weight": args.projlen_density_weight,
        "identity_density_weight": args.identity_density_weight,
        "kernel_hits": [state_record(state) for state in kernel_hits[: args.keep_best]],
        "best": [state_record(state) for state in best],
        "best_by_identity_defect": [
            state_record(state)
            for state in sorted(best, key=lambda item: (item.metrics.get("identity_defect", 10**9), item.objective))[
                : args.keep_best
            ]
        ],
    }
    write_json(output_dir / "summary.json", summary)


def state_record(state: BeamState) -> dict:
    return {
        "power": state.power,
        "factor_ids": list(state.factors),
        "length": len(state.factors),
        "metrics": state.metrics,
        "projlen_density": projlen_density(state.metrics, len(state.factors)),
        "objective": state.objective,
        "context_scalar": float(state_context(state)[0]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Braid-GPT-MinContext.")
    parser.add_argument("--braid-gpt-root", default=str(Path(__file__).resolve().parents[1] / "Braid-GPT"))
    sub = parser.add_subparsers(dest="command", required=True)

    predata = sub.add_parser("pretrain-data")
    predata.add_argument("--output-dir", required=True)
    predata.add_argument("--p", type=int, default=7)
    predata.add_argument("--n", type=int, default=4)
    predata.add_argument("--sequence-count", type=int, default=1_000_000)
    predata.add_argument("--min-length", type=int, default=8)
    predata.add_argument("--max-length", type=int, default=96)
    predata.add_argument("--max-factors", type=int, default=96)
    predata.add_argument("--seed", type=int, default=1)
    predata.add_argument("--progress-every", type=int, default=50_000)
    predata.set_defaults(func=generate_pretrain_data)

    pretrain = sub.add_parser("pretrain")
    pretrain.add_argument("--dataset", required=True)
    pretrain.add_argument("--output-dir", required=True)
    pretrain.add_argument("--p", type=int, default=7)
    pretrain.add_argument("--device", default="auto")
    pretrain.add_argument("--epochs", type=int, default=20)
    pretrain.add_argument("--batch-size", type=int, default=256)
    pretrain.add_argument("--lr", type=float, default=3e-4)
    pretrain.add_argument("--weight-decay", type=float, default=0.01)
    pretrain.add_argument("--validation-fraction", type=float, default=0.02)
    pretrain.add_argument("--d-model", type=int, default=256)
    pretrain.add_argument("--nhead", type=int, default=8)
    pretrain.add_argument("--num-layers", type=int, default=8)
    pretrain.add_argument("--dim-feedforward", type=int, default=1024)
    pretrain.add_argument("--dropout", type=float, default=0.10)
    pretrain.add_argument("--grad-clip", type=float, default=1.0)
    pretrain.add_argument("--seed", type=int, default=1)
    pretrain.set_defaults(func=train_pretrain)

    poldata = sub.add_parser("policy-data")
    poldata.add_argument("--author-repo", required=True)
    poldata.add_argument("--output-dir", required=True)
    poldata.add_argument("--p", type=int, default=7)
    poldata.add_argument("--n", type=int, default=4)
    poldata.add_argument("--r", type=int, default=1)
    poldata.add_argument("--state-count", type=int, default=100_000)
    poldata.add_argument("--min-length", type=int, default=12)
    poldata.add_argument("--max-length", type=int, default=72)
    poldata.add_argument("--max-factors", type=int, default=96)
    poldata.add_argument("--lookahead", type=int, default=2)
    poldata.add_argument("--rollouts-per-action", type=int, default=4)
    poldata.add_argument("--matrix-max-degree", type=int, default=256)
    poldata.add_argument("--projlen-density-weight", type=float, default=1.0)
    poldata.add_argument("--identity-density-weight", type=float, default=0.0)
    poldata.add_argument("--target-temperature", type=float, default=0.35)
    poldata.add_argument("--eval-batch-size", type=int, default=500)
    poldata.add_argument("--seed", type=int, default=1)
    poldata.add_argument("--progress-every", type=int, default=500)
    poldata.add_argument("--log-examples", type=int, default=200)
    poldata.set_defaults(func=generate_policy_data)

    finetune = sub.add_parser("finetune")
    finetune.add_argument("--dataset", required=True)
    finetune.add_argument("--output-dir", required=True)
    finetune.add_argument("--init-checkpoint", default="")
    finetune.add_argument("--p", type=int, default=7)
    finetune.add_argument("--device", default="auto")
    finetune.add_argument("--epochs", type=int, default=20)
    finetune.add_argument("--batch-size", type=int, default=128)
    finetune.add_argument("--lr", type=float, default=1e-4)
    finetune.add_argument("--weight-decay", type=float, default=0.01)
    finetune.add_argument("--value-loss-weight", type=float, default=0.15)
    finetune.add_argument("--validation-fraction", type=float, default=0.05)
    finetune.add_argument("--d-model", type=int, default=256)
    finetune.add_argument("--nhead", type=int, default=8)
    finetune.add_argument("--num-layers", type=int, default=8)
    finetune.add_argument("--dim-feedforward", type=int, default=1024)
    finetune.add_argument("--dropout", type=float, default=0.10)
    finetune.add_argument("--grad-clip", type=float, default=1.0)
    finetune.add_argument("--seed", type=int, default=1)
    finetune.set_defaults(func=train_finetune)

    search = sub.add_parser("generate")
    search.add_argument("--author-repo", required=True)
    search.add_argument("--checkpoint", required=True)
    search.add_argument("--output-dir", required=True)
    search.add_argument("--p", type=int, default=7)
    search.add_argument("--n", type=int, default=4)
    search.add_argument("--r", type=int, default=1)
    search.add_argument("--device", default="auto")
    search.add_argument("--start-mode", choices=("empty", "random", "both"), default="both")
    search.add_argument("--random-roots", type=int, default=128)
    search.add_argument("--root-min-length", type=int, default=12)
    search.add_argument("--root-max-length", type=int, default=48)
    search.add_argument("--seed-word", action="append", default=[])
    search.add_argument("--steps", type=int, default=96)
    search.add_argument("--beam-size", type=int, default=512)
    search.add_argument("--actions-per-state", type=int, default=4)
    search.add_argument("--keep-best", type=int, default=200)
    search.add_argument("--max-length", type=int, default=96)
    search.add_argument("--temperature", type=float, default=1.0)
    search.add_argument("--eval-batch-size", type=int, default=500)
    search.add_argument("--matrix-max-degree", type=int, default=256)
    search.add_argument("--projlen-density-weight", type=float, default=1.0)
    search.add_argument("--identity-density-weight", type=float, default=0.0)
    search.add_argument("--seed", type=int, default=1)
    search.add_argument("--stop-at-kernel", action="store_true")
    search.set_defaults(func=search_generate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

