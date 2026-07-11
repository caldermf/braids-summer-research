#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
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


REPO_ROOT = Path(__file__).resolve().parents[1]
STRUCTURAL_ROOT = REPO_ROOT / "structural-kernel-experiments"
DEFAULT_AUTHOR_REPO = STRUCTURAL_ROOT / "third_party" / "braids_project"
if str(STRUCTURAL_ROOT) not in sys.path:
    sys.path.insert(0, str(STRUCTURAL_ROOT))

from crispr_transformer.gnf import GNFAutomaton  # noqa: E402


def read_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_seed_word(value: str) -> tuple[int, tuple[int, ...]]:
    if ":" not in value:
        raise ValueError("seed words must have form POWER:f1,f2,...")
    power_text, factors_text = value.split(":", 1)
    return int(power_text), parse_int_list(factors_text)


def setup_author_imports(author_repo: Path):
    if not (author_repo / "peyl" / "braid.py").exists():
        raise FileNotFoundError(f"vendored peyl package is missing at {author_repo}")
    if str(author_repo) not in sys.path:
        sys.path.insert(0, str(author_repo))

    from peyl import polymat  # type: ignore
    from peyl.braid import GNF  # type: ignore
    from peyl.jonesrep import JonesCellRep  # type: ignore

    class PeylNamespace:
        pass

    PeylNamespace.JonesSummand = JonesCellRep
    PeylNamespace.GNF = GNF

    def evaluate_braids(rep, braids):
        indices_by_length: dict[int, list[int]] = {}
        index_location = []
        for index, braid in enumerate(braids):
            length = braid.canonical_length()
            bucket = indices_by_length.setdefault(length, [])
            index_location.append((length, len(bucket)))
            bucket.append(index)
        images_by_length = {
            length: rep.polymat_evaluate_braids_of_same_length(
                [braids[index] for index in indices]
            )
            for length, indices in indices_by_length.items()
        }
        return [images_by_length[length][local_index] for length, local_index in index_location]

    return PeylNamespace, polymat, evaluate_braids


def scalar_identity_metrics(polymat_module, image: np.ndarray) -> dict:
    projected = polymat_module.projectivise(image)
    width = int(projected.shape[-1])
    matrix_count = int(np.count_nonzero(projected))
    dim = int(projected.shape[0])
    diagonal = np.stack([projected[i, i, :] for i in range(dim)])
    scalar_poly = diagonal[0]
    diagonal_mismatch_terms = int(np.count_nonzero(diagonal - scalar_poly[None, :]))
    off_diagonal_terms = 0
    for row in range(dim):
        for column in range(dim):
            if row != column:
                off_diagonal_terms += int(np.count_nonzero(projected[row, column, :]))
    scalar_nonzero_degrees = int(np.count_nonzero(scalar_poly))
    scalar_extra_degrees = max(0, scalar_nonzero_degrees - 1)
    scalar_zero_penalty = 1 if scalar_nonzero_degrees == 0 else 0
    identity_defect = (
        off_diagonal_terms
        + diagonal_mismatch_terms
        + scalar_extra_degrees
        + scalar_zero_penalty
    )
    return {
        "projective_width": width,
        "scalar_identity": identity_defect == 0,
        "identity_defect": int(identity_defect),
        "off_diagonal_terms": int(off_diagonal_terms),
        "diagonal_mismatch_terms": int(diagonal_mismatch_terms),
        "scalar_nonzero_degrees": int(scalar_nonzero_degrees),
        "scalar_extra_degrees": int(scalar_extra_degrees),
        "nonzero_terms": matrix_count,
    }


def degeneracy_features(factors: Sequence[int]) -> dict:
    if not factors:
        return {
            "dominant_fraction": 0.0,
            "top_two_fraction": 0.0,
            "max_run_fraction": 0.0,
            "max_run_length": 0,
            "unique_fraction": 0.0,
            "repeated_bigram_fraction": 0.0,
            "period_at_most_2": False,
        }
    counts = Counter(factors)
    ordered_counts = sorted(counts.values(), reverse=True)
    dominant_fraction = ordered_counts[0] / len(factors)
    top_two_fraction = sum(ordered_counts[:2]) / len(factors)
    max_run = 1
    run = 1
    for left, right in zip(factors, factors[1:]):
        if left == right:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    period_at_most_2 = False
    if len(factors) >= 4:
        period_at_most_2 = any(
            all(factors[i] == factors[i % period] for i in range(len(factors)))
            for period in (1, 2)
        )
    bigrams = Counter(zip(factors, factors[1:]))
    repeated_bigram_fraction = (
        max(bigrams.values()) / max(1, len(factors) - 1) if bigrams else 0.0
    )
    return {
        "dominant_fraction": float(dominant_fraction),
        "top_two_fraction": float(top_two_fraction),
        "max_run_fraction": float(max_run / len(factors)),
        "max_run_length": int(max_run),
        "unique_fraction": float(len(counts) / len(factors)),
        "repeated_bigram_fraction": float(repeated_bigram_fraction),
        "period_at_most_2": bool(period_at_most_2),
    }


def score_metrics(
    metrics: dict,
    factors: Sequence[int],
    *,
    width_weight: float,
    min_meaningful_length: int,
    degeneracy_weight: float,
) -> float:
    degeneracy = degeneracy_features(factors)
    penalty = 0.0
    if len(factors) < min_meaningful_length:
        penalty += (min_meaningful_length - len(factors)) * 5.0
    penalty += max(0.0, degeneracy["dominant_fraction"] - 0.45) * 180.0
    penalty += max(0.0, degeneracy["top_two_fraction"] - 0.70) * 180.0
    penalty += max(0.0, degeneracy["max_run_fraction"] - 0.25) * 160.0
    penalty += max(0.0, degeneracy["max_run_length"] - 3) * 18.0
    penalty += max(0.0, 0.35 - degeneracy["unique_fraction"]) * 160.0
    penalty += max(0.0, degeneracy["repeated_bigram_fraction"] - 0.20) * 120.0
    if degeneracy["period_at_most_2"]:
        penalty += 40.0
    if metrics.get("scalar_identity"):
        penalty -= 10_000.0
    return (
        float(metrics["identity_defect"])
        + width_weight * float(metrics["projective_width"])
        + degeneracy_weight * penalty
    )


def normalized_context_features(
    *,
    power: int,
    factors: Sequence[int],
    metrics: dict,
    score: float,
) -> np.ndarray:
    degeneracy = degeneracy_features(factors)
    return np.array(
        [
            float(power % 2),
            min(len(factors), 256) / 256.0,
            math.log1p(max(0.0, float(metrics["identity_defect"]))) / 8.0,
            min(float(metrics["projective_width"]), 512.0) / 512.0,
            min(float(metrics["off_diagonal_terms"]), 512.0) / 512.0,
            min(float(metrics["diagonal_mismatch_terms"]), 256.0) / 256.0,
            min(float(metrics["scalar_nonzero_degrees"]), 256.0) / 256.0,
            max(-4.0, min(4.0, float(score) / 256.0)),
            float(degeneracy["dominant_fraction"]),
            float(degeneracy["max_run_fraction"]),
            1.0 if degeneracy["period_at_most_2"] else 0.0,
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class EvaluatedWord:
    power: int
    factors: tuple[int, ...]
    metrics: dict
    score: float
    tensor: np.ndarray


class ExactEvaluator:
    def __init__(
        self,
        *,
        author_repo: Path,
        p: int,
        n: int,
        r: int,
        max_degree: int,
        width_weight: float,
        min_meaningful_length: int,
        degeneracy_weight: float,
    ) -> None:
        peyl, polymat_module, evaluate_braids = setup_author_imports(author_repo)
        self.peyl = peyl
        self.polymat = polymat_module
        self.evaluate_braids = evaluate_braids
        self.rep = peyl.JonesSummand(n=n, r=r, p=p)
        self.p = int(p)
        self.n = int(n)
        self.r = int(r)
        self.dim = int(self.rep.dimension())
        self.max_degree = int(max_degree)
        self.width_weight = float(width_weight)
        self.min_meaningful_length = int(min_meaningful_length)
        self.degeneracy_weight = float(degeneracy_weight)

    def image_to_tensor(self, image: np.ndarray) -> np.ndarray:
        projected = self.polymat.projectivise(image) % self.p
        width = min(projected.shape[-1], self.max_degree)
        tensor = np.zeros((self.max_degree, self.dim, self.dim), dtype=np.uint8)
        tensor[:width, :, :] = np.transpose(projected[:, :, :width], (2, 0, 1)).astype(
            np.uint8,
            copy=False,
        )
        return tensor

    def evaluate_batch(
        self, words: Sequence[tuple[int, Sequence[int]]], *, batch_size: int
    ) -> list[EvaluatedWord]:
        output: list[EvaluatedWord] = []
        for start in range(0, len(words), batch_size):
            chunk = words[start : start + batch_size]
            braids = [
                self.peyl.GNF(n=self.n, power=int(power), factors=tuple(factors))
                for power, factors in chunk
            ]
            images = self.evaluate_braids(self.rep, braids)
            for (power, factors), image in zip(chunk, images):
                factors_tuple = tuple(int(value) for value in factors)
                metrics = scalar_identity_metrics(self.polymat, image)
                score = score_metrics(
                    metrics,
                    factors_tuple,
                    width_weight=self.width_weight,
                    min_meaningful_length=self.min_meaningful_length,
                    degeneracy_weight=self.degeneracy_weight,
                )
                output.append(
                    EvaluatedWord(
                        power=int(power),
                        factors=factors_tuple,
                        metrics=metrics,
                        score=score,
                        tensor=self.image_to_tensor(image),
                    )
                )
        return output


def random_suffixes_for_action(
    automaton: GNFAutomaton,
    *,
    first: int,
    lookahead: int,
    max_rollouts: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    if lookahead <= 1:
        return [(first,)]
    suffixes: list[tuple[int, ...]] = []
    attempts = max(max_rollouts * 4, 16)
    seen: set[tuple[int, ...]] = set()
    for _ in range(attempts):
        factors = [first]
        while len(factors) < lookahead:
            factors.append(rng.choice(automaton.successors[factors[-1]]))
        suffix = tuple(factors)
        if suffix not in seen:
            seen.add(suffix)
            suffixes.append(suffix)
        if len(suffixes) >= max_rollouts:
            break
    return suffixes or [(first,)]


def soft_target_from_scores(
    scores: np.ndarray, legal_mask: np.ndarray, temperature: float
) -> np.ndarray:
    legal_scores = scores[legal_mask]
    shifted = -(legal_scores - np.min(legal_scores)) / max(temperature, 1e-6)
    shifted -= np.max(shifted)
    probs = np.exp(shifted)
    probs /= np.sum(probs)
    target = np.zeros_like(scores, dtype=np.float32)
    target[legal_mask] = probs.astype(np.float32)
    return target


def generate_dataset(args: argparse.Namespace) -> None:
    start_time = time.time()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "examples.jsonl"
    examples_path.write_text("", encoding="utf-8")

    automaton = GNFAutomaton(args.n)
    evaluator = ExactEvaluator(
        author_repo=Path(args.author_repo),
        p=args.p,
        n=args.n,
        r=args.r,
        max_degree=args.max_degree,
        width_weight=args.width_weight,
        min_meaningful_length=args.min_meaningful_length,
        degeneracy_weight=args.degeneracy_weight,
    )

    tensors = np.zeros(
        (args.state_count, args.max_degree, evaluator.dim, evaluator.dim),
        dtype=np.uint8,
    )
    context = np.zeros((args.state_count, 11), dtype=np.float32)
    legal_masks = np.zeros((args.state_count, 24), dtype=bool)
    targets = np.zeros((args.state_count, 24), dtype=np.float32)
    labels = np.zeros((args.state_count,), dtype=np.int64)
    last_factors = np.zeros((args.state_count,), dtype=np.int64)
    value_targets = np.zeros((args.state_count,), dtype=np.float32)
    parent_scores = np.zeros((args.state_count,), dtype=np.float32)
    best_scores = np.zeros((args.state_count,), dtype=np.float32)
    lengths = np.zeros((args.state_count,), dtype=np.int16)
    powers = np.zeros((args.state_count,), dtype=np.int16)

    rows_for_log: list[dict] = []
    generated = 0
    attempts = 0
    while generated < args.state_count:
        attempts += 1
        length = rng.randint(args.min_length, args.max_length)
        factors = automaton.sample_uniform(length, rng)
        power = rng.choice((0, 1)) if args.power_mode == "both" else int(args.power_mode)
        parent = evaluator.evaluate_batch([(power, factors)], batch_size=1)[0]
        legal = tuple(automaton.successors[factors[-1]])
        candidate_words: list[tuple[int, tuple[int, ...], int]] = []
        for action in legal:
            for suffix in random_suffixes_for_action(
                automaton,
                first=action,
                lookahead=args.lookahead,
                max_rollouts=args.rollouts_per_action,
                rng=rng,
            ):
                candidate_words.append((action, factors + suffix, power))
        evaluated = evaluator.evaluate_batch(
            [(power, candidate) for action, candidate, power in candidate_words],
            batch_size=args.eval_batch_size,
        )
        action_scores = np.full((24,), np.inf, dtype=np.float32)
        action_best_metrics: dict[int, dict] = {}
        for (action, _, _), child in zip(candidate_words, evaluated):
            if child.score < action_scores[action]:
                action_scores[action] = np.float32(child.score)
                action_best_metrics[action] = child.metrics
        legal_mask = np.isfinite(action_scores)
        if not np.any(legal_mask):
            continue

        target = soft_target_from_scores(action_scores, legal_mask, args.target_temperature)
        best_action = int(np.nanargmin(action_scores))
        tensors[generated] = parent.tensor
        context[generated] = normalized_context_features(
            power=power,
            factors=factors,
            metrics=parent.metrics,
            score=parent.score,
        )
        legal_masks[generated] = legal_mask
        targets[generated] = target
        labels[generated] = best_action
        last_factors[generated] = factors[-1]
        value_targets[generated] = math.log1p(max(0.0, float(action_scores[best_action])))
        parent_scores[generated] = np.float32(parent.score)
        best_scores[generated] = action_scores[best_action]
        lengths[generated] = length
        powers[generated] = power

        if generated < args.log_examples:
            rows_for_log.append(
                {
                    "example_id": generated,
                    "power": power,
                    "factor_ids": list(factors),
                    "length": length,
                    "parent_metrics": parent.metrics,
                    "parent_score": parent.score,
                    "best_action": best_action,
                    "best_score": float(action_scores[best_action]),
                    "legal_actions": [int(value) for value in np.flatnonzero(legal_mask)],
                    "action_scores": {
                        str(int(index)): float(action_scores[index])
                        for index in np.flatnonzero(legal_mask)
                    },
                    "best_action_metrics": action_best_metrics.get(best_action, {}),
                }
            )
        generated += 1
        if generated % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "phase": "generate",
                        "generated": generated,
                        "attempts": attempts,
                        "elapsed_seconds": round(time.time() - start_time, 2),
                        "best_label_score": float(np.min(best_scores[:generated])),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    append_jsonl(examples_path, rows_for_log)
    np.savez_compressed(
        output_dir / "dataset.npz",
        tensors=tensors,
        context=context,
        legal_masks=legal_masks,
        targets=targets,
        labels=labels,
        last_factors=last_factors,
        value_targets=value_targets,
        parent_scores=parent_scores,
        best_scores=best_scores,
        lengths=lengths,
        powers=powers,
    )
    metadata = {
        "format": "exact-transformer-policy-dataset-v1",
        "p": args.p,
        "n": args.n,
        "r": args.r,
        "state_count": args.state_count,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "lookahead": args.lookahead,
        "rollouts_per_action": args.rollouts_per_action,
        "max_degree": args.max_degree,
        "width_weight": args.width_weight,
        "min_meaningful_length": args.min_meaningful_length,
        "degeneracy_weight": args.degeneracy_weight,
        "target_temperature": args.target_temperature,
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "label_histogram": dict(Counter(int(value) for value in labels)),
        "best_score_min": float(np.min(best_scores)),
        "best_score_median": float(np.median(best_scores)),
    }
    write_json(output_dir / "metadata.json", metadata)
    print(json.dumps({"phase": "done", **metadata}, sort_keys=True), flush=True)


def resolve_device(torch, device_arg: str):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_policy_model(torch, nn, config: dict):
    class MatrixPolicyTransformer(nn.Module):
        def __init__(self, cfg: dict):
            super().__init__()
            self.p = int(cfg["p"])
            self.max_degree = int(cfg["max_degree"])
            self.dim = int(cfg.get("matrix_size", 3))
            self.d_model = int(cfg["d_model"])
            self.value_emb = nn.Embedding(self.p, int(cfg["entry_dim"]))
            self.degree_emb = nn.Embedding(self.max_degree, self.d_model)
            self.entry_proj = nn.Sequential(
                nn.Linear(self.dim * self.dim * int(cfg["entry_dim"]), self.d_model),
                nn.GELU(),
                nn.Dropout(float(cfg["dropout"])),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=int(cfg["heads"]),
                dim_feedforward=int(cfg["ffn_dim"]),
                dropout=float(cfg["dropout"]),
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=int(cfg["layers"])
            )
            self.cls = nn.Parameter(torch.zeros(1, 1, self.d_model))
            self.last_factor_emb = nn.Embedding(24, self.d_model)
            self.context_proj = nn.Sequential(
                nn.Linear(11, self.d_model),
                nn.GELU(),
                nn.Dropout(float(cfg["dropout"])),
            )
            self.norm = nn.LayerNorm(self.d_model)
            self.policy_head = nn.Linear(self.d_model, 24)
            self.value_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.GELU(),
                nn.Linear(self.d_model // 2, 1),
            )
            nn.init.normal_(self.cls, mean=0.0, std=0.02)

        def forward(self, x, context, last_factor):
            batch, depth, dim, _ = x.shape
            coeff = self.value_emb(x.long()).reshape(batch, depth, -1)
            tokens = self.entry_proj(coeff)
            positions = torch.arange(depth, device=x.device)
            tokens = tokens + self.degree_emb(positions).unsqueeze(0)
            degree_has_support = x.ne(0).any(dim=(-1, -2))
            has_any = degree_has_support.any(dim=1)
            last_valid = depth - 1 - degree_has_support.flip(dims=[1]).long().argmax(dim=1)
            token_idx = torch.arange(depth, device=x.device).unsqueeze(0)
            valid = (token_idx <= last_valid.unsqueeze(1)) & has_any.unsqueeze(1)
            cls = self.cls.expand(batch, -1, -1)
            ctx = self.context_proj(context.float()) + self.last_factor_emb(last_factor.long())
            cls = cls + ctx.unsqueeze(1)
            hidden = torch.cat([cls, tokens], dim=1)
            pad_mask = torch.cat(
                [torch.zeros(batch, 1, dtype=torch.bool, device=x.device), ~valid],
                dim=1,
            )
            hidden = self.encoder(hidden, src_key_padding_mask=pad_mask)
            pooled = self.norm(hidden[:, 0])
            return self.policy_head(pooled), self.value_head(pooled).squeeze(-1)

    return MatrixPolicyTransformer(config)


def train_policy(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)
    dataset_path = Path(args.dataset)
    payload = np.load(dataset_path)
    device = resolve_device(torch, args.device)
    count = int(payload["tensors"].shape[0])
    permutation = np.random.default_rng(args.seed).permutation(count)
    split = max(1, int(count * (1.0 - args.validation_fraction)))
    train_idx = permutation[:split]
    val_idx = permutation[split:]
    if len(val_idx) == 0:
        val_idx = train_idx[: min(len(train_idx), 128)]

    def make_loader(indices, shuffle: bool):
        tensors = torch.tensor(payload["tensors"][indices], dtype=torch.long)
        context = torch.tensor(payload["context"][indices], dtype=torch.float32)
        last = torch.tensor(payload["last_factors"][indices], dtype=torch.long)
        legal = torch.tensor(payload["legal_masks"][indices], dtype=torch.bool)
        targets = torch.tensor(payload["targets"][indices], dtype=torch.float32)
        values = torch.tensor(payload["value_targets"][indices], dtype=torch.float32)
        return DataLoader(
            TensorDataset(tensors, context, last, legal, targets, values),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)
    metadata_path = dataset_path.parent / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    model_config = {
        "p": int(metadata.get("p", args.p)),
        "max_degree": int(payload["tensors"].shape[1]),
        "matrix_size": int(payload["tensors"].shape[2]),
        "entry_dim": args.entry_dim,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "ffn_dim": args.ffn_dim,
        "dropout": args.dropout,
    }
    model = build_policy_model(torch, nn, model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_top1 = 0
        total = 0
        for x, context, last, legal, targets, values in loader:
            x = x.to(device)
            context = context.to(device)
            last = last.to(device)
            legal = legal.to(device)
            targets = targets.to(device)
            values = values.to(device)
            with torch.set_grad_enabled(train):
                logits, pred_values = model(x, context, last)
                logits = logits.masked_fill(~legal, -1e9)
                log_probs = F.log_softmax(logits, dim=-1)
                policy_loss = -(targets * log_probs).sum(dim=-1).mean()
                value_loss = F.mse_loss(pred_values, values)
                loss = policy_loss + args.value_loss_weight * value_loss
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
            batch = x.shape[0]
            total_loss += float(loss.detach().cpu()) * batch
            total_policy += float(policy_loss.detach().cpu()) * batch
            total_value += float(value_loss.detach().cpu()) * batch
            total_top1 += int((torch.argmax(logits, dim=-1) == torch.argmax(targets, dim=-1)).sum().cpu())
            total += batch
        return {
            "loss": total_loss / max(1, total),
            "policy_loss": total_policy / max(1, total),
            "value_loss": total_value / max(1, total),
            "top1": total_top1 / max(1, total),
        }

    history = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(train_loader, True)
        val_stats = run_epoch(val_loader, False)
        row = {"epoch": epoch, "train": train_stats, "validation": val_stats}
        history.append(row)
        print(json.dumps({"phase": "train", **row}, sort_keys=True), flush=True)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(
                {
                    "format": "exact-transformer-policy-checkpoint-v1",
                    "model_config": model_config,
                    "model_state": model.state_dict(),
                    "dataset_metadata": metadata,
                    "history": history,
                },
                output_dir / "policy.pt",
            )
    write_json(output_dir / "train_summary.json", {"history": history, "best_validation_loss": best_val})


def load_policy_checkpoint(torch, nn, checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_policy_model(torch, nn, checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def guided_beam_search(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    candidates_path = output_dir / "candidates.jsonl"
    candidates_path.write_text("", encoding="utf-8")

    device = resolve_device(torch, args.device)
    model, checkpoint = load_policy_checkpoint(torch, nn, Path(args.checkpoint), device)
    model_config = checkpoint["model_config"]
    automaton = GNFAutomaton(args.n)
    evaluator = ExactEvaluator(
        author_repo=Path(args.author_repo),
        p=args.p,
        n=args.n,
        r=args.r,
        max_degree=int(model_config["max_degree"]),
        width_weight=args.width_weight,
        min_meaningful_length=args.min_meaningful_length,
        degeneracy_weight=args.degeneracy_weight,
    )

    roots: list[tuple[int, tuple[int, ...]]] = []
    for value in args.seed_word:
        roots.append(parse_seed_word(value))
    while len(roots) < args.root_count:
        length = rng.randint(args.root_min_length, args.root_max_length)
        power = rng.choice((0, 1))
        roots.append((power, automaton.sample_uniform(length, rng)))

    frontier = evaluator.evaluate_batch(roots, batch_size=args.eval_batch_size)
    best: list[EvaluatedWord] = sorted(frontier, key=lambda item: (item.score, len(item.factors)))[
        : args.keep_best
    ]
    seen = {(item.power % 2, item.factors) for item in frontier}
    kernel_hits: list[EvaluatedWord] = []

    for step in range(1, args.steps + 1):
        expansion_words: list[tuple[int, tuple[int, ...]]] = []
        if not frontier:
            break
        x = torch.tensor(np.stack([item.tensor for item in frontier]), dtype=torch.long, device=device)
        context = torch.tensor(
            np.stack(
                [
                    normalized_context_features(
                        power=item.power,
                        factors=item.factors,
                        metrics=item.metrics,
                        score=item.score,
                    )
                    for item in frontier
                ]
            ),
            dtype=torch.float32,
            device=device,
        )
        last = torch.tensor([item.factors[-1] for item in frontier], dtype=torch.long, device=device)
        legal = torch.zeros((len(frontier), 24), dtype=torch.bool, device=device)
        for index, item in enumerate(frontier):
            legal[index, list(automaton.successors[item.factors[-1]])] = True
        with torch.no_grad():
            logits, _ = model(x, context, last)
            logits = logits.masked_fill(~legal, -1e9)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for item, row_probs in zip(frontier, probs):
            legal_actions = list(automaton.successors[item.factors[-1]])
            ranked = sorted(legal_actions, key=lambda action: row_probs[action], reverse=True)
            for action in ranked[: args.actions_per_state]:
                child = item.factors + (int(action),)
                key = (item.power % 2, child)
                if key not in seen:
                    seen.add(key)
                    expansion_words.append((item.power, child))

        evaluated = evaluator.evaluate_batch(expansion_words, batch_size=args.eval_batch_size)
        append_jsonl(
            candidates_path,
            [
                {
                    "step": step,
                    "power": item.power,
                    "factor_ids": list(item.factors),
                    "length": len(item.factors),
                    "metrics": item.metrics,
                    "score": item.score,
                }
                for item in evaluated
            ],
        )
        kernel_hits.extend(item for item in evaluated if item.metrics.get("scalar_identity"))
        best = sorted(best + evaluated, key=lambda item: (item.score, len(item.factors)))[: args.keep_best]
        frontier = sorted(evaluated, key=lambda item: (item.score, len(item.factors)))[: args.beam_size]
        row = {
            "step": step,
            "frontier_size": len(frontier),
            "expanded": len(expansion_words),
            "best_score": best[0].score if best else None,
            "best_metrics": best[0].metrics if best else {},
            "best_length": len(best[0].factors) if best else None,
            "kernel_hits": len(kernel_hits),
        }
        append_jsonl(progress_path, [row])
        print(json.dumps({"phase": "beam", **row}, sort_keys=True), flush=True)
        if kernel_hits and args.stop_at_kernel:
            break

    summary = {
        "format": "exact-transformer-policy-beam-summary-v1",
        "checkpoint": str(args.checkpoint),
        "kernel_hits": [
            {
                "power": item.power,
                "factor_ids": list(item.factors),
                "length": len(item.factors),
                "metrics": item.metrics,
                "score": item.score,
            }
            for item in kernel_hits[: args.keep_best]
        ],
        "best": [
            {
                "power": item.power,
                "factor_ids": list(item.factors),
                "length": len(item.factors),
                "metrics": item.metrics,
                "score": item.score,
            }
            for item in best
        ],
    }
    write_json(output_dir / "summary.json", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact-supervised matrix policy experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate exact next-factor labels")
    gen.add_argument("--author-repo", default=str(DEFAULT_AUTHOR_REPO))
    gen.add_argument("--output-dir", required=True)
    gen.add_argument("--p", type=int, default=7)
    gen.add_argument("--n", type=int, default=4)
    gen.add_argument("--r", type=int, default=1)
    gen.add_argument("--state-count", type=int, default=20_000)
    gen.add_argument("--min-length", type=int, default=12)
    gen.add_argument("--max-length", type=int, default=40)
    gen.add_argument("--lookahead", type=int, default=2)
    gen.add_argument("--rollouts-per-action", type=int, default=4)
    gen.add_argument("--max-degree", type=int, default=192)
    gen.add_argument("--power-mode", default="both", choices=("0", "1", "both"))
    gen.add_argument("--width-weight", type=float, default=0.15)
    gen.add_argument("--min-meaningful-length", type=int, default=15)
    gen.add_argument("--degeneracy-weight", type=float, default=1.0)
    gen.add_argument("--target-temperature", type=float, default=8.0)
    gen.add_argument("--eval-batch-size", type=int, default=500)
    gen.add_argument("--seed", type=int, default=1)
    gen.add_argument("--progress-every", type=int, default=250)
    gen.add_argument("--log-examples", type=int, default=200)
    gen.set_defaults(func=generate_dataset)

    train = sub.add_parser("train", help="train the matrix policy transformer")
    train.add_argument("--dataset", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--p", type=int, default=7)
    train.add_argument("--device", default="auto")
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--weight-decay", type=float, default=1e-2)
    train.add_argument("--value-loss-weight", type=float, default=0.15)
    train.add_argument("--validation-fraction", type=float, default=0.10)
    train.add_argument("--entry-dim", type=int, default=24)
    train.add_argument("--d-model", type=int, default=192)
    train.add_argument("--layers", type=int, default=4)
    train.add_argument("--heads", type=int, default=6)
    train.add_argument("--ffn-dim", type=int, default=768)
    train.add_argument("--dropout", type=float, default=0.10)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=1)
    train.set_defaults(func=train_policy)

    search = sub.add_parser("search", help="run exact-verified model-guided beam search")
    search.add_argument("--author-repo", default=str(DEFAULT_AUTHOR_REPO))
    search.add_argument("--checkpoint", required=True)
    search.add_argument("--output-dir", required=True)
    search.add_argument("--p", type=int, default=7)
    search.add_argument("--n", type=int, default=4)
    search.add_argument("--r", type=int, default=1)
    search.add_argument("--device", default="auto")
    search.add_argument("--root-count", type=int, default=128)
    search.add_argument("--root-min-length", type=int, default=12)
    search.add_argument("--root-max-length", type=int, default=40)
    search.add_argument("--seed-word", action="append", default=[])
    search.add_argument("--steps", type=int, default=80)
    search.add_argument("--beam-size", type=int, default=512)
    search.add_argument("--actions-per-state", type=int, default=4)
    search.add_argument("--keep-best", type=int, default=200)
    search.add_argument("--eval-batch-size", type=int, default=500)
    search.add_argument("--width-weight", type=float, default=0.15)
    search.add_argument("--min-meaningful-length", type=int, default=15)
    search.add_argument("--degeneracy-weight", type=float, default=1.0)
    search.add_argument("--seed", type=int, default=1)
    search.add_argument("--stop-at-kernel", action="store_true")
    search.set_defaults(func=guided_beam_search)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
