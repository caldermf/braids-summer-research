#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
from functools import lru_cache
from itertools import permutations
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
STRUCTURAL_ROOT = REPO_ROOT / "structural-kernel-experiments"
DEFAULT_AUTHOR_REPO = STRUCTURAL_ROOT / "third_party" / "braids_project"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STRUCTURAL_ROOT) not in sys.path:
    sys.path.insert(0, str(STRUCTURAL_ROOT))


PAD_TOKEN = 0
BOS_TOKEN = 25
TOKEN_VOCAB_SIZE = 26
FACTOR_VOCAB_SIZE = 24
IGNORE_INDEX = -100


def has_peyl_package(path: Path) -> bool:
    return (path / "peyl" / "braid.py").exists() and (path / "peyl" / "jonesrep.py").exists()


def candidate_author_repos(requested: Path) -> list[Path]:
    return [
        requested,
        DEFAULT_AUTHOR_REPO,
        REPO_ROOT / "hybrid_of_reservoir_crispr_mcts_suffix" / "third_party" / "braids_project",
        REPO_ROOT / "CRISPR-Transformer-v3-wide-edit" / "third_party" / "braids_project",
        REPO_ROOT / "CRISPR-Transformer-v2" / "third_party" / "braids_project",
        REPO_ROOT / "CRISPR-Transformer" / "third_party" / "braids_project",
        REPO_ROOT / "annealed_reservoir_search" / "third_party" / "braids_project",
        REPO_ROOT.parent / "braids-project",
        REPO_ROOT.parent / "burau-experiments",
        REPO_ROOT.parent / "burau-experiments" / "beta",
        Path.home() / "braids-project",
        Path.home() / "burau-experiments",
        Path.home() / "burau-experiments" / "beta",
    ]


def resolve_author_repo(author_repo: Path) -> Path:
    tried: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidate_author_repos(author_repo.expanduser()):
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        tried.append(candidate)
        if has_peyl_package(candidate):
            return candidate
    tried_text = "\n  ".join(str(path) for path in tried)
    raise FileNotFoundError(
        "Could not find a peyl package with braid.py and jonesrep.py. Tried:\n"
        f"  {tried_text}"
    )


def identity_perm(n: int) -> tuple[int, ...]:
    return tuple(range(n))


def delta_perm(n: int) -> tuple[int, ...]:
    return tuple(range(n - 1, -1, -1))


def right_descent_set(perm: Sequence[int]) -> set[int]:
    return {index for index in range(len(perm) - 1) if perm[index] > perm[index + 1]}


def left_descent_set(perm: Sequence[int]) -> set[int]:
    inverse = [0] * len(perm)
    for position, value in enumerate(perm):
        inverse[value] = position
    return {index for index in range(len(perm) - 1) if inverse[index] > inverse[index + 1]}


class GNFAutomaton:
    """Self-contained legal-transition graph for left Garside normal forms."""

    def __init__(self, n: int = 4):
        self.n = n
        all_perms = list(permutations(range(n)))
        self.perm_to_id = {perm: index for index, perm in enumerate(all_perms)}
        self.id_to_perm = {index: perm for index, perm in enumerate(all_perms)}
        self.factor_ids = tuple(range(len(all_perms)))
        self.identity_id = self.perm_to_id[identity_perm(n)]
        self.delta_id = self.perm_to_id[delta_perm(n)]
        self.first_ids = tuple(
            factor_id
            for factor_id in self.factor_ids
            if factor_id != self.identity_id and (n == 2 or factor_id != self.delta_id)
        )
        self.successors = {
            factor_id: tuple(
                candidate
                for candidate in self.factor_ids
                if candidate != self.identity_id
                and (n == 2 or candidate != self.delta_id)
                and right_descent_set(self.id_to_perm[factor_id]).issuperset(
                    left_descent_set(self.id_to_perm[candidate])
                )
            )
            for factor_id in self.factor_ids
        }

    def is_legal(self, factor_ids: Sequence[int]) -> bool:
        if not factor_ids or factor_ids[0] not in self.first_ids:
            return False
        return all(
            right in self.successors[left]
            for left, right in zip(factor_ids, factor_ids[1:])
        )

    @lru_cache(maxsize=None)
    def can_finish(self, current: int, right: int | None, remaining: int) -> bool:
        if remaining < 0:
            return False
        if remaining == 0:
            return right is None or right in self.successors[current]
        return any(
            self.can_finish(next_factor, right, remaining - 1)
            for next_factor in self.successors[current]
        )

    def viable_next(
        self,
        left: int | None,
        right: int | None,
        remaining_after_choice: int,
    ) -> tuple[int, ...]:
        candidates = self.first_ids if left is None else self.successors[left]
        return tuple(
            candidate
            for candidate in candidates
            if self.can_finish(candidate, right, remaining_after_choice)
        )

    def sample_uniform(self, length: int, rng: random.Random) -> tuple[int, ...]:
        if length <= 0:
            raise ValueError("length must be positive")
        factors = [rng.choice(self.first_ids)]
        while len(factors) < length:
            factors.append(rng.choice(self.successors[factors[-1]]))
        return tuple(factors)

    def sample_bridge(
        self,
        left: int | None,
        right: int | None,
        length: int,
        rng: random.Random,
    ) -> tuple[int, ...]:
        if length <= 0:
            raise ValueError("bridge length must be positive")
        block = []
        current = left
        for offset in range(length):
            remaining = length - offset - 1
            viable = self.viable_next(current, right, remaining)
            if not viable:
                raise ValueError("no legal GNF bridge for the requested boundaries")
            current = rng.choice(viable)
            block.append(current)
        return tuple(block)


def setup_author_imports(author_repo: Path):
    author_repo = resolve_author_repo(author_repo)
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
            all(factors[index] == factors[index % period] for index in range(len(factors)))
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


def parse_seed_word(value: str) -> tuple[int, tuple[int, ...]]:
    if ":" not in value:
        raise ValueError("seed words must have form POWER:f1,f2,...")
    power_text, factors_text = value.split(":", 1)
    factors = tuple(int(part.strip()) for part in factors_text.split(",") if part.strip())
    return int(power_text), factors


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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def factor_to_token(factor_id: int) -> int:
    return int(factor_id) + 1


def token_to_factor(token_id: int) -> int:
    return int(token_id) - 1


def encode_prefix(factors: Sequence[int], max_factors: int) -> tuple[np.ndarray, int]:
    """Return BOS + factors, padded to max_factors + 1, and next-action position."""
    if len(factors) > max_factors:
        raise ValueError(f"prefix length {len(factors)} exceeds max_factors={max_factors}")
    tokens = np.zeros((max_factors + 1,), dtype=np.int16)
    tokens[0] = BOS_TOKEN
    if factors:
        tokens[1 : len(factors) + 1] = [factor_to_token(factor) for factor in factors]
    return tokens, len(factors)


def pretrain_labels(factors: Sequence[int], max_factors: int) -> np.ndarray:
    labels = np.full((max_factors + 1,), IGNORE_INDEX, dtype=np.int16)
    labels[: len(factors)] = [int(factor) for factor in factors]
    return labels


def empty_context(power: int) -> np.ndarray:
    context = np.zeros((11,), dtype=np.float32)
    context[0] = float(power % 2)
    return context


def make_allowed_next_matrix(automaton: GNFAutomaton) -> np.ndarray:
    allowed = np.zeros((TOKEN_VOCAB_SIZE, FACTOR_VOCAB_SIZE), dtype=bool)
    allowed[BOS_TOKEN, list(automaton.first_ids)] = True
    for factor_id, successors in automaton.successors.items():
        allowed[factor_to_token(factor_id), list(successors)] = True
    return allowed


def parse_block_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes:
        raise ValueError("expected at least one integer")
    return sizes


@dataclass
class BraidGPTConfig:
    p: int = 7
    factor_vocab_size: int = FACTOR_VOCAB_SIZE
    token_vocab_size: int = TOKEN_VOCAB_SIZE
    max_factors: int = 96
    context_dim: int = 11
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.10

    @property
    def max_context_tokens(self) -> int:
        return self.max_factors + 1

    def to_dict(self) -> dict:
        return asdict(self)


def build_braid_gpt(torch, nn, config: BraidGPTConfig):
    class BraidGPT(nn.Module):
        def __init__(self, cfg: BraidGPTConfig):
            super().__init__()
            self.config = cfg
            self.token_embedding = nn.Embedding(
                cfg.token_vocab_size,
                cfg.d_model,
                padding_idx=PAD_TOKEN,
            )
            self.position_embedding = nn.Embedding(cfg.max_context_tokens, cfg.d_model)
            self.context_projection = nn.Sequential(
                nn.Linear(cfg.context_dim, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
            self.final_norm = nn.LayerNorm(cfg.d_model)
            self.policy_head = nn.Linear(cfg.d_model, cfg.factor_vocab_size)
            self.value_head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Linear(cfg.d_model // 2, 1),
            )

        def causal_mask(self, length: int, device):
            mask = torch.full((length, length), float("-inf"), device=device)
            return torch.triu(mask, diagonal=1)

        def forward(self, tokens, context):
            if tokens.shape[1] > self.config.max_context_tokens:
                raise ValueError("input longer than model context")
            batch, width = tokens.shape
            positions = torch.arange(width, device=tokens.device)[None, :]
            hidden = self.token_embedding(tokens.long()) + self.position_embedding(positions)
            hidden[:, 0, :] = hidden[:, 0, :] + self.context_projection(context.float())
            padding_mask = tokens.eq(PAD_TOKEN)
            causal = self.causal_mask(width, tokens.device)
            hidden = self.encoder(
                hidden,
                mask=causal,
                src_key_padding_mask=padding_mask,
            )
            hidden = self.final_norm(hidden)
            return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    return BraidGPT(config)


def resolve_device(torch, device_arg: str):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_checkpoint(torch, nn, checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = BraidGPTConfig(**checkpoint["model_config"])
    model = build_braid_gpt(torch, nn, config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, checkpoint


def save_checkpoint(torch, path: Path, *, model, config: BraidGPTConfig, history: list[dict], extra: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "braid-gpt-checkpoint-v1",
            "model_config": config.to_dict(),
            "model_state": model.state_dict(),
            "history": history,
            **extra,
        },
        path,
    )


def generate_pretrain_data(args: argparse.Namespace) -> None:
    start = time.time()
    rng = random.Random(args.seed)
    automaton = GNFAutomaton(args.n)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens = np.zeros((args.sequence_count, args.max_factors + 1), dtype=np.int16)
    labels = np.full((args.sequence_count, args.max_factors + 1), IGNORE_INDEX, dtype=np.int16)
    context = np.zeros((args.sequence_count, 11), dtype=np.float32)
    lengths = np.zeros((args.sequence_count,), dtype=np.int16)
    powers = np.zeros((args.sequence_count,), dtype=np.int16)

    for index in range(args.sequence_count):
        length = rng.randint(args.min_length, args.max_length)
        factors = automaton.sample_uniform(length, rng)
        power = rng.choice((0, 1))
        tokens[index], _ = encode_prefix(factors, args.max_factors)
        labels[index] = pretrain_labels(factors, args.max_factors)
        context[index] = empty_context(power)
        lengths[index] = length
        powers[index] = power
        if (index + 1) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "phase": "pretrain_data",
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
        allowed_next=make_allowed_next_matrix(automaton),
    )
    metadata = {
        "format": "braid-gpt-pretrain-dataset-v1",
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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)
    payload = np.load(args.dataset)
    metadata_path = Path(args.dataset).parent / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    device = resolve_device(torch, args.device)
    config = BraidGPTConfig(
        p=int(metadata.get("p", args.p)),
        max_factors=int(payload["tokens"].shape[1] - 1),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    model = build_braid_gpt(torch, nn, config).to(device)
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
        print(json.dumps({"phase": "pretrain", **row}, sort_keys=True), flush=True)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(
                torch,
                output_dir / "braid_gpt_pretrained.pt",
                model=model,
                config=config,
                history=history,
                extra={"dataset_metadata": metadata, "stage": "pretrain"},
            )
    write_json(output_dir / "pretrain_summary.json", {"history": history, "best_validation_loss": best_val})


def random_suffixes_for_action(
    automaton: GNFAutomaton,
    *,
    first: int,
    lookahead: int,
    rollouts: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
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


def generate_policy_data(args: argparse.Namespace) -> None:
    start = time.time()
    rng = random.Random(args.seed)
    automaton = GNFAutomaton(args.n)
    evaluator = ExactEvaluator(
        author_repo=Path(args.author_repo),
        p=args.p,
        n=args.n,
        r=args.r,
        max_degree=args.matrix_max_degree,
        width_weight=args.width_weight,
        min_meaningful_length=args.min_meaningful_length,
        degeneracy_weight=args.degeneracy_weight,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "examples.jsonl"
    examples_path.write_text("", encoding="utf-8")

    tokens = np.zeros((args.state_count, args.max_factors + 1), dtype=np.int16)
    action_positions = np.zeros((args.state_count,), dtype=np.int16)
    context = np.zeros((args.state_count, 11), dtype=np.float32)
    legal_masks = np.zeros((args.state_count, FACTOR_VOCAB_SIZE), dtype=bool)
    targets = np.zeros((args.state_count, FACTOR_VOCAB_SIZE), dtype=np.float32)
    labels = np.zeros((args.state_count,), dtype=np.int16)
    value_targets = np.zeros((args.state_count,), dtype=np.float32)
    parent_scores = np.zeros((args.state_count,), dtype=np.float32)
    best_scores = np.zeros((args.state_count,), dtype=np.float32)
    lengths = np.zeros((args.state_count,), dtype=np.int16)
    powers = np.zeros((args.state_count,), dtype=np.int16)

    log_rows: list[dict] = []
    for index in range(args.state_count):
        length = rng.randint(args.min_length, args.max_length)
        factors = automaton.sample_uniform(length, rng)
        power = rng.choice((0, 1))
        parent = evaluator.evaluate_batch([(power, factors)], batch_size=1)[0]
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
        for (action, _), child in zip(candidate_words, evaluated):
            action_scores[action] = min(action_scores[action], np.float32(child.score))
        legal_mask = np.isfinite(action_scores)
        best_action = int(np.argmin(action_scores))
        token_row, action_position = encode_prefix(factors, args.max_factors)
        tokens[index] = token_row
        action_positions[index] = action_position
        context[index] = normalized_context_features(
            power=power,
            factors=factors,
            metrics=parent.metrics,
            score=parent.score,
        )
        legal_masks[index] = legal_mask
        targets[index] = soft_target(action_scores, legal_mask, args.target_temperature)
        labels[index] = best_action
        value_targets[index] = math.log1p(max(0.0, float(action_scores[best_action])))
        parent_scores[index] = np.float32(parent.score)
        best_scores[index] = action_scores[best_action]
        lengths[index] = length
        powers[index] = power
        if index < args.log_examples:
            log_rows.append(
                {
                    "example_id": index,
                    "power": power,
                    "factor_ids": list(factors),
                    "length": length,
                    "parent_metrics": parent.metrics,
                    "parent_score": parent.score,
                    "best_action": best_action,
                    "best_score": float(action_scores[best_action]),
                    "legal_actions": [int(value) for value in np.flatnonzero(legal_mask)],
                    "degeneracy": degeneracy_features(factors),
                }
            )
        if (index + 1) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "phase": "policy_data",
                        "generated": index + 1,
                        "best_score_min": float(np.min(best_scores[: index + 1])),
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
        parent_scores=parent_scores,
        best_scores=best_scores,
        lengths=lengths,
        powers=powers,
        allowed_next=make_allowed_next_matrix(automaton),
    )
    metadata = {
        "format": "braid-gpt-policy-dataset-v1",
        "n": args.n,
        "r": args.r,
        "p": args.p,
        "state_count": args.state_count,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "max_factors": args.max_factors,
        "lookahead": args.lookahead,
        "rollouts_per_action": args.rollouts_per_action,
        "width_weight": args.width_weight,
        "min_meaningful_length": args.min_meaningful_length,
        "degeneracy_weight": args.degeneracy_weight,
        "target_temperature": args.target_temperature,
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - start, 2),
        "best_score_min": float(np.min(best_scores)),
        "best_score_median": float(np.median(best_scores)),
        "label_histogram": dict(Counter(int(value) for value in labels)),
    }
    write_json(output_dir / "metadata.json", metadata)
    print(json.dumps({"phase": "done", **metadata}, sort_keys=True), flush=True)


def train_finetune(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)
    payload = np.load(args.dataset)
    metadata_path = Path(args.dataset).parent / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    device = resolve_device(torch, args.device)
    if args.init_checkpoint:
        model, config, checkpoint = load_checkpoint(torch, nn, Path(args.init_checkpoint), device)
        history = list(checkpoint.get("history", []))
    else:
        config = BraidGPTConfig(
            p=int(metadata.get("p", args.p)),
            max_factors=int(payload["tokens"].shape[1] - 1),
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        )
        model = build_braid_gpt(torch, nn, config).to(device)
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
        print(json.dumps({"phase": "finetune", **row}, sort_keys=True), flush=True)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(
                torch,
                output_dir / "braid_gpt_finetuned.pt",
                model=model,
                config=config,
                history=history,
                extra={"dataset_metadata": metadata, "stage": "finetune"},
            )
    write_json(output_dir / "finetune_summary.json", {"history": history, "best_validation_loss": best_val})


@dataclass(frozen=True)
class BeamState:
    power: int
    factors: tuple[int, ...]
    metrics: dict
    score: float


def state_context(state: BeamState) -> np.ndarray:
    if not state.factors:
        return empty_context(state.power)
    return normalized_context_features(
        power=state.power,
        factors=state.factors,
        metrics=state.metrics,
        score=state.score,
    )


def unique_ranked(states: Sequence[BeamState], limit: int) -> list[BeamState]:
    unique: dict[tuple[int, tuple[int, ...]], BeamState] = {}
    for state in states:
        key = (state.power % 2, state.factors)
        previous = unique.get(key)
        if previous is None or (state.score, len(state.factors)) < (
            previous.score,
            len(previous.factors),
        ):
            unique[key] = state
    return sorted(unique.values(), key=lambda item: (item.score, len(item.factors)))[:limit]


def search_generate(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn

    rng = random.Random(args.seed)
    device = resolve_device(torch, args.device)
    model, config, checkpoint = load_checkpoint(torch, nn, Path(args.checkpoint), device)
    automaton = GNFAutomaton(args.n)
    evaluator = ExactEvaluator(
        author_repo=Path(args.author_repo),
        p=args.p,
        n=args.n,
        r=args.r,
        max_degree=args.matrix_max_degree,
        width_weight=args.width_weight,
        min_meaningful_length=args.min_meaningful_length,
        degeneracy_weight=args.degeneracy_weight,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    candidates_path = output_dir / "candidates.jsonl"
    progress_path.write_text("", encoding="utf-8")
    candidates_path.write_text("", encoding="utf-8")

    frontier: list[BeamState] = []
    for value in args.seed_word:
        power, factors = parse_seed_word(value)
        evaluated = evaluator.evaluate_batch([(power, factors)], batch_size=1)[0]
        frontier.append(BeamState(power, factors, evaluated.metrics, evaluated.score))
    if args.start_mode in {"empty", "both"}:
        frontier.extend(BeamState(power, (), {}, 10_000.0) for power in (0, 1))
    if args.start_mode in {"random", "both"}:
        roots: list[tuple[int, tuple[int, ...]]] = []
        while len(roots) < args.random_roots:
            length = rng.randint(args.root_min_length, args.root_max_length)
            roots.append((rng.choice((0, 1)), automaton.sample_uniform(length, rng)))
        frontier.extend(
            BeamState(item.power, item.factors, item.metrics, item.score)
            for item in evaluator.evaluate_batch(roots, batch_size=args.eval_batch_size)
        )
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
        tokens = np.stack([encode_prefix(state.factors, config.max_factors)[0] for state in expandable])
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
        child_states = [BeamState(item.power, item.factors, item.metrics, item.score) for item in evaluated]
        append_jsonl(
            candidates_path,
            [
                {
                    "step": step,
                    "power": state.power,
                    "factor_ids": list(state.factors),
                    "length": len(state.factors),
                    "metrics": state.metrics,
                    "score": state.score,
                }
                for state in child_states
            ],
        )
        kernel_hits.extend(state for state in child_states if state.metrics.get("scalar_identity"))
        best = unique_ranked(best + child_states, args.keep_best)
        frontier = unique_ranked(child_states + best, args.beam_size)
        row = {
            "step": step,
            "expanded": len(children),
            "frontier_size": len(frontier),
            "best_score": best[0].score if best else None,
            "best_length": len(best[0].factors) if best else None,
            "best_metrics": best[0].metrics if best else {},
            "kernel_hits": len(kernel_hits),
        }
        append_jsonl(progress_path, [row])
        print(json.dumps({"phase": "generate", **row}, sort_keys=True), flush=True)
        if kernel_hits and args.stop_at_kernel:
            break

    summary = {
        "format": "braid-gpt-generation-summary-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_stage": checkpoint.get("stage"),
        "kernel_hits": [
            {
                "power": state.power,
                "factor_ids": list(state.factors),
                "length": len(state.factors),
                "metrics": state.metrics,
                "score": state.score,
            }
            for state in kernel_hits[: args.keep_best]
        ],
        "best": [
            {
                "power": state.power,
                "factor_ids": list(state.factors),
                "length": len(state.factors),
                "metrics": state.metrics,
                "score": state.score,
            }
            for state in best
        ],
    }
    write_json(output_dir / "summary.json", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Braid GPT: causal Garside-factor transformer.")
    sub = parser.add_subparsers(dest="command", required=True)

    predata = sub.add_parser("pretrain-data", help="generate large legal-GNF language data")
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

    pretrain = sub.add_parser("pretrain", help="pretrain causal braid language model")
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

    poldata = sub.add_parser("policy-data", help="generate exact next-factor policy data")
    poldata.add_argument("--author-repo", default=str(DEFAULT_AUTHOR_REPO))
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
    poldata.add_argument("--width-weight", type=float, default=0.15)
    poldata.add_argument("--min-meaningful-length", type=int, default=15)
    poldata.add_argument("--degeneracy-weight", type=float, default=1.0)
    poldata.add_argument("--target-temperature", type=float, default=8.0)
    poldata.add_argument("--eval-batch-size", type=int, default=500)
    poldata.add_argument("--seed", type=int, default=1)
    poldata.add_argument("--progress-every", type=int, default=500)
    poldata.add_argument("--log-examples", type=int, default=200)
    poldata.set_defaults(func=generate_policy_data)

    finetune = sub.add_parser("finetune", help="fine-tune on exact policy labels")
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

    search = sub.add_parser("generate", help="generate/search with a Braid GPT checkpoint")
    search.add_argument("--author-repo", default=str(DEFAULT_AUTHOR_REPO))
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
    search.add_argument("--width-weight", type=float, default=0.15)
    search.add_argument("--min-meaningful-length", type=int, default=15)
    search.add_argument("--degeneracy-weight", type=float, default=1.0)
    search.add_argument("--seed", type=int, default=1)
    search.add_argument("--stop-at-kernel", action="store_true")
    search.set_defaults(func=search_generate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
