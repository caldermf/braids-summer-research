#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import gzip
import importlib.util
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
DEFAULT_BRAID_GPT_ROOT = REPO_ROOT / "Braid-GPT"
DEFAULT_AUTHOR_REPO = REPO_ROOT / "hybrid_of_reservoir_crispr_mcts_suffix" / "third_party" / "braids_project"

PAD_TOKEN = 0
BOS_TOKEN = 25
TOKEN_VOCAB_SIZE = 26
FACTOR_VOCAB_SIZE = 24


def patch_functools_cache_for_old_python() -> None:
    if sys.version_info >= (3, 10):
        return
    if getattr(functools.cache, "_peyl_staticmethod_compatible", False):
        return
    original_cache = functools.cache

    def cache_compat(user_function):
        if isinstance(user_function, staticmethod):
            return staticmethod(original_cache(user_function.__func__))
        if isinstance(user_function, classmethod):
            return classmethod(original_cache(user_function.__func__))
        return original_cache(user_function)

    cache_compat._peyl_staticmethod_compatible = True  # type: ignore[attr-defined]
    functools.cache = cache_compat  # type: ignore[assignment]


def load_braid_gpt_module(braid_gpt_root: Path):
    module_path = braid_gpt_root / "braid_gpt.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find Braid-GPT script at {module_path}")
    spec = importlib.util.spec_from_file_location("braid_gpt_runtime_for_matrix_gpt", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Braid-GPT from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
        output["projlen"] = int(output.get("projective_width", 0))
    output.pop("projective_width", None)
    return output


def factor_to_token(factor_id: int) -> int:
    return int(factor_id) + 1


def encode_prefix(factors: Sequence[int], max_factors: int) -> tuple[np.ndarray, int]:
    if len(factors) > max_factors:
        raise ValueError(f"prefix length {len(factors)} exceeds max_factors={max_factors}")
    tokens = np.zeros((max_factors + 1,), dtype=np.int16)
    tokens[0] = BOS_TOKEN
    if factors:
        tokens[1 : len(factors) + 1] = [factor_to_token(factor) for factor in factors]
    return tokens, len(factors)


def make_allowed_next_matrix(automaton) -> np.ndarray:
    allowed = np.zeros((TOKEN_VOCAB_SIZE, FACTOR_VOCAB_SIZE), dtype=bool)
    allowed[BOS_TOKEN, list(automaton.first_ids)] = True
    for factor_id, successors in automaton.successors.items():
        allowed[factor_to_token(factor_id), list(successors)] = True
    return allowed


def soft_target(scores: np.ndarray, legal_mask: np.ndarray, temperature: float) -> np.ndarray:
    target = np.zeros((FACTOR_VOCAB_SIZE,), dtype=np.float32)
    legal_scores = scores[legal_mask]
    shifted = -(legal_scores - np.min(legal_scores)) / max(temperature, 1e-6)
    shifted -= np.max(shifted)
    probs = np.exp(shifted)
    probs /= np.sum(probs)
    target[legal_mask] = probs.astype(np.float32)
    return target


def degeneracy_penalty(bgpt, factors: Sequence[int]) -> float:
    deg = bgpt.degeneracy_features(factors)
    penalty = 0.0
    penalty += max(0.0, deg["dominant_fraction"] - 0.45) * 140.0
    penalty += max(0.0, deg["top_two_fraction"] - 0.70) * 140.0
    penalty += max(0.0, deg["max_run_fraction"] - 0.25) * 160.0
    penalty += max(0.0, deg["max_run_length"] - 4) * 14.0
    penalty += max(0.0, 0.12 - deg["unique_fraction"]) * 160.0
    penalty += max(0.0, deg["repeated_bigram_fraction"] - 0.22) * 100.0
    if deg["period_at_most_2"]:
        penalty += 80.0
    return float(penalty)


def objective_from_metrics(
    bgpt,
    metrics: dict,
    factors: Sequence[int],
    *,
    identity_weight: float,
    projlen_weight: float,
    identity_density_weight: float,
    projlen_density_weight: float,
    degeneracy_weight: float,
    min_length: int,
    kernel_bonus: float,
) -> float:
    length = max(1, len(factors))
    score = (
        identity_weight * float(metrics["identity_defect"])
        + projlen_weight * metric_projlen(metrics)
        + identity_density_weight * float(metrics["identity_defect"]) / length
        + projlen_density_weight * metric_projlen(metrics) / length
        + degeneracy_weight * degeneracy_penalty(bgpt, factors)
    )
    if len(factors) < min_length:
        score += (min_length - len(factors)) * 3.0
    if len(factors) == 0:
        score += 1000.0
    if metrics.get("scalar_identity") and len(factors) > 0:
        score -= kernel_bonus
    return float(score)


def matrix_and_residual_tensor(
    polymat_module,
    image: np.ndarray,
    *,
    p: int,
    max_degree: int,
) -> tuple[np.ndarray, int, int]:
    projected = polymat_module.projectivise(image) % p
    dim = int(projected.shape[0])
    width = min(int(projected.shape[-1]), int(max_degree))
    tensor = np.zeros((max_degree, 2, dim, dim), dtype=np.uint8)
    if width:
        raw = projected[:, :, :width].astype(np.int64, copy=False) % p
        residual = raw.copy()
        scalar = residual[0, 0, :].copy()
        for index in range(dim):
            residual[index, index, :] = (residual[index, index, :] - scalar) % p
        tensor[:width, 0, :, :] = np.transpose(raw, (2, 0, 1)).astype(np.uint8, copy=False)
        tensor[:width, 1, :, :] = np.transpose(residual, (2, 0, 1)).astype(np.uint8, copy=False)
    return tensor, width, dim


@dataclass(frozen=True)
class MatrixEvaluatedWord:
    power: int
    factors: tuple[int, ...]
    metrics: dict
    objective: float
    matrix_tensor: np.ndarray
    matrix_width: int


class MatrixEvaluator:
    def __init__(
        self,
        *,
        bgpt,
        author_repo: Path,
        p: int,
        n: int,
        r: int,
        matrix_max_degree: int,
        identity_weight: float,
        projlen_weight: float,
        identity_density_weight: float,
        projlen_density_weight: float,
        degeneracy_weight: float,
        min_length: int,
        kernel_bonus: float,
    ) -> None:
        self.bgpt = bgpt
        patch_functools_cache_for_old_python()
        self.peyl, self.polymat, self.evaluate_braids = bgpt.setup_author_imports(author_repo)
        self.rep = self.peyl.JonesSummand(n=n, r=r, p=p)
        self.p = int(p)
        self.n = int(n)
        self.r = int(r)
        self.dim = int(self.rep.dimension())
        self.matrix_max_degree = int(matrix_max_degree)
        self.identity_weight = float(identity_weight)
        self.projlen_weight = float(projlen_weight)
        self.identity_density_weight = float(identity_density_weight)
        self.projlen_density_weight = float(projlen_density_weight)
        self.degeneracy_weight = float(degeneracy_weight)
        self.min_length = int(min_length)
        self.kernel_bonus = float(kernel_bonus)

    def evaluate_batch(
        self,
        words: Sequence[tuple[int, Sequence[int]]],
        *,
        batch_size: int,
    ) -> list[MatrixEvaluatedWord]:
        output: list[MatrixEvaluatedWord] = []
        for start in range(0, len(words), batch_size):
            chunk = words[start : start + batch_size]
            braids = [
                self.peyl.GNF(n=self.n, power=int(power), factors=tuple(int(x) for x in factors))
                for power, factors in chunk
            ]
            images = self.evaluate_braids(self.rep, braids)
            for (power, factors), image in zip(chunk, images):
                factors_tuple = tuple(int(x) for x in factors)
                metrics = clean_metrics(self.bgpt.scalar_identity_metrics(self.polymat, image))
                matrix_tensor, matrix_width, _ = matrix_and_residual_tensor(
                    self.polymat,
                    image,
                    p=self.p,
                    max_degree=self.matrix_max_degree,
                )
                objective = objective_from_metrics(
                    self.bgpt,
                    metrics,
                    factors_tuple,
                    identity_weight=self.identity_weight,
                    projlen_weight=self.projlen_weight,
                    identity_density_weight=self.identity_density_weight,
                    projlen_density_weight=self.projlen_density_weight,
                    degeneracy_weight=self.degeneracy_weight,
                    min_length=self.min_length,
                    kernel_bonus=self.kernel_bonus,
                )
                output.append(
                    MatrixEvaluatedWord(
                        power=int(power),
                        factors=factors_tuple,
                        metrics=metrics,
                        objective=objective,
                        matrix_tensor=matrix_tensor,
                        matrix_width=matrix_width,
                    )
                )
        return output


def walk_factor_rows(obj) -> Iterable[dict]:
    if isinstance(obj, dict):
        if "factor_ids" in obj:
            yield obj
        if "powered_factors" in obj:
            yield {"factor_ids": obj["powered_factors"], "power": obj.get("powered_power", obj.get("power", 0))}
        if "final_factors" in obj:
            yield {"factor_ids": obj["final_factors"], "power": obj.get("final_power", obj.get("power", 0))}
        for value in obj.values():
            yield from walk_factor_rows(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_factor_rows(value)


def load_kernel_prefix_examples(
    paths: Sequence[str],
    *,
    max_examples: int,
    min_prefix_length: int,
    max_prefix_length: int,
) -> list[tuple[int, tuple[int, ...], int, str]]:
    examples: list[tuple[int, tuple[int, ...], int, str]] = []
    seen: set[tuple[int, tuple[int, ...], int]] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"):
            rows = list(iter_jsonl(path))
        else:
            rows = list(walk_factor_rows(read_json(path)))
        for row in rows:
            try:
                factors = tuple(int(x) for x in row["factor_ids"])
            except (KeyError, TypeError, ValueError):
                continue
            power = int(row.get("power", 0))
            if len(factors) < 2:
                continue
            upper = min(len(factors) - 1, max_prefix_length)
            for prefix_len in range(max(min_prefix_length, 1), upper + 1):
                prefix = factors[:prefix_len]
                label = int(factors[prefix_len])
                key = (power, prefix, label)
                if key in seen:
                    continue
                seen.add(key)
                examples.append((power, prefix, label, str(path)))
                if max_examples and len(examples) >= max_examples:
                    return examples
    return examples


def random_suffixes_for_action(automaton, *, first: int, lookahead: int, rollouts: int, rng: random.Random):
    if lookahead <= 1:
        return [(first,)]
    suffixes: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    attempts = max(rollouts * 6, 20)
    for _ in range(attempts):
        factors = [int(first)]
        while len(factors) < lookahead:
            factors.append(int(rng.choice(automaton.successors[factors[-1]])))
        suffix = tuple(factors)
        if suffix in seen:
            continue
        seen.add(suffix)
        suffixes.append(suffix)
        if len(suffixes) >= rollouts:
            break
    return suffixes or [(int(first),)]


def generate_policy_data(args: argparse.Namespace) -> None:
    start = time.time()
    bgpt = load_braid_gpt_module(Path(args.braid_gpt_root))
    rng = random.Random(args.seed)
    automaton = bgpt.GNFAutomaton(args.n)
    evaluator = MatrixEvaluator(
        bgpt=bgpt,
        author_repo=Path(args.author_repo),
        p=args.p,
        n=args.n,
        r=args.r,
        matrix_max_degree=args.matrix_max_degree,
        identity_weight=args.identity_weight,
        projlen_weight=args.projlen_weight,
        identity_density_weight=args.identity_density_weight,
        projlen_density_weight=args.projlen_density_weight,
        degeneracy_weight=args.degeneracy_weight,
        min_length=args.min_score_length,
        kernel_bonus=args.kernel_bonus,
    )
    kernel_examples = load_kernel_prefix_examples(
        args.kernel_source,
        max_examples=args.kernel_prefix_count,
        min_prefix_length=args.kernel_min_prefix_length,
        max_prefix_length=min(args.kernel_max_prefix_length, args.max_factors - 1),
    )
    total_count = int(args.state_count) + len(kernel_examples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "examples.jsonl"
    examples_path.write_text("", encoding="utf-8")

    tokens = np.zeros((total_count, args.max_factors + 1), dtype=np.int16)
    action_positions = np.zeros((total_count,), dtype=np.int16)
    matrix_tensors = np.zeros(
        (total_count, args.matrix_max_degree, 2, evaluator.dim, evaluator.dim),
        dtype=np.uint8,
    )
    matrix_widths = np.zeros((total_count,), dtype=np.int16)
    legal_masks = np.zeros((total_count, FACTOR_VOCAB_SIZE), dtype=bool)
    targets = np.zeros((total_count, FACTOR_VOCAB_SIZE), dtype=np.float32)
    labels = np.zeros((total_count,), dtype=np.int16)
    value_targets = np.zeros((total_count,), dtype=np.float32)
    basin_targets = np.zeros((total_count,), dtype=np.float32)
    parent_objectives = np.zeros((total_count,), dtype=np.float32)
    best_objectives = np.zeros((total_count,), dtype=np.float32)
    lengths = np.zeros((total_count,), dtype=np.int16)
    powers = np.zeros((total_count,), dtype=np.int16)
    sources = np.zeros((total_count,), dtype=np.int8)

    log_rows: list[dict] = []

    def fill_parent_fields(index: int, power: int, factors: tuple[int, ...], parent: MatrixEvaluatedWord) -> None:
        token_row, action_position = encode_prefix(factors, args.max_factors)
        tokens[index] = token_row
        action_positions[index] = action_position
        matrix_tensors[index] = parent.matrix_tensor
        matrix_widths[index] = parent.matrix_width
        parent_objectives[index] = np.float32(parent.objective)
        lengths[index] = len(factors)
        powers[index] = int(power)

    for index in range(args.state_count):
        length = rng.randint(args.min_length, args.max_length)
        factors = tuple(int(x) for x in automaton.sample_uniform(length, rng))
        power = rng.choice((0, 1))
        parent = evaluator.evaluate_batch([(power, factors)], batch_size=1)[0]
        legal = tuple(automaton.successors[factors[-1]]) if factors else tuple(automaton.first_ids)
        candidate_words: list[tuple[int, tuple[int, ...]]] = []
        for action in legal:
            for suffix in random_suffixes_for_action(
                automaton,
                first=int(action),
                lookahead=args.lookahead,
                rollouts=args.rollouts_per_action,
                rng=rng,
            ):
                candidate_words.append((int(action), factors + suffix))
        evaluated = evaluator.evaluate_batch(
            [(power, child_factors) for _, child_factors in candidate_words],
            batch_size=args.eval_batch_size,
        )
        action_scores = np.full((FACTOR_VOCAB_SIZE,), np.inf, dtype=np.float32)
        for (action, _), child in zip(candidate_words, evaluated):
            action_scores[action] = min(action_scores[action], np.float32(child.objective))
        legal_mask = np.isfinite(action_scores)
        best_action = int(np.argmin(action_scores))
        fill_parent_fields(index, power, factors, parent)
        legal_masks[index] = legal_mask
        targets[index] = soft_target(action_scores, legal_mask, args.target_temperature)
        labels[index] = best_action
        value_targets[index] = math.log1p(max(0.0, float(action_scores[best_action])))
        basin_targets[index] = np.float32(1.0 if action_scores[best_action] < parent.objective - args.basin_improvement_margin else 0.0)
        best_objectives[index] = action_scores[best_action]
        sources[index] = 0
        if index < args.log_examples:
            log_rows.append(
                {
                    "example_id": index,
                    "source": "random_policy",
                    "power": power,
                    "factor_ids": list(factors),
                    "length": length,
                    "parent_metrics": parent.metrics,
                    "parent_objective": parent.objective,
                    "best_action": best_action,
                    "best_objective": float(action_scores[best_action]),
                    "matrix_width": int(parent.matrix_width),
                    "legal_actions": [int(value) for value in np.flatnonzero(legal_mask)],
                    "degeneracy": bgpt.degeneracy_features(factors),
                }
            )
        if (index + 1) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "phase": "matrix_policy_data_random",
                        "generated": index + 1,
                        "best_objective_min": float(np.min(best_objectives[: index + 1])),
                        "elapsed_seconds": round(time.time() - start, 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    filled_kernel_examples = 0
    for power, factors, label, source in kernel_examples:
        parent = evaluator.evaluate_batch([(power, factors)], batch_size=1)[0]
        legal = tuple(automaton.successors[factors[-1]]) if factors else tuple(automaton.first_ids)
        if label not in legal:
            continue
        index = args.state_count + filled_kernel_examples
        fill_parent_fields(index, power, factors, parent)
        legal_mask = np.zeros((FACTOR_VOCAB_SIZE,), dtype=bool)
        legal_mask[list(legal)] = True
        target = np.zeros((FACTOR_VOCAB_SIZE,), dtype=np.float32)
        target[int(label)] = 1.0
        legal_masks[index] = legal_mask
        targets[index] = target
        labels[index] = int(label)
        value_targets[index] = 0.0
        basin_targets[index] = 1.0
        best_objectives[index] = 0.0
        sources[index] = 1
        filled_kernel_examples += 1
        if len(log_rows) < args.log_examples:
            log_rows.append(
                {
                    "example_id": index,
                    "source": "kernel_prefix",
                    "kernel_source": source,
                    "power": power,
                    "factor_ids": list(factors),
                    "length": len(factors),
                    "parent_metrics": parent.metrics,
                    "next_kernel_factor": int(label),
                    "matrix_width": int(parent.matrix_width),
                    "degeneracy": bgpt.degeneracy_features(factors),
                }
            )

    if filled_kernel_examples < len(kernel_examples):
        keep = args.state_count + filled_kernel_examples
        tokens = tokens[:keep]
        action_positions = action_positions[:keep]
        matrix_tensors = matrix_tensors[:keep]
        matrix_widths = matrix_widths[:keep]
        legal_masks = legal_masks[:keep]
        targets = targets[:keep]
        labels = labels[:keep]
        value_targets = value_targets[:keep]
        basin_targets = basin_targets[:keep]
        parent_objectives = parent_objectives[:keep]
        best_objectives = best_objectives[:keep]
        lengths = lengths[:keep]
        powers = powers[:keep]
        sources = sources[:keep]
        total_count = keep

    append_jsonl(examples_path, log_rows)
    np.savez_compressed(
        output_dir / "matrix_policy_dataset.npz",
        tokens=tokens,
        action_positions=action_positions,
        matrix_tensors=matrix_tensors,
        matrix_widths=matrix_widths,
        legal_masks=legal_masks,
        targets=targets,
        labels=labels,
        value_targets=value_targets,
        basin_targets=basin_targets,
        parent_objectives=parent_objectives,
        best_objectives=best_objectives,
        lengths=lengths,
        powers=powers,
        sources=sources,
        allowed_next=make_allowed_next_matrix(automaton),
    )
    metadata = {
        "format": "braid-matrix-gpt-policy-dataset-v1",
        "n": args.n,
        "r": args.r,
        "p": args.p,
        "state_count": args.state_count,
        "kernel_prefix_examples": filled_kernel_examples,
        "total_count": total_count,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "max_factors": args.max_factors,
        "matrix_max_degree": args.matrix_max_degree,
        "matrix_dim": evaluator.dim,
        "matrix_channels": ["projectivized_raw", "residual_to_scalar"],
        "lookahead": args.lookahead,
        "rollouts_per_action": args.rollouts_per_action,
        "target_temperature": args.target_temperature,
        "objective": {
            "identity_weight": args.identity_weight,
            "projlen_weight": args.projlen_weight,
            "identity_density_weight": args.identity_density_weight,
            "projlen_density_weight": args.projlen_density_weight,
            "degeneracy_weight": args.degeneracy_weight,
            "min_score_length": args.min_score_length,
            "kernel_bonus": args.kernel_bonus,
        },
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - start, 2),
        "label_histogram": dict(Counter(int(value) for value in labels)),
        "source_histogram": dict(Counter("kernel_prefix" if int(value) == 1 else "random_policy" for value in sources)),
    }
    write_json(output_dir / "metadata.json", metadata)
    print(json.dumps({"phase": "done", **metadata}, sort_keys=True), flush=True)


@dataclass
class MatrixGPTConfig:
    p: int = 5
    factor_vocab_size: int = FACTOR_VOCAB_SIZE
    token_vocab_size: int = TOKEN_VOCAB_SIZE
    max_factors: int = 128
    matrix_max_degree: int = 256
    matrix_dim: int = 3
    matrix_channels: int = 2
    d_model: int = 256
    nhead: int = 8
    braid_layers: int = 6
    matrix_layers: int = 3
    dim_feedforward: int = 1024
    dropout: float = 0.10

    @property
    def max_context_tokens(self) -> int:
        return self.max_factors + 1

    def to_dict(self) -> dict:
        return asdict(self)


def build_matrix_gpt(torch, nn, config: MatrixGPTConfig):
    class MatrixGPT(nn.Module):
        def __init__(self, cfg: MatrixGPTConfig):
            super().__init__()
            self.config = cfg
            self.token_embedding = nn.Embedding(cfg.token_vocab_size, cfg.d_model, padding_idx=PAD_TOKEN)
            self.position_embedding = nn.Embedding(cfg.max_context_tokens, cfg.d_model)
            self.matrix_degree_embedding = nn.Embedding(cfg.matrix_max_degree + 1, cfg.d_model)
            flat_matrix_dim = cfg.matrix_channels * cfg.matrix_dim * cfg.matrix_dim
            self.matrix_projection = nn.Sequential(
                nn.Linear(flat_matrix_dim, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, cfg.d_model),
            )
            self.matrix_cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            matrix_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            braid_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            try:
                self.matrix_encoder = nn.TransformerEncoder(
                    matrix_layer,
                    num_layers=cfg.matrix_layers,
                    enable_nested_tensor=False,
                )
                self.braid_encoder = nn.TransformerEncoder(
                    braid_layer,
                    num_layers=cfg.braid_layers,
                    enable_nested_tensor=False,
                )
            except TypeError:
                self.matrix_encoder = nn.TransformerEncoder(matrix_layer, num_layers=cfg.matrix_layers)
                self.braid_encoder = nn.TransformerEncoder(braid_layer, num_layers=cfg.braid_layers)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=cfg.d_model,
                num_heads=cfg.nhead,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.fusion_norm = nn.LayerNorm(cfg.d_model)
            self.final_norm = nn.LayerNorm(cfg.d_model)
            self.dropout = nn.Dropout(cfg.dropout)
            self.policy_head = nn.Linear(cfg.d_model, cfg.factor_vocab_size)
            self.value_head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Linear(cfg.d_model // 2, 1),
            )
            self.basin_head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Linear(cfg.d_model // 2, 1),
            )

        def causal_mask(self, length: int, device):
            mask = torch.ones((length, length), dtype=torch.bool, device=device)
            return torch.triu(mask, diagonal=1)

        def encode_matrix(self, matrices, matrix_widths):
            batch, width = matrices.shape[:2]
            x = matrices.float() / max(1, self.config.p - 1)
            x = x.reshape(batch, width, -1)
            degree_ids = torch.arange(width, device=matrices.device)[None, :]
            hidden = self.matrix_projection(x) + self.matrix_degree_embedding(degree_ids)
            cls = self.matrix_cls.expand(batch, -1, -1)
            hidden = torch.cat([cls, hidden], dim=1)
            degree_positions = torch.arange(width, device=matrices.device)[None, :]
            degree_mask = degree_positions >= matrix_widths[:, None].clamp(min=1)
            padding_mask = torch.cat(
                [torch.zeros((batch, 1), dtype=torch.bool, device=matrices.device), degree_mask],
                dim=1,
            )
            encoded = self.matrix_encoder(hidden, src_key_padding_mask=padding_mask)
            return encoded, padding_mask

        def forward(self, tokens, matrices, matrix_widths):
            if tokens.shape[1] > self.config.max_context_tokens:
                raise ValueError("input longer than model context")
            if matrices.shape[1] != self.config.matrix_max_degree:
                raise ValueError("matrix degree dimension does not match config")
            batch, seq_width = tokens.shape
            matrix_hidden, matrix_padding = self.encode_matrix(matrices, matrix_widths.long())
            positions = torch.arange(seq_width, device=tokens.device)[None, :]
            braid_hidden = self.token_embedding(tokens.long()) + self.position_embedding(positions)
            braid_hidden[:, 0, :] = braid_hidden[:, 0, :] + matrix_hidden[:, 0, :]
            braid_padding = tokens.eq(PAD_TOKEN)
            braid_hidden = self.braid_encoder(
                braid_hidden,
                mask=self.causal_mask(seq_width, tokens.device),
                src_key_padding_mask=braid_padding,
            )
            cross, _ = self.cross_attention(
                query=braid_hidden,
                key=matrix_hidden,
                value=matrix_hidden,
                key_padding_mask=matrix_padding,
                need_weights=False,
            )
            hidden = self.fusion_norm(braid_hidden + self.dropout(cross))
            hidden = self.final_norm(hidden)
            return (
                self.policy_head(hidden),
                self.value_head(hidden).squeeze(-1),
                self.basin_head(hidden).squeeze(-1),
            )

    return MatrixGPT(config)


def resolve_device(torch, device_arg: str):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def save_checkpoint(torch, path: Path, *, model, config: MatrixGPTConfig, history: list[dict], extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "braid-matrix-gpt-checkpoint-v1",
            "model_config": config.to_dict(),
            "model_state": model.state_dict(),
            "history": history,
            **extra,
        },
        path,
    )


def load_checkpoint(torch, nn, checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = MatrixGPTConfig(**checkpoint["model_config"])
    model = build_matrix_gpt(torch, nn, config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, checkpoint


def train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)
    payload = np.load(args.dataset)
    metadata_path = Path(args.dataset).parent / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    device = resolve_device(torch, args.device)
    config = MatrixGPTConfig(
        p=int(metadata.get("p", args.p)),
        max_factors=int(payload["tokens"].shape[1] - 1),
        matrix_max_degree=int(payload["matrix_tensors"].shape[1]),
        matrix_dim=int(payload["matrix_tensors"].shape[-1]),
        matrix_channels=int(payload["matrix_tensors"].shape[2]),
        d_model=args.d_model,
        nhead=args.nhead,
        braid_layers=args.braid_layers,
        matrix_layers=args.matrix_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    model = build_matrix_gpt(torch, nn, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    indices = np.random.default_rng(args.seed).permutation(payload["tokens"].shape[0])
    split = max(1, int(len(indices) * (1.0 - args.validation_fraction)))
    train_indices = indices[:split]
    val_indices = indices[split:] if split < len(indices) else indices[: min(1024, len(indices))]

    def loader(selected, shuffle):
        return DataLoader(
            TensorDataset(
                torch.tensor(payload["tokens"][selected], dtype=torch.long),
                torch.tensor(payload["matrix_tensors"][selected], dtype=torch.uint8),
                torch.tensor(payload["matrix_widths"][selected], dtype=torch.long),
                torch.tensor(payload["action_positions"][selected], dtype=torch.long),
                torch.tensor(payload["legal_masks"][selected], dtype=torch.bool),
                torch.tensor(payload["targets"][selected], dtype=torch.float32),
                torch.tensor(payload["labels"][selected], dtype=torch.long),
                torch.tensor(payload["value_targets"][selected], dtype=torch.float32),
                torch.tensor(payload["basin_targets"][selected], dtype=torch.float32),
            ),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    train_loader = loader(train_indices, True)
    val_loader = loader(val_indices, False)

    def run_epoch(data_loader, train_mode: bool):
        model.train(train_mode)
        totals = Counter()
        total = 0
        for batch in data_loader:
            (
                tokens,
                matrices,
                matrix_widths,
                positions,
                legal,
                targets,
                labels,
                values,
                basin,
            ) = [item.to(device) for item in batch]
            batch_index = torch.arange(tokens.shape[0], device=device)
            with torch.set_grad_enabled(train_mode):
                logits_all, values_all, basin_all = model(tokens, matrices, matrix_widths)
                logits = logits_all[batch_index, positions].masked_fill(~legal, -1e9)
                predicted_values = values_all[batch_index, positions]
                predicted_basin = basin_all[batch_index, positions]
                log_probs = F.log_softmax(logits, dim=-1)
                policy_loss = -(targets * log_probs).sum(dim=-1).mean()
                value_loss = F.mse_loss(predicted_values, values)
                basin_loss = F.binary_cross_entropy_with_logits(predicted_basin, basin)
                loss = (
                    policy_loss
                    + args.value_loss_weight * value_loss
                    + args.basin_loss_weight * basin_loss
                )
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
            batch_size = tokens.shape[0]
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["policy_loss"] += float(policy_loss.detach().cpu()) * batch_size
            totals["value_loss"] += float(value_loss.detach().cpu()) * batch_size
            totals["basin_loss"] += float(basin_loss.detach().cpu()) * batch_size
            totals["top1"] += int((torch.argmax(logits, dim=-1) == labels).sum().detach().cpu())
            total += batch_size
        return {
            "loss": totals["loss"] / max(1, total),
            "policy_loss": totals["policy_loss"] / max(1, total),
            "value_loss": totals["value_loss"] / max(1, total),
            "basin_loss": totals["basin_loss"] / max(1, total),
            "top1": totals["top1"] / max(1, total),
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
        print(json.dumps({"phase": "matrix_gpt_train", **row}, sort_keys=True), flush=True)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(
                torch,
                output_dir / "braid_matrix_gpt.pt",
                model=model,
                config=config,
                history=history,
                extra={"dataset_metadata": metadata, "stage": "matrix_train"},
            )
    write_json(output_dir / "training_summary.json", {"history": history, "best_validation_loss": best_val})


@dataclass(frozen=True)
class BeamState:
    power: int
    factors: tuple[int, ...]
    metrics: dict
    objective: float
    matrix_width: int = 1


def unique_ranked(states: Sequence[BeamState], limit: int) -> list[BeamState]:
    unique: dict[tuple[int, tuple[int, ...]], BeamState] = {}
    for state in states:
        key = (state.power % 2, state.factors)
        previous = unique.get(key)
        if previous is None or (state.objective, -len(state.factors)) < (
            previous.objective,
            -len(previous.factors),
        ):
            unique[key] = state
    return sorted(unique.values(), key=lambda item: (item.objective, -len(item.factors)))[:limit]


def state_record(state: BeamState) -> dict:
    return {
        "power": state.power,
        "factor_ids": list(state.factors),
        "length": len(state.factors),
        "metrics": state.metrics,
        "objective": state.objective,
        "matrix_width": state.matrix_width,
    }


def parse_seed_word(value: str) -> tuple[int, tuple[int, ...]]:
    if ":" not in value:
        raise ValueError("seed words must have form POWER:f1,f2,...")
    power_text, factors_text = value.split(":", 1)
    factors = tuple(int(part.strip()) for part in factors_text.split(",") if part.strip())
    return int(power_text), factors


def search_generate(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn

    bgpt = load_braid_gpt_module(Path(args.braid_gpt_root))
    rng = random.Random(args.seed)
    device = resolve_device(torch, args.device)
    model, config, checkpoint = load_checkpoint(torch, nn, Path(args.checkpoint), device)
    automaton = bgpt.GNFAutomaton(args.n)
    evaluator = MatrixEvaluator(
        bgpt=bgpt,
        author_repo=Path(args.author_repo),
        p=args.p,
        n=args.n,
        r=args.r,
        matrix_max_degree=config.matrix_max_degree,
        identity_weight=args.identity_weight,
        projlen_weight=args.projlen_weight,
        identity_density_weight=args.identity_density_weight,
        projlen_density_weight=args.projlen_density_weight,
        degeneracy_weight=args.degeneracy_weight,
        min_length=args.min_score_length,
        kernel_bonus=args.kernel_bonus,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    candidates_path = output_dir / "candidates.jsonl"
    progress_path.write_text("", encoding="utf-8")
    candidates_path.write_text("", encoding="utf-8")

    def evaluated_states(words: Sequence[tuple[int, tuple[int, ...]]]) -> list[tuple[BeamState, np.ndarray]]:
        evaluated = evaluator.evaluate_batch(words, batch_size=args.eval_batch_size)
        out: list[tuple[BeamState, np.ndarray]] = []
        for item in evaluated:
            out.append(
                (
                    BeamState(
                        power=item.power,
                        factors=item.factors,
                        metrics=item.metrics,
                        objective=item.objective,
                        matrix_width=item.matrix_width,
                    ),
                    item.matrix_tensor,
                )
            )
        return out

    frontier_pairs: list[tuple[BeamState, np.ndarray]] = []
    for value in args.seed_word:
        frontier_pairs.extend(evaluated_states([parse_seed_word(value)]))
    if args.start_mode in {"empty", "both"}:
        frontier_pairs.extend(evaluated_states([(0, tuple()), (1, tuple())]))
    if args.start_mode in {"random", "both"}:
        roots: list[tuple[int, tuple[int, ...]]] = []
        while len(roots) < args.random_roots:
            length = rng.randint(args.root_min_length, args.root_max_length)
            roots.append((rng.choice((0, 1)), tuple(int(x) for x in automaton.sample_uniform(length, rng))))
        frontier_pairs.extend(evaluated_states(roots))

    ranked_frontier = unique_ranked([state for state, _ in frontier_pairs], args.beam_size)
    matrix_by_key = {
        (state.power % 2, state.factors): matrix for state, matrix in frontier_pairs
    }
    frontier = ranked_frontier
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
        missing = [
            (state.power, state.factors)
            for state in expandable
            if (state.power % 2, state.factors) not in matrix_by_key
        ]
        if missing:
            for state, matrix in evaluated_states(missing):
                matrix_by_key[(state.power % 2, state.factors)] = matrix
        tokens = np.stack([encode_prefix(state.factors, config.max_factors)[0] for state in expandable])
        positions = np.array([len(state.factors) for state in expandable], dtype=np.int64)
        matrices = np.stack([matrix_by_key[(state.power % 2, state.factors)] for state in expandable])
        widths = np.array([state.matrix_width for state in expandable], dtype=np.int64)
        legal_masks = np.zeros((len(expandable), FACTOR_VOCAB_SIZE), dtype=bool)
        for index, state in enumerate(expandable):
            if not state.factors:
                legal_masks[index, list(automaton.first_ids)] = True
            else:
                legal_masks[index, list(automaton.successors[state.factors[-1]])] = True
        with torch.no_grad():
            logits_all, values_all, basin_all = model(
                torch.tensor(tokens, dtype=torch.long, device=device),
                torch.tensor(matrices, dtype=torch.uint8, device=device),
                torch.tensor(widths, dtype=torch.long, device=device),
            )
            batch_index = torch.arange(len(expandable), device=device)
            positions_t = torch.tensor(positions, dtype=torch.long, device=device)
            logits = logits_all[batch_index, positions_t]
            values = values_all[batch_index, positions_t]
            basin_logits = basin_all[batch_index, positions_t]
            legal_tensor = torch.tensor(legal_masks, dtype=torch.bool, device=device)
            logits = logits.masked_fill(~legal_tensor, -1e9)
            policy = torch.softmax(logits / max(args.temperature, 1e-6), dim=-1)
            basin = torch.sigmoid(basin_logits)
            model_scores = (
                torch.log(policy.clamp_min(1e-9))
                + args.basin_prior_weight * basin[:, None]
                - args.value_prior_weight * values[:, None]
            ).detach().cpu().numpy()
        children: list[tuple[int, tuple[int, ...]]] = []
        for state, row, legal_mask in zip(expandable, model_scores, legal_masks):
            legal_actions = np.flatnonzero(legal_mask)
            ranked = sorted(legal_actions, key=lambda action: row[action], reverse=True)[: args.actions_per_state]
            for action in ranked:
                factors = state.factors + (int(action),)
                key = (state.power % 2, factors)
                if key in seen:
                    continue
                seen.add(key)
                children.append((state.power, factors))
        child_pairs = evaluated_states(children) if children else []
        child_states = [state for state, _ in child_pairs]
        for state, matrix in child_pairs:
            matrix_by_key[(state.power % 2, state.factors)] = matrix
        append_jsonl(
            candidates_path,
            [
                {
                    "step": step,
                    **state_record(state),
                    "degeneracy": bgpt.degeneracy_features(state.factors),
                }
                for state in child_states
            ],
        )
        kernel_hits.extend(state for state in child_states if state.metrics.get("scalar_identity") and len(state.factors) > 0)
        best = unique_ranked(best + child_states, args.keep_best)
        frontier = unique_ranked(child_states + best, args.beam_size)
        row = {
            "phase": "matrix_gpt_generate",
            "step": step,
            "expanded": len(children),
            "frontier_size": len(frontier),
            "best_objective": best[0].objective if best else None,
            "best_length": len(best[0].factors) if best else None,
            "best_metrics": best[0].metrics if best else {},
            "kernel_hits": len(kernel_hits),
        }
        append_jsonl(progress_path, [row])
        print(json.dumps(row, sort_keys=True), flush=True)
        if kernel_hits and args.stop_at_kernel:
            break

    summary = {
        "format": "braid-matrix-gpt-generation-summary-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_stage": checkpoint.get("stage"),
        "kernel_hits": [state_record(state) for state in kernel_hits[: args.keep_best]],
        "best": [state_record(state) for state in best],
        "best_by_identity_defect": [
            state_record(state)
            for state in sorted(best, key=lambda item: (item.metrics.get("identity_defect", 10**9), item.objective))[
                : args.keep_best
            ]
        ],
        "objective": {
            "identity_weight": args.identity_weight,
            "projlen_weight": args.projlen_weight,
            "identity_density_weight": args.identity_density_weight,
            "projlen_density_weight": args.projlen_density_weight,
            "degeneracy_weight": args.degeneracy_weight,
            "min_score_length": args.min_score_length,
            "kernel_bonus": args.kernel_bonus,
        },
    }
    write_json(output_dir / "summary.json", summary)


def add_objective_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--projlen-weight", type=float, default=0.25)
    parser.add_argument("--identity-density-weight", type=float, default=8.0)
    parser.add_argument("--projlen-density-weight", type=float, default=4.0)
    parser.add_argument("--degeneracy-weight", type=float, default=1.0)
    parser.add_argument("--min-score-length", type=int, default=45)
    parser.add_argument("--kernel-bonus", type=float, default=10000.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Braid-Matrix-GPT: Garside tokens plus peyl matrix/residual tensors.")
    parser.add_argument("--braid-gpt-root", default=str(DEFAULT_BRAID_GPT_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("policy-data")
    data.add_argument("--author-repo", default=str(DEFAULT_AUTHOR_REPO))
    data.add_argument("--output-dir", required=True)
    data.add_argument("--p", type=int, default=5)
    data.add_argument("--n", type=int, default=4)
    data.add_argument("--r", type=int, default=1)
    data.add_argument("--state-count", type=int, default=100_000)
    data.add_argument("--min-length", type=int, default=12)
    data.add_argument("--max-length", type=int, default=72)
    data.add_argument("--max-factors", type=int, default=128)
    data.add_argument("--lookahead", type=int, default=2)
    data.add_argument("--rollouts-per-action", type=int, default=4)
    data.add_argument("--matrix-max-degree", type=int, default=256)
    data.add_argument("--target-temperature", type=float, default=0.35)
    data.add_argument("--basin-improvement-margin", type=float, default=25.0)
    data.add_argument("--eval-batch-size", type=int, default=500)
    data.add_argument("--kernel-source", action="append", default=[])
    data.add_argument("--kernel-prefix-count", type=int, default=0)
    data.add_argument("--kernel-min-prefix-length", type=int, default=8)
    data.add_argument("--kernel-max-prefix-length", type=int, default=80)
    data.add_argument("--seed", type=int, default=1)
    data.add_argument("--progress-every", type=int, default=500)
    data.add_argument("--log-examples", type=int, default=200)
    add_objective_args(data)
    data.set_defaults(func=generate_policy_data)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--p", type=int, default=5)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--value-loss-weight", type=float, default=0.15)
    train_parser.add_argument("--basin-loss-weight", type=float, default=0.20)
    train_parser.add_argument("--validation-fraction", type=float, default=0.05)
    train_parser.add_argument("--d-model", type=int, default=256)
    train_parser.add_argument("--nhead", type=int, default=8)
    train_parser.add_argument("--braid-layers", type=int, default=6)
    train_parser.add_argument("--matrix-layers", type=int, default=3)
    train_parser.add_argument("--dim-feedforward", type=int, default=1024)
    train_parser.add_argument("--dropout", type=float, default=0.10)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--seed", type=int, default=1)
    train_parser.set_defaults(func=train)

    search = sub.add_parser("generate")
    search.add_argument("--author-repo", default=str(DEFAULT_AUTHOR_REPO))
    search.add_argument("--checkpoint", required=True)
    search.add_argument("--output-dir", required=True)
    search.add_argument("--p", type=int, default=5)
    search.add_argument("--n", type=int, default=4)
    search.add_argument("--r", type=int, default=1)
    search.add_argument("--device", default="auto")
    search.add_argument("--start-mode", choices=("empty", "random", "both"), default="both")
    search.add_argument("--random-roots", type=int, default=256)
    search.add_argument("--root-min-length", type=int, default=12)
    search.add_argument("--root-max-length", type=int, default=60)
    search.add_argument("--seed-word", action="append", default=[])
    search.add_argument("--steps", type=int, default=120)
    search.add_argument("--beam-size", type=int, default=1024)
    search.add_argument("--actions-per-state", type=int, default=8)
    search.add_argument("--keep-best", type=int, default=200)
    search.add_argument("--max-length", type=int, default=128)
    search.add_argument("--temperature", type=float, default=1.10)
    search.add_argument("--value-prior-weight", type=float, default=0.05)
    search.add_argument("--basin-prior-weight", type=float, default=0.30)
    search.add_argument("--eval-batch-size", type=int, default=500)
    search.add_argument("--seed", type=int, default=1)
    search.add_argument("--stop-at-kernel", action="store_true")
    add_objective_args(search)
    search.set_defaults(func=search_generate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
