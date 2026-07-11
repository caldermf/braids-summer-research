#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import importlib.util
import itertools
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
DEFAULT_MATRIX_GPT_ROOT = REPO_ROOT / "Braid-Matrix-GPT"
DEFAULT_BRAID_GPT_ROOT = REPO_ROOT / "Braid-GPT"
DEFAULT_AUTHOR_REPO = REPO_ROOT / "hybrid_of_reservoir_crispr_mcts_suffix" / "third_party" / "braids_project"

PAD_TOKEN = 0
BOS_TOKEN = 25
TOKEN_VOCAB_SIZE = 26
FACTOR_VOCAB_SIZE = 24
ACTION_TYPES = {"replace": 0, "left": 1, "right": 2, "insert": 3}
ACTION_NAMES = {value: key for key, value in ACTION_TYPES.items()}


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_matrix_gpt(matrix_gpt_root: Path):
    return load_module("matrix_gpt_runtime_for_teacher_reservoir", matrix_gpt_root / "matrix_gpt.py")


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


def parse_int_list(value: str) -> tuple[int, ...]:
    output = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not output:
        raise ValueError("expected at least one integer")
    return output


def row_factor_power(row: dict) -> tuple[int, tuple[int, ...]] | None:
    for key in ("factor_ids", "final_factors", "powered_factors", "factors"):
        if key in row:
            try:
                factors = tuple(int(value) for value in row[key])
            except (TypeError, ValueError):
                return None
            power = int(row.get("power", row.get("final_power", row.get("powered_power", 0))))
            return power, factors
    return None


def walk_factor_rows(obj) -> Iterable[dict]:
    if isinstance(obj, dict):
        parsed = row_factor_power(obj)
        if parsed is not None:
            power, factors = parsed
            yield {
                "power": power,
                "factor_ids": list(factors),
                "metrics": obj.get("metrics", obj.get("exact_metrics", {})),
                "objective": obj.get("objective"),
                "source": obj.get("source"),
            }
        for value in obj.values():
            yield from walk_factor_rows(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_factor_rows(value)


@dataclass(frozen=True)
class SeedCandidate:
    seed_id: int
    power: int
    factors: tuple[int, ...]
    source: str
    metrics: dict
    objective: float | None


@dataclass(frozen=True)
class MoveProposal:
    action_type: str
    position: int
    delete_width: int
    insert_factors: tuple[int, ...]
    child_factors: tuple[int, ...]
    source: str


@dataclass(frozen=True)
class SearchState:
    power: int
    factors: tuple[int, ...]
    metrics: dict
    objective: float
    matrix_width: int


def metric_projlen(metrics: dict) -> int:
    return int(metrics.get("projlen", metrics.get("projective_width", 10**9)))


def rank_seed_tuple(row: SeedCandidate) -> tuple[float, int, int, int]:
    objective = float(row.objective) if row.objective is not None else float(row.metrics.get("identity_defect", 10**9))
    return (
        objective,
        int(row.metrics.get("identity_defect", 10**9)),
        metric_projlen(row.metrics),
        len(row.factors),
    )


def load_seed_candidates(
    paths: Sequence[str],
    *,
    min_length: int,
    max_length: int,
    limit: int,
) -> list[SeedCandidate]:
    rows: list[SeedCandidate] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            print(json.dumps({"phase": "missing_seed_source", "path": str(path)}), flush=True)
            continue
        if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"):
            iterator = iter_jsonl(path)
        else:
            iterator = walk_factor_rows(read_json(path))
        for row in iterator:
            parsed = row_factor_power(row)
            if parsed is None:
                continue
            power, factors = parsed
            if len(factors) < min_length or len(factors) > max_length:
                continue
            key = (power % 2, factors)
            if key in seen:
                continue
            seen.add(key)
            objective = row.get("objective")
            rows.append(
                SeedCandidate(
                    seed_id=len(rows),
                    power=int(power),
                    factors=factors,
                    source=str(row.get("source", path)),
                    metrics=dict(row.get("metrics", row.get("exact_metrics", {})) or {}),
                    objective=float(objective) if objective is not None else None,
                )
            )
    rows.sort(key=rank_seed_tuple)
    return rows[:limit] if limit else rows


def factor_to_token(factor_id: int) -> int:
    return int(factor_id) + 1


def encode_factors(factors: Sequence[int], max_factors: int) -> np.ndarray:
    if len(factors) > max_factors:
        raise ValueError(f"length {len(factors)} exceeds max_factors={max_factors}")
    tokens = np.zeros((max_factors + 1,), dtype=np.int16)
    tokens[0] = BOS_TOKEN
    if factors:
        tokens[1 : len(factors) + 1] = [factor_to_token(item) for item in factors]
    return tokens


def make_evaluator(args: argparse.Namespace, mgpt, bgpt, *, matrix_max_degree: int | None = None):
    return mgpt.MatrixEvaluator(
        bgpt=bgpt,
        author_repo=Path(args.author_repo),
        p=args.p,
        n=args.n,
        r=args.r,
        matrix_max_degree=int(matrix_max_degree if matrix_max_degree is not None else args.matrix_max_degree),
        identity_weight=args.identity_weight,
        projlen_weight=args.projlen_weight,
        identity_density_weight=args.identity_density_weight,
        projlen_density_weight=args.projlen_density_weight,
        degeneracy_weight=args.degeneracy_weight,
        min_length=args.min_score_length,
        kernel_bonus=args.kernel_bonus,
    )


def apply_edit(
    factors: Sequence[int],
    *,
    position: int,
    delete_width: int,
    insert_factors: Sequence[int],
) -> tuple[int, ...]:
    return (
        tuple(int(x) for x in factors[:position])
        + tuple(int(x) for x in insert_factors)
        + tuple(int(x) for x in factors[position + delete_width :])
    )


def legal_bridge_for_edit(automaton, factors: Sequence[int], position: int, delete_width: int, width: int, rng: random.Random):
    left = int(factors[position - 1]) if position > 0 else None
    right_index = position + delete_width
    right = int(factors[right_index]) if right_index < len(factors) else None
    return tuple(int(x) for x in automaton.sample_bridge(left, right, width, rng))


def sample_unique_bridge(
    automaton,
    factors: Sequence[int],
    position: int,
    delete_width: int,
    insert_width: int,
    rng: random.Random,
    *,
    attempts: int,
) -> tuple[int, ...] | None:
    old = tuple(int(x) for x in factors[position : position + delete_width])
    for _ in range(attempts):
        try:
            bridge = legal_bridge_for_edit(automaton, factors, position, delete_width, insert_width, rng)
        except ValueError:
            return None
        if bridge != old:
            return bridge
    return None


def propose_teacher_moves(
    *,
    automaton,
    rng: random.Random,
    factors: tuple[int, ...],
    max_factors: int,
    right_lengths: Sequence[int],
    left_lengths: Sequence[int],
    window_widths: Sequence[int],
    right_samples_per_length: int,
    left_samples_per_length: int,
    windows_per_width: int,
    bridge_attempts: int,
) -> list[MoveProposal]:
    proposals: list[MoveProposal] = []
    seen_children: set[tuple[int, ...]] = set()

    def add(action_type: str, position: int, delete_width: int, insert: tuple[int, ...], source: str) -> None:
        if len(insert) == 0:
            return
        child = apply_edit(factors, position=position, delete_width=delete_width, insert_factors=insert)
        if len(child) > max_factors or child == factors or not automaton.is_legal(child):
            return
        if child in seen_children:
            return
        seen_children.add(child)
        proposals.append(
            MoveProposal(
                action_type=action_type,
                position=int(position),
                delete_width=int(delete_width),
                insert_factors=insert,
                child_factors=child,
                source=source,
            )
        )

    for length in right_lengths:
        if len(factors) + length > max_factors:
            continue
        for sample_id in range(right_samples_per_length):
            insert = sample_unique_bridge(
                automaton,
                factors,
                len(factors),
                0,
                int(length),
                rng,
                attempts=bridge_attempts,
            )
            if insert is not None:
                add("right", len(factors), 0, insert, f"right_len{length}_sample{sample_id}")

    for length in left_lengths:
        if len(factors) + length > max_factors:
            continue
        for sample_id in range(left_samples_per_length):
            insert = sample_unique_bridge(
                automaton,
                factors,
                0,
                0,
                int(length),
                rng,
                attempts=bridge_attempts,
            )
            if insert is not None:
                add("left", 0, 0, insert, f"left_len{length}_sample{sample_id}")

    for width in window_widths:
        width = int(width)
        if width <= 0 or width > len(factors):
            continue
        starts = list(range(0, len(factors) - width + 1))
        rng.shuffle(starts)
        for sample_id, position in enumerate(starts[:windows_per_width]):
            insert = sample_unique_bridge(
                automaton,
                factors,
                position,
                width,
                width,
                rng,
                attempts=bridge_attempts,
            )
            if insert is not None:
                add("replace", position, width, insert, f"replace_w{width}_sample{sample_id}")

    return proposals


def state_record(state: SearchState) -> dict:
    return {
        "power": int(state.power),
        "factor_ids": list(state.factors),
        "length": len(state.factors),
        "metrics": state.metrics,
        "objective": float(state.objective),
        "matrix_width": int(state.matrix_width),
    }


def evaluated_to_state(item) -> SearchState:
    return SearchState(
        power=int(item.power),
        factors=tuple(int(x) for x in item.factors),
        metrics=dict(item.metrics),
        objective=float(item.objective),
        matrix_width=int(item.matrix_width),
    )


def score_tuple(state: SearchState) -> tuple[float, int, int, int]:
    return (
        float(state.objective),
        int(state.metrics.get("identity_defect", 10**9)),
        metric_projlen(state.metrics),
        len(state.factors),
    )


def mine_teacher(args: argparse.Namespace) -> None:
    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    moves_path = output_dir / "teacher_moves.jsonl"
    all_path = output_dir / "scored_moves.jsonl"
    moves_path.write_text("", encoding="utf-8")
    all_path.write_text("", encoding="utf-8")

    mgpt = load_matrix_gpt(Path(args.matrix_gpt_root))
    bgpt = mgpt.load_braid_gpt_module(Path(args.braid_gpt_root))
    automaton = bgpt.GNFAutomaton(args.n)
    evaluator = make_evaluator(args, mgpt, bgpt)
    rng = random.Random(args.seed)

    seeds = load_seed_candidates(
        args.seed_source,
        min_length=args.min_seed_length,
        max_length=args.max_seed_length,
        limit=args.seed_limit,
    )
    if not seeds:
        raise RuntimeError("No seed candidates loaded")

    right_lengths = parse_int_list(args.right_lengths)
    left_lengths = parse_int_list(args.left_lengths)
    window_widths = parse_int_list(args.window_widths)
    selected = seeds[: args.seed_limit] if args.seed_limit else seeds

    accepted_count = 0
    scored_count = 0
    parent_evaluated = evaluator.evaluate_batch(
        [(seed.power, seed.factors) for seed in selected],
        batch_size=args.eval_batch_size,
    )
    parent_by_id = {seed.seed_id: evaluated_to_state(item) for seed, item in zip(selected, parent_evaluated)}
    progress_every = max(1, args.progress_every)

    for seed_index, seed in enumerate(selected, start=1):
        parent = parent_by_id[seed.seed_id]
        proposals = propose_teacher_moves(
            automaton=automaton,
            rng=rng,
            factors=parent.factors,
            max_factors=args.max_factors,
            right_lengths=right_lengths,
            left_lengths=left_lengths,
            window_widths=window_widths,
            right_samples_per_length=args.right_samples_per_length,
            left_samples_per_length=args.left_samples_per_length,
            windows_per_width=args.windows_per_width,
            bridge_attempts=args.bridge_attempts,
        )
        if not proposals:
            continue
        child_eval = evaluator.evaluate_batch(
            [(parent.power, proposal.child_factors) for proposal in proposals],
            batch_size=args.eval_batch_size,
        )
        scored_rows: list[dict] = []
        accepted_rows: list[dict] = []
        for proposal, child_item in zip(proposals, child_eval):
            child = evaluated_to_state(child_item)
            objective_delta = child.objective - parent.objective
            identity_delta = int(child.metrics.get("identity_defect", 10**9)) - int(
                parent.metrics.get("identity_defect", 10**9)
            )
            projlen_delta = metric_projlen(child.metrics) - metric_projlen(parent.metrics)
            row = {
                "seed_id": seed.seed_id,
                "seed_rank": seed_index - 1,
                "seed_source": seed.source,
                "p": args.p,
                "n": args.n,
                "r": args.r,
                "power": parent.power,
                "parent_factors": list(parent.factors),
                "parent_length": len(parent.factors),
                "parent_metrics": parent.metrics,
                "parent_objective": parent.objective,
                "action_type": proposal.action_type,
                "position": proposal.position,
                "delete_width": proposal.delete_width,
                "insert_width": len(proposal.insert_factors),
                "insert_factors": list(proposal.insert_factors),
                "child_factors": list(child.factors),
                "child_length": len(child.factors),
                "child_metrics": child.metrics,
                "child_objective": child.objective,
                "objective_delta": objective_delta,
                "identity_delta": identity_delta,
                "projlen_delta": projlen_delta,
                "move_source": proposal.source,
            }
            scored_rows.append(row)
            improved = (
                child.metrics.get("scalar_identity")
                or objective_delta < -args.min_objective_improvement
                or identity_delta < -args.min_identity_improvement
            )
            if improved:
                accepted_rows.append(row)
        scored_rows.sort(key=lambda row: (row["child_objective"], row["child_metrics"]["identity_defect"], row["child_metrics"]["projlen"]))
        accepted_rows.sort(key=lambda row: (row["child_objective"], row["child_metrics"]["identity_defect"], row["child_metrics"]["projlen"]))
        append_jsonl(all_path, scored_rows[: args.keep_scored_per_seed])
        append_jsonl(moves_path, accepted_rows[: args.keep_teacher_per_seed])
        scored_count += len(scored_rows)
        accepted_count += min(len(accepted_rows), args.keep_teacher_per_seed)
        if seed_index % progress_every == 0 or seed_index == len(selected):
            print(
                json.dumps(
                    {
                        "phase": "mine_teacher",
                        "seeds_done": seed_index,
                        "seeds_total": len(selected),
                        "scored_moves": scored_count,
                        "teacher_moves": accepted_count,
                        "elapsed_seconds": round(time.time() - start_time, 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    metadata = {
        "format": "braid-teacher-reservoir-moves-v1",
        "p": args.p,
        "n": args.n,
        "r": args.r,
        "seed_sources": args.seed_source,
        "seed_count": len(selected),
        "teacher_moves": accepted_count,
        "scored_moves": scored_count,
        "right_lengths": list(right_lengths),
        "left_lengths": list(left_lengths),
        "window_widths": list(window_widths),
        "max_factors": args.max_factors,
        "objective": objective_metadata(args),
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    write_json(output_dir / "metadata.json", metadata)
    print(json.dumps({"phase": "done", **metadata}, sort_keys=True), flush=True)


def objective_metadata(args: argparse.Namespace) -> dict:
    return {
        "identity_weight": args.identity_weight,
        "projlen_weight": args.projlen_weight,
        "identity_density_weight": args.identity_density_weight,
        "projlen_density_weight": args.projlen_density_weight,
        "degeneracy_weight": args.degeneracy_weight,
        "min_score_length": args.min_score_length,
        "kernel_bonus": args.kernel_bonus,
    }


def build_teacher_data(args: argparse.Namespace) -> None:
    start_time = time.time()
    mgpt = load_matrix_gpt(Path(args.matrix_gpt_root))
    bgpt = mgpt.load_braid_gpt_module(Path(args.braid_gpt_root))
    evaluator = make_evaluator(args, mgpt, bgpt)
    rows: list[dict] = []
    for raw_path in args.teacher_moves:
        path = Path(raw_path)
        rows.extend(iter_jsonl(path))
    rows = [
        row
        for row in rows
        if row.get("action_type") in ACTION_TYPES
        and int(row.get("insert_width", 0)) <= args.max_insert_width
        and int(row.get("delete_width", 0)) <= args.max_delete_width
        and int(row.get("parent_length", 0)) <= args.max_factors
    ]
    if args.max_examples and len(rows) > args.max_examples:
        rng = random.Random(args.seed)
        rows = rng.sample(rows, args.max_examples)
    if not rows:
        raise RuntimeError("No usable teacher rows")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "examples.jsonl"
    examples_path.write_text("", encoding="utf-8")

    total = len(rows)
    tokens = np.zeros((total, args.max_factors + 1), dtype=np.int16)
    matrix_tensors = np.zeros((total, args.matrix_max_degree, 2, evaluator.dim, evaluator.dim), dtype=np.uint8)
    matrix_widths = np.zeros((total,), dtype=np.int16)
    p_values = np.full((total,), args.p, dtype=np.int16)
    powers = np.zeros((total,), dtype=np.int16)
    lengths = np.zeros((total,), dtype=np.int16)
    action_types = np.zeros((total,), dtype=np.int8)
    positions = np.zeros((total,), dtype=np.int16)
    delete_widths = np.zeros((total,), dtype=np.int8)
    insert_widths = np.zeros((total,), dtype=np.int8)
    insert_factors = np.full((total, args.max_insert_width), -100, dtype=np.int16)
    parent_objectives = np.zeros((total,), dtype=np.float32)
    child_objectives = np.zeros((total,), dtype=np.float32)

    log_rows: list[dict] = []
    evaluated_count = 0
    for start in range(0, total, args.eval_batch_size):
        batch = rows[start : start + args.eval_batch_size]
        evaluated = evaluator.evaluate_batch(
            [
                (int(row["power"]), tuple(int(x) for x in row["parent_factors"]))
                for row in batch
            ],
            batch_size=args.eval_batch_size,
        )
        for offset, (row, parent) in enumerate(zip(batch, evaluated)):
            index = start + offset
            factors = tuple(int(x) for x in row["parent_factors"])
            tokens[index] = encode_factors(factors, args.max_factors)
            matrix_tensors[index] = parent.matrix_tensor
            matrix_widths[index] = parent.matrix_width
            powers[index] = int(row["power"])
            lengths[index] = len(factors)
            action_types[index] = ACTION_TYPES[str(row["action_type"])]
            positions[index] = int(row["position"])
            delete_widths[index] = int(row["delete_width"])
            insert = tuple(int(x) for x in row["insert_factors"])
            insert_widths[index] = len(insert)
            insert_factors[index, : len(insert)] = list(insert)
            parent_objectives[index] = np.float32(row["parent_objective"])
            child_objectives[index] = np.float32(row["child_objective"])
            if len(log_rows) < args.log_examples:
                log_rows.append(row)
        evaluated_count += len(batch)
        print(
            json.dumps(
                {
                    "phase": "build_teacher_data",
                    "evaluated": evaluated_count,
                    "total": total,
                    "elapsed_seconds": round(time.time() - start_time, 2),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    append_jsonl(examples_path, log_rows)
    np.savez_compressed(
        output_dir / "teacher_dataset.npz",
        tokens=tokens,
        matrix_tensors=matrix_tensors,
        matrix_widths=matrix_widths,
        p_values=p_values,
        powers=powers,
        lengths=lengths,
        action_types=action_types,
        positions=positions,
        delete_widths=delete_widths,
        insert_widths=insert_widths,
        insert_factors=insert_factors,
        parent_objectives=parent_objectives,
        child_objectives=child_objectives,
    )
    metadata = {
        "format": "braid-teacher-reservoir-dataset-v1",
        "p": args.p,
        "n": args.n,
        "r": args.r,
        "example_count": total,
        "teacher_moves": args.teacher_moves,
        "max_factors": args.max_factors,
        "max_insert_width": args.max_insert_width,
        "max_delete_width": args.max_delete_width,
        "matrix_max_degree": args.matrix_max_degree,
        "matrix_dim": evaluator.dim,
        "action_histogram": dict(Counter(str(ACTION_NAMES[int(value)]) for value in action_types)),
        "insert_width_histogram": dict(Counter(int(value) for value in insert_widths)),
        "delete_width_histogram": dict(Counter(int(value) for value in delete_widths)),
        "objective": objective_metadata(args),
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    write_json(output_dir / "metadata.json", metadata)
    print(json.dumps({"phase": "done", **metadata}, sort_keys=True), flush=True)


@dataclass
class TeacherReservoirConfig:
    p_max: int = 31
    token_vocab_size: int = TOKEN_VOCAB_SIZE
    factor_vocab_size: int = FACTOR_VOCAB_SIZE
    action_type_count: int = len(ACTION_TYPES)
    max_factors: int = 160
    max_insert_width: int = 8
    max_delete_width: int = 8
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


def resolve_device(torch, device_arg: str):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_model(torch, nn, config: TeacherReservoirConfig):
    class TeacherReservoirModel(nn.Module):
        def __init__(self, cfg: TeacherReservoirConfig):
            super().__init__()
            self.config = cfg
            self.token_embedding = nn.Embedding(cfg.token_vocab_size, cfg.d_model, padding_idx=PAD_TOKEN)
            self.position_embedding = nn.Embedding(cfg.max_context_tokens, cfg.d_model)
            self.p_embedding = nn.Embedding(cfg.p_max + 1, cfg.d_model)
            self.matrix_degree_embedding = nn.Embedding(cfg.matrix_max_degree + 1, cfg.d_model)
            self.matrix_projection = nn.Sequential(
                nn.Linear(cfg.matrix_channels * cfg.matrix_dim * cfg.matrix_dim, cfg.d_model),
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
                self.matrix_encoder = nn.TransformerEncoder(matrix_layer, num_layers=cfg.matrix_layers, enable_nested_tensor=False)
                self.braid_encoder = nn.TransformerEncoder(braid_layer, num_layers=cfg.braid_layers, enable_nested_tensor=False)
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
            self.action_head = nn.Linear(cfg.d_model, cfg.action_type_count)
            self.position_head = nn.Linear(cfg.d_model, 1)
            self.delete_width_head = nn.Linear(cfg.d_model, cfg.max_delete_width + 1)
            self.insert_width_head = nn.Linear(cfg.d_model, cfg.max_insert_width)
            self.factor_head = nn.Linear(cfg.d_model, cfg.max_insert_width * cfg.factor_vocab_size)
            self.value_head = nn.Linear(cfg.d_model, 1)

        def encode_matrix(self, matrices, matrix_widths, p_values):
            batch, width = matrices.shape[:2]
            denom = (p_values.float().clamp(min=2.0) - 1.0).view(batch, 1, 1, 1, 1)
            x = matrices.float() / denom
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

        def forward(self, tokens, matrices, matrix_widths, p_values):
            batch, seq_width = tokens.shape
            matrix_hidden, matrix_padding = self.encode_matrix(matrices, matrix_widths.long(), p_values.long())
            positions = torch.arange(seq_width, device=tokens.device)[None, :]
            braid_hidden = self.token_embedding(tokens.long()) + self.position_embedding(positions)
            braid_hidden[:, 0, :] = braid_hidden[:, 0, :] + matrix_hidden[:, 0, :] + self.p_embedding(
                p_values.long().clamp(min=0, max=self.config.p_max)
            )
            braid_padding = tokens.eq(PAD_TOKEN)
            braid_padding[:, 0] = False
            braid_hidden = self.braid_encoder(braid_hidden, src_key_padding_mask=braid_padding)
            cross, _ = self.cross_attention(
                query=braid_hidden,
                key=matrix_hidden,
                value=matrix_hidden,
                key_padding_mask=matrix_padding,
                need_weights=False,
            )
            hidden = self.final_norm(self.fusion_norm(braid_hidden + self.dropout(cross)))
            pooled = hidden[:, 0, :]
            slot_hidden = hidden[:, : self.config.max_context_tokens, :]
            return {
                "action_logits": self.action_head(pooled),
                "position_logits": self.position_head(slot_hidden).squeeze(-1),
                "delete_width_logits": self.delete_width_head(slot_hidden),
                "insert_width_logits": self.insert_width_head(slot_hidden),
                "factor_logits": self.factor_head(slot_hidden).view(
                    batch,
                    self.config.max_context_tokens,
                    self.config.max_insert_width,
                    self.config.factor_vocab_size,
                ),
                "value": self.value_head(pooled).squeeze(-1),
            }

    return TeacherReservoirModel(config)


def save_checkpoint(torch, path: Path, *, model, config: TeacherReservoirConfig, history: list[dict], extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "braid-teacher-reservoir-checkpoint-v1",
            "model_config": config.to_dict(),
            "model_state": model.state_dict(),
            "history": history,
            **extra,
        },
        path,
    )


def load_checkpoint(torch, nn, checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = TeacherReservoirConfig(**checkpoint["model_config"])
    model = build_model(torch, nn, config).to(device)
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
    indices = np.arange(payload["tokens"].shape[0])
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(indices)
    split = max(1, int(indices.size * (1.0 - args.validation_fraction)))
    train_indices = indices[:split]
    val_indices = indices[split:] if split < indices.size else indices[: min(1024, indices.size)]
    device = resolve_device(torch, args.device)
    config = TeacherReservoirConfig(
        p_max=args.p_max,
        max_factors=int(payload["tokens"].shape[1] - 1),
        max_insert_width=int(payload["insert_factors"].shape[1]),
        max_delete_width=int(metadata.get("max_delete_width", int(np.max(payload["delete_widths"])))),
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
    model = build_model(torch, nn, config).to(device)
    history: list[dict] = []
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        old_config = TeacherReservoirConfig(**checkpoint["model_config"])
        if old_config.to_dict() != config.to_dict():
            raise RuntimeError("init checkpoint config does not match current dataset/model config")
        model.load_state_dict(checkpoint["model_state"])
        history = list(checkpoint.get("history", []))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def loader(selected, shuffle):
        return DataLoader(
            TensorDataset(
                torch.tensor(payload["tokens"][selected], dtype=torch.long),
                torch.tensor(payload["matrix_tensors"][selected], dtype=torch.uint8),
                torch.tensor(payload["matrix_widths"][selected], dtype=torch.long),
                torch.tensor(payload["p_values"][selected], dtype=torch.long),
                torch.tensor(payload["lengths"][selected], dtype=torch.long),
                torch.tensor(payload["action_types"][selected], dtype=torch.long),
                torch.tensor(payload["positions"][selected], dtype=torch.long),
                torch.tensor(payload["delete_widths"][selected], dtype=torch.long),
                torch.tensor(payload["insert_widths"][selected], dtype=torch.long),
                torch.tensor(payload["insert_factors"][selected], dtype=torch.long),
                torch.tensor(payload["parent_objectives"][selected], dtype=torch.float32),
                torch.tensor(payload["child_objectives"][selected], dtype=torch.float32),
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
        factor_slots = 0
        for batch in data_loader:
            (
                tokens,
                matrices,
                matrix_widths,
                p_values,
                lengths,
                action_types,
                positions,
                delete_widths,
                insert_widths,
                insert_factors,
                parent_objectives,
                child_objectives,
            ) = [item.to(device) for item in batch]
            batch_ids = torch.arange(tokens.shape[0], device=device)
            with torch.set_grad_enabled(train_mode):
                out = model(tokens, matrices, matrix_widths, p_values)
                position_ids = torch.arange(config.max_context_tokens, device=device)[None, :]
                position_mask = position_ids <= lengths[:, None].clamp(max=config.max_factors)
                position_logits = out["position_logits"].masked_fill(~position_mask, -1e9)
                selected_delete_logits = out["delete_width_logits"][batch_ids, positions]
                selected_insert_logits = out["insert_width_logits"][batch_ids, positions]
                selected_factor_logits = out["factor_logits"][batch_ids, positions]
                value_target = torch.log1p(torch.clamp(parent_objectives - child_objectives, min=0.0))
                action_loss = F.cross_entropy(out["action_logits"], action_types)
                position_loss = F.cross_entropy(position_logits, positions)
                delete_loss = F.cross_entropy(selected_delete_logits, delete_widths)
                insert_loss = F.cross_entropy(selected_insert_logits, insert_widths - 1)
                factor_loss = F.cross_entropy(
                    selected_factor_logits.reshape(-1, config.factor_vocab_size),
                    insert_factors.reshape(-1),
                    ignore_index=-100,
                )
                value_loss = F.smooth_l1_loss(out["value"], value_target)
                loss = (
                    args.action_loss_weight * action_loss
                    + args.position_loss_weight * position_loss
                    + args.delete_loss_weight * delete_loss
                    + args.insert_loss_weight * insert_loss
                    + args.factor_loss_weight * factor_loss
                    + args.value_loss_weight * value_loss
                )
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
            size = tokens.shape[0]
            pred_positions = torch.argmax(position_logits, dim=-1)
            pred_deletes = torch.argmax(selected_delete_logits, dim=-1)
            pred_inserts = torch.argmax(selected_insert_logits, dim=-1) + 1
            pred_actions = torch.argmax(out["action_logits"], dim=-1)
            pred_factors = torch.argmax(selected_factor_logits, dim=-1)
            factor_mask = insert_factors.ne(-100)
            factor_correct = (pred_factors.eq(insert_factors) & factor_mask).sum()
            exact_factor = ((pred_factors.eq(insert_factors) | ~factor_mask).all(dim=-1)).sum()
            exact_action = (
                pred_actions.eq(action_types)
                & pred_positions.eq(positions)
                & pred_deletes.eq(delete_widths)
                & pred_inserts.eq(insert_widths)
                & (pred_factors.eq(insert_factors) | ~factor_mask).all(dim=-1)
            ).sum()
            factor_slots += int(factor_mask.sum().detach().cpu())
            totals["loss"] += float(loss.detach().cpu()) * size
            totals["action_loss"] += float(action_loss.detach().cpu()) * size
            totals["position_loss"] += float(position_loss.detach().cpu()) * size
            totals["delete_loss"] += float(delete_loss.detach().cpu()) * size
            totals["insert_loss"] += float(insert_loss.detach().cpu()) * size
            totals["factor_loss"] += float(factor_loss.detach().cpu()) * size
            totals["value_loss"] += float(value_loss.detach().cpu()) * size
            totals["action_top1"] += int(pred_actions.eq(action_types).sum().detach().cpu())
            totals["position_top1"] += int(pred_positions.eq(positions).sum().detach().cpu())
            totals["delete_top1"] += int(pred_deletes.eq(delete_widths).sum().detach().cpu())
            totals["insert_top1"] += int(pred_inserts.eq(insert_widths).sum().detach().cpu())
            totals["factor_exact"] += int(exact_factor.detach().cpu())
            totals["factor_slot_top1"] += int(factor_correct.detach().cpu())
            totals["exact_action"] += int(exact_action.detach().cpu())
            total += size
        return {
            "loss": totals["loss"] / max(1, total),
            "action_loss": totals["action_loss"] / max(1, total),
            "position_loss": totals["position_loss"] / max(1, total),
            "delete_loss": totals["delete_loss"] / max(1, total),
            "insert_loss": totals["insert_loss"] / max(1, total),
            "factor_loss": totals["factor_loss"] / max(1, total),
            "value_loss": totals["value_loss"] / max(1, total),
            "action_top1": totals["action_top1"] / max(1, total),
            "position_top1": totals["position_top1"] / max(1, total),
            "delete_top1": totals["delete_top1"] / max(1, total),
            "insert_top1": totals["insert_top1"] / max(1, total),
            "factor_exact": totals["factor_exact"] / max(1, total),
            "factor_slot_top1": totals["factor_slot_top1"] / max(1, factor_slots),
            "exact_action": totals["exact_action"] / max(1, total),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(train_loader, True)
        val_stats = run_epoch(val_loader, False)
        row = {"phase": "teacher_reservoir_train", "epoch": epoch, "train": train_stats, "validation": val_stats}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(
                torch,
                output_dir / "teacher_reservoir.pt",
                model=model,
                config=config,
                history=history,
                extra={"dataset_metadata": metadata, "best_validation_loss": best_val},
            )
    write_json(output_dir / "training_summary.json", {"history": history, "best_validation_loss": best_val})


def top_factor_combos(
    logits: np.ndarray,
    *,
    width: int,
    choices_per_slot: int,
    limit: int,
) -> list[tuple[tuple[int, ...], float]]:
    slot_choices: list[list[tuple[int, float]]] = []
    for slot in range(width):
        order = np.argsort(logits[slot])[::-1][:choices_per_slot]
        slot_choices.append([(int(index), float(logits[slot, index])) for index in order])
    combos: list[tuple[tuple[int, ...], float]] = []
    for product in itertools.product(*slot_choices):
        factors = tuple(item[0] for item in product)
        combos.append((factors, sum(item[1] for item in product)))
    combos.sort(key=lambda item: item[1], reverse=True)
    return combos[:limit]


def propose_model_edits(
    *,
    automaton,
    rng: random.Random,
    state: SearchState,
    out: dict[str, np.ndarray],
    positions_per_state: int,
    delete_widths_per_position: int,
    insert_widths_per_position: int,
    factor_choices_per_slot: int,
    edits_per_state: int,
    random_bridge_per_position: int,
    max_factors: int,
) -> list[tuple[tuple[int, ...], float, dict]]:
    length = len(state.factors)
    position_scores = out["position_logits"][: length + 1].copy()
    position_order = np.argsort(position_scores)[::-1][:positions_per_state]
    action_bonus = float(np.max(out["action_logits"]))
    proposals: list[tuple[tuple[int, ...], float, dict]] = []
    seen: set[tuple[int, ...]] = set()
    for position in position_order:
        position = int(position)
        max_delete = min(out["delete_width_logits"].shape[-1] - 1, length - position)
        delete_order = np.argsort(out["delete_width_logits"][position, : max_delete + 1])[::-1][:delete_widths_per_position]
        insert_order = np.argsort(out["insert_width_logits"][position])[::-1][:insert_widths_per_position]
        for delete_width in delete_order:
            delete_width = int(delete_width)
            for insert_index in insert_order:
                insert_width = int(insert_index) + 1
                if length - delete_width + insert_width > max_factors:
                    continue
                combos = top_factor_combos(
                    out["factor_logits"][position],
                    width=insert_width,
                    choices_per_slot=factor_choices_per_slot,
                    limit=max(1, edits_per_state // max(1, positions_per_state)),
                )
                for insert, factor_score in combos:
                    child = apply_edit(
                        state.factors,
                        position=position,
                        delete_width=delete_width,
                        insert_factors=insert,
                    )
                    if child in seen or child == state.factors or not automaton.is_legal(child):
                        continue
                    seen.add(child)
                    score = (
                        action_bonus
                        + float(position_scores[position])
                        + float(out["delete_width_logits"][position, delete_width])
                        + float(out["insert_width_logits"][position, insert_width - 1])
                        + factor_score
                    )
                    proposals.append(
                        (
                            child,
                            score,
                            {
                                "position": position,
                                "delete_width": delete_width,
                                "insert_width": insert_width,
                                "insert_factors": list(insert),
                                "source": "model",
                            },
                        )
                    )
                for sample_id in range(random_bridge_per_position):
                    try:
                        bridge = legal_bridge_for_edit(automaton, state.factors, position, delete_width, insert_width, rng)
                    except ValueError:
                        continue
                    child = apply_edit(
                        state.factors,
                        position=position,
                        delete_width=delete_width,
                        insert_factors=bridge,
                    )
                    if child in seen or child == state.factors or not automaton.is_legal(child):
                        continue
                    seen.add(child)
                    score = (
                        float(position_scores[position])
                        + float(out["delete_width_logits"][position, delete_width])
                        + float(out["insert_width_logits"][position, insert_width - 1])
                        - 0.25 * (sample_id + 1)
                    )
                    proposals.append(
                        (
                            child,
                            score,
                            {
                                "position": position,
                                "delete_width": delete_width,
                                "insert_width": insert_width,
                                "insert_factors": list(bridge),
                                "source": "legal_bridge_sample",
                            },
                        )
                    )
    proposals.sort(key=lambda item: item[1], reverse=True)
    return proposals[:edits_per_state]


def unique_best(states: Sequence[SearchState]) -> list[SearchState]:
    by_key: dict[tuple[int, tuple[int, ...]], SearchState] = {}
    for state in states:
        key = (state.power % 2, state.factors)
        old = by_key.get(key)
        if old is None or score_tuple(state) < score_tuple(old):
            by_key[key] = state
    return list(by_key.values())


def select_diverse(
    states: Sequence[SearchState],
    *,
    global_limit: int,
    per_length_keep: int,
    objective_keep: int,
    identity_keep: int,
    projlen_keep: int,
    random_keep: int,
    rng: random.Random,
) -> list[SearchState]:
    unique = unique_best(states)
    if per_length_keep <= 0:
        return sorted(unique, key=score_tuple)[:global_limit]
    by_length: dict[int, list[SearchState]] = {}
    for state in unique:
        by_length.setdefault(len(state.factors), []).append(state)
    selected: dict[tuple[int, tuple[int, ...]], SearchState] = {}
    for length, bucket in by_length.items():
        local: list[SearchState] = []
        local.extend(sorted(bucket, key=score_tuple)[:objective_keep])
        local.extend(
            sorted(
                bucket,
                key=lambda state: (
                    int(state.metrics.get("identity_defect", 10**9)),
                    metric_projlen(state.metrics),
                    float(state.objective),
                ),
            )[:identity_keep]
        )
        local.extend(
            sorted(
                bucket,
                key=lambda state: (
                    metric_projlen(state.metrics),
                    int(state.metrics.get("identity_defect", 10**9)),
                    float(state.objective),
                ),
            )[:projlen_keep]
        )
        remaining = [state for state in bucket if (state.power % 2, state.factors) not in {(item.power % 2, item.factors) for item in local}]
        if random_keep > 0 and remaining:
            local.extend(rng.sample(remaining, min(random_keep, len(remaining))))
        local = sorted(unique_best(local), key=score_tuple)[:per_length_keep]
        for state in local:
            selected[(state.power % 2, state.factors)] = state
    output = sorted(selected.values(), key=score_tuple)
    return output[:global_limit] if global_limit else output


def length_counts(states: Sequence[SearchState]) -> dict[str, int]:
    counts = Counter(len(state.factors) for state in states)
    return {str(length): int(counts[length]) for length in sorted(counts)}


def search(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn

    start_time = time.time()
    mgpt = load_matrix_gpt(Path(args.matrix_gpt_root))
    bgpt = mgpt.load_braid_gpt_module(Path(args.braid_gpt_root))
    automaton = bgpt.GNFAutomaton(args.n)
    rng = random.Random(args.seed)
    device = resolve_device(torch, args.device)
    model, config, checkpoint = load_checkpoint(torch, nn, Path(args.checkpoint), device)
    evaluator = make_evaluator(args, mgpt, bgpt, matrix_max_degree=config.matrix_max_degree)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    candidates_path = output_dir / "candidates.jsonl"
    progress_path.write_text("", encoding="utf-8")
    candidates_path.write_text("", encoding="utf-8")
    search_min_length = args.search_min_length if args.search_min_length > 0 else args.root_min_length
    search_max_length = args.search_max_length if args.search_max_length > 0 else args.root_max_length

    roots: list[tuple[int, tuple[int, ...]]] = []
    seed_candidates = load_seed_candidates(
        args.seed_source,
        min_length=args.root_min_length,
        max_length=args.root_max_length,
        limit=args.seed_roots,
    )
    roots.extend((seed.power, seed.factors) for seed in seed_candidates)
    while len(roots) < args.seed_roots + args.random_roots:
        length = rng.randint(args.root_min_length, args.root_max_length)
        roots.append((rng.choice((0, 1)), tuple(int(x) for x in automaton.sample_uniform(length, rng))))
    if not roots:
        raise RuntimeError("No roots were created")

    def evaluate_words(words: Sequence[tuple[int, tuple[int, ...]]]) -> list[tuple[SearchState, np.ndarray]]:
        if not words:
            return []
        evaluated = evaluator.evaluate_batch(words, batch_size=args.eval_batch_size)
        return [(evaluated_to_state(item), item.matrix_tensor) for item in evaluated]

    root_pairs = evaluate_words(roots)
    matrix_by_key = {(state.power % 2, state.factors): matrix for state, matrix in root_pairs}
    frontier = select_diverse(
        [state for state, _ in root_pairs],
        global_limit=args.beam_size,
        per_length_keep=args.per_length_keep,
        objective_keep=args.objective_keep,
        identity_keep=args.identity_keep,
        projlen_keep=args.projlen_keep,
        random_keep=args.random_keep,
        rng=rng,
    )
    best = select_diverse(
        frontier,
        global_limit=args.keep_best,
        per_length_keep=args.best_per_length_keep,
        objective_keep=args.objective_keep,
        identity_keep=args.identity_keep,
        projlen_keep=args.projlen_keep,
        random_keep=args.random_keep,
        rng=rng,
    )
    seen = {(state.power % 2, state.factors) for state in frontier}
    kernel_hits = [state for state in best if state.metrics.get("scalar_identity") and len(state.factors) > 0]

    for step in range(1, args.steps + 1):
        expandable = [state for state in frontier if len(state.factors) <= config.max_factors]
        missing = [
            (state.power, state.factors)
            for state in expandable
            if (state.power % 2, state.factors) not in matrix_by_key
        ]
        for state, matrix in evaluate_words(missing):
            matrix_by_key[(state.power % 2, state.factors)] = matrix
        child_words: list[tuple[int, tuple[int, ...]]] = []
        child_meta: dict[tuple[int, tuple[int, ...]], dict] = {}
        child_parent: dict[tuple[int, tuple[int, ...]], SearchState] = {}
        for batch_start in range(0, len(expandable), args.policy_batch_size):
            batch_states = expandable[batch_start : batch_start + args.policy_batch_size]
            tokens = np.stack([encode_factors(state.factors, config.max_factors) for state in batch_states])
            matrices = np.stack([matrix_by_key[(state.power % 2, state.factors)] for state in batch_states])
            widths = np.array([state.matrix_width for state in batch_states], dtype=np.int64)
            p_values = np.full((len(batch_states),), args.model_p if args.model_p > 0 else args.p, dtype=np.int64)
            with torch.no_grad():
                raw = model(
                    torch.tensor(tokens, dtype=torch.long, device=device),
                    torch.tensor(matrices, dtype=torch.uint8, device=device),
                    torch.tensor(widths, dtype=torch.long, device=device),
                    torch.tensor(p_values, dtype=torch.long, device=device),
                )
            raw_np = {key: value.detach().cpu().numpy() for key, value in raw.items()}
            for local_index, state in enumerate(batch_states):
                one = {key: value[local_index] for key, value in raw_np.items()}
                proposals = propose_model_edits(
                    automaton=automaton,
                    rng=rng,
                    state=state,
                    out=one,
                    positions_per_state=args.positions_per_state,
                    delete_widths_per_position=args.delete_widths_per_position,
                    insert_widths_per_position=args.insert_widths_per_position,
                    factor_choices_per_slot=args.factor_choices_per_slot,
                    edits_per_state=args.edits_per_state,
                    random_bridge_per_position=args.random_bridge_per_position,
                    max_factors=config.max_factors,
                )
                for child_factors, model_score, meta in proposals:
                    if len(child_factors) < search_min_length or len(child_factors) > search_max_length:
                        continue
                    key = (state.power % 2, child_factors)
                    if key in seen:
                        continue
                    seen.add(key)
                    meta["model_score"] = model_score
                    child_words.append((state.power, child_factors))
                    child_meta[key] = meta
                    child_parent[key] = state
        child_pairs = evaluate_words(child_words)
        accepted: list[SearchState] = []
        candidate_rows: list[dict] = []
        for child, matrix in child_pairs:
            key = (child.power % 2, child.factors)
            parent = child_parent[key]
            if args.accept_only_improvements and child.objective >= parent.objective:
                continue
            matrix_by_key[key] = matrix
            accepted.append(child)
            candidate_rows.append(
                {
                    "step": step,
                    "parent": state_record(parent),
                    "objective_delta": child.objective - parent.objective,
                    "edit": child_meta.get(key, {}),
                    **state_record(child),
                    "degeneracy": bgpt.degeneracy_features(child.factors),
                }
            )
        append_jsonl(candidates_path, candidate_rows)
        kernel_hits.extend(state for state in accepted if state.metrics.get("scalar_identity") and len(state.factors) > 0)
        best = select_diverse(
            best + accepted,
            global_limit=args.keep_best,
            per_length_keep=args.best_per_length_keep,
            objective_keep=args.objective_keep,
            identity_keep=args.identity_keep,
            projlen_keep=args.projlen_keep,
            random_keep=args.random_keep,
            rng=rng,
        )
        frontier = select_diverse(
            best + accepted,
            global_limit=args.beam_size,
            per_length_keep=args.per_length_keep,
            objective_keep=args.objective_keep,
            identity_keep=args.identity_keep,
            projlen_keep=args.projlen_keep,
            random_keep=args.random_keep,
            rng=rng,
        )
        row = {
            "phase": "teacher_reservoir_search",
            "step": step,
            "expanded": len(child_words),
            "accepted_children": len(accepted),
            "frontier_size": len(frontier),
            "frontier_length_counts": length_counts(frontier),
            "best_length_counts": length_counts(best),
            "best_objective": best[0].objective if best else None,
            "best_length": len(best[0].factors) if best else None,
            "best_metrics": best[0].metrics if best else {},
            "kernel_hits": len(kernel_hits),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        append_jsonl(progress_path, [row])
        print(json.dumps(row, sort_keys=True), flush=True)
        if kernel_hits and args.stop_at_kernel:
            break

    summary = {
        "format": "braid-teacher-reservoir-search-summary-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": checkpoint.get("dataset_metadata", {}),
        "kernel_hits": [state_record(state) for state in kernel_hits[: args.keep_best]],
        "best": [state_record(state) for state in best],
        "best_by_identity_defect": [
            state_record(state)
            for state in sorted(
                best,
                key=lambda state: (
                    int(state.metrics.get("identity_defect", 10**9)),
                    metric_projlen(state.metrics),
                    float(state.objective),
                ),
            )
        ],
        "search": {
            "steps": args.steps,
            "beam_size": args.beam_size,
            "keep_best": args.keep_best,
            "per_length_keep": args.per_length_keep,
            "best_per_length_keep": args.best_per_length_keep,
            "objective_keep": args.objective_keep,
            "identity_keep": args.identity_keep,
            "projlen_keep": args.projlen_keep,
            "random_keep": args.random_keep,
            "positions_per_state": args.positions_per_state,
            "delete_widths_per_position": args.delete_widths_per_position,
            "insert_widths_per_position": args.insert_widths_per_position,
            "factor_choices_per_slot": args.factor_choices_per_slot,
            "edits_per_state": args.edits_per_state,
            "random_bridge_per_position": args.random_bridge_per_position,
            "accept_only_improvements": args.accept_only_improvements,
            "search_min_length": search_min_length,
            "search_max_length": search_max_length,
            "evaluator_p": args.p,
            "model_p": args.model_p if args.model_p > 0 else args.p,
        },
        "objective": objective_metadata(args),
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
    parser = argparse.ArgumentParser(description="Exact-teacher, transformer-guided reservoir search for braid kernels.")
    parser.add_argument("--matrix-gpt-root", default=str(DEFAULT_MATRIX_GPT_ROOT))
    parser.add_argument("--braid-gpt-root", default=str(DEFAULT_BRAID_GPT_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    mine = sub.add_parser("mine-teacher")
    mine.add_argument("--author-repo", default=str(DEFAULT_AUTHOR_REPO))
    mine.add_argument("--seed-source", action="append", default=[])
    mine.add_argument("--output-dir", required=True)
    mine.add_argument("--p", type=int, default=7)
    mine.add_argument("--n", type=int, default=4)
    mine.add_argument("--r", type=int, default=1)
    mine.add_argument("--seed-limit", type=int, default=2000)
    mine.add_argument("--min-seed-length", type=int, default=40)
    mine.add_argument("--max-seed-length", type=int, default=180)
    mine.add_argument("--max-factors", type=int, default=180)
    mine.add_argument("--matrix-max-degree", type=int, default=256)
    mine.add_argument("--right-lengths", default="1,2,3,4,5,6")
    mine.add_argument("--left-lengths", default="1,2,3,4,5,6")
    mine.add_argument("--window-widths", default="2,3,4,5,6,7,8")
    mine.add_argument("--right-samples-per-length", type=int, default=12)
    mine.add_argument("--left-samples-per-length", type=int, default=12)
    mine.add_argument("--windows-per-width", type=int, default=24)
    mine.add_argument("--bridge-attempts", type=int, default=80)
    mine.add_argument("--min-objective-improvement", type=float, default=0.05)
    mine.add_argument("--min-identity-improvement", type=int, default=1)
    mine.add_argument("--keep-teacher-per-seed", type=int, default=16)
    mine.add_argument("--keep-scored-per-seed", type=int, default=32)
    mine.add_argument("--eval-batch-size", type=int, default=500)
    mine.add_argument("--progress-every", type=int, default=25)
    mine.add_argument("--seed", type=int, default=1)
    add_objective_args(mine)
    mine.set_defaults(func=mine_teacher)

    data = sub.add_parser("build-data")
    data.add_argument("--author-repo", default=str(DEFAULT_AUTHOR_REPO))
    data.add_argument("--teacher-moves", action="append", default=[])
    data.add_argument("--output-dir", required=True)
    data.add_argument("--p", type=int, default=7)
    data.add_argument("--n", type=int, default=4)
    data.add_argument("--r", type=int, default=1)
    data.add_argument("--max-examples", type=int, default=0)
    data.add_argument("--max-factors", type=int, default=180)
    data.add_argument("--max-insert-width", type=int, default=8)
    data.add_argument("--max-delete-width", type=int, default=8)
    data.add_argument("--matrix-max-degree", type=int, default=256)
    data.add_argument("--eval-batch-size", type=int, default=500)
    data.add_argument("--log-examples", type=int, default=200)
    data.add_argument("--seed", type=int, default=1)
    add_objective_args(data)
    data.set_defaults(func=build_teacher_data)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--init-checkpoint", default="")
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--batch-size", type=int, default=96)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--action-loss-weight", type=float, default=0.5)
    train_parser.add_argument("--position-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--delete-loss-weight", type=float, default=0.5)
    train_parser.add_argument("--insert-loss-weight", type=float, default=0.5)
    train_parser.add_argument("--factor-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--value-loss-weight", type=float, default=0.15)
    train_parser.add_argument("--validation-fraction", type=float, default=0.05)
    train_parser.add_argument("--d-model", type=int, default=256)
    train_parser.add_argument("--nhead", type=int, default=8)
    train_parser.add_argument("--braid-layers", type=int, default=6)
    train_parser.add_argument("--matrix-layers", type=int, default=3)
    train_parser.add_argument("--dim-feedforward", type=int, default=1024)
    train_parser.add_argument("--dropout", type=float, default=0.10)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--p-max", type=int, default=31)
    train_parser.add_argument("--seed", type=int, default=1)
    train_parser.set_defaults(func=train)

    search_parser = sub.add_parser("search")
    search_parser.add_argument("--author-repo", default=str(DEFAULT_AUTHOR_REPO))
    search_parser.add_argument("--checkpoint", required=True)
    search_parser.add_argument("--seed-source", action="append", default=[])
    search_parser.add_argument("--output-dir", required=True)
    search_parser.add_argument("--p", type=int, default=7)
    search_parser.add_argument("--model-p", type=int, default=0)
    search_parser.add_argument("--n", type=int, default=4)
    search_parser.add_argument("--r", type=int, default=1)
    search_parser.add_argument("--device", default="auto")
    search_parser.add_argument("--root-min-length", type=int, default=50)
    search_parser.add_argument("--root-max-length", type=int, default=160)
    search_parser.add_argument("--search-min-length", type=int, default=0)
    search_parser.add_argument("--search-max-length", type=int, default=0)
    search_parser.add_argument("--seed-roots", type=int, default=1000)
    search_parser.add_argument("--random-roots", type=int, default=512)
    search_parser.add_argument("--steps", type=int, default=80)
    search_parser.add_argument("--beam-size", type=int, default=6000)
    search_parser.add_argument("--keep-best", type=int, default=1500)
    search_parser.add_argument("--per-length-keep", type=int, default=80)
    search_parser.add_argument("--best-per-length-keep", type=int, default=40)
    search_parser.add_argument("--objective-keep", type=int, default=36)
    search_parser.add_argument("--identity-keep", type=int, default=18)
    search_parser.add_argument("--projlen-keep", type=int, default=18)
    search_parser.add_argument("--random-keep", type=int, default=16)
    search_parser.add_argument("--positions-per-state", type=int, default=12)
    search_parser.add_argument("--delete-widths-per-position", type=int, default=3)
    search_parser.add_argument("--insert-widths-per-position", type=int, default=3)
    search_parser.add_argument("--factor-choices-per-slot", type=int, default=2)
    search_parser.add_argument("--edits-per-state", type=int, default=32)
    search_parser.add_argument("--random-bridge-per-position", type=int, default=2)
    search_parser.add_argument("--eval-batch-size", type=int, default=400)
    search_parser.add_argument("--policy-batch-size", type=int, default=128)
    search_parser.add_argument("--accept-only-improvements", action="store_true")
    search_parser.add_argument("--stop-at-kernel", action="store_true")
    search_parser.add_argument("--matrix-max-degree", type=int, default=256)
    search_parser.add_argument("--seed", type=int, default=1)
    add_objective_args(search_parser)
    search_parser.set_defaults(func=search)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
