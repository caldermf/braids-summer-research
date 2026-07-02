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


def load_matrix_gpt_module(matrix_gpt_root: Path):
    module_path = matrix_gpt_root / "matrix_gpt.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find Matrix-GPT script at {module_path}")
    spec = importlib.util.spec_from_file_location("matrix_gpt_runtime_for_diffusion_repair", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Matrix-GPT from {module_path}")
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


def parse_weight_spec(spec: str, *, default: dict[int, float]) -> dict[int, float]:
    if not spec:
        return dict(default)
    weights: dict[int, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("weight specs must look like 1:0.5,2:1.0")
        key, value = part.split(":", 1)
        weights[int(key)] = float(value)
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0:
        raise ValueError("at least one noise level must have positive weight")
    return weights


def weighted_choice(weights: dict[int, float], rng: random.Random) -> int:
    total = sum(max(0.0, value) for value in weights.values())
    draw = rng.random() * total
    seen = 0.0
    for key in sorted(weights):
        seen += max(0.0, weights[key])
        if draw <= seen:
            return key
    return sorted(weights)[-1]


def hamming_distance(left: Sequence[int], right: Sequence[int]) -> int:
    return sum(int(a != b) for a, b in zip(left, right)) + abs(len(left) - len(right))


def replace_window(factors: Sequence[int], start: int, width: int, replacement: Sequence[int]) -> tuple[int, ...]:
    return tuple(factors[:start]) + tuple(int(x) for x in replacement) + tuple(factors[start + width :])


@dataclass(frozen=True)
class CleanKernel:
    kernel_id: int
    power: int
    factors: tuple[int, ...]
    source: str


@dataclass(frozen=True)
class CorruptedRepairExample:
    clean: CleanKernel
    corrupted_factors: tuple[int, ...]
    noise_level: int
    corruption_kind: str
    target_start: int
    target_width: int
    target_factors: tuple[int, ...]
    hamming_to_clean: int


def row_factor_power(row: dict) -> tuple[int, tuple[int, ...]] | None:
    keys = ("factor_ids", "final_factors", "powered_factors", "factors")
    factors = None
    for key in keys:
        if key in row:
            factors = row[key]
            break
    if factors is None:
        return None
    try:
        factors_tuple = tuple(int(value) for value in factors)
    except (TypeError, ValueError):
        return None
    power = int(row.get("power", row.get("final_power", row.get("powered_power", 0))))
    return power, factors_tuple


def row_has_explicit_power(row: dict) -> bool:
    return any(key in row and row[key] is not None for key in ("power", "final_power", "powered_power"))


def walk_factor_rows(obj) -> Iterable[dict]:
    if isinstance(obj, dict):
        parsed = row_factor_power(obj)
        if parsed is not None:
            power, factors = parsed
            yield {
                "power": power,
                "factor_ids": factors,
                "power_unspecified": not row_has_explicit_power(obj),
            }
        for value in obj.values():
            yield from walk_factor_rows(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_factor_rows(value)


def load_kernel_rows(paths: Sequence[str]) -> Iterable[tuple[int, tuple[int, ...], str]]:
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            print(json.dumps({"phase": "missing_kernel_source", "path": str(path)}), flush=True)
            continue
        if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"):
            rows = iter_jsonl(path)
        else:
            rows = walk_factor_rows(read_json(path))
        for row in rows:
            parsed = row_factor_power(row)
            if parsed is None:
                continue
            power, factors = parsed
            powers = (0, 1) if row.get("power_unspecified", not row_has_explicit_power(row)) else (int(power),)
            for candidate_power in powers:
                yield int(candidate_power), tuple(int(x) for x in factors), str(path)


def load_clean_kernels(
    *,
    mgpt,
    bgpt,
    evaluator,
    kernel_sources: Sequence[str],
    min_length: int,
    max_length: int,
    max_kernels: int,
    verify: bool,
    reject_degenerate: bool,
    augment_repeats: int,
    augment_rotations_per_kernel: int,
) -> list[CleanKernel]:
    seen: set[tuple[int, tuple[int, ...]]] = set()
    raw_kernels: list[CleanKernel] = []
    for power, factors, source in load_kernel_rows(kernel_sources):
        if len(factors) < min_length or len(factors) > max_length:
            continue
        key = (power % 2, factors)
        if key in seen:
            continue
        if reject_degenerate:
            deg = bgpt.degeneracy_features(factors)
            if deg["period_at_most_2"] or deg["unique_fraction"] < 0.08:
                continue
        seen.add(key)
        raw_kernels.append(CleanKernel(len(raw_kernels), int(power), factors, source))
        if max_kernels and len(raw_kernels) >= max_kernels:
            break
    if not raw_kernels:
        raise RuntimeError("No clean kernel candidates were loaded from --kernel-source paths")
    if not verify:
        return raw_kernels
    verified: list[CleanKernel] = []
    for start in range(0, len(raw_kernels), 500):
        chunk = raw_kernels[start : start + 500]
        evaluated = evaluator.evaluate_batch(
            [(item.power, item.factors) for item in chunk],
            batch_size=500,
        )
        for clean, result in zip(chunk, evaluated):
            if result.metrics.get("scalar_identity") and len(clean.factors) > 0:
                verified.append(CleanKernel(len(verified), clean.power, clean.factors, clean.source))
    if not verified:
        raise RuntimeError("Kernel sources loaded, but none verified as scalar identity for this p")
    automaton = bgpt.GNFAutomaton(evaluator.n)
    augmented_candidates: list[CleanKernel] = []
    seen_verified = {(item.power % 2, item.factors) for item in verified}
    if augment_repeats > 1:
        for clean in verified:
            for repeat in range(2, augment_repeats + 1):
                factors = clean.factors * repeat
                if len(factors) > max_length or not automaton.is_legal(factors):
                    continue
                for power in (0, 1):
                    key = (power % 2, factors)
                    if key in seen_verified:
                        continue
                    seen_verified.add(key)
                    augmented_candidates.append(
                        CleanKernel(
                            len(verified) + len(augmented_candidates),
                            power,
                            factors,
                            f"{clean.source}#repeat{repeat}",
                        )
                    )
    if augment_rotations_per_kernel > 0:
        for clean in verified:
            if len(clean.factors) < 4:
                continue
            offsets = sorted(
                {
                    1 + (index * max(1, len(clean.factors) // max(1, augment_rotations_per_kernel)))
                    % (len(clean.factors) - 1)
                    for index in range(augment_rotations_per_kernel)
                }
            )
            for offset in offsets:
                factors = clean.factors[offset:] + clean.factors[:offset]
                if len(factors) > max_length or not automaton.is_legal(factors):
                    continue
                for power in (0, 1):
                    key = (power % 2, factors)
                    if key in seen_verified:
                        continue
                    seen_verified.add(key)
                    augmented_candidates.append(
                        CleanKernel(
                            len(verified) + len(augmented_candidates),
                            power,
                            factors,
                            f"{clean.source}#rotation{offset}",
                        )
                    )
    if augmented_candidates:
        kept_augmented: list[CleanKernel] = []
        for start in range(0, len(augmented_candidates), 500):
            chunk = augmented_candidates[start : start + 500]
            evaluated = evaluator.evaluate_batch(
                [(item.power, item.factors) for item in chunk],
                batch_size=500,
            )
            for clean, result in zip(chunk, evaluated):
                if result.metrics.get("scalar_identity") and len(clean.factors) > 0:
                    kept_augmented.append(
                        CleanKernel(
                            len(verified) + len(kept_augmented),
                            clean.power,
                            clean.factors,
                            clean.source,
                        )
                    )
        verified.extend(kept_augmented)
    return verified


def sample_disjoint_windows(
    *,
    length: int,
    count: int,
    min_width: int,
    max_width: int,
    rng: random.Random,
    require_gap: bool = True,
) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    attempts = 0
    max_width = min(max_width, length)
    while len(windows) < count and attempts < 2000:
        attempts += 1
        width = rng.randint(min_width, max_width)
        start = rng.randint(0, length - width)
        left = start - (1 if require_gap else 0)
        right = start + width + (1 if require_gap else 0)
        if all(right <= other_start or left >= other_start + other_width for other_start, other_width in windows):
            windows.append((start, width))
    if len(windows) < count:
        raise ValueError("could not sample disjoint windows")
    return sorted(windows)


def sample_different_bridge(
    automaton,
    current: Sequence[int],
    start: int,
    width: int,
    rng: random.Random,
    *,
    attempts: int = 80,
) -> tuple[int, ...] | None:
    left = int(current[start - 1]) if start > 0 else None
    right = int(current[start + width]) if start + width < len(current) else None
    old = tuple(int(x) for x in current[start : start + width])
    for _ in range(attempts):
        try:
            bridge = tuple(int(x) for x in automaton.sample_bridge(left, right, width, rng))
        except ValueError:
            return None
        if bridge != old:
            return bridge
    return None


def corrupt_by_windows(
    automaton,
    clean: CleanKernel,
    windows: Sequence[tuple[int, int]],
    rng: random.Random,
    *,
    max_repair_window: int,
    corruption_kind: str,
) -> CorruptedRepairExample | None:
    current = list(clean.factors)
    changed_windows: list[tuple[int, int]] = []
    for start, width in sorted(windows):
        bridge = sample_different_bridge(automaton, current, start, width, rng)
        if bridge is None:
            return None
        current[start : start + width] = list(bridge)
        changed_windows.append((start, width))
        if not automaton.is_legal(current):
            return None
    corrupted = tuple(current)
    if corrupted == clean.factors:
        return None
    target_candidates: list[tuple[int, int]] = []
    for start, width in changed_windows:
        if width > max_repair_window:
            continue
        repaired = replace_window(corrupted, start, width, clean.factors[start : start + width])
        if automaton.is_legal(repaired) and hamming_distance(repaired, clean.factors) < hamming_distance(corrupted, clean.factors):
            target_candidates.append((start, width))
    if not target_candidates:
        return None
    target_start, target_width = rng.choice(target_candidates)
    return CorruptedRepairExample(
        clean=clean,
        corrupted_factors=corrupted,
        noise_level=0,
        corruption_kind=corruption_kind,
        target_start=int(target_start),
        target_width=int(target_width),
        target_factors=tuple(int(x) for x in clean.factors[target_start : target_start + target_width]),
        hamming_to_clean=hamming_distance(corrupted, clean.factors),
    )


def make_corruption(
    automaton,
    clean: CleanKernel,
    noise_level: int,
    rng: random.Random,
    *,
    max_repair_window: int,
) -> CorruptedRepairExample | None:
    length = len(clean.factors)
    if length < 2:
        return None
    for _ in range(200):
        try:
            if noise_level == 1:
                windows = sample_disjoint_windows(
                    length=length,
                    count=1,
                    min_width=1,
                    max_width=1,
                    rng=rng,
                    require_gap=False,
                )
                kind = "single_factor"
            elif noise_level == 2:
                count = 2 if length >= 5 else 1
                windows = sample_disjoint_windows(
                    length=length,
                    count=count,
                    min_width=1,
                    max_width=1,
                    rng=rng,
                    require_gap=True,
                )
                kind = "scattered_factors"
            elif noise_level == 3:
                windows = sample_disjoint_windows(
                    length=length,
                    count=1,
                    min_width=2,
                    max_width=min(max_repair_window, 4),
                    rng=rng,
                    require_gap=False,
                )
                kind = "local_window"
            elif noise_level == 4:
                width = rng.randint(2, min(max_repair_window, max(2, length // 3), length))
                start = 0 if rng.random() < 0.5 else length - width
                windows = [(start, width)]
                kind = "prefix_suffix"
            elif noise_level == 5:
                count = 2 if length < 40 else rng.randint(2, 3)
                windows = sample_disjoint_windows(
                    length=length,
                    count=count,
                    min_width=2,
                    max_width=min(max_repair_window, 4),
                    rng=rng,
                    require_gap=True,
                )
                kind = "multiple_windows"
            elif noise_level == 6:
                target_changed = max(4, int(length * rng.uniform(0.30, 0.55)))
                windows = []
                changed = 0
                attempts = 0
                while changed < target_changed and attempts < 200:
                    attempts += 1
                    width = rng.randint(1, min(max_repair_window, 4, length))
                    start = rng.randint(0, length - width)
                    left = start - 1
                    right = start + width + 1
                    if all(right <= other_start or left >= other_start + other_width for other_start, other_width in windows):
                        windows.append((start, width))
                        changed += width
                if not windows:
                    continue
                windows = sorted(windows)
                kind = "heavy_same_length"
            else:
                raise ValueError(f"unsupported noise level {noise_level}")
        except ValueError:
            continue
        example = corrupt_by_windows(
            automaton,
            clean,
            windows,
            rng,
            max_repair_window=max_repair_window,
            corruption_kind=kind,
        )
        if example is not None:
            return CorruptedRepairExample(
                clean=example.clean,
                corrupted_factors=example.corrupted_factors,
                noise_level=int(noise_level),
                corruption_kind=example.corruption_kind,
                target_start=example.target_start,
                target_width=example.target_width,
                target_factors=example.target_factors,
                hamming_to_clean=example.hamming_to_clean,
            )
    return None


def generate_data(args: argparse.Namespace) -> None:
    start_time = time.time()
    mgpt = load_matrix_gpt_module(Path(args.matrix_gpt_root))
    bgpt = mgpt.load_braid_gpt_module(Path(args.braid_gpt_root))
    rng = random.Random(args.seed)
    automaton = bgpt.GNFAutomaton(args.n)
    evaluator = mgpt.MatrixEvaluator(
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
    clean_kernels = load_clean_kernels(
        mgpt=mgpt,
        bgpt=bgpt,
        evaluator=evaluator,
        kernel_sources=args.kernel_source,
        min_length=args.min_kernel_length,
        max_length=min(args.max_kernel_length, args.max_factors),
        max_kernels=args.max_kernels,
        verify=not args.no_verify_clean_kernels,
        reject_degenerate=not args.keep_degenerate_kernels,
        augment_repeats=args.augment_repeats,
        augment_rotations_per_kernel=args.augment_rotations_per_kernel,
    )
    noise_weights = parse_weight_spec(
        args.noise_level_weights,
        default={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0},
    )
    examples: list[CorruptedRepairExample] = []
    attempts = 0
    while len(examples) < args.example_count and attempts < args.example_count * 50:
        attempts += 1
        clean = rng.choice(clean_kernels)
        level = weighted_choice(noise_weights, rng)
        example = make_corruption(
            automaton,
            clean,
            level,
            rng,
            max_repair_window=args.max_repair_window,
        )
        if example is None:
            continue
        examples.append(example)
        if len(examples) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "phase": "corrupt_examples",
                        "examples": len(examples),
                        "attempts": attempts,
                        "noise_histogram": dict(Counter(item.noise_level for item in examples)),
                        "elapsed_seconds": round(time.time() - start_time, 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if len(examples) < args.example_count:
        raise RuntimeError(f"Only generated {len(examples)} examples after {attempts} attempts")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "examples.jsonl"
    log_path.write_text("", encoding="utf-8")

    total_count = len(examples)
    tokens = np.zeros((total_count, args.max_factors + 1), dtype=np.int16)
    matrix_tensors = np.zeros(
        (total_count, args.matrix_max_degree, 2, evaluator.dim, evaluator.dim),
        dtype=np.uint8,
    )
    matrix_widths = np.zeros((total_count,), dtype=np.int16)
    p_values = np.full((total_count,), args.p, dtype=np.int16)
    noise_levels = np.zeros((total_count,), dtype=np.int8)
    lengths = np.zeros((total_count,), dtype=np.int16)
    powers = np.zeros((total_count,), dtype=np.int16)
    target_starts = np.zeros((total_count,), dtype=np.int16)
    target_widths = np.zeros((total_count,), dtype=np.int8)
    target_factors = np.full((total_count, args.max_repair_window), -100, dtype=np.int16)
    parent_objectives = np.zeros((total_count,), dtype=np.float32)
    hamming_to_clean = np.zeros((total_count,), dtype=np.int16)
    clean_kernel_ids = np.zeros((total_count,), dtype=np.int32)

    evaluated_count = 0
    rows: list[dict] = []
    for batch_start in range(0, total_count, args.eval_batch_size):
        batch = examples[batch_start : batch_start + args.eval_batch_size]
        evaluated = evaluator.evaluate_batch(
            [(item.clean.power, item.corrupted_factors) for item in batch],
            batch_size=args.eval_batch_size,
        )
        for offset, (example, result) in enumerate(zip(batch, evaluated)):
            index = batch_start + offset
            token_row, _ = mgpt.encode_prefix(example.corrupted_factors, args.max_factors)
            tokens[index] = token_row
            matrix_tensors[index] = result.matrix_tensor
            matrix_widths[index] = result.matrix_width
            noise_levels[index] = int(example.noise_level)
            lengths[index] = len(example.corrupted_factors)
            powers[index] = int(example.clean.power)
            target_starts[index] = int(example.target_start)
            target_widths[index] = int(example.target_width)
            target_factors[index, : example.target_width] = list(example.target_factors)
            parent_objectives[index] = np.float32(result.objective)
            hamming_to_clean[index] = int(example.hamming_to_clean)
            clean_kernel_ids[index] = int(example.clean.kernel_id)
            if len(rows) < args.log_examples:
                repaired_once = replace_window(
                    example.corrupted_factors,
                    example.target_start,
                    example.target_width,
                    example.target_factors,
                )
                rows.append(
                    {
                        "example_id": index,
                        "p": args.p,
                        "power": example.clean.power,
                        "noise_level": example.noise_level,
                        "corruption_kind": example.corruption_kind,
                        "clean_kernel_id": example.clean.kernel_id,
                        "clean_source": example.clean.source,
                        "length": len(example.corrupted_factors),
                        "hamming_to_clean": example.hamming_to_clean,
                        "target_start": example.target_start,
                        "target_width": example.target_width,
                        "target_factors": list(example.target_factors),
                        "corrupted_factors": list(example.corrupted_factors),
                        "one_step_repaired_legal": automaton.is_legal(repaired_once),
                        "parent_metrics": result.metrics,
                        "parent_objective": result.objective,
                        "matrix_width": int(result.matrix_width),
                    }
                )
        evaluated_count += len(batch)
        if evaluated_count % args.progress_every == 0 or evaluated_count == total_count:
            print(
                json.dumps(
                    {
                        "phase": "evaluate_corruptions",
                        "evaluated": evaluated_count,
                        "total": total_count,
                        "elapsed_seconds": round(time.time() - start_time, 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    append_jsonl(log_path, rows)
    np.savez_compressed(
        output_dir / "diffusion_repair_dataset.npz",
        tokens=tokens,
        matrix_tensors=matrix_tensors,
        matrix_widths=matrix_widths,
        p_values=p_values,
        noise_levels=noise_levels,
        lengths=lengths,
        powers=powers,
        target_starts=target_starts,
        target_widths=target_widths,
        target_factors=target_factors,
        parent_objectives=parent_objectives,
        hamming_to_clean=hamming_to_clean,
        clean_kernel_ids=clean_kernel_ids,
    )
    metadata = {
        "format": "braid-diffusion-repair-dataset-v1",
        "n": args.n,
        "r": args.r,
        "p": args.p,
        "example_count": total_count,
        "clean_kernel_count": len(clean_kernels),
        "kernel_sources": args.kernel_source,
        "augment_repeats": args.augment_repeats,
        "augment_rotations_per_kernel": args.augment_rotations_per_kernel,
        "min_kernel_length": args.min_kernel_length,
        "max_kernel_length": args.max_kernel_length,
        "max_factors": args.max_factors,
        "max_repair_window": args.max_repair_window,
        "matrix_max_degree": args.matrix_max_degree,
        "matrix_dim": evaluator.dim,
        "matrix_channels": ["projectivized_raw", "residual_to_scalar"],
        "noise_level_weights": noise_weights,
        "noise_histogram": dict(Counter(int(value) for value in noise_levels)),
        "target_width_histogram": dict(Counter(int(value) for value in target_widths)),
        "hamming_median": float(np.median(hamming_to_clean)),
        "hamming_max": int(np.max(hamming_to_clean)),
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
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    write_json(output_dir / "metadata.json", metadata)
    print(json.dumps({"phase": "done", **metadata}, sort_keys=True), flush=True)


def merge_data(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    if not args.dataset:
        raise RuntimeError("At least one --dataset is required")
    payloads = []
    metadatas = []
    for raw_path in args.dataset:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = np.load(path)
        metadata_path = path.parent / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {"dataset": str(path)}
        payloads.append((path, payload))
        metadatas.append(metadata)
    shape_keys = ("tokens", "matrix_tensors", "target_factors")
    reference_shapes = {key: payloads[0][1][key].shape[1:] for key in shape_keys}
    for path, payload in payloads:
        for key, suffix in reference_shapes.items():
            if payload[key].shape[1:] != suffix:
                raise RuntimeError(
                    f"{path} has incompatible {key} shape {payload[key].shape[1:]}; expected {suffix}"
                )
    arrays_by_key: dict[str, list[np.ndarray]] = {}
    source_records: list[dict] = []
    selected_total = 0
    for source_index, (path, payload) in enumerate(payloads):
        p_values = payload["p_values"]
        noise_levels = payload["noise_levels"]
        local_indices: list[int] = []
        for p_value in sorted(set(int(value) for value in p_values)):
            p_indices = np.flatnonzero(p_values == p_value)
            if args.max_examples_per_p > 0 and p_indices.size > args.max_examples_per_p:
                p_indices = rng.choice(p_indices, size=args.max_examples_per_p, replace=False)
            if args.max_examples_per_p_noise > 0:
                bounded: list[np.ndarray] = []
                for noise_level in sorted(set(int(value) for value in noise_levels[p_indices])):
                    group = p_indices[noise_levels[p_indices] == noise_level]
                    if group.size > args.max_examples_per_p_noise:
                        group = rng.choice(group, size=args.max_examples_per_p_noise, replace=False)
                    bounded.append(group)
                p_indices = np.concatenate(bounded) if bounded else np.array([], dtype=np.int64)
            local_indices.extend(int(index) for index in p_indices)
        local_indices_array = np.array(sorted(set(local_indices)), dtype=np.int64)
        if local_indices_array.size == 0:
            continue
        selected_total += int(local_indices_array.size)
        for key in payload.files:
            arrays_by_key.setdefault(key, []).append(payload[key][local_indices_array])
        source_records.append(
            {
                "dataset": str(path),
                "metadata": metadatas[source_index],
                "available_count": int(payload["tokens"].shape[0]),
                "selected_count": int(local_indices_array.size),
                "selected_p_histogram": dict(Counter(int(value) for value in payload["p_values"][local_indices_array])),
                "selected_noise_histogram": dict(Counter(int(value) for value in payload["noise_levels"][local_indices_array])),
            }
        )
    if selected_total == 0:
        raise RuntimeError("No examples selected from input datasets")
    merged = {key: np.concatenate(values, axis=0) for key, values in arrays_by_key.items()}
    order = rng.permutation(merged["tokens"].shape[0])
    for key in merged:
        merged[key] = merged[key][order]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "diffusion_repair_dataset.npz", **merged)
    metadata = {
        "format": "braid-diffusion-repair-merged-dataset-v1",
        "source_records": source_records,
        "total_count": int(merged["tokens"].shape[0]),
        "p_histogram": dict(Counter(int(value) for value in merged["p_values"])),
        "noise_histogram": dict(Counter(int(value) for value in merged["noise_levels"])),
        "max_examples_per_p": args.max_examples_per_p,
        "max_examples_per_p_noise": args.max_examples_per_p_noise,
        "seed": args.seed,
    }
    write_json(output_dir / "metadata.json", metadata)
    print(json.dumps({"phase": "done", **metadata}, sort_keys=True), flush=True)


@dataclass
class DiffusionRepairConfig:
    p_max: int = 31
    token_vocab_size: int = TOKEN_VOCAB_SIZE
    factor_vocab_size: int = FACTOR_VOCAB_SIZE
    max_factors: int = 128
    max_repair_window: int = 4
    matrix_max_degree: int = 256
    matrix_dim: int = 3
    matrix_channels: int = 2
    max_noise_level: int = 6
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


def build_repair_model(torch, nn, config: DiffusionRepairConfig):
    class BraidDiffusionRepair(nn.Module):
        def __init__(self, cfg: DiffusionRepairConfig):
            super().__init__()
            self.config = cfg
            self.token_embedding = nn.Embedding(cfg.token_vocab_size, cfg.d_model, padding_idx=PAD_TOKEN)
            self.position_embedding = nn.Embedding(cfg.max_context_tokens, cfg.d_model)
            self.p_embedding = nn.Embedding(cfg.p_max + 1, cfg.d_model)
            self.noise_embedding = nn.Embedding(cfg.max_noise_level + 1, cfg.d_model)
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
            self.position_head = nn.Linear(cfg.d_model, 1)
            self.width_head = nn.Linear(cfg.d_model, cfg.max_repair_window)
            self.factor_head = nn.Linear(cfg.d_model, cfg.max_repair_window * cfg.factor_vocab_size)

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

        def forward(self, tokens, matrices, matrix_widths, p_values, noise_levels):
            if tokens.shape[1] > self.config.max_context_tokens:
                raise ValueError("input longer than model context")
            batch, seq_width = tokens.shape
            matrix_hidden, matrix_padding = self.encode_matrix(matrices, matrix_widths.long(), p_values.long())
            positions = torch.arange(seq_width, device=tokens.device)[None, :]
            braid_hidden = self.token_embedding(tokens.long()) + self.position_embedding(positions)
            context = self.p_embedding(p_values.long().clamp(min=0, max=self.config.p_max))
            context = context + self.noise_embedding(noise_levels.long().clamp(min=0, max=self.config.max_noise_level))
            braid_hidden[:, 0, :] = braid_hidden[:, 0, :] + matrix_hidden[:, 0, :] + context
            braid_padding = tokens.eq(PAD_TOKEN)
            braid_hidden = self.braid_encoder(braid_hidden, src_key_padding_mask=braid_padding)
            cross, _ = self.cross_attention(
                query=braid_hidden,
                key=matrix_hidden,
                value=matrix_hidden,
                key_padding_mask=matrix_padding,
                need_weights=False,
            )
            hidden = self.final_norm(self.fusion_norm(braid_hidden + self.dropout(cross)))
            factor_hidden = hidden[:, 1 : self.config.max_factors + 1, :]
            position_logits = self.position_head(factor_hidden).squeeze(-1)
            width_logits = self.width_head(factor_hidden)
            factor_logits = self.factor_head(factor_hidden)
            factor_logits = factor_logits.view(
                batch,
                self.config.max_factors,
                self.config.max_repair_window,
                self.config.factor_vocab_size,
            )
            return position_logits, width_logits, factor_logits

    return BraidDiffusionRepair(config)


def resolve_device(torch, device_arg: str):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def save_checkpoint(torch, path: Path, *, model, config: DiffusionRepairConfig, history: list[dict], extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "braid-diffusion-repair-checkpoint-v1",
            "model_config": config.to_dict(),
            "model_state": model.state_dict(),
            "history": history,
            **extra,
        },
        path,
    )


def load_checkpoint(torch, nn, checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = DiffusionRepairConfig(**checkpoint["model_config"])
    model = build_repair_model(torch, nn, config).to(device)
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
    mask = payload["noise_levels"] <= args.max_noise_level
    selected_all = np.flatnonzero(mask)
    if selected_all.size == 0:
        raise RuntimeError(f"No examples with noise_level <= {args.max_noise_level}")
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(selected_all)
    split = max(1, int(len(indices) * (1.0 - args.validation_fraction)))
    train_indices = indices[:split]
    val_indices = indices[split:] if split < len(indices) else indices[: min(1024, len(indices))]
    device = resolve_device(torch, args.device)
    config = DiffusionRepairConfig(
        p_max=args.p_max,
        max_factors=int(payload["tokens"].shape[1] - 1),
        max_repair_window=int(payload["target_factors"].shape[1]),
        matrix_max_degree=int(payload["matrix_tensors"].shape[1]),
        matrix_dim=int(payload["matrix_tensors"].shape[-1]),
        matrix_channels=int(payload["matrix_tensors"].shape[2]),
        max_noise_level=max(6, int(np.max(payload["noise_levels"]))),
        d_model=args.d_model,
        nhead=args.nhead,
        braid_layers=args.braid_layers,
        matrix_layers=args.matrix_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    model = build_repair_model(torch, nn, config).to(device)
    history: list[dict] = []
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        old_config = DiffusionRepairConfig(**checkpoint["model_config"])
        if old_config.to_dict() != config.to_dict():
            raise RuntimeError("init checkpoint config does not match current training config")
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
                torch.tensor(payload["noise_levels"][selected], dtype=torch.long),
                torch.tensor(payload["lengths"][selected], dtype=torch.long),
                torch.tensor(payload["target_starts"][selected], dtype=torch.long),
                torch.tensor(payload["target_widths"][selected], dtype=torch.long),
                torch.tensor(payload["target_factors"][selected], dtype=torch.long),
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
        valid_factor_total = 0
        for batch in data_loader:
            (
                tokens,
                matrices,
                matrix_widths,
                p_values,
                noise_levels,
                lengths,
                starts,
                widths,
                target_factors,
            ) = [item.to(device) for item in batch]
            batch_index = torch.arange(tokens.shape[0], device=device)
            with torch.set_grad_enabled(train_mode):
                position_logits, width_logits, factor_logits = model(
                    tokens,
                    matrices,
                    matrix_widths,
                    p_values,
                    noise_levels,
                )
                position_ids = torch.arange(config.max_factors, device=device)[None, :]
                position_mask = position_ids < lengths[:, None]
                masked_position_logits = position_logits.masked_fill(~position_mask, -1e9)
                position_loss = F.cross_entropy(masked_position_logits, starts)
                selected_width_logits = width_logits[batch_index, starts]
                width_loss = F.cross_entropy(selected_width_logits, widths - 1)
                selected_factor_logits = factor_logits[batch_index, starts]
                factor_loss = F.cross_entropy(
                    selected_factor_logits.reshape(-1, config.factor_vocab_size),
                    target_factors.reshape(-1),
                    ignore_index=-100,
                )
                loss = (
                    args.position_loss_weight * position_loss
                    + args.width_loss_weight * width_loss
                    + args.factor_loss_weight * factor_loss
                )
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
            batch_size = tokens.shape[0]
            pred_starts = torch.argmax(masked_position_logits, dim=-1)
            pred_widths = torch.argmax(selected_width_logits, dim=-1) + 1
            pred_factors = torch.argmax(selected_factor_logits, dim=-1)
            factor_mask = target_factors.ne(-100)
            factor_correct = (pred_factors.eq(target_factors) & factor_mask).sum()
            valid_factor_total += int(factor_mask.sum().detach().cpu())
            exact_factor = ((pred_factors.eq(target_factors) | ~factor_mask).all(dim=-1)).sum()
            exact_action = (
                pred_starts.eq(starts)
                & pred_widths.eq(widths)
                & (pred_factors.eq(target_factors) | ~factor_mask).all(dim=-1)
            ).sum()
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["position_loss"] += float(position_loss.detach().cpu()) * batch_size
            totals["width_loss"] += float(width_loss.detach().cpu()) * batch_size
            totals["factor_loss"] += float(factor_loss.detach().cpu()) * batch_size
            totals["position_top1"] += int(pred_starts.eq(starts).sum().detach().cpu())
            totals["width_top1"] += int(pred_widths.eq(widths).sum().detach().cpu())
            totals["factor_exact"] += int(exact_factor.detach().cpu())
            totals["factor_slot_top1"] += int(factor_correct.detach().cpu())
            totals["exact_action"] += int(exact_action.detach().cpu())
            total += batch_size
        return {
            "loss": totals["loss"] / max(1, total),
            "position_loss": totals["position_loss"] / max(1, total),
            "width_loss": totals["width_loss"] / max(1, total),
            "factor_loss": totals["factor_loss"] / max(1, total),
            "position_top1": totals["position_top1"] / max(1, total),
            "width_top1": totals["width_top1"] / max(1, total),
            "factor_exact": totals["factor_exact"] / max(1, total),
            "factor_slot_top1": totals["factor_slot_top1"] / max(1, valid_factor_total),
            "exact_action": totals["exact_action"] / max(1, total),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(train_loader, True)
        val_stats = run_epoch(val_loader, False)
        row = {
            "phase": "diffusion_repair_train",
            "stage_max_noise_level": args.max_noise_level,
            "epoch": epoch,
            "train": train_stats,
            "validation": val_stats,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(
                torch,
                output_dir / "braid_diffusion_repair.pt",
                model=model,
                config=config,
                history=history,
                extra={
                    "dataset_metadata": metadata,
                    "stage_max_noise_level": args.max_noise_level,
                    "selected_examples": int(selected_all.size),
                },
            )
    write_json(
        output_dir / "training_summary.json",
        {
            "history": history,
            "best_validation_loss": best_val,
            "stage_max_noise_level": args.max_noise_level,
            "selected_examples": int(selected_all.size),
        },
    )


@dataclass(frozen=True)
class RepairState:
    power: int
    factors: tuple[int, ...]
    metrics: dict
    objective: float
    matrix_width: int = 1


def state_record(state: RepairState) -> dict:
    return {
        "power": int(state.power),
        "factor_ids": list(state.factors),
        "length": len(state.factors),
        "metrics": state.metrics,
        "objective": float(state.objective),
        "matrix_width": int(state.matrix_width),
    }


def unique_ranked(states: Sequence[RepairState], limit: int) -> list[RepairState]:
    unique: dict[tuple[int, tuple[int, ...]], RepairState] = {}
    for state in states:
        key = (state.power % 2, state.factors)
        previous = unique.get(key)
        if previous is None or (state.objective, state.metrics.get("identity_defect", 10**9)) < (
            previous.objective,
            previous.metrics.get("identity_defect", 10**9),
        ):
            unique[key] = state
    return sorted(
        unique.values(),
        key=lambda item: (item.objective, item.metrics.get("identity_defect", 10**9), item.metrics.get("projlen", 10**9)),
    )[:limit]


def parse_seed_word(value: str) -> tuple[int, tuple[int, ...]]:
    if ":" not in value:
        raise ValueError("seed words must have form POWER:f1,f2,...")
    power_text, factors_text = value.split(":", 1)
    factors = tuple(int(part.strip()) for part in factors_text.split(",") if part.strip())
    return int(power_text), factors


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
        score = sum(item[1] for item in product)
        combos.append((factors, score))
    combos.sort(key=lambda item: item[1], reverse=True)
    return combos[:limit]


def propose_edits_for_state(
    *,
    automaton,
    rng: random.Random,
    state: RepairState,
    position_logits: np.ndarray,
    width_logits: np.ndarray,
    factor_logits: np.ndarray,
    positions_per_state: int,
    widths_per_position: int,
    factor_choices_per_slot: int,
    edits_per_state: int,
    bridge_samples_per_edit: int,
) -> list[tuple[tuple[int, ...], float, dict]]:
    length = len(state.factors)
    if length == 0:
        return []
    position_scores = position_logits[:length].copy()
    ranked_positions = np.argsort(position_scores)[::-1][:positions_per_state]
    proposed: list[tuple[tuple[int, ...], float, dict]] = []
    for start in ranked_positions:
        max_width = min(width_logits.shape[-1], length - int(start))
        width_order = np.argsort(width_logits[start, :max_width])[::-1][:widths_per_position]
        for width_index in width_order:
            width = int(width_index) + 1
            combos = top_factor_combos(
                factor_logits[int(start)],
                width=width,
                choices_per_slot=factor_choices_per_slot,
                limit=max(1, edits_per_state // max(1, positions_per_state)),
            )
            for replacement, factor_score in combos:
                if tuple(state.factors[int(start) : int(start) + width]) == replacement:
                    continue
                child = replace_window(state.factors, int(start), width, replacement)
                if not automaton.is_legal(child):
                    continue
                score = float(position_scores[int(start)] + width_logits[int(start), width_index] + factor_score)
                proposed.append(
                    (
                        child,
                        score,
                        {
                            "start": int(start),
                            "width": width,
                            "replacement": list(replacement),
                            "model_score": score,
                        },
                    )
                )
            left = int(state.factors[int(start) - 1]) if int(start) > 0 else None
            right_index = int(start) + width
            right = int(state.factors[right_index]) if right_index < length else None
            for sample_id in range(bridge_samples_per_edit):
                try:
                    replacement = tuple(int(x) for x in automaton.sample_bridge(left, right, width, rng))
                except ValueError:
                    continue
                if tuple(state.factors[int(start) : int(start) + width]) == replacement:
                    continue
                child = replace_window(state.factors, int(start), width, replacement)
                if not automaton.is_legal(child):
                    continue
                score = float(position_scores[int(start)] + width_logits[int(start), width_index] - 0.25 * (sample_id + 1))
                proposed.append(
                    (
                        child,
                        score,
                        {
                            "start": int(start),
                            "width": width,
                            "replacement": list(replacement),
                            "model_score": score,
                            "fallback": "legal_bridge_sample",
                        },
                    )
                )
    proposed.sort(key=lambda item: item[1], reverse=True)
    return proposed[:edits_per_state]


def search(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn

    mgpt = load_matrix_gpt_module(Path(args.matrix_gpt_root))
    bgpt = mgpt.load_braid_gpt_module(Path(args.braid_gpt_root))
    rng = random.Random(args.seed)
    device = resolve_device(torch, args.device)
    model, config, checkpoint = load_checkpoint(torch, nn, Path(args.checkpoint), device)
    automaton = bgpt.GNFAutomaton(args.n)
    evaluator = mgpt.MatrixEvaluator(
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

    def evaluated_states(words: Sequence[tuple[int, tuple[int, ...]]]) -> list[tuple[RepairState, np.ndarray]]:
        if not words:
            return []
        evaluated = evaluator.evaluate_batch(words, batch_size=args.eval_batch_size)
        out: list[tuple[RepairState, np.ndarray]] = []
        for item in evaluated:
            out.append(
                (
                    RepairState(
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

    start_words: list[tuple[int, tuple[int, ...]]] = []
    for value in args.seed_word:
        start_words.append(parse_seed_word(value))
    if args.start_mode in {"random", "both"}:
        while len(start_words) < args.random_roots:
            length = rng.randint(args.root_min_length, args.root_max_length)
            start_words.append((rng.choice((0, 1)), tuple(int(x) for x in automaton.sample_uniform(length, rng))))
    if args.start_mode in {"corrupted-kernels", "both"}:
        clean_kernels = load_clean_kernels(
            mgpt=mgpt,
            bgpt=bgpt,
            evaluator=evaluator,
            kernel_sources=args.kernel_source,
            min_length=args.root_min_length,
            max_length=min(args.root_max_length, config.max_factors),
            max_kernels=args.max_kernels,
            verify=not args.no_verify_clean_kernels,
            reject_degenerate=not args.keep_degenerate_kernels,
            augment_repeats=args.augment_repeats,
            augment_rotations_per_kernel=args.augment_rotations_per_kernel,
        )
        roots_made = 0
        while roots_made < args.corrupted_roots:
            clean = rng.choice(clean_kernels)
            level = rng.randint(args.root_min_noise_level, args.root_max_noise_level)
            example = make_corruption(
                automaton,
                clean,
                level,
                rng,
                max_repair_window=config.max_repair_window,
            )
            if example is None:
                continue
            start_words.append((example.clean.power, example.corrupted_factors))
            roots_made += 1
    if not start_words:
        raise RuntimeError("No start words generated")

    start_pairs = evaluated_states(start_words)
    matrix_by_key = {(state.power % 2, state.factors): matrix for state, matrix in start_pairs}
    frontier = unique_ranked([state for state, _ in start_pairs], args.beam_size)
    best = unique_ranked(frontier, args.keep_best)
    seen = {(state.power % 2, state.factors) for state in frontier}
    kernel_hits: list[RepairState] = [state for state in frontier if state.metrics.get("scalar_identity") and len(state.factors) > 0]
    start_time = time.time()

    for step in range(1, args.steps + 1):
        expandable = [state for state in frontier if 0 < len(state.factors) <= config.max_factors]
        if not expandable:
            break
        missing = [
            (state.power, state.factors)
            for state in expandable
            if (state.power % 2, state.factors) not in matrix_by_key
        ]
        for state, matrix in evaluated_states(missing):
            matrix_by_key[(state.power % 2, state.factors)] = matrix
        tokens = np.stack([mgpt.encode_prefix(state.factors, config.max_factors)[0] for state in expandable])
        matrices = np.stack([matrix_by_key[(state.power % 2, state.factors)] for state in expandable])
        widths = np.array([state.matrix_width for state in expandable], dtype=np.int64)
        p_values = np.full((len(expandable),), args.p, dtype=np.int64)
        noise_values = np.full((len(expandable),), args.inference_noise_level, dtype=np.int64)
        with torch.no_grad():
            position_logits, width_logits, factor_logits = model(
                torch.tensor(tokens, dtype=torch.long, device=device),
                torch.tensor(matrices, dtype=torch.uint8, device=device),
                torch.tensor(widths, dtype=torch.long, device=device),
                torch.tensor(p_values, dtype=torch.long, device=device),
                torch.tensor(noise_values, dtype=torch.long, device=device),
            )
        position_np = position_logits.detach().cpu().numpy()
        width_np = width_logits.detach().cpu().numpy()
        factor_np = factor_logits.detach().cpu().numpy()

        child_words: list[tuple[int, tuple[int, ...]]] = []
        child_meta: dict[tuple[int, tuple[int, ...]], dict] = {}
        parent_by_child: dict[tuple[int, tuple[int, ...]], RepairState] = {}
        for index, state in enumerate(expandable):
            proposals = propose_edits_for_state(
                automaton=automaton,
                rng=rng,
                state=state,
                position_logits=position_np[index],
                width_logits=width_np[index],
                factor_logits=factor_np[index],
                positions_per_state=args.positions_per_state,
                widths_per_position=args.widths_per_position,
                factor_choices_per_slot=args.factor_choices_per_slot,
                edits_per_state=args.edits_per_state,
                bridge_samples_per_edit=args.bridge_samples_per_edit,
            )
            for child_factors, _, meta in proposals:
                key = (state.power % 2, child_factors)
                if key in seen:
                    continue
                seen.add(key)
                child_words.append((state.power, child_factors))
                child_meta[key] = meta
                parent_by_child[key] = state
        child_pairs = evaluated_states(child_words)
        child_states: list[RepairState] = []
        candidate_rows: list[dict] = []
        for state, matrix in child_pairs:
            key = (state.power % 2, state.factors)
            parent = parent_by_child[key]
            if args.accept_only_improvements and state.objective >= parent.objective:
                continue
            child_states.append(state)
            matrix_by_key[key] = matrix
            candidate_rows.append(
                {
                    "step": step,
                    "parent_objective": parent.objective,
                    "objective_delta": state.objective - parent.objective,
                    "edit": child_meta.get(key, {}),
                    **state_record(state),
                    "degeneracy": bgpt.degeneracy_features(state.factors),
                }
            )
        append_jsonl(candidates_path, candidate_rows)
        kernel_hits.extend(state for state in child_states if state.metrics.get("scalar_identity") and len(state.factors) > 0)
        best = unique_ranked(best + child_states, args.keep_best)
        frontier = unique_ranked(child_states + best, args.beam_size)
        row = {
            "phase": "diffusion_repair_search",
            "step": step,
            "expanded": len(child_words),
            "accepted_children": len(child_states),
            "frontier_size": len(frontier),
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
        "format": "braid-diffusion-repair-search-summary-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_stage_max_noise_level": checkpoint.get("stage_max_noise_level"),
        "kernel_hits": [state_record(state) for state in kernel_hits[: args.keep_best]],
        "best": [state_record(state) for state in best],
        "best_by_identity_defect": [
            state_record(state)
            for state in sorted(
                best,
                key=lambda item: (
                    item.metrics.get("identity_defect", 10**9),
                    item.metrics.get("projlen", 10**9),
                    item.objective,
                ),
            )[: args.keep_best]
        ],
        "search": {
            "start_mode": args.start_mode,
            "steps": args.steps,
            "beam_size": args.beam_size,
            "positions_per_state": args.positions_per_state,
            "widths_per_position": args.widths_per_position,
            "factor_choices_per_slot": args.factor_choices_per_slot,
            "edits_per_state": args.edits_per_state,
            "bridge_samples_per_edit": args.bridge_samples_per_edit,
            "accept_only_improvements": args.accept_only_improvements,
            "inference_noise_level": args.inference_noise_level,
        },
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
    parser = argparse.ArgumentParser(
        description="Braid-Diffusion-Repair: discrete denoising transformer for corrupted kernel braids."
    )
    parser.add_argument("--matrix-gpt-root", default=str(DEFAULT_MATRIX_GPT_ROOT))
    parser.add_argument("--braid-gpt-root", default=str(DEFAULT_BRAID_GPT_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data")
    data.add_argument("--author-repo", default=str(DEFAULT_AUTHOR_REPO))
    data.add_argument("--output-dir", required=True)
    data.add_argument("--kernel-source", action="append", default=[])
    data.add_argument("--p", type=int, default=5)
    data.add_argument("--n", type=int, default=4)
    data.add_argument("--r", type=int, default=1)
    data.add_argument("--example-count", type=int, default=100_000)
    data.add_argument("--max-kernels", type=int, default=0)
    data.add_argument("--min-kernel-length", type=int, default=12)
    data.add_argument("--max-kernel-length", type=int, default=128)
    data.add_argument("--max-factors", type=int, default=128)
    data.add_argument("--max-repair-window", type=int, default=4)
    data.add_argument("--matrix-max-degree", type=int, default=256)
    data.add_argument("--noise-level-weights", default="")
    data.add_argument("--eval-batch-size", type=int, default=500)
    data.add_argument("--seed", type=int, default=1)
    data.add_argument("--progress-every", type=int, default=500)
    data.add_argument("--log-examples", type=int, default=200)
    data.add_argument("--no-verify-clean-kernels", action="store_true")
    data.add_argument("--keep-degenerate-kernels", action="store_true")
    data.add_argument("--augment-repeats", type=int, default=2)
    data.add_argument("--augment-rotations-per-kernel", type=int, default=8)
    add_objective_args(data)
    data.set_defaults(func=generate_data)

    merge = sub.add_parser("merge-data")
    merge.add_argument("--dataset", action="append", default=[])
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--max-examples-per-p", type=int, default=0)
    merge.add_argument("--max-examples-per-p-noise", type=int, default=0)
    merge.add_argument("--seed", type=int, default=1)
    merge.set_defaults(func=merge_data)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--init-checkpoint", default="")
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--epochs", type=int, default=6)
    train_parser.add_argument("--max-noise-level", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--position-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--width-loss-weight", type=float, default=0.35)
    train_parser.add_argument("--factor-loss-weight", type=float, default=1.0)
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
    search_parser.add_argument("--output-dir", required=True)
    search_parser.add_argument("--kernel-source", action="append", default=[])
    search_parser.add_argument("--p", type=int, default=5)
    search_parser.add_argument("--n", type=int, default=4)
    search_parser.add_argument("--r", type=int, default=1)
    search_parser.add_argument("--device", default="auto")
    search_parser.add_argument("--start-mode", choices=("random", "corrupted-kernels", "both"), default="corrupted-kernels")
    search_parser.add_argument("--seed-word", action="append", default=[])
    search_parser.add_argument("--random-roots", type=int, default=256)
    search_parser.add_argument("--corrupted-roots", type=int, default=512)
    search_parser.add_argument("--root-min-length", type=int, default=35)
    search_parser.add_argument("--root-max-length", type=int, default=90)
    search_parser.add_argument("--root-min-noise-level", type=int, default=1)
    search_parser.add_argument("--root-max-noise-level", type=int, default=4)
    search_parser.add_argument("--max-kernels", type=int, default=0)
    search_parser.add_argument("--no-verify-clean-kernels", action="store_true")
    search_parser.add_argument("--keep-degenerate-kernels", action="store_true")
    search_parser.add_argument("--augment-repeats", type=int, default=2)
    search_parser.add_argument("--augment-rotations-per-kernel", type=int, default=8)
    search_parser.add_argument("--steps", type=int, default=40)
    search_parser.add_argument("--beam-size", type=int, default=1024)
    search_parser.add_argument("--keep-best", type=int, default=200)
    search_parser.add_argument("--positions-per-state", type=int, default=8)
    search_parser.add_argument("--widths-per-position", type=int, default=2)
    search_parser.add_argument("--factor-choices-per-slot", type=int, default=2)
    search_parser.add_argument("--edits-per-state", type=int, default=16)
    search_parser.add_argument("--bridge-samples-per-edit", type=int, default=1)
    search_parser.add_argument("--inference-noise-level", type=int, default=4)
    search_parser.add_argument("--eval-batch-size", type=int, default=500)
    search_parser.add_argument("--accept-only-improvements", action="store_true")
    search_parser.add_argument("--stop-at-kernel", action="store_true")
    search_parser.add_argument("--seed", type=int, default=1)
    add_objective_args(search_parser)
    search_parser.set_defaults(func=search)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
