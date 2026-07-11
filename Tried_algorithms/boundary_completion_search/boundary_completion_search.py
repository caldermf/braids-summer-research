#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import gzip
import heapq
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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRAID_GPT_ROOT = REPO_ROOT / "Braid-GPT"
DEFAULT_AUTHOR_REPO = REPO_ROOT / "hybrid_of_reservoir_crispr_mcts_suffix" / "third_party" / "braids_project"

FlatMatrix = tuple[int, ...]
MatrixTuple = tuple[FlatMatrix, ...]
Fingerprint = tuple[int, ...]


@dataclass(frozen=True)
class CoreCandidate:
    core_id: int
    source_candidate_id: str
    power: int
    factors: tuple[int, ...]
    length: int
    metrics: dict
    source: str


@dataclass(frozen=True)
class CompletionWord:
    source: str
    length: int
    factors: tuple[int, ...]
    matrices: MatrixTuple


@dataclass(frozen=True)
class FiniteCompletion:
    mode: str
    core_id: int
    source_candidate_id: str
    left_factors: tuple[int, ...]
    right_factors: tuple[int, ...]
    finite_scalar_defect: int
    finite_scalar_flags: tuple[bool, ...]
    finite_score: float
    total_completion_length: int
    source: str


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
    spec = importlib.util.spec_from_file_location("braid_gpt_runtime_for_boundary_completion", module_path)
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


def parse_int_list(value: str) -> tuple[int, ...]:
    output = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not output:
        raise ValueError("expected at least one integer")
    return output


def parse_modes(value: str) -> tuple[str, ...]:
    allowed = {"right", "left", "both"}
    modes = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not modes:
        raise ValueError("expected at least one mode")
    invalid = sorted(set(modes) - allowed)
    if invalid:
        raise ValueError(f"unknown mode(s): {', '.join(invalid)}")
    return tuple(dict.fromkeys(modes))


def parse_t_values(value: str, p: int) -> tuple[int, ...]:
    if value.strip():
        raw = parse_int_list(value)
    elif p > 3:
        raw = tuple(range(2, p))
    else:
        raw = tuple(range(1, p))
    output = tuple(dict.fromkeys(int(item) % p for item in raw))
    if not output:
        raise ValueError("at least one t-specialization is required")
    if any(item == 0 for item in output):
        raise ValueError("all t-specializations must be nonzero")
    return output


def metric_projlen(metrics: dict) -> int:
    return int(metrics.get("projlen", metrics.get("projective_width", 0)))


def clean_metrics(metrics: dict) -> dict:
    output = dict(metrics)
    if "projlen" not in output:
        output["projlen"] = int(output.get("projective_width", 0))
    output.pop("projective_width", None)
    return output


def core_key(row: dict) -> tuple[int, tuple[int, ...]] | None:
    if "powered_power" in row and "powered_factors" in row:
        return int(row["powered_power"]), tuple(int(x) for x in row["powered_factors"])
    if "power" in row and "factor_ids" in row:
        return int(row["power"]), tuple(int(x) for x in row["factor_ids"])
    if "power" in row and "factors" in row:
        return int(row["power"]), tuple(int(x) for x in row["factors"])
    return None


def core_source_id(row: dict, fallback: int) -> str:
    for key in ("candidate_id", "match_id", "word_digest", "source"):
        if key in row:
            return str(row[key])
    return str(fallback)


def walk_candidate_rows(obj) -> Iterable[dict]:
    if isinstance(obj, dict):
        if core_key(obj) is not None and "metrics" in obj:
            yield obj
        for value in obj.values():
            yield from walk_candidate_rows(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_candidate_rows(value)


def load_core_candidates(
    *,
    path: Path,
    limit: int,
    min_length: int,
    max_length: int,
    max_identity_defect: int,
    max_projlen: int,
) -> list[CoreCandidate]:
    if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"):
        raw_rows = list(iter_jsonl(path))
    else:
        try:
            raw_rows = list(walk_candidate_rows(read_json(path)))
        except json.JSONDecodeError:
            raw_rows = list(iter_jsonl(path))

    cores: list[CoreCandidate] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for index, row in enumerate(raw_rows):
        key = core_key(row)
        if key is None:
            continue
        power, factors = key
        metrics = clean_metrics(row.get("metrics", row.get("exact_metrics", {})))
        if not metrics:
            continue
        length = len(factors)
        if length < min_length or length > max_length:
            continue
        if int(metrics["identity_defect"]) > max_identity_defect:
            continue
        if metric_projlen(metrics) > max_projlen:
            continue
        canonical_key = (power, factors)
        if canonical_key in seen:
            continue
        seen.add(canonical_key)
        cores.append(
            CoreCandidate(
                core_id=len(cores),
                source_candidate_id=core_source_id(row, index),
                power=power,
                factors=factors,
                length=length,
                metrics=metrics,
                source=str(row.get("source", path.name)),
            )
        )

    cores.sort(
        key=lambda core: (
            int(core.metrics["identity_defect"]),
            metric_projlen(core.metrics),
            float(core.metrics["identity_defect"]) / max(1, core.length),
            core.length,
        )
    )
    return cores[:limit]


def identity_flat(dim: int) -> FlatMatrix:
    return tuple(1 if row == col else 0 for row in range(dim) for col in range(dim))


def mat_mul_flat(left: FlatMatrix, right: FlatMatrix, dim: int, p: int) -> FlatMatrix:
    out: list[int] = []
    for row in range(dim):
        row_offset = row * dim
        for col in range(dim):
            value = 0
            for mid in range(dim):
                value += left[row_offset + mid] * right[mid * dim + col]
            out.append(value % p)
    return tuple(out)


def mat_inv_flat(matrix: FlatMatrix, dim: int, p: int) -> FlatMatrix:
    rows = [
        [matrix[row * dim + col] % p for col in range(dim)]
        + [1 if row == col else 0 for col in range(dim)]
        for row in range(dim)
    ]
    for col in range(dim):
        pivot = None
        for row in range(col, dim):
            if rows[row][col] % p:
                pivot = row
                break
        if pivot is None:
            raise ValueError("singular matrix")
        if pivot != col:
            rows[col], rows[pivot] = rows[pivot], rows[col]
        inv = pow(rows[col][col] % p, -1, p)
        rows[col] = [(value * inv) % p for value in rows[col]]
        for row in range(dim):
            if row == col:
                continue
            factor = rows[row][col] % p
            if factor:
                rows[row] = [
                    (rows[row][idx] - factor * rows[col][idx]) % p
                    for idx in range(2 * dim)
                ]
    return tuple(rows[row][dim + col] % p for row in range(dim) for col in range(dim))


def normalize_flat(matrix: FlatMatrix, p: int) -> FlatMatrix:
    for value in matrix:
        value %= p
        if value:
            inv = pow(value, -1, p)
            return tuple((entry * inv) % p for entry in matrix)
    raise ValueError("zero matrix cannot be projectively normalized")


def specialize_polymat(poly_matrix: np.ndarray, t_value: int, p: int) -> FlatMatrix:
    dim = int(poly_matrix.shape[0])
    powers = np.array(
        [pow(t_value % p, degree, p) for degree in range(poly_matrix.shape[-1])],
        dtype=np.int64,
    )
    specialized = np.tensordot(poly_matrix.astype(np.int64) % p, powers, axes=([-1], [0])) % p
    return tuple(int(value) % p for value in specialized.reshape(dim * dim))


def scalar_defect_flat(matrix: FlatMatrix, dim: int, p: int) -> int:
    scalar = matrix[0] % p
    defect = 0
    if scalar == 0:
        defect += 1
    for row in range(dim):
        for col in range(dim):
            value = matrix[row * dim + col] % p
            if row == col:
                if value != scalar:
                    defect += 1
            elif value != 0:
                defect += 1
    return defect


def is_scalar_flat(matrix: FlatMatrix, dim: int, p: int) -> bool:
    return scalar_defect_flat(matrix, dim, p) == 0


class FiniteBoundaryRep:
    def __init__(
        self,
        *,
        bgpt,
        author_repo: Path,
        n: int,
        r: int,
        p: int,
        t_values: Sequence[int],
    ) -> None:
        patch_functools_cache_for_old_python()
        self.bgpt = bgpt
        self.peyl, self.polymat, self.evaluate_braids = bgpt.setup_author_imports(author_repo)
        self.n = int(n)
        self.r = int(r)
        self.p = int(p)
        self.t_values = tuple(int(value) % self.p for value in t_values)
        self.rep = self.peyl.JonesSummand(n=self.n, r=self.r, p=self.p)
        self.dim = int(self.rep.dimension())
        self.table = self.peyl.GNF._nf_table(self.n)
        self.identity_mats = tuple(identity_flat(self.dim) for _ in self.t_values)

        self.delta_mats: dict[int, MatrixTuple] = {}
        for power in range(self.table.tau_order):
            delta_poly = self.rep._polymat_delta_power(self.peyl.GNF, power)
            self.delta_mats[power] = tuple(
                specialize_polymat(delta_poly, t_value, self.p) for t_value in self.t_values
            )

        self.factor_mats: dict[int, MatrixTuple] = {}
        for factor_id in range(self.table.order):
            factor_poly = self.rep._polymat_braid_factor(self.peyl.GNF, factor_id)
            self.factor_mats[factor_id] = tuple(
                specialize_polymat(factor_poly, t_value, self.p) for t_value in self.t_values
            )

    def braid(self, power: int, factors: Sequence[int]):
        return self.peyl.GNF(n=self.n, power=int(power), factors=tuple(int(x) for x in factors))

    def multiply_tuples(self, left: MatrixTuple, right: MatrixTuple) -> MatrixTuple:
        return tuple(mat_mul_flat(a, b, self.dim, self.p) for a, b in zip(left, right))

    def inverse_tuple(self, matrices: MatrixTuple) -> MatrixTuple:
        return tuple(mat_inv_flat(matrix, self.dim, self.p) for matrix in matrices)

    def key(self, matrices: MatrixTuple) -> Fingerprint:
        chunks = [normalize_flat(matrix, self.p) for matrix in matrices]
        return tuple(entry for chunk in chunks for entry in chunk)

    def evaluate_factors(self, power: int, factors: Sequence[int]) -> MatrixTuple:
        matrices = self.delta_mats[int(power) % self.table.tau_order]
        for factor_id in factors:
            matrices = self.multiply_tuples(matrices, self.factor_mats[int(factor_id)])
        return matrices

    def exact_metrics_for_braids(self, braids: Sequence, batch_size: int) -> list[dict]:
        output: list[dict] = []
        for start in range(0, len(braids), batch_size):
            chunk = braids[start : start + batch_size]
            images = self.evaluate_braids(self.rep, chunk)
            for image in images:
                output.append(clean_metrics(self.bgpt.scalar_identity_metrics(self.polymat, image)))
        return output


def finite_scalar_summary(matrices: MatrixTuple, finite: FiniteBoundaryRep) -> tuple[int, tuple[bool, ...]]:
    defects = tuple(scalar_defect_flat(matrix, finite.dim, finite.p) for matrix in matrices)
    return sum(defects), tuple(defect == 0 for defect in defects)


def finite_score(
    *,
    finite_scalar_defect: int,
    total_completion_length: int,
    core_identity_defect: int,
    core_length: int,
    finite_exact_match_bonus: float,
) -> float:
    density = finite_scalar_defect / max(1, total_completion_length + core_length)
    core_density = core_identity_defect / max(1, core_length)
    return (
        float(finite_scalar_defect)
        + 0.35 * density
        + 0.03 * total_completion_length
        + 0.01 * core_density
        - finite_exact_match_bonus
    )


def push_completion(
    heap: list[tuple[float, int, FiniteCompletion]],
    item: FiniteCompletion,
    *,
    limit: int,
    counter: int,
) -> None:
    packed = (-item.finite_score, counter, item)
    if len(heap) < limit:
        heapq.heappush(heap, packed)
    elif packed > heap[0]:
        heapq.heapreplace(heap, packed)


def unique_sample_words(
    *,
    table,
    length: int,
    count: int,
    exhaustive_up_to: int,
    rng: random.Random,
) -> Iterable[tuple[str, tuple[int, ...]]]:
    if length < 0:
        raise ValueError("length cannot be negative")
    if length == 0:
        yield "empty", tuple()
        return
    if length <= exhaustive_up_to:
        for index, factors in enumerate(table.normal_forms(length)):
            yield f"exhaustive_len{length}_{index}", tuple(int(x) for x in factors)
        return
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    max_attempts = max(1000, count * 50)
    while len(seen) < count and attempts < max_attempts:
        attempts += 1
        factors = tuple(int(x) for x in table.sample(length, rng))
        if factors in seen:
            continue
        seen.add(factors)
        yield f"random_len{length}_{len(seen)}", factors


def build_completion_pool(
    *,
    finite: FiniteBoundaryRep,
    lengths: Sequence[int],
    samples_per_length: int,
    exhaustive_up_to: int,
    rng: random.Random,
    label: str,
) -> list[CompletionWord]:
    words: list[CompletionWord] = []
    seen: set[tuple[int, ...]] = set()
    for length in lengths:
        for source, factors in unique_sample_words(
            table=finite.table,
            length=int(length),
            count=samples_per_length,
            exhaustive_up_to=exhaustive_up_to,
            rng=rng,
        ):
            if factors in seen:
                continue
            seen.add(factors)
            words.append(
                CompletionWord(
                    source=f"{label}:{source}",
                    length=len(factors),
                    factors=factors,
                    matrices=finite.evaluate_factors(0, factors),
                )
            )
    return words


def completion_to_row(item: FiniteCompletion) -> dict:
    return {
        "mode": item.mode,
        "core_id": item.core_id,
        "source_candidate_id": item.source_candidate_id,
        "left_factors": list(item.left_factors),
        "left_length": len(item.left_factors),
        "right_factors": list(item.right_factors),
        "right_length": len(item.right_factors),
        "finite_scalar_defect": item.finite_scalar_defect,
        "finite_scalar_flags": list(item.finite_scalar_flags),
        "finite_score": item.finite_score,
        "total_completion_length": item.total_completion_length,
        "source": item.source,
    }


def core_to_row(core: CoreCandidate) -> dict:
    return {
        "core_id": core.core_id,
        "source_candidate_id": core.source_candidate_id,
        "power": core.power,
        "factors": list(core.factors),
        "length": core.length,
        "metrics": core.metrics,
        "source": core.source,
    }


def run_search(args: argparse.Namespace) -> None:
    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    finite_path = output_dir / "finite_survivors.jsonl"
    exact_path = output_dir / "exact_completions.jsonl"
    finite_path.write_text("", encoding="utf-8")
    exact_path.write_text("", encoding="utf-8")

    bgpt = load_braid_gpt_module(Path(args.braid_gpt_root))
    t_values = parse_t_values(args.t_values, args.p)
    modes = parse_modes(args.modes)
    rng = random.Random(args.seed)
    finite = FiniteBoundaryRep(
        bgpt=bgpt,
        author_repo=Path(args.author_repo),
        n=args.n,
        r=args.r,
        p=args.p,
        t_values=t_values,
    )

    cores = load_core_candidates(
        path=Path(args.candidate_path),
        limit=args.candidate_limit,
        min_length=args.min_core_length,
        max_length=args.max_core_length,
        max_identity_defect=args.max_core_identity_defect,
        max_projlen=args.max_core_projlen,
    )
    if not cores:
        raise RuntimeError("no usable core candidates survived the input filters")

    left_lengths = parse_int_list(args.left_lengths)
    right_lengths = parse_int_list(args.right_lengths)
    left_pool = build_completion_pool(
        finite=finite,
        lengths=left_lengths if any(mode in modes for mode in ("left", "both")) else (0,),
        samples_per_length=args.left_samples_per_length,
        exhaustive_up_to=args.exhaustive_up_to,
        rng=rng,
        label="left",
    )
    right_pool = build_completion_pool(
        finite=finite,
        lengths=right_lengths if any(mode in modes for mode in ("right", "both")) else (0,),
        samples_per_length=args.right_samples_per_length,
        exhaustive_up_to=args.exhaustive_up_to,
        rng=rng,
        label="right",
    )

    metadata = {
        "format": "boundary-completion-search-v1",
        "candidate_path": str(Path(args.candidate_path)),
        "n": args.n,
        "r": args.r,
        "p": args.p,
        "t_values": list(t_values),
        "modes": list(modes),
        "candidate_limit": args.candidate_limit,
        "selected_cores": len(cores),
        "min_core_length": args.min_core_length,
        "max_core_length": args.max_core_length,
        "max_core_identity_defect": args.max_core_identity_defect,
        "max_core_projlen": args.max_core_projlen,
        "left_lengths": list(left_lengths),
        "right_lengths": list(right_lengths),
        "left_pool_size": len(left_pool),
        "right_pool_size": len(right_pool),
        "left_samples_per_length": args.left_samples_per_length,
        "right_samples_per_length": args.right_samples_per_length,
        "both_random_pairs_per_core": args.both_random_pairs_per_core,
        "max_finite_survivors": args.max_finite_survivors,
        "max_exact_checks": args.max_exact_checks,
        "min_final_length": args.min_final_length,
        "max_final_length": args.max_final_length,
        "seed": args.seed,
    }
    write_json(output_dir / "metadata.json", metadata)
    append_jsonl(output_dir / "selected_cores.jsonl", [core_to_row(core) for core in cores])

    print(
        json.dumps(
            {
                "phase": "setup",
                "selected_cores": len(cores),
                "left_pool_size": len(left_pool),
                "right_pool_size": len(right_pool),
                "modes": list(modes),
                "elapsed_seconds": round(time.time() - start, 2),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    core_mats = {
        core.core_id: finite.evaluate_factors(core.power, core.factors)
        for core in cores
    }

    finite_heap: list[tuple[float, int, FiniteCompletion]] = []
    finite_counter = 0
    finite_evaluated = Counter()
    finite_exact_matches = Counter()
    last_progress = time.time()
    right_inverse_index: dict[Fingerprint, list[CompletionWord]] = {}
    singular_right_for_both = 0
    if "both" in modes:
        for right in right_pool:
            try:
                key = finite.key(finite.inverse_tuple(right.matrices))
            except ValueError:
                singular_right_for_both += 1
                continue
            right_inverse_index.setdefault(key, []).append(right)

    for core_index, core in enumerate(cores, start=1):
        matrices_core = core_mats[core.core_id]
        core_defect = int(core.metrics["identity_defect"])

        if "right" in modes:
            for right in right_pool:
                matrices = finite.multiply_tuples(matrices_core, right.matrices)
                defect, flags = finite_scalar_summary(matrices, finite)
                finite_counter += 1
                finite_evaluated["right"] += 1
                if all(flags):
                    finite_exact_matches["right"] += 1
                completion = FiniteCompletion(
                    mode="right",
                    core_id=core.core_id,
                    source_candidate_id=core.source_candidate_id,
                    left_factors=tuple(),
                    right_factors=right.factors,
                    finite_scalar_defect=defect,
                    finite_scalar_flags=flags,
                    finite_score=finite_score(
                        finite_scalar_defect=defect,
                        total_completion_length=right.length,
                        core_identity_defect=core_defect,
                        core_length=core.length,
                        finite_exact_match_bonus=args.finite_exact_match_bonus if all(flags) else 0.0,
                    ),
                    total_completion_length=right.length,
                    source=right.source,
                )
                push_completion(
                    finite_heap,
                    completion,
                    limit=args.max_finite_survivors,
                    counter=finite_counter,
                )

        if "left" in modes:
            for left in left_pool:
                matrices = finite.multiply_tuples(left.matrices, matrices_core)
                defect, flags = finite_scalar_summary(matrices, finite)
                finite_counter += 1
                finite_evaluated["left"] += 1
                if all(flags):
                    finite_exact_matches["left"] += 1
                completion = FiniteCompletion(
                    mode="left",
                    core_id=core.core_id,
                    source_candidate_id=core.source_candidate_id,
                    left_factors=left.factors,
                    right_factors=tuple(),
                    finite_scalar_defect=defect,
                    finite_scalar_flags=flags,
                    finite_score=finite_score(
                        finite_scalar_defect=defect,
                        total_completion_length=left.length,
                        core_identity_defect=core_defect,
                        core_length=core.length,
                        finite_exact_match_bonus=args.finite_exact_match_bonus if all(flags) else 0.0,
                    ),
                    total_completion_length=left.length,
                    source=left.source,
                )
                push_completion(
                    finite_heap,
                    completion,
                    limit=args.max_finite_survivors,
                    counter=finite_counter,
                )

        if "both" in modes:
            both_matches = 0
            for left in left_pool:
                left_core = finite.multiply_tuples(left.matrices, matrices_core)
                try:
                    key = finite.key(left_core)
                except ValueError:
                    continue
                for right in right_inverse_index.get(key, []):
                    matrices = finite.multiply_tuples(left_core, right.matrices)
                    defect, flags = finite_scalar_summary(matrices, finite)
                    finite_counter += 1
                    finite_evaluated["both_mitm"] += 1
                    finite_exact_matches["both_mitm"] += int(all(flags))
                    both_matches += 1
                    total_length = left.length + right.length
                    completion = FiniteCompletion(
                        mode="both",
                        core_id=core.core_id,
                        source_candidate_id=core.source_candidate_id,
                        left_factors=left.factors,
                        right_factors=right.factors,
                        finite_scalar_defect=defect,
                        finite_scalar_flags=flags,
                        finite_score=finite_score(
                            finite_scalar_defect=defect,
                            total_completion_length=total_length,
                            core_identity_defect=core_defect,
                            core_length=core.length,
                            finite_exact_match_bonus=args.finite_exact_match_bonus if all(flags) else 0.0,
                        ),
                        total_completion_length=total_length,
                        source=f"both_mitm:{left.source}+{right.source}",
                    )
                    push_completion(
                        finite_heap,
                        completion,
                        limit=args.max_finite_survivors,
                        counter=finite_counter,
                    )

            for pair_index in range(args.both_random_pairs_per_core):
                if not left_pool or not right_pool:
                    break
                left = rng.choice(left_pool)
                right = rng.choice(right_pool)
                matrices = finite.multiply_tuples(
                    finite.multiply_tuples(left.matrices, matrices_core),
                    right.matrices,
                )
                defect, flags = finite_scalar_summary(matrices, finite)
                finite_counter += 1
                finite_evaluated["both_random"] += 1
                if all(flags):
                    finite_exact_matches["both_random"] += 1
                total_length = left.length + right.length
                completion = FiniteCompletion(
                    mode="both",
                    core_id=core.core_id,
                    source_candidate_id=core.source_candidate_id,
                    left_factors=left.factors,
                    right_factors=right.factors,
                    finite_scalar_defect=defect,
                    finite_scalar_flags=flags,
                    finite_score=finite_score(
                        finite_scalar_defect=defect,
                        total_completion_length=total_length,
                        core_identity_defect=core_defect,
                        core_length=core.length,
                        finite_exact_match_bonus=args.finite_exact_match_bonus if all(flags) else 0.0,
                    ),
                    total_completion_length=total_length,
                    source=(
                        f"both_random:{pair_index}:{left.source}+{right.source}:"
                        f"mitm_matches_{both_matches}:singular_right_{singular_right_for_both}"
                    ),
                )
                push_completion(
                    finite_heap,
                    completion,
                    limit=args.max_finite_survivors,
                    counter=finite_counter,
                )

        now = time.time()
        if now - last_progress >= args.progress_interval_seconds or core_index == len(cores):
            print(
                json.dumps(
                    {
                        "phase": "finite_scan",
                        "cores_done": core_index,
                        "cores_total": len(cores),
                        "finite_evaluated": dict(finite_evaluated),
                        "finite_exact_matches": dict(finite_exact_matches),
                        "finite_heap_size": len(finite_heap),
                        "elapsed_seconds": round(now - start, 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_progress = now

    finite_survivors = sorted((item[2] for item in finite_heap), key=lambda item: item.finite_score)
    append_jsonl(finite_path, [completion_to_row(item) for item in finite_survivors])

    exact_braids = []
    exact_sources: list[FiniteCompletion] = []
    seen_final: set[tuple[int, tuple[int, ...]]] = set()
    skipped_final_length = 0

    core_by_id = {core.core_id: core for core in cores}
    identity = finite.braid(0, tuple())
    for item in finite_survivors:
        core = core_by_id[item.core_id]
        left_braid = finite.braid(0, item.left_factors) if item.left_factors else identity
        core_braid = finite.braid(core.power, core.factors)
        right_braid = finite.braid(0, item.right_factors) if item.right_factors else identity
        final_braid = left_braid * core_braid * right_braid
        final_key = (int(final_braid.power), tuple(int(x) for x in final_braid.factors))
        if final_key in seen_final:
            continue
        seen_final.add(final_key)
        final_length = len(tuple(final_braid.factors))
        if final_length < args.min_final_length or final_length > args.max_final_length:
            skipped_final_length += 1
            continue
        exact_braids.append(final_braid)
        exact_sources.append(item)
        if len(exact_braids) >= args.max_exact_checks:
            break

    exact_metrics = finite.exact_metrics_for_braids(exact_braids, batch_size=args.exact_batch_size)
    exact_rows: list[dict] = []
    kernel_hits: list[dict] = []
    for index, (item, braid, metrics) in enumerate(zip(exact_sources, exact_braids, exact_metrics)):
        core = core_by_id[item.core_id]
        final_factors = tuple(int(x) for x in braid.factors)
        row = {
            "completion_id": index,
            **completion_to_row(item),
            "core": core_to_row(core),
            "final_power": int(braid.power),
            "final_factors": list(final_factors),
            "final_length": len(final_factors),
            "exact_metrics": metrics,
            "objective": float(metrics["identity_defect"])
            + args.exact_projlen_weight * metric_projlen(metrics)
            + args.exact_density_weight
            * (
                float(metrics["identity_defect"]) + metric_projlen(metrics)
            )
            / max(1, len(final_factors)),
            "usable_kernel_hit": bool(metrics["scalar_identity"]) and len(final_factors) > 0,
        }
        exact_rows.append(row)
        if row["usable_kernel_hit"]:
            kernel_hits.append(row)
            print(json.dumps({"phase": "exact_kernel", **row}, sort_keys=True), flush=True)
            if args.stop_after_kernel:
                break

    append_jsonl(exact_path, exact_rows)

    best_by_objective = sorted(
        exact_rows,
        key=lambda row: (
            float(row["objective"]),
            int(row["exact_metrics"]["identity_defect"]),
            metric_projlen(row["exact_metrics"]),
            int(row["final_length"]),
        ),
    )[: args.top_output]
    best_by_identity_defect = sorted(
        exact_rows,
        key=lambda row: (
            int(row["exact_metrics"]["identity_defect"]),
            metric_projlen(row["exact_metrics"]),
            int(row["final_length"]),
        ),
    )[: args.top_output]
    best_by_mode = {}
    for mode in modes:
        rows = [row for row in exact_rows if row["mode"] == mode]
        best_by_mode[mode] = sorted(
            rows,
            key=lambda row: (
                int(row["exact_metrics"]["identity_defect"]),
                metric_projlen(row["exact_metrics"]),
                int(row["final_length"]),
            ),
        )[: min(args.top_output, 25)]

    summary = {
        "format": "boundary-completion-search-summary-v1",
        "metadata": metadata,
        "counts": {
            "selected_cores": len(cores),
            "finite_evaluated": dict(finite_evaluated),
            "finite_exact_matches": dict(finite_exact_matches),
            "finite_survivors_kept": len(finite_survivors),
            "exact_checked": len(exact_rows),
            "exact_skipped_final_length": skipped_final_length,
            "usable_kernel_hits": len(kernel_hits),
        },
        "kernel_hits": kernel_hits[: args.top_output],
        "best_by_objective": best_by_objective,
        "best_by_identity_defect": best_by_identity_defect,
        "best_by_mode": best_by_mode,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    write_json(output_dir / "summary.json", summary)
    print(
        json.dumps({"phase": "done", **summary["counts"], "elapsed_seconds": summary["elapsed_seconds"]}, sort_keys=True),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Given near-miss core braids, search for short left/right/two-sided "
            "Garside completions that make the finite-specialized image closer "
            "to projectively scalar, then exact-verify the best completions."
        )
    )
    parser.add_argument("--braid-gpt-root", default=str(DEFAULT_BRAID_GPT_ROOT))
    parser.add_argument("--author-repo", default=str(DEFAULT_AUTHOR_REPO))
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--t-values", default="")
    parser.add_argument("--modes", default="right,left,both")
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--min-core-length", type=int, default=15)
    parser.add_argument("--max-core-length", type=int, default=220)
    parser.add_argument("--max-core-identity-defect", type=int, default=200)
    parser.add_argument("--max-core-projlen", type=int, default=200)
    parser.add_argument("--left-lengths", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--right-lengths", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--left-samples-per-length", type=int, default=2000)
    parser.add_argument("--right-samples-per-length", type=int, default=2000)
    parser.add_argument("--exhaustive-up-to", type=int, default=3)
    parser.add_argument("--both-random-pairs-per-core", type=int, default=2000)
    parser.add_argument("--finite-exact-match-bonus", type=float, default=20.0)
    parser.add_argument("--max-finite-survivors", type=int, default=5000)
    parser.add_argument("--max-exact-checks", type=int, default=1000)
    parser.add_argument("--exact-batch-size", type=int, default=64)
    parser.add_argument("--min-final-length", type=int, default=20)
    parser.add_argument("--max-final-length", type=int, default=260)
    parser.add_argument("--exact-projlen-weight", type=float, default=0.2)
    parser.add_argument("--exact-density-weight", type=float, default=3.0)
    parser.add_argument("--top-output", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--progress-interval-seconds", type=float, default=30.0)
    parser.add_argument("--stop-after-kernel", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
