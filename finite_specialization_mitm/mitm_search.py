#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
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


FlatMatrix = tuple[int, ...]
MatrixTuple = tuple[FlatMatrix, ...]
Fingerprint = tuple[int, ...]


@dataclass(frozen=True)
class SeedWord:
    label: str
    power: int
    factors: tuple[int, ...]
    source: str


@dataclass(frozen=True)
class LeftRecord:
    power: int
    parity: int
    factors: tuple[int, ...]
    allowed_first_right: frozenset[int]
    source: str


@dataclass(frozen=True)
class SplitSeed:
    label: str
    power: int
    left: tuple[int, ...]
    right: tuple[int, ...]
    source: str


def setup_author_imports(author_repo: Path):
    if not (author_repo / "peyl" / "braid.py").exists():
        raise FileNotFoundError(f"vendored peyl package is missing at {author_repo}")
    if str(author_repo) not in sys.path:
        sys.path.insert(0, str(author_repo))

    import peyl  # type: ignore
    from peyl import polymat  # type: ignore
    from peyl.braid import GNF  # type: ignore
    from peyl.braidsearch import evaluate_braids, symmetric_table  # type: ignore
    from peyl.permutations import SymmetricGroup  # type: ignore

    return peyl, polymat, GNF, SymmetricGroup, evaluate_braids, symmetric_table


def read_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_int_list(value: str) -> tuple[int, ...]:
    output = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not output:
        raise ValueError("expected at least one integer")
    return output


def parse_power_parities(value: str) -> tuple[int, ...]:
    parities = tuple(sorted({int(part.strip()) % 2 for part in value.split(",") if part.strip()}))
    if not parities:
        raise ValueError("expected at least one power parity")
    return parities


def parse_seed_word(value: str, index: int) -> SeedWord:
    if ":" not in value:
        raise ValueError("seed words must have form POWER:f1,f2,... or LABEL=POWER:f1,f2,...")
    label = f"cli_seed_{index}"
    body = value
    if "=" in value and value.index("=") < value.index(":"):
        label, body = value.split("=", 1)
        label = label.strip() or label
    power_text, factor_text = body.split(":", 1)
    factors = tuple(int(part.strip()) for part in factor_text.split(",") if part.strip())
    if not factors:
        raise ValueError(f"seed word {value!r} has no factors")
    return SeedWord(label=label, power=int(power_text), factors=factors, source=value)


def extract_json_objects_from_text(text: str) -> list[dict]:
    objects: list[dict] = []
    depth = 0
    start: int | None = None
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        objects.append(json.loads(text[start : index + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = None
    return objects


def walk_seed_dicts(obj) -> Iterable[dict]:
    if isinstance(obj, dict):
        if "power" in obj and "factor_ids" in obj:
            yield obj
        for value in obj.values():
            yield from walk_seed_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_seed_dicts(value)


def load_checkpoint_seed_words(path: Path, *, limit: int) -> list[SeedWord]:
    if path.suffix == ".txt":
        objects = extract_json_objects_from_text(path.read_text(encoding="utf-8"))
    else:
        objects = [read_json(path)]

    seeds: list[SeedWord] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for obj in objects:
        for row in walk_seed_dicts(obj):
            try:
                power = int(row["power"])
                factors = tuple(int(value) for value in row["factor_ids"])
            except (KeyError, TypeError, ValueError):
                continue
            if not factors:
                continue
            key = (power, factors)
            if key in seen:
                continue
            seen.add(key)
            label = f"checkpoint_seed_{len(seeds)}"
            seeds.append(
                SeedWord(
                    label=label,
                    power=power,
                    factors=factors,
                    source=f"{path}:{row.get('candidate_id', row.get('depth', len(factors)))}",
                )
            )
            if len(seeds) >= limit:
                return seeds
    return seeds


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


def identity_flat(dim: int) -> FlatMatrix:
    return tuple(1 if row == col else 0 for row in range(dim) for col in range(dim))


def mat_mul_flat(left: FlatMatrix, right: FlatMatrix, dim: int, p: int) -> FlatMatrix:
    out = []
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


def is_scalar_flat(matrix: FlatMatrix, dim: int, p: int) -> bool:
    scalar = matrix[0] % p
    if scalar == 0:
        return False
    for row in range(dim):
        for col in range(dim):
            value = matrix[row * dim + col] % p
            if row == col and value != scalar:
                return False
            if row != col and value != 0:
                return False
    return True


def specialize_polymat(poly_matrix: np.ndarray, t_value: int, p: int) -> FlatMatrix:
    dim = int(poly_matrix.shape[0])
    powers = np.array(
        [pow(t_value % p, degree, p) for degree in range(poly_matrix.shape[-1])],
        dtype=np.int64,
    )
    specialized = np.tensordot(poly_matrix.astype(np.int64) % p, powers, axes=([-1], [0])) % p
    return tuple(int(value) % p for value in specialized.reshape(dim * dim))


class FiniteSpecialization:
    def __init__(self, *, author_repo: Path, n: int, r: int, p: int, t_values: Sequence[int]):
        if p <= 1:
            raise ValueError("finite specialization MITM needs a prime p > 1")
        self.author_repo = author_repo
        self.n = n
        self.r = r
        self.p = p
        self.t_values = tuple(int(value) % p for value in t_values)
        if not self.t_values:
            raise ValueError("at least one t-specialization is required")
        if any(value == 0 for value in self.t_values):
            raise ValueError("all t-specializations must be nonzero")

        (
            self.peyl,
            self.polymat,
            self.GNF,
            self.SymmetricGroup,
            self.evaluate_braids,
            symmetric_table,
        ) = setup_author_imports(author_repo)
        self.rep = self.peyl.JonesSummand(n=n, r=r, p=p)
        self.dim = int(self.rep.dimension())
        table = symmetric_table(self.rep)
        nf_table = self.GNF._nf_table(n)
        w0 = self.SymmetricGroup(n).longest_element()
        self.identity_mats = tuple(identity_flat(self.dim) for _ in self.t_values)
        self.delta_mats = tuple(
            specialize_polymat(table[w0], t_value, p) for t_value in self.t_values
        )
        self.factor_mats: dict[int, MatrixTuple] = {}
        for factor_id, perm in enumerate(nf_table.divs):
            self.factor_mats[factor_id] = tuple(
                specialize_polymat(table[perm], t_value, p) for t_value in self.t_values
            )

    def multiply_tuples(self, left: MatrixTuple, right: MatrixTuple) -> MatrixTuple:
        return tuple(
            mat_mul_flat(a, b, self.dim, self.p)
            for a, b in zip(left, right)
        )

    def inverse_tuple(self, matrices: MatrixTuple) -> MatrixTuple:
        return tuple(mat_inv_flat(matrix, self.dim, self.p) for matrix in matrices)

    def key(self, matrices: MatrixTuple) -> Fingerprint:
        chunks = [normalize_flat(matrix, self.p) for matrix in matrices]
        return tuple(entry for chunk in chunks for entry in chunk)

    def evaluate_factors(self, factors: Sequence[int], *, power_parity: int = 0) -> MatrixTuple:
        matrices = self.delta_mats if power_parity % 2 else self.identity_mats
        for factor_id in factors:
            matrices = self.multiply_tuples(matrices, self.factor_mats[int(factor_id)])
        return matrices

    def finite_scalar_flags(self, matrices: MatrixTuple) -> list[bool]:
        return [is_scalar_flat(matrix, self.dim, self.p) for matrix in matrices]

    def exact_metrics(self, *, power: int, factors: Sequence[int]) -> dict:
        braid = self.GNF(n=self.n, power=int(power), factors=tuple(int(x) for x in factors))
        image = self.evaluate_braids(self.rep, [braid])[0]
        return scalar_identity_metrics(self.polymat, image)


def unique_random_sequences(
    *,
    automaton: GNFAutomaton,
    length: int,
    requested_count: int,
    rng: random.Random,
) -> tuple[list[tuple[int, ...]], int]:
    if length <= 0:
        raise ValueError("MITM half lengths must be positive")
    if requested_count <= 0:
        return [], 0

    seen: set[tuple[int, ...]] = set()
    sequences: list[tuple[int, ...]] = []
    attempts = 0
    max_attempts = max(1000, requested_count * 100)
    while len(sequences) < requested_count and attempts < max_attempts:
        attempts += 1
        factors = automaton.sample_uniform(length, rng)
        if factors in seen:
            continue
        seen.add(factors)
        sequences.append(factors)
    return sequences, attempts


def split_seed_words(
    *,
    seeds: Sequence[SeedWord],
    automaton: GNFAutomaton,
    left_length: int,
    right_length: int,
    allowed_parities: set[int],
) -> list[SplitSeed]:
    total_length = left_length + right_length
    splits: list[SplitSeed] = []
    for seed in seeds:
        if len(seed.factors) != total_length:
            continue
        left = seed.factors[:left_length]
        right = seed.factors[left_length:]
        if not left or not right:
            continue
        if not automaton.is_legal(left):
            continue
        if not automaton.is_legal(right):
            continue
        if right[0] not in automaton.successors[left[-1]]:
            continue
        for parity in sorted(allowed_parities):
            power = seed.power if seed.power % 2 == parity else parity
            splits.append(
                SplitSeed(
                    label=f"{seed.label}_parity{parity}",
                    power=power,
                    left=left,
                    right=right,
                    source=f"{seed.source}:recorded_power_{seed.power}",
                )
            )
    return splits


def word_digest(power: int, factors: Sequence[int]) -> str:
    encoded = json.dumps([int(power), list(map(int, factors))], separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()


def make_match_row(
    *,
    match_id: int,
    source: str,
    left: LeftRecord,
    right: Sequence[int],
    exact_metrics: dict,
    finite_flags: list[bool],
) -> dict:
    factors = tuple(left.factors) + tuple(int(x) for x in right)
    return {
        "match_id": match_id,
        "source": source,
        "power": int(left.power),
        "power_parity": int(left.parity),
        "length": len(factors),
        "left_length": len(left.factors),
        "right_length": len(right),
        "factor_ids": list(factors),
        "left_source": left.source,
        "word_digest": word_digest(left.power, factors),
        "finite_scalar_at_t_values": finite_flags,
        "exact_metrics": exact_metrics,
    }


def build_left_index(
    *,
    finite: FiniteSpecialization,
    automaton: GNFAutomaton,
    left_sequences: Sequence[tuple[int, ...]],
    split_seeds: Sequence[SplitSeed],
    power_parities: Sequence[int],
    max_records_per_key: int,
) -> tuple[dict[Fingerprint, list[LeftRecord]], dict]:
    index: dict[Fingerprint, list[LeftRecord]] = defaultdict(list)
    capped = 0
    singular = 0
    inserted = 0
    duplicate_records: set[tuple[int, tuple[int, ...], str]] = set()

    def add_record(power: int, factors: tuple[int, ...], source: str) -> None:
        nonlocal capped, singular, inserted
        parity = power % 2
        try:
            matrices = finite.evaluate_factors(factors, power_parity=parity)
            needed = finite.key(finite.inverse_tuple(matrices))
        except ValueError:
            singular += 1
            return
        record_key = (power, factors, source)
        if record_key in duplicate_records:
            return
        duplicate_records.add(record_key)
        bucket = index[needed]
        if max_records_per_key > 0 and len(bucket) >= max_records_per_key:
            capped += 1
            return
        bucket.append(
            LeftRecord(
                power=power,
                parity=parity,
                factors=factors,
                allowed_first_right=frozenset(automaton.successors[factors[-1]]),
                source=source,
            )
        )
        inserted += 1

    for split in split_seeds:
        add_record(split.power, split.left, f"planted_left:{split.label}:{split.source}")

    for parity in power_parities:
        power = int(parity)
        for factors in left_sequences:
            add_record(power, factors, f"random_left:parity_{parity}")

    bucket_sizes = [len(bucket) for bucket in index.values()]
    stats = {
        "left_records": inserted,
        "left_index_keys": len(index),
        "left_singular": singular,
        "left_capped_records": capped,
        "max_bucket_size": max(bucket_sizes) if bucket_sizes else 0,
        "mean_bucket_size": (sum(bucket_sizes) / len(bucket_sizes)) if bucket_sizes else 0.0,
    }
    return index, stats


def run_search(args: argparse.Namespace) -> dict:
    start_time = time.time()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matches_path = output_dir / "matches.jsonl"
    matches_path.write_text("", encoding="utf-8")

    t_values = parse_int_list(args.t_values) if args.t_values else tuple(range(2, args.p))
    power_parities = parse_power_parities(args.power_parities)
    automaton = GNFAutomaton(args.n)
    finite = FiniteSpecialization(
        author_repo=Path(args.author_repo),
        n=args.n,
        r=args.r,
        p=args.p,
        t_values=t_values,
    )

    seed_words: list[SeedWord] = []
    for index, value in enumerate(args.seed_word):
        seed_words.append(parse_seed_word(value, index))
    for path in args.checkpoint_seed:
        seed_words.extend(load_checkpoint_seed_words(Path(path), limit=args.checkpoint_seed_limit))

    split_seeds = (
        split_seed_words(
            seeds=seed_words,
            automaton=automaton,
            left_length=args.left_length,
            right_length=args.right_length,
            allowed_parities=set(power_parities),
        )
        if args.include_seed_splits
        else []
    )

    left_sequences, left_attempts = unique_random_sequences(
        automaton=automaton,
        length=args.left_length,
        requested_count=args.left_samples,
        rng=rng,
    )
    right_sequences, right_attempts = unique_random_sequences(
        automaton=automaton,
        length=args.right_length,
        requested_count=args.right_samples,
        rng=rng,
    )

    for split in split_seeds:
        if split.right not in right_sequences:
            right_sequences.insert(0, split.right)

    print(
        json.dumps(
            {
                "phase": "setup",
                "p": args.p,
                "t_values": list(t_values),
                "power_parities": list(power_parities),
                "loaded_seed_words": len(seed_words),
                "usable_split_seeds": len(split_seeds),
                "left_sequences": len(left_sequences),
                "right_sequences": len(right_sequences),
                "left_attempts": left_attempts,
                "right_attempts": right_attempts,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    left_index, left_stats = build_left_index(
        finite=finite,
        automaton=automaton,
        left_sequences=left_sequences,
        split_seeds=split_seeds,
        power_parities=power_parities,
        max_records_per_key=args.max_left_records_per_key,
    )

    print(
        json.dumps({"phase": "left_index", **left_stats}, sort_keys=True),
        flush=True,
    )

    match_count = 0
    exact_kernel_count = 0
    right_singular = 0
    boundary_rejected = 0
    seen_matches: set[tuple[int, tuple[int, ...]]] = set()
    best_matches: list[dict] = []
    matches_by_source: Counter[str] = Counter()
    last_progress = time.time()

    split_seed_by_right = {split.right: split for split in split_seeds}

    for right_index, right in enumerate(right_sequences, start=1):
        try:
            right_matrices = finite.evaluate_factors(right, power_parity=0)
            right_key = finite.key(right_matrices)
        except ValueError:
            right_singular += 1
            continue

        candidates = left_index.get(right_key)
        if candidates:
            for left in candidates:
                if right[0] not in left.allowed_first_right:
                    boundary_rejected += 1
                    continue
                factors = left.factors + right
                match_key = (left.power, factors)
                if match_key in seen_matches:
                    continue
                seen_matches.add(match_key)
                product = finite.multiply_tuples(
                    finite.evaluate_factors(left.factors, power_parity=left.parity),
                    right_matrices,
                )
                finite_flags = finite.finite_scalar_flags(product)
                exact_metrics = finite.exact_metrics(power=left.power, factors=factors)
                source = "finite_match"
                if right in split_seed_by_right or left.source.startswith("planted_left:"):
                    source = "planted_seed_match"
                row = make_match_row(
                    match_id=match_count,
                    source=source,
                    left=left,
                    right=right,
                    exact_metrics=exact_metrics,
                    finite_flags=finite_flags,
                )
                append_jsonl(matches_path, row)
                best_matches.append(row)
                best_matches.sort(
                    key=lambda item: (
                        int(item["exact_metrics"]["identity_defect"]),
                        int(item["exact_metrics"]["projective_width"]),
                        int(item["length"]),
                    )
                )
                del best_matches[args.top_output :]
                matches_by_source[source] += 1
                match_count += 1
                if exact_metrics["scalar_identity"]:
                    exact_kernel_count += 1
                    print(json.dumps({"phase": "exact_kernel", **row}, sort_keys=True), flush=True)
                    if args.stop_after_exact_kernel:
                        break
                if args.max_matches > 0 and match_count >= args.max_matches:
                    break
        if args.stop_after_exact_kernel and exact_kernel_count:
            break
        if args.max_matches > 0 and match_count >= args.max_matches:
            break
        now = time.time()
        if now - last_progress >= args.progress_interval_seconds:
            print(
                json.dumps(
                    {
                        "phase": "right_scan",
                        "right_scanned": right_index,
                        "right_total": len(right_sequences),
                        "finite_matches": match_count,
                        "exact_kernels": exact_kernel_count,
                        "elapsed_seconds": round(now - start_time, 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_progress = now

    elapsed = time.time() - start_time
    summary = {
        "format": "finite-specialization-mitm-summary-v1",
        "metadata": {
            "author_repo": str(Path(args.author_repo)),
            "n": args.n,
            "r": args.r,
            "p": args.p,
            "t_values": list(t_values),
            "left_length": args.left_length,
            "right_length": args.right_length,
            "left_samples_requested": args.left_samples,
            "right_samples_requested": args.right_samples,
            "left_sequences_actual": len(left_sequences),
            "right_sequences_actual": len(right_sequences),
            "left_attempts": left_attempts,
            "right_attempts": right_attempts,
            "power_parities": list(power_parities),
            "seed": args.seed,
            "loaded_seed_words": len(seed_words),
            "usable_split_seeds": len(split_seeds),
            "include_seed_splits": args.include_seed_splits,
            "elapsed_seconds": round(elapsed, 2),
            "max_left_records_per_key": args.max_left_records_per_key,
        },
        "left_index": left_stats,
        "scan": {
            "finite_matches": match_count,
            "exact_kernels": exact_kernel_count,
            "right_singular": right_singular,
            "boundary_rejected": boundary_rejected,
            "matches_by_source": dict(matches_by_source),
        },
        "split_seeds": [
            {
                "label": split.label,
                "source": split.source,
                "power": split.power,
                "left_length": len(split.left),
                "right_length": len(split.right),
            }
            for split in split_seeds
        ],
        "best_matches": best_matches,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"phase": "done", **summary["scan"]}, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Meet-in-the-middle search for projectively scalar Burau/Jones images "
            "using finite t-specialization fingerprints, followed by exact verification."
        )
    )
    parser.add_argument("--author-repo", default=str(DEFAULT_AUTHOR_REPO))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--t-values", default="")
    parser.add_argument("--left-length", type=int, required=True)
    parser.add_argument("--right-length", type=int, required=True)
    parser.add_argument("--left-samples", type=int, default=100_000)
    parser.add_argument("--right-samples", type=int, default=100_000)
    parser.add_argument("--power-parities", default="0,1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seed-word", action="append", default=[])
    parser.add_argument("--checkpoint-seed", action="append", default=[])
    parser.add_argument("--checkpoint-seed-limit", type=int, default=20)
    parser.add_argument("--include-seed-splits", action="store_true")
    parser.add_argument("--max-left-records-per-key", type=int, default=128)
    parser.add_argument("--max-matches", type=int, default=1000)
    parser.add_argument("--top-output", type=int, default=100)
    parser.add_argument("--stop-after-exact-kernel", action="store_true")
    parser.add_argument("--progress-interval-seconds", type=float, default=30.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
