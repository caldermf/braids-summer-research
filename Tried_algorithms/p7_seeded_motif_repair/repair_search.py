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


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    power: int
    factor_ids: tuple[int, ...]
    source: str
    stage: str
    parent_id: int | None = None
    metadata: dict | None = None

    @property
    def length(self) -> int:
        return len(self.factor_ids)

    def key(self) -> tuple[int, tuple[int, ...]]:
        return self.power, self.factor_ids

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "power": self.power,
            "factor_ids": list(self.factor_ids),
            "length": self.length,
            "source": self.source,
            "stage": self.stage,
            "parent_id": self.parent_id,
            "metadata": self.metadata or {},
        }


class CandidateFactory:
    def __init__(self) -> None:
        self.next_id = 0

    def make(
        self,
        *,
        power: int,
        factor_ids: Sequence[int],
        source: str,
        stage: str,
        parent_id: int | None = None,
        metadata: dict | None = None,
    ) -> Candidate:
        candidate = Candidate(
            candidate_id=self.next_id,
            power=int(power),
            factor_ids=tuple(int(value) for value in factor_ids),
            source=source,
            stage=stage,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        self.next_id += 1
        return candidate


def _read_json(path: str | Path) -> dict:
    input_path = Path(path)
    if input_path.suffix == ".gz":
        with gzip.open(input_path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(input_path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _iter_jsonl(path: str | Path) -> Iterable[dict]:
    input_path = Path(path)
    opener = gzip.open if input_path.suffix == ".gz" else open
    with opener(input_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _parse_int_list(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_seed_word(value: str) -> dict:
    if ":" not in value:
        raise ValueError("--seed-word must have form POWER:f1,f2,...")
    power_text, factors_text = value.split(":", 1)
    return {
        "power": int(power_text),
        "factor_ids": list(_parse_int_list(factors_text)),
        "source": "cli_seed_word",
        "metrics": {},
    }


def image_fingerprint(image: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(tuple(image.shape)).encode("ascii"))
    digest.update(image.dtype.str.encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def balanced_power(factor_count: int) -> int:
    return -(factor_count // 2)


def candidate_powers(
    *,
    parent_power: int,
    factor_count: int,
    power_mode: str,
    power_offsets: Sequence[int],
) -> tuple[int, ...]:
    bases: list[int] = []
    if power_mode in {"inherit", "both"}:
        bases.append(parent_power)
    if power_mode in {"balanced", "both"}:
        bases.append(balanced_power(factor_count))
    if not bases:
        bases.append(parent_power)
    powers = []
    seen_parities = set()
    for base in bases:
        for offset in power_offsets:
            power = base + offset
            parity = power % 2
            if parity in seen_parities:
                continue
            seen_parities.add(parity)
            powers.append(power)
    return tuple(sorted(powers))


def setup_author_imports(author_repo: Path):
    if not (author_repo / "peyl" / "braid.py").exists():
        raise FileNotFoundError(f"vendored peyl package is missing at {author_repo}")
    if str(author_repo) not in sys.path:
        sys.path.insert(0, str(author_repo))
    import peyl  # type: ignore
    from peyl import polymat  # type: ignore
    from peyl.braidsearch import evaluate_braid_factors, evaluate_braids  # type: ignore

    return peyl, polymat, evaluate_braid_factors, evaluate_braids


def scalar_identity_metrics(polymat_module, image: np.ndarray) -> dict:
    projected = polymat_module.projectivise(image)
    width = int(projected.shape[-1])
    matrix_count = int(np.count_nonzero(projected))

    if projected.shape[0] != projected.shape[1]:
        return {
            "projective_width": width,
            "scalar_identity": False,
            "identity_defect": matrix_count,
            "nonzero_terms": matrix_count,
            "reason": "non_square_matrix",
        }

    dim = projected.shape[0]
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
    scalar_identity = identity_defect == 0
    scalar = None
    degree = None
    if scalar_identity:
        nonzero = np.flatnonzero(scalar_poly)
        degree = int(nonzero[0])
        scalar = int(scalar_poly[degree])
    return {
        "projective_width": width,
        "scalar_identity": bool(scalar_identity),
        "identity_defect": int(identity_defect),
        "off_diagonal_terms": off_diagonal_terms,
        "diagonal_mismatch_terms": diagonal_mismatch_terms,
        "scalar_nonzero_degrees": scalar_nonzero_degrees,
        "scalar_extra_degrees": scalar_extra_degrees,
        "nonzero_terms": matrix_count,
        "scalar": scalar,
        "scalar_degree": degree,
    }


def row_key(row: dict) -> tuple[int, tuple[int, ...]]:
    return int(row["power"]), tuple(int(value) for value in row["factor_ids"])


def projective_search_key(power: int, factor_ids: Sequence[int]) -> tuple[int, tuple[int, ...]]:
    return int(power) % 2, tuple(int(value) for value in factor_ids)


def projective_row_key(row: dict) -> tuple[int, tuple[int, ...]]:
    return projective_search_key(int(row["power"]), row["factor_ids"])


def defect_key(row: dict) -> tuple[int, int, int, int]:
    metrics = row.get("metrics", {})
    return (
        int(metrics.get("identity_defect", 10**12)),
        int(metrics.get("projective_width", 10**12)),
        int(row.get("length", len(row.get("factor_ids", [])))),
        int(row.get("candidate_id", 10**12)),
    )


def width_key(row: dict) -> tuple[int, int, int, int]:
    metrics = row.get("metrics", {})
    return (
        int(metrics.get("projective_width", 10**12)),
        int(metrics.get("identity_defect", 10**12)),
        int(row.get("length", len(row.get("factor_ids", [])))),
        int(row.get("candidate_id", 10**12)),
    )


def minimal_row(row: dict) -> dict:
    return {
        "candidate_id": row.get("candidate_id"),
        "power": int(row["power"]),
        "factor_ids": list(row["factor_ids"]),
        "length": int(row.get("length", len(row["factor_ids"]))),
        "source": row.get("source"),
        "stage": row.get("stage"),
        "parent_id": row.get("parent_id"),
        "metadata": row.get("metadata", {}),
        "metrics": row.get("metrics", {}),
    }


class SummaryAccumulator:
    def __init__(self, top_n: int) -> None:
        self.top_n = top_n
        self.count = 0
        self.projective_width_histogram: Counter[int] = Counter()
        self.identity_defect_histogram: Counter[int] = Counter()
        self.source_histogram: Counter[str] = Counter()
        self.length_histogram: Counter[int] = Counter()
        self.best_by_identity_defect: list[dict] = []
        self.best_by_projective_width: list[dict] = []
        self.kernel_candidates: list[dict] = []

    def observe(self, rows: Sequence[dict]) -> None:
        self.count += len(rows)
        for row in rows:
            metrics = row["metrics"]
            self.projective_width_histogram[int(metrics["projective_width"])] += 1
            self.identity_defect_histogram[int(metrics["identity_defect"])] += 1
            self.source_histogram[str(row.get("source", "unknown"))] += 1
            self.length_histogram[int(row["length"])] += 1
            if metrics.get("scalar_identity"):
                self.kernel_candidates.append(minimal_row(row))

        self.best_by_identity_defect = sorted(
            self.best_by_identity_defect + [minimal_row(row) for row in rows],
            key=defect_key,
        )[: self.top_n]
        self.best_by_projective_width = sorted(
            self.best_by_projective_width + [minimal_row(row) for row in rows],
            key=width_key,
        )[: self.top_n]
        self.kernel_candidates = sorted(self.kernel_candidates, key=defect_key)[
            : self.top_n
        ]

    def best_defect(self) -> int | None:
        if not self.best_by_identity_defect:
            return None
        return int(self.best_by_identity_defect[0]["metrics"]["identity_defect"])

    def best_width(self) -> int | None:
        if not self.best_by_projective_width:
            return None
        return int(self.best_by_projective_width[0]["metrics"]["projective_width"])

    def to_dict(self) -> dict:
        return {
            "evaluated_candidates": self.count,
            "kernel_hits": len(self.kernel_candidates),
            "projective_width_histogram": dict(
                sorted(self.projective_width_histogram.items())
            ),
            "identity_defect_histogram": dict(
                sorted(self.identity_defect_histogram.items())[:100]
            ),
            "source_histogram": dict(sorted(self.source_histogram.items())),
            "length_histogram": dict(sorted(self.length_histogram.items())),
            "best_by_identity_defect": self.best_by_identity_defect,
            "best_by_projective_width": self.best_by_projective_width,
            "kernel_candidates": self.kernel_candidates,
        }


class ProjectiveCollisionIndex:
    def __init__(self, *, rep, polymat_module, evaluate_braid_factors, max_records: int):
        self.rep = rep
        self.polymat = polymat_module
        self.evaluate_braid_factors = evaluate_braid_factors
        self.max_records = max_records
        self.representatives: dict[str, dict] = {}
        self.records_seen = 0
        self.raw_collisions = 0
        self.duplicate_words = 0
        self.central_power_quotients = 0
        self.trivial_quotients = 0
        self.verified_kernel_quotients = 0
        self.failed_verifications = 0
        self.collision_records: list[dict] = []

    def observe(self, *, braid, row: dict, image: np.ndarray) -> None:
        self.records_seen += 1
        projected = self.polymat.projectivise(image)
        fingerprint = image_fingerprint(projected)
        previous = self.representatives.get(fingerprint)
        if previous is None:
            self.representatives[fingerprint] = {"braid": braid, "row": minimal_row(row)}
            return

        self.raw_collisions += 1
        if row_key(previous["row"]) == row_key(row):
            self.duplicate_words += 1
            return

        quotient = braid * previous["braid"].inv()
        if quotient == braid.identity(braid.n):
            self.trivial_quotients += 1
            return
        if quotient.canonical_length() == 0:
            self.central_power_quotients += 1
            return

        quotient_image = self.evaluate_braid_factors(self.rep, quotient)
        metrics = scalar_identity_metrics(self.polymat, quotient_image)
        if metrics["scalar_identity"]:
            self.verified_kernel_quotients += 1
        else:
            self.failed_verifications += 1

        if len(self.collision_records) < self.max_records:
            quotient_power, _ = quotient.canonical_decomposition()
            self.collision_records.append(
                {
                    "fingerprint": fingerprint,
                    "verified_projective_identity": bool(metrics["scalar_identity"]),
                    "quotient_metrics": metrics,
                    "left": minimal_row(row),
                    "right": previous["row"],
                    "quotient": {
                        "power": int(quotient_power),
                        "factor_ids": [int(value) for value in quotient.factors],
                        "garside_length": int(quotient.canonical_length()),
                    },
                }
            )

    def summary(self) -> dict:
        return {
            "records_seen": self.records_seen,
            "unique_fingerprints": len(self.representatives),
            "raw_collisions": self.raw_collisions,
            "duplicate_words": self.duplicate_words,
            "central_power_quotients": self.central_power_quotients,
            "trivial_quotients": self.trivial_quotients,
            "verified_kernel_quotients": self.verified_kernel_quotients,
            "failed_verifications": self.failed_verifications,
            "stored_collision_records": len(self.collision_records),
        }


class Evaluator:
    def __init__(
        self,
        *,
        author_repo: Path,
        p: int,
        n: int,
        r: int,
        collision_index: bool,
        max_collision_records: int,
    ) -> None:
        peyl, polymat_module, evaluate_braid_factors, evaluate_braids = (
            setup_author_imports(author_repo)
        )
        self.peyl = peyl
        self.polymat = polymat_module
        self.evaluate_braid_factors = evaluate_braid_factors
        self.evaluate_braids = evaluate_braids
        self.rep = peyl.JonesSummand(n=n, r=r, p=p)
        self.p = p
        self.n = n
        self.r = r
        self.collision_index = (
            ProjectiveCollisionIndex(
                rep=self.rep,
                polymat_module=self.polymat,
                evaluate_braid_factors=self.evaluate_braid_factors,
                max_records=max_collision_records,
            )
            if collision_index
            else None
        )

    def evaluate(
        self,
        candidates: Sequence[Candidate],
        *,
        batch_size: int,
        output_path: Path,
        phase: str,
    ) -> list[dict]:
        rows: list[dict] = []
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start : start + batch_size]
            braids = [
                self.peyl.GNF(n=self.n, power=candidate.power, factors=candidate.factor_ids)
                for candidate in chunk
            ]
            images = self.evaluate_braids(self.rep, braids)
            chunk_rows = []
            for candidate, braid, image in zip(chunk, braids, images):
                row = {
                    **candidate.to_dict(),
                    "target_p": self.p,
                    "n": self.n,
                    "r": self.r,
                    "phase": phase,
                    "metrics": scalar_identity_metrics(self.polymat, image),
                }
                if self.collision_index is not None:
                    self.collision_index.observe(braid=braid, row=row, image=image)
                rows.append(row)
                chunk_rows.append(row)
            _append_jsonl(output_path, chunk_rows)
        return rows


def load_seed_rows(
    *,
    seed_evaluations: Path | None,
    seed_summary: Path | None,
    seed_words: Sequence[str],
    top_seeds: int,
) -> list[dict]:
    rows: list[dict] = []
    if seed_evaluations is not None:
        rows.extend(_iter_jsonl(seed_evaluations))
    if seed_summary is not None:
        payload = _read_json(seed_summary)
        evaluation_summary = payload.get("evaluation_summary", {})
        rows.extend(evaluation_summary.get("best_by_identity_defect", []))
        rows.extend(evaluation_summary.get("best_by_projective_width", []))
        rows.extend(evaluation_summary.get("kernel_candidates", []))
    rows.extend(_parse_seed_word(value) for value in seed_words)
    if not rows:
        raise ValueError("no seeds found; pass --seed-evaluations, --seed-summary, or --seed-word")

    unique: dict[tuple[int, tuple[int, ...]], dict] = {}
    for row in rows:
        if "factor_ids" not in row or "power" not in row:
            continue
        row = minimal_row(
            {
                **row,
                "length": len(row["factor_ids"]),
                "metrics": row.get("metrics", {}),
            }
        )
        key = projective_row_key(row)
        current = unique.get(key)
        if current is None or defect_key(row) < defect_key(current):
            unique[key] = row

    ordered_by_defect = sorted(unique.values(), key=defect_key)
    ordered_by_width = sorted(unique.values(), key=width_key)
    selected: dict[tuple[int, tuple[int, ...]], dict] = {}
    for ordering in (ordered_by_defect, ordered_by_width):
        for row in ordering:
            selected.setdefault(projective_row_key(row), row)
            if len(selected) >= top_seeds:
                break
        if len(selected) >= top_seeds:
            break
    return list(selected.values())


def filter_legal_seed_rows(seed_rows: Sequence[dict], automaton: GNFAutomaton) -> list[dict]:
    legal = []
    for row in seed_rows:
        factors = tuple(int(value) for value in row["factor_ids"])
        if automaton.is_legal(factors):
            legal.append(row)
    if not legal:
        raise ValueError("none of the selected seed rows are legal GNF factor sequences")
    return legal


def add_candidate(
    output: list[Candidate],
    seen: set[tuple[int, tuple[int, ...]]],
    factory: CandidateFactory,
    *,
    power: int,
    factors: tuple[int, ...],
    source: str,
    stage: str,
    parent_id: int | None,
    metadata: dict,
    automaton: GNFAutomaton,
) -> None:
    if not automaton.is_legal(factors):
        return
    key = projective_search_key(int(power), factors)
    if key in seen:
        return
    seen.add(key)
    output.append(
        factory.make(
            power=power,
            factor_ids=factors,
            source=source,
            stage=stage,
            parent_id=parent_id,
            metadata=metadata,
        )
    )


def select_population(
    rows: Sequence[dict],
    *,
    population_size: int,
    rng: random.Random,
    defect_fraction: float = 0.50,
    width_fraction: float = 0.25,
) -> list[dict]:
    unique: dict[tuple[int, tuple[int, ...]], dict] = {}
    for row in rows:
        key = projective_row_key(row)
        current = unique.get(key)
        if current is None or defect_key(row) < defect_key(current):
            unique[key] = row
    values = list(unique.values())
    selected: dict[tuple[int, tuple[int, ...]], dict] = {}

    def add(row: dict) -> None:
        if len(selected) < population_size:
            selected.setdefault(projective_row_key(row), row)

    defect_quota = max(1, int(population_size * defect_fraction))
    width_quota = max(1, int(population_size * width_fraction))
    for row in sorted(values, key=defect_key)[:defect_quota]:
        add(row)
    for row in sorted(values, key=width_key)[:width_quota]:
        add(row)

    buckets: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for row in values:
        if projective_row_key(row) in selected:
            continue
        width_bin = int(row["metrics"]["projective_width"]) // 8
        buckets[(int(row["length"]), int(row["factor_ids"][-1]), width_bin)].append(row)
    for bucket_rows in buckets.values():
        bucket_rows.sort(key=defect_key)
    bucket_keys = list(buckets)
    rng.shuffle(bucket_keys)
    while len(selected) < population_size and bucket_keys:
        remaining = []
        for key in bucket_keys:
            bucket = buckets[key]
            if bucket:
                add(bucket.pop(0))
            if bucket:
                remaining.append(key)
            if len(selected) >= population_size:
                break
        bucket_keys = remaining

    if len(selected) < population_size:
        for row in sorted(values, key=defect_key):
            add(row)
            if len(selected) >= population_size:
                break
    return list(selected.values())


def seed_candidates(
    seed_rows: Sequence[dict],
    *,
    factory: CandidateFactory,
    stage: str,
    seen: set[tuple[int, tuple[int, ...]]],
    automaton: GNFAutomaton,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for row in seed_rows:
        add_candidate(
            candidates,
            seen,
            factory,
            power=int(row["power"]),
            factors=tuple(int(value) for value in row["factor_ids"]),
            source="seed",
            stage=stage,
            parent_id=None,
            metadata={
                "seed_candidate_id": row.get("candidate_id"),
                "seed_source": row.get("source"),
                "seed_stage": row.get("stage"),
                "seed_metrics": row.get("metrics", {}),
            },
            automaton=automaton,
        )
    return candidates


def generate_suffix_children(
    frontier: Sequence[dict],
    *,
    factory: CandidateFactory,
    seen: set[tuple[int, tuple[int, ...]]],
    automaton: GNFAutomaton,
    rng: random.Random,
    children_per_parent: int,
    power_mode: str,
    power_offsets: Sequence[int],
    extra_depth: int,
) -> list[Candidate]:
    children: list[Candidate] = []
    for parent in frontier:
        factors = tuple(int(value) for value in parent["factor_ids"])
        successors = list(automaton.successors[factors[-1]])
        rng.shuffle(successors)
        if children_per_parent > 0:
            successors = successors[:children_per_parent]
        for next_factor in successors:
            child_factors = factors + (int(next_factor),)
            for power in candidate_powers(
                parent_power=int(parent["power"]),
                factor_count=len(child_factors),
                power_mode=power_mode,
                power_offsets=power_offsets,
            ):
                add_candidate(
                    children,
                    seen,
                    factory,
                    power=power,
                    factors=child_factors,
                    source="suffix_append",
                    stage="suffix",
                    parent_id=int(parent["candidate_id"]),
                    metadata={
                        "extra_depth": extra_depth,
                        "appended_factor": int(next_factor),
                    },
                    automaton=automaton,
                )
    return children


def legal_single_replacements(
    automaton: GNFAutomaton,
    word: tuple[int, ...],
    position: int,
) -> tuple[int, ...]:
    left = word[position - 1] if position > 0 else None
    right = word[position + 1] if position + 1 < len(word) else None
    choices = automaton.first_ids if left is None else automaton.successors[left]
    output = []
    for choice in choices:
        if choice == word[position]:
            continue
        if right is None or right in automaton.successors[choice]:
            output.append(choice)
    return tuple(output)


def sample_insertion(
    automaton: GNFAutomaton,
    word: tuple[int, ...],
    *,
    max_insert: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], dict] | None:
    size = rng.randint(1, max_insert)
    start = rng.randint(0, len(word))
    left = word[start - 1] if start > 0 else None
    right = word[start] if start < len(word) else None
    try:
        block = automaton.sample_bridge(left, right, size, rng)
    except ValueError:
        return None
    mutated = word[:start] + block + word[start:]
    if not automaton.is_legal(mutated):
        return None
    return mutated, {"edit": "insert", "start": start, "insert_length": size, "block": list(block)}


def sample_deletion(
    automaton: GNFAutomaton,
    word: tuple[int, ...],
    *,
    max_delete: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], dict] | None:
    if len(word) <= 1:
        return None
    size = rng.randint(1, min(max_delete, len(word) - 1))
    start = rng.randint(0, len(word) - size)
    mutated = word[:start] + word[start + size :]
    if not mutated or not automaton.is_legal(mutated):
        return None
    return mutated, {"edit": "delete", "start": start, "delete_length": size}


def sample_bridge_replace(
    automaton: GNFAutomaton,
    word: tuple[int, ...],
    *,
    max_bridge: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], dict] | None:
    size = rng.randint(1, min(max_bridge, len(word)))
    start = rng.randint(0, len(word) - size)
    end = start + size
    left = word[start - 1] if start > 0 else None
    right = word[end] if end < len(word) else None
    try:
        block = automaton.sample_bridge(left, right, size, rng)
    except ValueError:
        return None
    if block == word[start:end]:
        return None
    mutated = word[:start] + block + word[end:]
    if not automaton.is_legal(mutated):
        return None
    return mutated, {
        "edit": "bridge_replace",
        "start": start,
        "delete_length": size,
        "insert_length": size,
        "block": list(block),
    }


def sample_single_replace(
    automaton: GNFAutomaton,
    word: tuple[int, ...],
    *,
    rng: random.Random,
) -> tuple[tuple[int, ...], dict] | None:
    positions = list(range(len(word)))
    rng.shuffle(positions)
    for position in positions:
        choices = list(legal_single_replacements(automaton, word, position))
        if not choices:
            continue
        replacement = rng.choice(choices)
        mutated = word[:position] + (replacement,) + word[position + 1 :]
        if automaton.is_legal(mutated):
            return mutated, {
                "edit": "single_replace",
                "position": position,
                "old": word[position],
                "new": replacement,
            }
    return None


def sample_crispr_mutation(
    automaton: GNFAutomaton,
    word: tuple[int, ...],
    *,
    max_delete: int,
    max_insert: int,
    max_bridge: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], dict] | None:
    operators = ["single_replace", "bridge_replace", "insert", "delete"]
    rng.shuffle(operators)
    for operator in operators:
        if operator == "single_replace":
            result = sample_single_replace(automaton, word, rng=rng)
        elif operator == "bridge_replace":
            result = sample_bridge_replace(
                automaton, word, max_bridge=max_bridge, rng=rng
            )
        elif operator == "insert":
            result = sample_insertion(automaton, word, max_insert=max_insert, rng=rng)
        else:
            result = sample_deletion(automaton, word, max_delete=max_delete, rng=rng)
        if result is not None:
            return result
    return None


def generate_crispr_children(
    population: Sequence[dict],
    *,
    factory: CandidateFactory,
    seen: set[tuple[int, tuple[int, ...]]],
    automaton: GNFAutomaton,
    rng: random.Random,
    mutations_per_parent: int,
    max_delete: int,
    max_insert: int,
    max_bridge: int,
    power_mode: str,
    power_offsets: Sequence[int],
    generation: int,
) -> list[Candidate]:
    children: list[Candidate] = []
    for parent in population:
        parent_factors = tuple(int(value) for value in parent["factor_ids"])
        attempts = 0
        accepted = 0
        while accepted < mutations_per_parent and attempts < mutations_per_parent * 12:
            attempts += 1
            result = sample_crispr_mutation(
                automaton,
                parent_factors,
                max_delete=max_delete,
                max_insert=max_insert,
                max_bridge=max_bridge,
                rng=rng,
            )
            if result is None:
                continue
            child_factors, edit_metadata = result
            accepted += 1
            for power in candidate_powers(
                parent_power=int(parent["power"]),
                factor_count=len(child_factors),
                power_mode=power_mode,
                power_offsets=power_offsets,
            ):
                add_candidate(
                    children,
                    seen,
                    factory,
                    power=power,
                    factors=child_factors,
                    source=f"crispr_{edit_metadata['edit']}",
                    stage="crispr",
                    parent_id=int(parent["candidate_id"]),
                    metadata={
                        "generation": generation,
                        "parent_metrics": parent.get("metrics", {}),
                        **edit_metadata,
                    },
                    automaton=automaton,
                )
    return children


def reset_output_files(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("evaluations.jsonl", "progress.jsonl"):
        path = output_dir / name
        if path.exists():
            path.unlink()


def run_suffix(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    rng = random.Random(args.seed)
    output = args.output_dir
    reset_output_files(output)

    automaton = GNFAutomaton(args.n)
    seed_rows = filter_legal_seed_rows(
        load_seed_rows(
            seed_evaluations=args.seed_evaluations,
            seed_summary=args.seed_summary,
            seed_words=args.seed_word,
            top_seeds=args.top_seeds,
        ),
        automaton,
    )
    _write_json(
        output / "seeds.json",
        {
            "format": "p7-seeded-motif-repair-seeds-v1",
            "seed_count": len(seed_rows),
            "seeds": seed_rows,
        },
    )

    evaluator = Evaluator(
        author_repo=args.author_repo,
        p=args.target_p,
        n=args.n,
        r=args.r,
        collision_index=not args.no_collision_index,
        max_collision_records=args.max_collision_records,
    )
    accumulator = SummaryAccumulator(args.top_output)
    factory = CandidateFactory()
    seen: set[tuple[int, tuple[int, ...]]] = set()
    seed_eval_candidates = seed_candidates(
        seed_rows,
        factory=factory,
        stage="suffix_seed",
        seen=seen,
        automaton=automaton,
    )
    seed_eval_rows = evaluator.evaluate(
        seed_eval_candidates,
        batch_size=args.batch_size,
        output_path=output / "evaluations.jsonl",
        phase="suffix_seed",
    )
    accumulator.observe(seed_eval_rows)
    frontier = select_population(
        seed_eval_rows,
        population_size=args.frontier_size,
        rng=rng,
    )

    _append_jsonl(
        output / "progress.jsonl",
        [
            {
                "phase": "suffix_seed",
                "evaluated": len(seed_eval_rows),
                "frontier_size": len(frontier),
                "best_width": accumulator.best_width(),
                "best_defect": accumulator.best_defect(),
                "kernel_hits": len(accumulator.kernel_candidates),
                "collision_summary": evaluator.collision_index.summary()
                if evaluator.collision_index
                else None,
            }
        ],
    )

    halt_reason = "max_extra_length"
    for extra_depth in range(1, args.max_extra_length + 1):
        children = generate_suffix_children(
            frontier,
            factory=factory,
            seen=seen,
            automaton=automaton,
            rng=rng,
            children_per_parent=args.children_per_parent,
            power_mode=args.power_mode,
            power_offsets=_parse_int_list(args.power_offsets),
            extra_depth=extra_depth,
        )
        if not children:
            halt_reason = "no_new_children"
            break
        child_rows = evaluator.evaluate(
            children,
            batch_size=args.batch_size,
            output_path=output / "evaluations.jsonl",
            phase=f"suffix_depth_{extra_depth}",
        )
        accumulator.observe(child_rows)
        frontier = select_population(
            child_rows,
            population_size=args.frontier_size,
            rng=rng,
        )
        progress = {
            "phase": "suffix",
            "extra_depth": extra_depth,
            "generated": len(children),
            "evaluated": len(child_rows),
            "frontier_size": len(frontier),
            "best_width": accumulator.best_width(),
            "best_defect": accumulator.best_defect(),
            "kernel_hits": len(accumulator.kernel_candidates),
            "collision_summary": evaluator.collision_index.summary()
            if evaluator.collision_index
            else None,
        }
        _append_jsonl(output / "progress.jsonl", [progress])
        print(json.dumps(progress), flush=True)
        if accumulator.kernel_candidates and not args.no_stop_at_kernel:
            halt_reason = "kernel_found"
            break

    summary = {
        "format": "p7-seeded-suffix-reservoir-summary-v1",
        "metadata": {
            "target_p": args.target_p,
            "n": args.n,
            "r": args.r,
            "seed": args.seed,
            "seed_evaluations": str(args.seed_evaluations)
            if args.seed_evaluations
            else None,
            "seed_summary": str(args.seed_summary) if args.seed_summary else None,
            "author_repo": str(args.author_repo),
            "top_seeds": args.top_seeds,
            "max_extra_length": args.max_extra_length,
            "frontier_size": args.frontier_size,
            "children_per_parent": args.children_per_parent,
            "power_mode": args.power_mode,
            "power_offsets": list(_parse_int_list(args.power_offsets)),
            "batch_size": args.batch_size,
            "halt_reason": halt_reason,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        "evaluation_summary": accumulator.to_dict(),
        "collision_summary": evaluator.collision_index.summary()
        if evaluator.collision_index
        else None,
        "collision_records": evaluator.collision_index.collision_records
        if evaluator.collision_index
        else [],
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps({"summary": str(output / "summary.json"), "halt_reason": halt_reason}), flush=True)


def run_crispr(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    rng = random.Random(args.seed)
    output = args.output_dir
    reset_output_files(output)

    automaton = GNFAutomaton(args.n)
    seed_rows = filter_legal_seed_rows(
        load_seed_rows(
            seed_evaluations=args.seed_evaluations,
            seed_summary=args.seed_summary,
            seed_words=args.seed_word,
            top_seeds=args.top_seeds,
        ),
        automaton,
    )
    _write_json(
        output / "seeds.json",
        {
            "format": "p7-seeded-motif-crispr-seeds-v1",
            "seed_count": len(seed_rows),
            "seeds": seed_rows,
        },
    )

    evaluator = Evaluator(
        author_repo=args.author_repo,
        p=args.target_p,
        n=args.n,
        r=args.r,
        collision_index=not args.no_collision_index,
        max_collision_records=args.max_collision_records,
    )
    accumulator = SummaryAccumulator(args.top_output)
    factory = CandidateFactory()
    seen: set[tuple[int, tuple[int, ...]]] = set()
    initial_candidates = seed_candidates(
        seed_rows,
        factory=factory,
        stage="crispr_seed",
        seen=seen,
        automaton=automaton,
    )
    initial_rows = evaluator.evaluate(
        initial_candidates,
        batch_size=args.batch_size,
        output_path=output / "evaluations.jsonl",
        phase="crispr_seed",
    )
    accumulator.observe(initial_rows)
    population = select_population(
        initial_rows,
        population_size=args.population_size,
        rng=rng,
    )

    halt_reason = "generations"
    best_defect_seen = accumulator.best_defect()
    stagnant_for = 0
    for generation in range(1, args.generations + 1):
        children = generate_crispr_children(
            population,
            factory=factory,
            seen=seen,
            automaton=automaton,
            rng=rng,
            mutations_per_parent=args.mutations_per_parent,
            max_delete=args.max_delete,
            max_insert=args.max_insert,
            max_bridge=args.max_bridge,
            power_mode=args.power_mode,
            power_offsets=_parse_int_list(args.power_offsets),
            generation=generation,
        )
        if not children:
            halt_reason = "no_new_children"
            break
        child_rows = evaluator.evaluate(
            children,
            batch_size=args.batch_size,
            output_path=output / "evaluations.jsonl",
            phase=f"crispr_generation_{generation}",
        )
        accumulator.observe(child_rows)
        population = select_population(
            population + child_rows,
            population_size=args.population_size,
            rng=rng,
        )
        current_best = accumulator.best_defect()
        if best_defect_seen is None or (current_best is not None and current_best < best_defect_seen):
            best_defect_seen = current_best
            stagnant_for = 0
        else:
            stagnant_for += 1
        progress = {
            "phase": "crispr",
            "generation": generation,
            "generated": len(children),
            "evaluated": len(child_rows),
            "population_size": len(population),
            "best_width": accumulator.best_width(),
            "best_defect": accumulator.best_defect(),
            "kernel_hits": len(accumulator.kernel_candidates),
            "stagnant_for": stagnant_for,
            "collision_summary": evaluator.collision_index.summary()
            if evaluator.collision_index
            else None,
        }
        _append_jsonl(output / "progress.jsonl", [progress])
        print(json.dumps(progress), flush=True)
        if accumulator.kernel_candidates and not args.no_stop_at_kernel:
            halt_reason = "kernel_found"
            break
        if stagnant_for >= args.stagnation_generations:
            population = select_population(
                accumulator.best_by_identity_defect
                + accumulator.best_by_projective_width
                + child_rows,
                population_size=args.population_size,
                rng=rng,
            )
            stagnant_for = 0

    summary = {
        "format": "p7-seeded-motif-crispr-summary-v1",
        "metadata": {
            "target_p": args.target_p,
            "n": args.n,
            "r": args.r,
            "seed": args.seed,
            "seed_evaluations": str(args.seed_evaluations)
            if args.seed_evaluations
            else None,
            "seed_summary": str(args.seed_summary) if args.seed_summary else None,
            "author_repo": str(args.author_repo),
            "top_seeds": args.top_seeds,
            "population_size": args.population_size,
            "generations": args.generations,
            "mutations_per_parent": args.mutations_per_parent,
            "max_delete": args.max_delete,
            "max_insert": args.max_insert,
            "max_bridge": args.max_bridge,
            "power_mode": args.power_mode,
            "power_offsets": list(_parse_int_list(args.power_offsets)),
            "batch_size": args.batch_size,
            "halt_reason": halt_reason,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        "evaluation_summary": accumulator.to_dict(),
        "collision_summary": evaluator.collision_index.summary()
        if evaluator.collision_index
        else None,
        "collision_records": evaluator.collision_index.collision_records
        if evaluator.collision_index
        else [],
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps({"summary": str(output / "summary.json"), "halt_reason": halt_reason}), flush=True)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed-evaluations", type=Path)
    parser.add_argument("--seed-summary", type=Path)
    parser.add_argument("--seed-word", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path, default=DEFAULT_AUTHOR_REPO)
    parser.add_argument("--target-p", type=int, default=7)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--top-seeds", type=int, default=32)
    parser.add_argument("--power-mode", choices=("inherit", "balanced", "both"), default="both")
    parser.add_argument("--power-offsets", default="0")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--top-output", type=int, default=200)
    parser.add_argument("--no-collision-index", action="store_true")
    parser.add_argument("--max-collision-records", type=int, default=200)
    parser.add_argument("--no-stop-at-kernel", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grow p=7 searches from p=5 motif near-misses, then optionally run "
            "a legal CRISPR-style edit repair stage."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    suffix = commands.add_parser("suffix", help="seeded suffix reservoir")
    add_common_args(suffix)
    suffix.add_argument("--max-extra-length", type=int, default=80)
    suffix.add_argument("--frontier-size", type=int, default=1024)
    suffix.add_argument(
        "--children-per-parent",
        type=int,
        default=0,
        help="0 means try every legal next factor.",
    )

    crispr = commands.add_parser("crispr", help="CRISPR edit repair from best rows")
    add_common_args(crispr)
    crispr.add_argument("--population-size", type=int, default=512)
    crispr.add_argument("--generations", type=int, default=80)
    crispr.add_argument("--mutations-per-parent", type=int, default=16)
    crispr.add_argument("--max-delete", type=int, default=8)
    crispr.add_argument("--max-insert", type=int, default=8)
    crispr.add_argument("--max-bridge", type=int, default=8)
    crispr.add_argument("--stagnation-generations", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "suffix":
        run_suffix(args)
    elif args.command == "crispr":
        run_crispr(args)
    else:
        raise ValueError(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
