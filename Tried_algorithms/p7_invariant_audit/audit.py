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
class AuditCandidate:
    label: str
    origin: str
    power: int
    factor_ids: tuple[int, ...]
    source_id: str
    source_metrics: dict
    source_metadata: dict

    @property
    def length(self) -> int:
        return len(self.factor_ids)

    def key(self) -> tuple[str, int, tuple[int, ...]]:
        return self.label, self.power, self.factor_ids


def _read_json(path: str | Path) -> dict:
    input_path = Path(path)
    if input_path.suffix == ".gz":
        with gzip.open(input_path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(input_path.read_text(encoding="utf-8"))


def _append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _iter_jsonl(path: str | Path) -> Iterable[dict]:
    input_path = Path(path)
    opener = gzip.open if input_path.suffix == ".gz" else open
    with opener(input_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected LABEL=PATH, got {value!r}")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def _parse_seed_word(value: str) -> AuditCandidate:
    if "=" not in value or ":" not in value:
        raise ValueError("expected LABEL=POWER:f1,f2,...")
    label, rest = value.split("=", 1)
    power_text, factors_text = rest.split(":", 1)
    factors = tuple(int(part.strip()) for part in factors_text.split(",") if part.strip())
    return AuditCandidate(
        label=label.strip(),
        origin="cli_seed_word",
        power=int(power_text),
        factor_ids=factors,
        source_id=value,
        source_metrics={},
        source_metadata={},
    )


def parse_length_bands(value: str) -> tuple[tuple[int, int], ...]:
    bands = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            lo = hi = int(part)
        else:
            lo_text, hi_text = part.split(":", 1)
            lo = int(lo_text)
            hi = int(hi_text)
        if lo > hi or lo < 1:
            raise ValueError(f"invalid length band {part!r}")
        bands.append((lo, hi))
    if not bands:
        raise ValueError("at least one length band is required")
    return tuple(bands)


def setup_author_imports(author_repo: Path):
    if not (author_repo / "peyl" / "braid.py").exists():
        raise FileNotFoundError(f"vendored peyl package is missing at {author_repo}")
    if str(author_repo) not in sys.path:
        sys.path.insert(0, str(author_repo))
    import peyl  # type: ignore
    from peyl import polymat  # type: ignore
    from peyl.braidsearch import evaluate_braids  # type: ignore
    from peyl.permutations import (  # type: ignore
        compose,
        cycle_type,
        identity,
        length as permutation_length,
        longest_element,
        parity as permutation_parity,
    )

    return {
        "peyl": peyl,
        "polymat": polymat,
        "evaluate_braids": evaluate_braids,
        "compose": compose,
        "cycle_type": cycle_type,
        "identity": identity,
        "permutation_length": permutation_length,
        "longest_element": longest_element,
        "permutation_parity": permutation_parity,
    }


def source_score(row: dict) -> tuple[int, int, int, int]:
    metrics = row.get("metrics", {})
    return (
        int(metrics.get("identity_defect", 10**12)),
        int(metrics.get("projective_width", 10**12)),
        int(row.get("length", len(row.get("factor_ids", [])))),
        int(row.get("candidate_id", 10**12)),
    )


def source_width_score(row: dict) -> tuple[int, int, int, int]:
    metrics = row.get("metrics", {})
    return (
        int(metrics.get("projective_width", 10**12)),
        int(metrics.get("identity_defect", 10**12)),
        int(row.get("length", len(row.get("factor_ids", [])))),
        int(row.get("candidate_id", 10**12)),
    )


def add_top(bucket: list[dict], row: dict, *, limit: int, key_fn) -> None:
    bucket.append(row)
    bucket.sort(key=key_fn)
    del bucket[limit:]


def load_evaluation_candidates(
    *,
    label: str,
    path: Path,
    bands: Sequence[tuple[int, int]],
    top_per_band: int,
    include_width_tops: bool,
    min_length: int,
    max_length: int | None,
) -> list[AuditCandidate]:
    by_defect: dict[tuple[int, int], list[dict]] = {band: [] for band in bands}
    by_width: dict[tuple[int, int], list[dict]] = {band: [] for band in bands}
    total = 0
    kept = 0
    for row in _iter_jsonl(path):
        total += 1
        factors = row.get("factor_ids") or []
        length = int(row.get("length", len(factors)))
        if length < min_length or (max_length is not None and length > max_length):
            continue
        if "power" not in row or not factors:
            continue
        for band in bands:
            lo, hi = band
            if lo <= length <= hi:
                add_top(by_defect[band], row, limit=top_per_band, key_fn=source_score)
                if include_width_tops:
                    add_top(
                        by_width[band],
                        row,
                        limit=top_per_band,
                        key_fn=source_width_score,
                    )
                kept += 1

    selected: dict[tuple[int, tuple[int, ...], str], AuditCandidate] = {}
    for band in bands:
        rows = list(by_defect[band]) + list(by_width[band])
        for row in rows:
            factors = tuple(int(value) for value in row["factor_ids"])
            source_id = f"{path}:{row.get('candidate_id', 'unknown')}"
            key = (int(row["power"]), factors, source_id)
            selected[key] = AuditCandidate(
                label=label,
                origin="evaluations_jsonl",
                power=int(row["power"]),
                factor_ids=factors,
                source_id=source_id,
                source_metrics=row.get("metrics", {}),
                source_metadata={
                    "path": str(path),
                    "candidate_id": row.get("candidate_id"),
                    "source": row.get("source"),
                    "stage": row.get("stage"),
                    "phase": row.get("phase"),
                    "parent_id": row.get("parent_id"),
                    "metadata": row.get("metadata", {}),
                    "length_band": list(band),
                },
            )
    print(
        json.dumps(
            {
                "loaded_evaluations": str(path),
                "label": label,
                "rows_seen": total,
                "band_memberships_seen": kept,
                "selected": len(selected),
            }
        ),
        flush=True,
    )
    return list(selected.values())


def load_checkpoint_candidates(
    *,
    label: str,
    path: Path,
    limit: int,
    min_length: int,
    max_length: int | None,
) -> list[AuditCandidate]:
    payload = _read_json(path)
    output = []

    def maybe_add(origin: str, index: int, power: int, factors: Sequence[int], metadata: dict) -> None:
        if not factors:
            return
        length = len(factors)
        if length < min_length or (max_length is not None and length > max_length):
            return
        output.append(
            AuditCandidate(
                label=label,
                origin=origin,
                power=int(power),
                factor_ids=tuple(int(value) for value in factors),
                source_id=f"{path}:{origin}:{index}",
                source_metrics=metadata.get("metrics", {}),
                source_metadata={**metadata, "path": str(path), "record_index": index},
            )
        )

    for index, record in enumerate(payload.get("collision_records", [])):
        quotient = record.get("quotient", {})
        maybe_add(
            "checkpoint_collision_quotient",
            index,
            int(quotient.get("power", 0)),
            quotient.get("factor_ids", []),
            {
                "verified_projective_identity": record.get("verified_projective_identity"),
                "match": record.get("match", {}),
                "quotient_garside_length": quotient.get("garside_length"),
            },
        )
        if len(output) >= limit:
            break

    remaining = max(0, limit - len(output))
    if remaining:
        kernel_candidates = payload.get("kernel_candidates", [])
        def kernel_sort_key(row: dict) -> tuple[int, int]:
            return (
                int(row.get("author_projlen", row.get("projective_width", 10**9))),
                len(row.get("factor_ids", [])),
            )

        for index, record in enumerate(sorted(kernel_candidates, key=kernel_sort_key)[:remaining]):
            maybe_add(
                "checkpoint_kernel_candidate",
                index,
                int(record.get("power", 0)),
                record.get("factor_ids", []),
                {
                    "author_projlen": record.get("author_projlen"),
                    "depth": record.get("depth"),
                    "bucket": record.get("bucket"),
                },
            )

    print(
        json.dumps(
            {
                "loaded_checkpoint": str(path),
                "label": label,
                "selected": len(output),
            }
        ),
        flush=True,
    )
    return output[:limit]


def dedupe_candidates(candidates: Sequence[AuditCandidate]) -> list[AuditCandidate]:
    unique: dict[tuple[str, int, tuple[int, ...]], AuditCandidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.key(), candidate)
    return list(unique.values())


def factor_word_invariants(
    *,
    factor_ids: tuple[int, ...],
    power: int,
    automaton: GNFAutomaton,
    min_meaningful_length: int,
) -> dict:
    counts = Counter(factor_ids)
    dominant_factor, dominant_count = counts.most_common(1)[0]
    max_run_factor = factor_ids[0]
    max_run_length = 1
    current_factor = factor_ids[0]
    current_length = 1
    for factor in factor_ids[1:]:
        if factor == current_factor:
            current_length += 1
        else:
            if current_length > max_run_length:
                max_run_factor = current_factor
                max_run_length = current_length
            current_factor = factor
            current_length = 1
    if current_length > max_run_length:
        max_run_factor = current_factor
        max_run_length = current_length

    exact_period = None
    for period in range(1, min(32, len(factor_ids)) + 1):
        if len(factor_ids) % period == 0:
            block = factor_ids[:period]
            if block * (len(factor_ids) // period) == factor_ids:
                exact_period = period
                break

    legal = automaton.is_legal(factor_ids)
    flags = {
        "shorter_than_min_meaningful": len(factor_ids) < min_meaningful_length,
        "dominant_factor_over_70pct": dominant_count / len(factor_ids) >= 0.70,
        "run_over_half_word": max_run_length / len(factor_ids) >= 0.50,
        "exact_period_at_most_2": exact_period is not None and exact_period <= 2,
        "not_legal_gnf_factor_sequence": not legal,
    }
    return {
        "length": len(factor_ids),
        "power": power,
        "delta_power_parity": power % 2,
        "legal_gnf_factor_sequence": legal,
        "unique_factors": len(counts),
        "dominant_factor": dominant_factor,
        "dominant_factor_count": dominant_count,
        "dominant_factor_fraction": dominant_count / len(factor_ids),
        "max_run_factor": max_run_factor,
        "max_run_length": max_run_length,
        "max_run_fraction": max_run_length / len(factor_ids),
        "exact_period": exact_period,
        "factor_histogram": dict(sorted(counts.items())),
        "degeneracy_flags": flags,
        "degeneracy_score": sum(1 for value in flags.values() if value),
    }


def permutation_invariants(candidate, modules: dict, braid) -> dict:
    n = braid.n
    identity = modules["identity"]
    compose = modules["compose"]
    longest_element = modules["longest_element"]
    cycle_type = modules["cycle_type"]
    permutation_length = modules["permutation_length"]
    permutation_parity = modules["permutation_parity"]

    perm = identity(n)
    w0 = longest_element(n)
    if candidate.power % 2:
        perm = compose(w0, perm)
    factor_perms = []
    for factor in braid.canonical_factors():
        factor_perms.append(tuple(int(value) for value in factor.word))
        perm = compose(tuple(int(value) for value in factor.word), perm)
    writhe = candidate.power * (n * (n - 1) // 2)
    writhe += sum(permutation_length(perm_word) for perm_word in factor_perms)
    return {
        "underlying_permutation": list(perm),
        "underlying_cycle_type": list(cycle_type(perm)),
        "underlying_permutation_length": permutation_length(perm),
        "underlying_permutation_parity": permutation_parity(perm),
        "artin_writhe_from_gnf": writhe,
        "writhe_mod_dimension": writhe % 3,
        "factor_permutation_lengths": [
            permutation_length(perm_word) for perm_word in factor_perms
        ],
    }


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


def matrix_rank_mod_p(matrix: np.ndarray, p: int) -> int:
    mat = np.array(matrix, dtype=np.int64) % p
    rows, cols = mat.shape
    rank = 0
    pivot_col = 0
    while rank < rows and pivot_col < cols:
        pivot = None
        for row in range(rank, rows):
            if mat[row, pivot_col] % p:
                pivot = row
                break
        if pivot is None:
            pivot_col += 1
            continue
        if pivot != rank:
            mat[[rank, pivot]] = mat[[pivot, rank]]
        inv = pow(int(mat[rank, pivot_col]), -1, p)
        mat[rank, :] = (mat[rank, :] * inv) % p
        for row in range(rows):
            if row != rank and mat[row, pivot_col] % p:
                mat[row, :] = (mat[row, :] - mat[row, pivot_col] * mat[rank, :]) % p
        rank += 1
        pivot_col += 1
    return rank


def evaluate_poly_at(poly: np.ndarray, t_value: int, p: int) -> int:
    total = 0
    power = 1
    t_value %= p
    for coeff in poly:
        total = (total + int(coeff) * power) % p
        power = (power * t_value) % p
    return total % p


def specialization_invariants(projected: np.ndarray, *, p: int, t_values: Sequence[int]) -> list[dict]:
    dim = projected.shape[0]
    output = []
    for t_value in t_values:
        mat = np.zeros((dim, dim), dtype=np.int64)
        for row in range(dim):
            for col in range(dim):
                mat[row, col] = evaluate_poly_at(projected[row, col, :], t_value, p)
        scalar = int(mat[0, 0] % p)
        residual = mat.copy() % p
        for i in range(dim):
            residual[i, i] = (residual[i, i] - scalar) % p
        output.append(
            {
                "t": int(t_value),
                "scalar": scalar,
                "is_scalar": bool(np.count_nonzero(residual % p) == 0),
                "residual_nonzero_entries": int(np.count_nonzero(residual % p)),
                "residual_rank": int(matrix_rank_mod_p(residual, p)),
                "matrix_rank": int(matrix_rank_mod_p(mat, p)),
            }
        )
    return output


def residual_support(projected: np.ndarray, *, max_terms: int) -> dict:
    dim = projected.shape[0]
    scalar_poly = projected[0, 0, :]
    offdiag = []
    diag_mismatch = []

    for row in range(dim):
        for col in range(dim):
            if row == col:
                continue
            residual = projected[row, col, :]
            nz = np.flatnonzero(residual)
            for degree in nz[: max(0, max_terms - len(offdiag))]:
                offdiag.append(
                    {
                        "row": row,
                        "col": col,
                        "degree": int(degree),
                        "coeff": int(residual[degree]),
                    }
                )
            if len(offdiag) >= max_terms:
                break
        if len(offdiag) >= max_terms:
            break

    for row in range(dim):
        residual = projected[row, row, :] - scalar_poly
        nz = np.flatnonzero(residual)
        for degree in nz[: max(0, max_terms - len(diag_mismatch))]:
            diag_mismatch.append(
                {
                    "row": row,
                    "col": row,
                    "degree": int(degree),
                    "coeff": int(residual[degree]),
                }
            )
        if len(diag_mismatch) >= max_terms:
            break

    entry_ranges = []
    for row in range(dim):
        for col in range(dim):
            nz = np.flatnonzero(projected[row, col, :])
            if len(nz):
                entry_ranges.append(
                    {
                        "row": row,
                        "col": col,
                        "min_degree": int(nz[0]),
                        "max_degree": int(nz[-1]),
                        "terms": int(len(nz)),
                    }
                )
    scalar_nz = np.flatnonzero(scalar_poly)
    return {
        "scalar_poly_support": [
            {"degree": int(degree), "coeff": int(scalar_poly[degree])}
            for degree in scalar_nz[:max_terms]
        ],
        "offdiag_support_sample": offdiag[:max_terms],
        "diag_mismatch_support_sample": diag_mismatch[:max_terms],
        "entry_degree_ranges": entry_ranges,
    }


def image_fingerprint(image: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(tuple(image.shape)).encode("ascii"))
    digest.update(image.dtype.str.encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def audit_candidates(
    candidates: Sequence[AuditCandidate],
    *,
    modules: dict,
    p: int,
    n: int,
    r: int,
    automaton: GNFAutomaton,
    batch_size: int,
    t_values: Sequence[int],
    min_meaningful_length: int,
    max_residual_terms: int,
    rows_path: Path,
) -> list[dict]:
    peyl = modules["peyl"]
    polymat_module = modules["polymat"]
    evaluate_braids = modules["evaluate_braids"]
    rep = peyl.JonesSummand(n=n, r=r, p=p)

    if rows_path.exists():
        rows_path.unlink()

    audited = []
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start : start + batch_size]
        braids = [
            peyl.GNF(n=n, power=candidate.power, factors=candidate.factor_ids)
            for candidate in chunk
        ]
        images = evaluate_braids(rep, braids)
        chunk_rows = []
        for candidate, braid, image in zip(chunk, braids, images):
            projected = polymat_module.projectivise(image)
            metrics = scalar_identity_metrics(polymat_module, image)
            word = factor_word_invariants(
                factor_ids=candidate.factor_ids,
                power=candidate.power,
                automaton=automaton,
                min_meaningful_length=min_meaningful_length,
            )
            row = {
                "label": candidate.label,
                "origin": candidate.origin,
                "source_id": candidate.source_id,
                "target_p": p,
                "n": n,
                "r": r,
                "power": candidate.power,
                "factor_ids": list(candidate.factor_ids),
                "length": candidate.length,
                "source_metrics": candidate.source_metrics,
                "source_metadata": candidate.source_metadata,
                "exact_metrics": metrics,
                "word_invariants": word,
                "permutation_invariants": permutation_invariants(candidate, modules, braid),
                "specializations": specialization_invariants(projected, p=p, t_values=t_values),
                "residual_support": residual_support(projected, max_terms=max_residual_terms),
                "projective_fingerprint": image_fingerprint(projected),
            }
            audited.append(row)
            chunk_rows.append(row)
        _append_jsonl(rows_path, chunk_rows)
        print(
            json.dumps(
                {
                    "audited": min(start + len(chunk), len(candidates)),
                    "total": len(candidates),
                }
            ),
            flush=True,
        )
    return audited


def generate_random_controls(
    candidates: Sequence[AuditCandidate],
    *,
    automaton: GNFAutomaton,
    controls_per_length: int,
    seed: int,
    label: str,
) -> list[AuditCandidate]:
    if controls_per_length <= 0:
        return []
    rng = random.Random(seed)
    lengths = sorted({candidate.length for candidate in candidates})
    controls = []
    seen = set()
    for length in lengths:
        attempts = 0
        while len([c for c in controls if c.length == length]) < controls_per_length:
            attempts += 1
            if attempts > controls_per_length * 200:
                break
            factors = automaton.sample_uniform(length, rng)
            power = -(length // 2)
            key = (power, factors)
            if key in seen:
                continue
            seen.add(key)
            controls.append(
                AuditCandidate(
                    label=label,
                    origin="random_control",
                    power=power,
                    factor_ids=factors,
                    source_id=f"random:{length}:{len(controls)}",
                    source_metrics={},
                    source_metadata={"matched_length": length},
                )
            )
    return controls


def summarise_rows(rows: Sequence[dict], *, top_n: int) -> dict:
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    def exact_defect_key(row: dict) -> tuple[int, int, int, str]:
        metrics = row["exact_metrics"]
        return (
            int(metrics["identity_defect"]),
            int(metrics["projective_width"]),
            int(row["length"]),
            row["source_id"],
        )

    summary_by_label = {}
    for label, label_rows in sorted(by_label.items()):
        degenerate_counts = Counter()
        for row in label_rows:
            for flag, value in row["word_invariants"]["degeneracy_flags"].items():
                if value:
                    degenerate_counts[flag] += 1
        summary_by_label[label] = {
            "count": len(label_rows),
            "length_histogram": dict(sorted(Counter(row["length"] for row in label_rows).items())),
            "identity_defect_histogram": dict(
                sorted(Counter(row["exact_metrics"]["identity_defect"] for row in label_rows).items())[:50]
            ),
            "projective_width_histogram": dict(
                sorted(Counter(row["exact_metrics"]["projective_width"] for row in label_rows).items())[:50]
            ),
            "degeneracy_flag_counts": dict(sorted(degenerate_counts.items())),
            "best_by_exact_defect": sorted(label_rows, key=exact_defect_key)[:top_n],
            "best_meaningful_by_exact_defect": [
                row
                for row in sorted(label_rows, key=exact_defect_key)
                if not row["word_invariants"]["degeneracy_flags"]["shorter_than_min_meaningful"]
                and row["word_invariants"]["degeneracy_score"] == 0
            ][:top_n],
        }

    return {
        "total_rows": len(rows),
        "labels": summary_by_label,
        "global_best_by_exact_defect": sorted(rows, key=exact_defect_key)[:top_n],
        "global_best_meaningful_by_exact_defect": [
            row
            for row in sorted(rows, key=exact_defect_key)
            if not row["word_invariants"]["degeneracy_flags"]["shorter_than_min_meaningful"]
            and row["word_invariants"]["degeneracy_score"] == 0
        ][:top_n],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit algebraic invariants of p=7 candidate braids and controls. "
            "This is diagnostic only; it performs no mutation/search."
        )
    )
    parser.add_argument("--evaluation", action="append", default=[], help="LABEL=path/to/evaluations.jsonl")
    parser.add_argument("--checkpoint", action="append", default=[], help="LABEL=path/to/frontier.json.gz")
    parser.add_argument("--seed-word", action="append", default=[], help="LABEL=POWER:f1,f2,...")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path, default=DEFAULT_AUTHOR_REPO)
    parser.add_argument("--target-p", type=int, default=7)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--length-bands", default="1:4,5:14,15:35,36:80,81:160,161:320")
    parser.add_argument("--top-per-band", type=int, default=25)
    parser.add_argument("--no-width-tops", action="store_true")
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--checkpoint-limit", type=int, default=100)
    parser.add_argument("--controls-per-length", type=int, default=8)
    parser.add_argument("--control-label", default="random_control")
    parser.add_argument("--t-values", default="0,1,2,3,4,5,6")
    parser.add_argument("--min-meaningful-length", type=int, default=15)
    parser.add_argument("--max-residual-terms", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--top-output", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bands = parse_length_bands(args.length_bands)
    candidates: list[AuditCandidate] = []
    for value in args.evaluation:
        label, path = _parse_labeled_path(value)
        candidates.extend(
            load_evaluation_candidates(
                label=label,
                path=path,
                bands=bands,
                top_per_band=args.top_per_band,
                include_width_tops=not args.no_width_tops,
                min_length=args.min_length,
                max_length=args.max_length,
            )
        )
    for value in args.checkpoint:
        label, path = _parse_labeled_path(value)
        candidates.extend(
            load_checkpoint_candidates(
                label=label,
                path=path,
                limit=args.checkpoint_limit,
                min_length=args.min_length,
                max_length=args.max_length,
            )
        )
    candidates.extend(_parse_seed_word(value) for value in args.seed_word)
    candidates = dedupe_candidates(candidates)
    if not candidates:
        raise ValueError("no candidates selected for audit")

    automaton = GNFAutomaton(args.n)
    controls = generate_random_controls(
        candidates,
        automaton=automaton,
        controls_per_length=args.controls_per_length,
        seed=args.seed,
        label=args.control_label,
    )
    all_candidates = dedupe_candidates([*candidates, *controls])

    modules = setup_author_imports(args.author_repo)
    rows = audit_candidates(
        all_candidates,
        modules=modules,
        p=args.target_p,
        n=args.n,
        r=args.r,
        automaton=automaton,
        batch_size=args.batch_size,
        t_values=tuple(int(value) for value in args.t_values.split(",") if value.strip()),
        min_meaningful_length=args.min_meaningful_length,
        max_residual_terms=args.max_residual_terms,
        rows_path=args.output_dir / "audit_rows.jsonl",
    )

    summary = {
        "format": "p7-invariant-audit-summary-v1",
        "metadata": {
            "target_p": args.target_p,
            "n": args.n,
            "r": args.r,
            "author_repo": str(args.author_repo),
            "seed": args.seed,
            "length_bands": [list(band) for band in bands],
            "top_per_band": args.top_per_band,
            "include_width_tops": not args.no_width_tops,
            "min_length": args.min_length,
            "max_length": args.max_length,
            "checkpoint_limit": args.checkpoint_limit,
            "controls_per_length": args.controls_per_length,
            "t_values": [int(value) for value in args.t_values.split(",") if value.strip()],
            "min_meaningful_length": args.min_meaningful_length,
            "max_residual_terms": args.max_residual_terms,
            "batch_size": args.batch_size,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "input_candidate_count": len(candidates),
            "control_candidate_count": len(controls),
            "audited_candidate_count": len(all_candidates),
        },
        "summary": summarise_rows(rows, top_n=args.top_output),
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "summary": str(args.output_dir / "summary.json"),
                "rows": str(args.output_dir / "audit_rows.jsonl"),
                "audited": len(rows),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
