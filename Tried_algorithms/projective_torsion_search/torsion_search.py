#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import heapq
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRAID_GPT_ROOT = REPO_ROOT / "Braid-GPT"
DEFAULT_AUTHOR_REPO = REPO_ROOT / "hybrid_of_reservoir_crispr_mcts_suffix" / "third_party" / "braids_project"

FlatMatrix = tuple[int, ...]
MatrixTuple = tuple[FlatMatrix, ...]


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


@dataclass(frozen=True)
class SeedWord:
    label: str
    power: int
    factors: tuple[int, ...]


@dataclass(frozen=True)
class FiniteSurvivor:
    base_power: int
    base_factors: tuple[int, ...]
    source: str
    projective_order: int
    order_multiple: int
    exact_exponent: int
    per_t_orders: tuple[int, ...]
    raw_powered_length: int
    finite_score: float


def load_braid_gpt_module(braid_gpt_root: Path):
    module_path = braid_gpt_root / "braid_gpt.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find Braid-GPT script at {module_path}")
    spec = importlib.util.spec_from_file_location("braid_gpt_runtime_for_torsion", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Braid-GPT from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def parse_t_values(value: str, p: int) -> tuple[int, ...]:
    if value.strip():
        raw = parse_int_list(value)
    elif p > 3:
        # t=1 often creates extra degeneracy, so the default follows the MITM
        # experiment and uses nonzero, non-one specializations.
        raw = tuple(range(2, p))
    else:
        raw = tuple(range(1, p))
    output = tuple(dict.fromkeys(int(item) % p for item in raw))
    if not output:
        raise ValueError("at least one t-specialization is required")
    if any(item == 0 for item in output):
        raise ValueError("all t-specializations must be nonzero")
    return output


def parse_seed_word(value: str, index: int) -> SeedWord:
    if ":" not in value:
        raise ValueError("seed words must have form POWER:f1,f2,... or LABEL=POWER:f1,f2,...")
    label = f"seed_{index}"
    body = value
    if "=" in value and value.index("=") < value.index(":"):
        label, body = value.split("=", 1)
        label = label.strip() or f"seed_{index}"
    power_text, factor_text = body.split(":", 1)
    factors = tuple(int(part.strip()) for part in factor_text.split(",") if part.strip())
    if not factors:
        raise ValueError(f"seed word {value!r} has no factors")
    return SeedWord(label=label, power=int(power_text), factors=factors)


def clean_metrics(metrics: dict) -> dict:
    output = dict(metrics)
    if "projlen" not in output:
        output["projlen"] = int(output.get("projective_width", 0))
    output.pop("projective_width", None)
    return output


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


def lcm(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result = result * int(value) // math.gcd(result, int(value))
    return result


class FiniteProjectiveRep:
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
        self.bgpt = bgpt
        patch_functools_cache_for_old_python()
        self.peyl, self.polymat, self.evaluate_braids = bgpt.setup_author_imports(author_repo)
        self.n = int(n)
        self.r = int(r)
        self.p = int(p)
        self.t_values = tuple(int(value) % self.p for value in t_values)
        self.rep = self.peyl.JonesSummand(n=self.n, r=self.r, p=self.p)
        self.dim = int(self.rep.dimension())
        self.table = self.peyl.GNF._nf_table(self.n)
        self.identity_tuple = tuple(identity_flat(self.dim) for _ in self.t_values)
        self.projective_identity_tuple = self.normalize_tuple(self.identity_tuple)

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

    def multiply_tuples(self, left: MatrixTuple, right: MatrixTuple) -> MatrixTuple:
        return tuple(mat_mul_flat(a, b, self.dim, self.p) for a, b in zip(left, right))

    def normalize_tuple(self, matrices: MatrixTuple) -> MatrixTuple:
        return tuple(normalize_flat(matrix, self.p) for matrix in matrices)

    def multiply_projective(self, left: MatrixTuple, right: MatrixTuple) -> MatrixTuple:
        return self.normalize_tuple(self.multiply_tuples(left, right))

    def evaluate_factors(self, power: int, factors: Sequence[int]) -> MatrixTuple:
        matrices = self.delta_mats[int(power) % self.table.tau_order]
        for factor_id in factors:
            matrices = self.multiply_tuples(matrices, self.factor_mats[int(factor_id)])
        return matrices

    def projective_order_one_matrix(self, matrix: FlatMatrix, max_order: int) -> int | None:
        base = normalize_flat(matrix, self.p)
        current = base
        identity = normalize_flat(identity_flat(self.dim), self.p)
        for order in range(1, max_order + 1):
            if current == identity:
                return order
            current = normalize_flat(mat_mul_flat(current, base, self.dim, self.p), self.p)
        return None

    def projective_order(self, matrices: MatrixTuple, max_order: int) -> tuple[int | None, tuple[int, ...]]:
        per_t: list[int] = []
        for matrix in matrices:
            order = self.projective_order_one_matrix(matrix, max_order)
            if order is None:
                return None, tuple(per_t)
            per_t.append(order)
        aggregate = lcm(per_t)
        if aggregate > max_order:
            return None, tuple(per_t)

        base = self.normalize_tuple(matrices)
        current = base
        for order in range(1, max_order + 1):
            if current == self.projective_identity_tuple:
                return order, tuple(per_t)
            current = self.multiply_projective(current, base)
        return None, tuple(per_t)

    def powered_braid(self, power: int, factors: Sequence[int], exponent: int):
        base = self.peyl.GNF(n=self.n, power=int(power), factors=tuple(int(x) for x in factors))
        return base ** int(exponent)

    def exact_metrics_for_braids(self, braids: Sequence, batch_size: int) -> list[dict]:
        output: list[dict] = []
        for start in range(0, len(braids), batch_size):
            chunk = braids[start : start + batch_size]
            images = self.evaluate_braids(self.rep, chunk)
            for image in images:
                output.append(clean_metrics(self.bgpt.scalar_identity_metrics(self.polymat, image)))
        return output


def finite_score(*, raw_powered_length: int, target_powered_length: int, base_length: int, order: int) -> float:
    return (
        abs(raw_powered_length - target_powered_length)
        + 0.15 * base_length
        + 0.05 * order
    )


def iter_words_for_length(
    *,
    table,
    length: int,
    samples: int,
    exhaustive_up_to: int,
    rng: random.Random,
) -> Iterable[tuple[str, tuple[int, ...]]]:
    if length <= exhaustive_up_to:
        for index, factors in enumerate(table.normal_forms(length)):
            yield f"exhaustive_len{length}_{index}", tuple(int(x) for x in factors)
        return

    seen: set[tuple[int, ...]] = set()
    attempts = 0
    max_attempts = max(samples * 50, 1000)
    while len(seen) < samples and attempts < max_attempts:
        attempts += 1
        factors = tuple(int(x) for x in table.sample(length, rng))
        if factors in seen:
            continue
        seen.add(factors)
        yield f"random_len{length}_{len(seen)}", factors


def push_survivor(
    heap: list[tuple[float, int, FiniteSurvivor]],
    survivor: FiniteSurvivor,
    *,
    limit: int,
    counter: int,
) -> None:
    # heapq is a min-heap. Store negative score so the worst kept candidate is
    # at the top and can be replaced by a better finite survivor.
    item = (-survivor.finite_score, counter, survivor)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def braid_is_pure_delta_power(braid) -> bool:
    return len(tuple(braid.factors)) == 0 and int(braid.power) != 0


def braid_is_identity(braid) -> bool:
    return len(tuple(braid.factors)) == 0 and int(braid.power) == 0


def survivor_to_row(survivor: FiniteSurvivor) -> dict:
    return {
        "base_power": survivor.base_power,
        "base_factors": list(survivor.base_factors),
        "base_length": len(survivor.base_factors),
        "source": survivor.source,
        "projective_order": survivor.projective_order,
        "order_multiple": survivor.order_multiple,
        "exact_exponent": survivor.exact_exponent,
        "per_t_orders": list(survivor.per_t_orders),
        "raw_powered_length": survivor.raw_powered_length,
        "finite_score": survivor.finite_score,
    }


def run_search(args: argparse.Namespace) -> None:
    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bgpt = load_braid_gpt_module(Path(args.braid_gpt_root))
    t_values = parse_t_values(args.t_values, args.p)
    base_powers = parse_int_list(args.base_powers)
    order_multiples = parse_int_list(args.order_multiples)
    rng = random.Random(args.seed)

    finite = FiniteProjectiveRep(
        bgpt=bgpt,
        author_repo=Path(args.author_repo),
        n=args.n,
        r=args.r,
        p=args.p,
        t_values=t_values,
    )

    metadata = {
        "format": "projective-torsion-search-v1",
        "n": args.n,
        "r": args.r,
        "p": args.p,
        "t_values": list(t_values),
        "base_powers": list(base_powers),
        "order_multiples": list(order_multiples),
        "min_length": args.min_length,
        "max_length": args.max_length,
        "samples_per_length": args.samples_per_length,
        "exhaustive_up_to": args.exhaustive_up_to,
        "max_projective_order": args.max_projective_order,
        "min_projective_order": args.min_projective_order,
        "min_powered_length": args.min_powered_length,
        "max_powered_length": args.max_powered_length,
        "target_powered_length": args.target_powered_length,
        "max_finite_survivors": args.max_finite_survivors,
        "max_exact_checks": args.max_exact_checks,
        "reject_pure_delta_powers": args.reject_pure_delta_powers,
        "seed": args.seed,
    }
    write_json(output_dir / "metadata.json", metadata)

    finite_heap: list[tuple[float, int, FiniteSurvivor]] = []
    finite_seen: set[tuple[int, tuple[int, ...], int]] = set()
    finite_survivor_counter = 0
    tested = 0
    finite_hits_seen = 0
    last_progress = time.time()

    seed_words = [parse_seed_word(value, index) for index, value in enumerate(args.seed_word)]
    seeded_by_length: dict[int, list[SeedWord]] = {}
    for seed in seed_words:
        seeded_by_length.setdefault(len(seed.factors), []).append(seed)

    for length in range(args.min_length, args.max_length + 1):
        word_iter = iter_words_for_length(
            table=finite.table,
            length=length,
            samples=args.samples_per_length,
            exhaustive_up_to=args.exhaustive_up_to,
            rng=rng,
        )
        rows_this_length = 0
        for source, factors in word_iter:
            rows_this_length += 1
            for base_power in base_powers:
                tested += 1
                matrices = finite.evaluate_factors(base_power, factors)
                order, per_t_orders = finite.projective_order(
                    matrices,
                    max_order=args.max_projective_order,
                )
                if order is None or order < args.min_projective_order:
                    continue
                for multiple in order_multiples:
                    exact_exponent = int(order) * int(multiple)
                    raw_powered_length = len(factors) * exact_exponent
                    if raw_powered_length < args.min_powered_length:
                        continue
                    if raw_powered_length > args.max_powered_length:
                        continue
                    key = (int(base_power), factors, exact_exponent)
                    if key in finite_seen:
                        continue
                    finite_seen.add(key)
                    finite_hits_seen += 1
                    survivor = FiniteSurvivor(
                        base_power=int(base_power),
                        base_factors=factors,
                        source=source,
                        projective_order=int(order),
                        order_multiple=int(multiple),
                        exact_exponent=exact_exponent,
                        per_t_orders=tuple(int(x) for x in per_t_orders),
                        raw_powered_length=raw_powered_length,
                        finite_score=finite_score(
                            raw_powered_length=raw_powered_length,
                            target_powered_length=args.target_powered_length,
                            base_length=len(factors),
                            order=exact_exponent,
                        ),
                    )
                    finite_survivor_counter += 1
                    push_survivor(
                        finite_heap,
                        survivor,
                        limit=args.max_finite_survivors,
                        counter=finite_survivor_counter,
                    )
                    if args.write_all_finite_survivors:
                        append_jsonl(output_dir / "finite_survivors_all.jsonl", [survivor_to_row(survivor)])

            now = time.time()
            if now - last_progress >= args.progress_interval_seconds:
                row = {
                    "phase": "scan",
                    "length": length,
                    "rows_this_length": rows_this_length,
                    "tested": tested,
                    "finite_hits_seen": finite_hits_seen,
                    "finite_heap_size": len(finite_heap),
                    "elapsed_seconds": round(now - start, 2),
                }
                print(json.dumps(row, sort_keys=True), flush=True)
                last_progress = now

        for seed in seeded_by_length.get(length, []):
            for base_power in [seed.power]:
                tested += 1
                factors = seed.factors
                matrices = finite.evaluate_factors(base_power, factors)
                order, per_t_orders = finite.projective_order(
                    matrices,
                    max_order=args.max_projective_order,
                )
                if order is None or order < args.min_projective_order:
                    continue
                for multiple in order_multiples:
                    exact_exponent = int(order) * int(multiple)
                    raw_powered_length = len(factors) * exact_exponent
                    if raw_powered_length < args.min_powered_length or raw_powered_length > args.max_powered_length:
                        continue
                    survivor = FiniteSurvivor(
                        base_power=int(base_power),
                        base_factors=factors,
                        source=seed.label,
                        projective_order=int(order),
                        order_multiple=int(multiple),
                        exact_exponent=exact_exponent,
                        per_t_orders=tuple(int(x) for x in per_t_orders),
                        raw_powered_length=raw_powered_length,
                        finite_score=finite_score(
                            raw_powered_length=raw_powered_length,
                            target_powered_length=args.target_powered_length,
                            base_length=len(factors),
                            order=exact_exponent,
                        ),
                    )
                    finite_hits_seen += 1
                    finite_survivor_counter += 1
                    push_survivor(
                        finite_heap,
                        survivor,
                        limit=args.max_finite_survivors,
                        counter=finite_survivor_counter,
                    )

        print(
            json.dumps(
                {
                    "phase": "length_done",
                    "length": length,
                    "tested": tested,
                    "finite_hits_seen": finite_hits_seen,
                    "finite_heap_size": len(finite_heap),
                    "elapsed_seconds": round(time.time() - start, 2),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    survivors = sorted((item[2] for item in finite_heap), key=lambda item: item.finite_score)
    append_jsonl(output_dir / "finite_survivors.jsonl", [survivor_to_row(item) for item in survivors])

    exact_rows: list[dict] = []
    exact_braids = []
    exact_sources: list[tuple[FiniteSurvivor, object]] = []
    seen_powered: set[tuple[int, tuple[int, ...]]] = set()
    skipped_exact = 0

    for survivor in survivors:
        powered = finite.powered_braid(
            survivor.base_power,
            survivor.base_factors,
            survivor.exact_exponent,
        )
        powered_key = (int(powered.power), tuple(int(x) for x in powered.factors))
        if powered_key in seen_powered:
            continue
        seen_powered.add(powered_key)
        powered_length = len(tuple(powered.factors))
        if powered_length < args.min_exact_canonical_length:
            skipped_exact += 1
            continue
        if powered_length > args.max_exact_canonical_length:
            skipped_exact += 1
            continue
        if args.reject_pure_delta_powers and braid_is_pure_delta_power(powered):
            skipped_exact += 1
            continue
        exact_braids.append(powered)
        exact_sources.append((survivor, powered))
        if len(exact_braids) >= args.max_exact_checks:
            break

    exact_metrics = finite.exact_metrics_for_braids(exact_braids, batch_size=args.exact_batch_size)
    kernel_hits: list[dict] = []
    for index, ((survivor, powered), metrics) in enumerate(zip(exact_sources, exact_metrics)):
        is_identity = braid_is_identity(powered)
        is_pure_delta = braid_is_pure_delta_power(powered)
        usable_kernel_hit = bool(metrics["scalar_identity"]) and not is_identity and not (
            args.reject_pure_delta_powers and is_pure_delta
        )
        row = {
            "candidate_id": index,
            "base_power": survivor.base_power,
            "base_factors": list(survivor.base_factors),
            "base_length": len(survivor.base_factors),
            "source": survivor.source,
            "projective_order": survivor.projective_order,
            "order_multiple": survivor.order_multiple,
            "exact_exponent": survivor.exact_exponent,
            "per_t_orders": list(survivor.per_t_orders),
            "raw_powered_length": survivor.raw_powered_length,
            "finite_score": survivor.finite_score,
            "powered_power": int(powered.power),
            "powered_factors": [int(x) for x in powered.factors],
            "powered_length": len(tuple(powered.factors)),
            "is_braid_identity": is_identity,
            "is_pure_delta_power": is_pure_delta,
            "usable_kernel_hit": usable_kernel_hit,
            "metrics": metrics,
        }
        exact_rows.append(row)
        if usable_kernel_hit:
            kernel_hits.append(row)

    append_jsonl(output_dir / "exact_candidates.jsonl", exact_rows)
    best_by_defect = sorted(
        exact_rows,
        key=lambda row: (
            int(row["metrics"]["identity_defect"]),
            int(row["metrics"]["projlen"]),
            int(row["powered_length"]),
        ),
    )[: args.top_output]
    best_by_density = sorted(
        exact_rows,
        key=lambda row: (
            float(row["metrics"]["identity_defect"]) / max(1, int(row["powered_length"])),
            float(row["metrics"]["projlen"]) / max(1, int(row["powered_length"])),
            int(row["powered_length"]),
        ),
    )[: args.top_output]

    summary = {
        "format": "projective-torsion-search-summary-v1",
        "metadata": metadata,
        "counts": {
            "finite_words_tested": tested,
            "finite_hits_seen": finite_hits_seen,
            "finite_survivors_kept": len(survivors),
            "exact_checked": len(exact_rows),
            "exact_skipped": skipped_exact,
            "usable_kernel_hits": len(kernel_hits),
        },
        "kernel_hits": kernel_hits[: args.top_output],
        "best_by_identity_defect": best_by_defect,
        "best_by_identity_density": best_by_density,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"phase": "done", **summary["counts"], "elapsed_seconds": summary["elapsed_seconds"]}, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search for short Garside words whose finite-specialized projective "
            "Burau/Jones images have small order, then exact-verify the powered braid."
        )
    )
    parser.add_argument("--braid-gpt-root", default=str(DEFAULT_BRAID_GPT_ROOT))
    parser.add_argument("--author-repo", default=str(DEFAULT_AUTHOR_REPO))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--t-values", default="")
    parser.add_argument("--base-powers", default="0")
    parser.add_argument("--order-multiples", default="1,2")
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--samples-per-length", type=int, default=20000)
    parser.add_argument("--exhaustive-up-to", type=int, default=3)
    parser.add_argument("--max-projective-order", type=int, default=256)
    parser.add_argument("--min-projective-order", type=int, default=2)
    parser.add_argument("--min-powered-length", type=int, default=16)
    parser.add_argument("--max-powered-length", type=int, default=192)
    parser.add_argument("--target-powered-length", type=int, default=80)
    parser.add_argument("--min-exact-canonical-length", type=int, default=2)
    parser.add_argument("--max-exact-canonical-length", type=int, default=240)
    parser.add_argument("--max-finite-survivors", type=int, default=5000)
    parser.add_argument("--max-exact-checks", type=int, default=1000)
    parser.add_argument("--exact-batch-size", type=int, default=64)
    parser.add_argument("--top-output", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seed-word", action="append", default=[])
    parser.add_argument("--write-all-finite-survivors", action="store_true")
    parser.add_argument("--progress-interval-seconds", type=float, default=30.0)
    parser.add_argument(
        "--reject-pure-delta-powers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject exact candidates that normalize to a pure nonzero Delta power.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
