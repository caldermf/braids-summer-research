#!/usr/bin/env python3
"""
Hybrid reservoir/frontier search for low-projlen Burau kernel candidates.

This is intentionally not MCTS. It keeps the part of the paper's algorithm that
seems structurally important: a population frontier organized by Garside length
and projlen buckets. Within each bucket, however, it keeps candidates using a
new score that mixes:

  - length-relative projlen surprise,
  - recent projlen drops,
  - recent downward slope,
  - distance to the projective identity/Delta frontier.

The goal is to rediscover p=5 with less blind luck than pure reservoir sampling,
then use the same frontier machinery as a testbed for p >= 7.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peyl.braid_data import (
    GNF,
    append_factor_to_burau_matrix,
    delta_burau_matrix,
    factor_ids_to_artin_word,
    factor_ids_to_perms,
    identity_burau_matrix,
    polynomial_matrix_degree_bounds,
    polynomial_matrix_projlen,
    projective_kernel_match,
    serialize_prefix_state,
    simple_factor_burau_table,
    simple_factor_id_maps,
    valid_first_factor_ids,
    valid_suffix_factor_ids,
)


@dataclass
class OnlineProjlenBaseline:
    max_depth: int
    counts: List[int] = field(init=False)
    means: List[float] = field(init=False)
    m2: List[float] = field(init=False)

    def __post_init__(self) -> None:
        self.counts = [0 for _ in range(self.max_depth + 1)]
        self.means = [0.0 for _ in range(self.max_depth + 1)]
        self.m2 = [0.0 for _ in range(self.max_depth + 1)]

    def add(self, depth: int, projlen: int) -> None:
        if depth < 0 or depth > self.max_depth:
            return
        self.counts[depth] += 1
        count = self.counts[depth]
        delta = float(projlen) - self.means[depth]
        self.means[depth] += delta / count
        delta2 = float(projlen) - self.means[depth]
        self.m2[depth] += delta * delta2

    def mean(self, depth: int) -> float:
        if depth < 0 or depth > self.max_depth or self.counts[depth] == 0:
            return 2.0 * depth
        return self.means[depth]

    def std(self, depth: int) -> float:
        if depth < 0 or depth > self.max_depth or self.counts[depth] < 2:
            return 1.0
        return max(1.0, math.sqrt(self.m2[depth] / (self.counts[depth] - 1)))

    def surprise(self, depth: int, projlen: int) -> float:
        return self.mean(depth) - float(projlen)

    def surprise_z(self, depth: int, projlen: int) -> float:
        return self.surprise(depth, projlen) / self.std(depth)

    def to_json(self) -> List[dict]:
        return [
            {
                "depth": depth,
                "count": self.counts[depth],
                "mean_projlen": self.mean(depth),
                "std_projlen": self.std(depth),
            }
            for depth in range(1, self.max_depth + 1)
            if self.counts[depth] > 0
        ]


@dataclass
class PeriodicFrontierConfig:
    p: int = 5
    n: int = 4
    max_depth: int = 65
    baseline_samples: int = 2048
    bucket_size: int = 256
    use_best: int = 50000
    projlen_bucket_width: int = 1
    random_keep_rate: float = 0.02
    slope_window: int = 8
    surprise_z_weight: float = 1.0
    surprise_per_depth_weight: float = 0.1
    low_projlen_weight: float = 0.25
    drop_weight: float = 0.25
    slope_weight: float = 0.75
    periodic_frontier_weight: float = 4.0
    exact_periodic_bonus: float = 1000.0
    stop_at_kernel: bool = True
    max_kernel_hits: int = 20
    seed: int = 1
    output_dir: str = "results/periodic_frontier_reservoir"


@dataclass
class Candidate:
    factor_ids: List[int]
    burau_matrix: object
    projlen_history: List[int]
    depth: int
    projlen: int
    typical_projlen: float
    surprise: float
    surprise_z: float
    recent_drop: float
    recent_slope: float
    identity_distance: float
    delta_distance: float
    periodic_distance: float
    score: float
    kernel_match: dict


class ScoredReservoirBucket:
    def __init__(self, max_size: int, rng: random.Random, random_keep_rate: float):
        self.max_size = max_size
        self.rng = rng
        self.random_keep_rate = random_keep_rate
        self.seen = 0
        self.items: List[Candidate] = []

    def add(self, candidate: Candidate) -> None:
        self.seen += 1
        if len(self.items) < self.max_size:
            self.items.append(candidate)
            return

        worst_idx, worst = min(enumerate(self.items), key=lambda item: item[1].score)
        if candidate.score > worst.score:
            self.items[worst_idx] = candidate
            return

        if self.random_keep_rate <= 0:
            return

        reservoir_probability = self.random_keep_rate * min(1.0, self.max_size / max(1, self.seen))
        if self.rng.random() < reservoir_probability:
            self.items[self.rng.randrange(len(self.items))] = candidate

    def best(self) -> Candidate:
        return max(self.items, key=lambda item: item.score)


def normalize_entry(entry: dict, p: int) -> dict:
    return {
        int(exp): int(coeff) % p
        for exp, coeff in entry.items()
        if int(coeff) % p != 0
    }


def entry_width(entry: dict) -> int:
    if not entry:
        return 0
    exps = list(entry.keys())
    return max(exps) - min(exps) + 1


def monomial_mul_entry(scalar: Tuple[int, int], entry: dict, p: int) -> dict:
    scalar_exp, scalar_coeff = scalar
    out = {}
    for exp, coeff in entry.items():
        value = (scalar_coeff * coeff) % p
        if value:
            out[exp + scalar_exp] = value
    return out


def entry_difference(actual: dict, expected: dict, p: int) -> float:
    actual = normalize_entry(actual, p)
    expected = normalize_entry(expected, p)
    if actual == expected:
        return 0.0
    actual_terms = set(actual.items())
    expected_terms = set(expected.items())
    support_penalty = len(actual_terms.symmetric_difference(expected_terms))
    width_penalty = abs(entry_width(actual) - entry_width(expected))
    return float(support_penalty + 0.25 * width_penalty)


def projective_identity_distance(poly_mat, p: int, n: int) -> float:
    size = n - 1
    penalty = 0.0
    diagonal_scalars: List[Tuple[int, int]] = []

    for i in range(size):
        for j in range(size):
            entry = normalize_entry(poly_mat[i][j], p)
            if i != j:
                penalty += 2.0 * len(entry) + 0.25 * entry_width(entry)
                continue

            if len(entry) == 1:
                diagonal_scalars.append(next(iter(entry.items())))
            else:
                penalty += 2.0 + 1.5 * len(entry) + 0.25 * entry_width(entry)

    if not diagonal_scalars:
        return penalty + 3.0 * size

    counts: Dict[Tuple[int, int], int] = {}
    for scalar in diagonal_scalars:
        counts[scalar] = counts.get(scalar, 0) + 1
    ref_exp, ref_coeff = max(counts, key=counts.get)

    for exp, coeff in diagonal_scalars:
        if exp == ref_exp and coeff == ref_coeff:
            continue
        penalty += 1.0 + 0.1 * min(20, abs(exp - ref_exp))
        if coeff != ref_coeff:
            penalty += 0.5

    return penalty


def projective_target_distance(poly_mat, target_mat, p: int, n: int) -> float:
    size = n - 1
    scalar: Optional[Tuple[int, int]] = None
    penalty = 0.0

    for i in range(size):
        for j in range(size):
            target = normalize_entry(target_mat[i][j], p)
            actual = normalize_entry(poly_mat[i][j], p)
            if not target or not actual:
                continue
            if len(target) == 1 and len(actual) == 1:
                target_exp, target_coeff = next(iter(target.items()))
                actual_exp, actual_coeff = next(iter(actual.items()))
                scalar = (
                    actual_exp - target_exp,
                    (actual_coeff * pow(target_coeff, -1, p)) % p,
                )
                break
        if scalar is not None:
            break

    if scalar is None:
        penalty += 5.0

    for i in range(size):
        for j in range(size):
            target = normalize_entry(target_mat[i][j], p)
            actual = normalize_entry(poly_mat[i][j], p)
            if not target:
                penalty += 2.0 * len(actual) + 0.25 * entry_width(actual)
                continue
            if scalar is None:
                if not actual:
                    penalty += 3.0
                else:
                    penalty += 1.0 + abs(len(actual) - len(target)) + 0.25 * entry_width(actual)
                continue
            expected = monomial_mul_entry(scalar, target, p)
            penalty += entry_difference(actual, expected, p)

    return penalty


def recent_slope(projlen_history: List[int], window: int) -> float:
    if len(projlen_history) < 2:
        return 0.0
    width = min(max(2, window), len(projlen_history))
    old = projlen_history[-width]
    new = projlen_history[-1]
    return max(0.0, float(old - new) / float(width - 1))


class PeriodicFrontierSearch:
    def __init__(self, config: PeriodicFrontierConfig):
        if config.n < 2:
            raise ValueError("n must be at least 2")
        if config.p <= 1:
            raise ValueError("p must be at least 2")
        if config.bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        if config.use_best <= 0:
            raise ValueError("use_best must be positive")
        if config.projlen_bucket_width <= 0:
            raise ValueError("projlen_bucket_width must be positive")
        if config.random_keep_rate < 0:
            raise ValueError("random_keep_rate must be nonnegative")

        self.config = config
        self.rng = random.Random(config.seed)
        self.simple_table = simple_factor_burau_table(p=config.p, n=config.n)
        self.delta_factor_id = simple_factor_id_maps(config.n)[0][GNF.delta_perm(config.n)]
        self.delta_target = delta_burau_matrix(p=config.p, n=config.n)
        self.baseline = self.estimate_baseline()

        self.best_candidate: Optional[Candidate] = None
        self.best_projlen_by_depth: Dict[int, int] = {}
        self.best_score_by_depth: Dict[int, float] = {}
        self.kernel_hits: List[dict] = []

        self.run_dir = self.create_run_directory()
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2)
        with (self.run_dir / "typical_projlen_by_depth.json").open("w", encoding="utf-8") as f:
            json.dump(self.baseline.to_json(), f, indent=2)
        self.depth_log_path = self.run_dir / "depth_summaries.jsonl"

    def create_run_directory(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = Path(self.config.output_dir)
        run_dir = base_dir / f"periodic_frontier_{timestamp}_seed{self.config.seed}"
        suffix = 1
        unique_run_dir = run_dir
        while unique_run_dir.exists():
            suffix += 1
            unique_run_dir = base_dir / f"{run_dir.name}_{suffix}"
        unique_run_dir.mkdir(parents=True, exist_ok=False)
        return unique_run_dir

    def first_actions(self) -> List[int]:
        if self.config.n == 2:
            return [self.delta_factor_id]
        return valid_first_factor_ids(n=self.config.n)

    def legal_actions_from_factors(self, factor_ids: List[int]) -> List[int]:
        if self.config.n == 2:
            return [self.delta_factor_id]
        if not factor_ids:
            return self.first_actions()
        return valid_suffix_factor_ids(factor_ids[-1], n=self.config.n)

    def estimate_baseline(self) -> OnlineProjlenBaseline:
        baseline = OnlineProjlenBaseline(self.config.max_depth)
        for _ in range(self.config.baseline_samples):
            factor_ids: List[int] = []
            matrix = identity_burau_matrix(p=self.config.p, n=self.config.n)
            for depth in range(1, self.config.max_depth + 1):
                actions = self.legal_actions_from_factors(factor_ids)
                if not actions:
                    break
                action = self.rng.choice(actions)
                factor_ids = factor_ids + [action]
                matrix = append_factor_to_burau_matrix(
                    current_matrix=matrix,
                    factor_id=action,
                    simple_table=self.simple_table,
                    p=self.config.p,
                )
                baseline.add(depth, polynomial_matrix_projlen(matrix))
        return baseline

    def score_candidate(
        self,
        factor_ids: List[int],
        matrix,
        parent: Optional[Candidate],
    ) -> Candidate:
        depth = len(factor_ids)
        projlen = polynomial_matrix_projlen(matrix)
        typical = self.baseline.mean(depth)
        surprise = self.baseline.surprise(depth, projlen)
        surprise_z = self.baseline.surprise_z(depth, projlen)
        kernel_match = projective_kernel_match(matrix, p=self.config.p, n=self.config.n)

        if parent is None:
            history = [projlen]
            recent_drop = 0.0
        else:
            history = parent.projlen_history + [projlen]
            recent_drop = max(0.0, float(parent.projlen - projlen))
        slope = recent_slope(history, self.config.slope_window)

        identity_distance = projective_identity_distance(matrix, p=self.config.p, n=self.config.n)
        delta_distance = projective_target_distance(matrix, self.delta_target, p=self.config.p, n=self.config.n)
        periodic_distance = min(identity_distance, delta_distance)
        frontier_closeness = 1.0 / (1.0 + periodic_distance)
        low_projlen_advantage = max(0.0, 1.0 - float(projlen) / max(1.0, typical))

        score = 0.0
        score += self.config.surprise_z_weight * surprise_z
        score += self.config.surprise_per_depth_weight * surprise / max(1.0, float(depth))
        score += self.config.low_projlen_weight * low_projlen_advantage
        score += self.config.drop_weight * recent_drop
        score += self.config.slope_weight * slope
        score += self.config.periodic_frontier_weight * frontier_closeness
        if kernel_match.get("matches"):
            score += self.config.exact_periodic_bonus

        return Candidate(
            factor_ids=factor_ids,
            burau_matrix=matrix,
            projlen_history=history,
            depth=depth,
            projlen=projlen,
            typical_projlen=typical,
            surprise=surprise,
            surprise_z=surprise_z,
            recent_drop=recent_drop,
            recent_slope=slope,
            identity_distance=identity_distance,
            delta_distance=delta_distance,
            periodic_distance=periodic_distance,
            score=score,
            kernel_match=kernel_match,
        )

    def bucket_key(self, candidate: Candidate) -> Tuple[int, int]:
        proj_bucket = candidate.projlen // self.config.projlen_bucket_width
        return candidate.depth, proj_bucket

    def add_to_buckets(
        self,
        buckets: Dict[Tuple[int, int], ScoredReservoirBucket],
        candidate: Candidate,
    ) -> None:
        key = self.bucket_key(candidate)
        if key not in buckets:
            buckets[key] = ScoredReservoirBucket(
                max_size=self.config.bucket_size,
                rng=self.rng,
                random_keep_rate=self.config.random_keep_rate,
            )
        buckets[key].add(candidate)

    def select_frontier(self, buckets: Dict[Tuple[int, int], ScoredReservoirBucket]) -> List[Candidate]:
        ordered_buckets = sorted(
            buckets.items(),
            key=lambda item: (
                item[0][1],
                item[1].best().periodic_distance,
                -item[1].best().score,
            ),
        )
        selected: List[Candidate] = []
        for _, bucket in ordered_buckets:
            candidates = sorted(bucket.items, key=lambda item: item.score, reverse=True)
            for candidate in candidates:
                selected.append(candidate)
                if len(selected) >= self.config.use_best:
                    return selected
        return selected

    def candidate_to_json(self, candidate: Candidate) -> dict:
        min_degree, max_degree = polynomial_matrix_degree_bounds(candidate.burau_matrix)
        state = serialize_prefix_state(
            candidate.factor_ids,
            poly_mat=candidate.burau_matrix,
            p=self.config.p,
            n=self.config.n,
        )
        return {
            "score": candidate.score,
            "depth": candidate.depth,
            "projlen": candidate.projlen,
            "typical_projlen": candidate.typical_projlen,
            "surprise": candidate.surprise,
            "surprise_z": candidate.surprise_z,
            "recent_drop": candidate.recent_drop,
            "recent_slope": candidate.recent_slope,
            "identity_distance": candidate.identity_distance,
            "delta_distance": candidate.delta_distance,
            "periodic_distance": candidate.periodic_distance,
            "kernel_match": candidate.kernel_match,
            "burau_min_degree": min_degree,
            "burau_max_degree": max_degree,
            "factor_ids": [int(item) for item in candidate.factor_ids],
            "gnf_factors": [list(perm) for perm in factor_ids_to_perms(candidate.factor_ids, n=self.config.n)],
            "artin_word": factor_ids_to_artin_word(candidate.factor_ids, d=0, n=self.config.n),
            "state": state,
        }

    def update_best(self, candidate: Candidate) -> None:
        self.best_projlen_by_depth[candidate.depth] = min(
            candidate.projlen,
            self.best_projlen_by_depth.get(candidate.depth, candidate.projlen),
        )
        self.best_score_by_depth[candidate.depth] = max(
            candidate.score,
            self.best_score_by_depth.get(candidate.depth, candidate.score),
        )
        if self.best_candidate is None:
            self.best_candidate = candidate
            return
        if candidate.score > self.best_candidate.score:
            self.best_candidate = candidate

    def save_progress(self) -> None:
        if self.best_candidate is not None:
            with (self.run_dir / "best_candidate.json").open("w", encoding="utf-8") as f:
                json.dump(self.candidate_to_json(self.best_candidate), f, indent=2)
        with (self.run_dir / "kernel_hits.json").open("w", encoding="utf-8") as f:
            json.dump(self.kernel_hits, f, indent=2)

    def log_depth(self, summary: dict) -> None:
        with self.depth_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

    def run(self) -> dict:
        start_time = time.perf_counter()
        root = Candidate(
            factor_ids=[],
            burau_matrix=identity_burau_matrix(p=self.config.p, n=self.config.n),
            projlen_history=[],
            depth=0,
            projlen=0,
            typical_projlen=0.0,
            surprise=0.0,
            surprise_z=0.0,
            recent_drop=0.0,
            recent_slope=0.0,
            identity_distance=0.0,
            delta_distance=0.0,
            periodic_distance=0.0,
            score=0.0,
            kernel_match={"matches": False, "kernel_type": None, "delta_power": None, "scalar": None},
        )
        frontier = [root]

        for depth in range(1, self.config.max_depth + 1):
            depth_start = time.perf_counter()
            buckets: Dict[Tuple[int, int], ScoredReservoirBucket] = {}
            generated = 0
            depth_kernel_hits = 0
            parents_expanded = len(frontier)

            for parent in frontier:
                for action in self.legal_actions_from_factors(parent.factor_ids):
                    child_factor_ids = parent.factor_ids + [action]
                    child_matrix = append_factor_to_burau_matrix(
                        current_matrix=parent.burau_matrix,
                        factor_id=action,
                        simple_table=self.simple_table,
                        p=self.config.p,
                    )
                    child = self.score_candidate(child_factor_ids, child_matrix, parent if parent.depth else None)
                    generated += 1
                    self.add_to_buckets(buckets, child)
                    self.update_best(child)

                    if child.kernel_match.get("matches"):
                        depth_kernel_hits += 1
                        if len(self.kernel_hits) < self.config.max_kernel_hits:
                            self.kernel_hits.append(self.candidate_to_json(child))

            frontier = self.select_frontier(buckets)
            best_depth_candidate = max(frontier, key=lambda item: item.score) if frontier else None
            min_projlen = min((candidate.projlen for candidate in frontier), default=None)
            summary = {
                "depth": depth,
                "parents_expanded": parents_expanded,
                "generated_children": generated,
                "num_buckets": len(buckets),
                "frontier_size": len(frontier),
                "min_frontier_projlen": min_projlen,
                "best_score_candidate_projlen": self.best_candidate.projlen if self.best_candidate else None,
                "best_score_candidate_depth": self.best_candidate.depth if self.best_candidate else None,
                "best_score_candidate_score": self.best_candidate.score if self.best_candidate else None,
                "kernel_hits_this_depth": depth_kernel_hits,
                "kernel_hits_total": len(self.kernel_hits),
                "elapsed_depth_sec": round(time.perf_counter() - depth_start, 4),
                "elapsed_total_sec": round(time.perf_counter() - start_time, 4),
            }
            if best_depth_candidate is not None:
                summary.update(
                    {
                        "best_depth_score": best_depth_candidate.score,
                        "best_depth_projlen": best_depth_candidate.projlen,
                        "best_depth_surprise_z": best_depth_candidate.surprise_z,
                        "best_depth_periodic_distance": best_depth_candidate.periodic_distance,
                    }
                )
            self.log_depth(summary)
            self.save_progress()
            print(json.dumps(summary), flush=True)

            if self.kernel_hits and self.config.stop_at_kernel:
                break
            if not frontier:
                break

        final_summary = {
            "run_dir": str(self.run_dir),
            "p": self.config.p,
            "n": self.config.n,
            "max_depth": self.config.max_depth,
            "best_score_candidate_projlen": self.best_candidate.projlen if self.best_candidate else None,
            "best_score_candidate_depth": self.best_candidate.depth if self.best_candidate else None,
            "best_score": self.best_candidate.score if self.best_candidate else None,
            "best_projlen_by_depth": {str(k): v for k, v in sorted(self.best_projlen_by_depth.items())},
            "best_score_by_depth": {str(k): v for k, v in sorted(self.best_score_by_depth.items())},
            "num_kernel_hits": len(self.kernel_hits),
            "elapsed_sec": round(time.perf_counter() - start_time, 4),
        }
        with (self.run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2)
        self.save_progress()
        return final_summary


def parse_args() -> PeriodicFrontierConfig:
    parser = argparse.ArgumentParser(description="Run periodic-frontier reservoir search.")
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=65)
    parser.add_argument("--baseline-samples", type=int, default=2048)
    parser.add_argument("--bucket-size", type=int, default=256)
    parser.add_argument("--use-best", type=int, default=50000)
    parser.add_argument("--projlen-bucket-width", type=int, default=1)
    parser.add_argument("--random-keep-rate", type=float, default=0.02)
    parser.add_argument("--slope-window", type=int, default=8)
    parser.add_argument("--surprise-z-weight", type=float, default=1.0)
    parser.add_argument("--surprise-per-depth-weight", type=float, default=0.1)
    parser.add_argument("--low-projlen-weight", type=float, default=0.25)
    parser.add_argument("--drop-weight", type=float, default=0.25)
    parser.add_argument("--slope-weight", type=float, default=0.75)
    parser.add_argument("--periodic-frontier-weight", type=float, default=4.0)
    parser.add_argument("--exact-periodic-bonus", type=float, default=1000.0)
    parser.add_argument("--no-stop-at-kernel", action="store_true")
    parser.add_argument("--max-kernel-hits", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="results/periodic_frontier_reservoir")
    args = parser.parse_args()
    return PeriodicFrontierConfig(
        p=args.p,
        n=args.n,
        max_depth=args.max_depth,
        baseline_samples=args.baseline_samples,
        bucket_size=args.bucket_size,
        use_best=args.use_best,
        projlen_bucket_width=args.projlen_bucket_width,
        random_keep_rate=args.random_keep_rate,
        slope_window=args.slope_window,
        surprise_z_weight=args.surprise_z_weight,
        surprise_per_depth_weight=args.surprise_per_depth_weight,
        low_projlen_weight=args.low_projlen_weight,
        drop_weight=args.drop_weight,
        slope_weight=args.slope_weight,
        periodic_frontier_weight=args.periodic_frontier_weight,
        exact_periodic_bonus=args.exact_periodic_bonus,
        stop_at_kernel=not args.no_stop_at_kernel,
        max_kernel_hits=args.max_kernel_hits,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def main() -> None:
    search = PeriodicFrontierSearch(parse_args())
    print(json.dumps(search.run(), indent=2), flush=True)


if __name__ == "__main__":
    main()
