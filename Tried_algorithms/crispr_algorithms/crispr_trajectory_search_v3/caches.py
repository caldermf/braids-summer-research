from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import replace
from typing import Iterable

from .models import Trajectory, TrajectoryEvaluation


class EvaluationCache:
    """Exact factor-word cache shared by all islands and the MCTS finisher."""

    def __init__(self, max_size: int = 250_000):
        self.max_size = max_size
        self._evaluations: OrderedDict[bytes | tuple[int, ...], TrajectoryEvaluation] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def key(factor_ids: tuple[int, ...]) -> bytes | tuple[int, ...]:
        if factor_ids and max(factor_ids) <= 255:
            return bytes(factor_ids)
        return factor_ids

    def evaluate(self, evaluator, trajectories: Iterable[Trajectory]) -> list[TrajectoryEvaluation]:
        trajectory_list = list(trajectories)
        missing: dict[bytes | tuple[int, ...], Trajectory] = {}
        for trajectory in trajectory_list:
            key = self.key(trajectory.factor_ids)
            if key in self._evaluations:
                self.hits += 1
                self._evaluations.move_to_end(key)
            elif key not in missing:
                missing[key] = trajectory

        if missing:
            fresh = evaluator.evaluate(missing.values())
            self.misses += len(fresh)
            for evaluation in fresh:
                key = self.key(evaluation.trajectory.factor_ids)
                self._evaluations[key] = evaluation
                self._evaluations.move_to_end(key)
            while len(self._evaluations) > self.max_size:
                self._evaluations.popitem(last=False)
                self.evictions += 1

        output = []
        for trajectory in trajectory_list:
            key = self.key(trajectory.factor_ids)
            cached = self._evaluations.get(key)
            if cached is None:
                # A single oversized request can evict an early item from the
                # same request. Re-evaluate only that rare configuration.
                cached = evaluator.evaluate([trajectory])[0]
                self._evaluations[key] = cached
            scores = dict(cached.island_scores)
            output.append(
                replace(
                    cached,
                    trajectory=trajectory,
                    island_scores=scores,
                    score=scores[trajectory.island],
                )
            )
        return output

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "entries": len(self._evaluations),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hits / total if total else 0.0,
        }


class SeenWordCache:
    """Collision-free global cache because the complete factor tuple is the key."""

    def __init__(self):
        self._seen: set[bytes | tuple[int, ...]] = set()
        self.rejections = 0

    def add(self, factor_ids: tuple[int, ...]) -> bool:
        key = EvaluationCache.key(factor_ids)
        if key in self._seen:
            self.rejections += 1
            return False
        self._seen.add(key)
        return True

    def contains(self, factor_ids: tuple[int, ...]) -> bool:
        return EvaluationCache.key(factor_ids) in self._seen

    def stats(self) -> dict:
        return {"entries": len(self._seen), "duplicate_rejections": self.rejections}


class MatrixNoveltyArchive:
    """Bounded archive of final Burau-state fingerprints."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._fingerprints: OrderedDict[str, None] = OrderedDict()
        self.collisions = 0

    def assign(self, evaluations: list[TrajectoryEvaluation]) -> None:
        counts = Counter(item.matrix_fingerprint for item in evaluations)
        for evaluation in evaluations:
            fingerprint = evaluation.matrix_fingerprint
            unseen = fingerprint not in self._fingerprints
            evaluation.novelty = (2.0 if unseen else 0.0) + 1.0 / counts[fingerprint]
            if unseen:
                self._fingerprints[fingerprint] = None
        while len(self._fingerprints) > self.max_size:
            self._fingerprints.popitem(last=False)

    def stats(self) -> dict:
        return {
            "entries": len(self._fingerprints),
            "fingerprint_collisions_observed": self.collisions,
        }


class MatrixStateTranspositionTable:
    """
    MCTS table keyed by final matrix fingerprint plus GNF boundary information.

    Distinct factor words sharing a key are retained in a bucket, so a hash
    collision never causes a candidate to be silently discarded.
    """

    def __init__(self):
        self._buckets: dict[tuple[str, int, int, int], set[tuple[int, ...]]] = defaultdict(set)
        self.hits = 0
        self.collisions = 0

    def add(self, evaluation: TrajectoryEvaluation, remaining_budget: int) -> bool:
        factors = evaluation.trajectory.factor_ids
        key = (
            evaluation.matrix_fingerprint,
            factors[-1],
            len(factors),
            remaining_budget,
        )
        bucket = self._buckets[key]
        if factors in bucket:
            self.hits += 1
            return False
        if bucket:
            self.collisions += 1
        bucket.add(factors)
        return True

    def stats(self) -> dict:
        return {
            "states": sum(len(bucket) for bucket in self._buckets.values()),
            "keys": len(self._buckets),
            "hits": self.hits,
            "collision_buckets": self.collisions,
        }


class FinishingQueue:
    """Shared set of the most promising low-projlen full trajectories."""

    def __init__(self, max_size: int, threshold: int):
        self.max_size = max_size
        self.threshold = threshold
        self._items: dict[tuple[int, ...], TrajectoryEvaluation] = {}

    @staticmethod
    def rank(evaluation: TrajectoryEvaluation) -> tuple:
        return (
            evaluation.has_kernel,
            -evaluation.final_projlen,
            -evaluation.terminal_weighted_area,
            evaluation.terminal_collapse,
            evaluation.novelty,
        )

    def update(self, evaluations: Iterable[TrajectoryEvaluation]) -> None:
        for evaluation in evaluations:
            if (
                evaluation.final_projlen > self.threshold
                and evaluation.min_late_projlen > self.threshold
            ):
                continue
            key = evaluation.trajectory.factor_ids
            existing = self._items.get(key)
            if existing is None or self.rank(evaluation) > self.rank(existing):
                self._items[key] = evaluation
        if len(self._items) > self.max_size:
            ordered = sorted(self._items.values(), key=self.rank, reverse=True)
            self._items = {
                item.trajectory.factor_ids: item
                for item in ordered[: self.max_size]
            }

    def members(self) -> list[TrajectoryEvaluation]:
        return sorted(self._items.values(), key=self.rank, reverse=True)

    def stats(self) -> dict:
        return {"size": len(self._items), "threshold": self.threshold}
