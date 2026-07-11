from __future__ import annotations

import random
from dataclasses import dataclass

from .models import Candidate


class UniformReservoir:
    def __init__(self, capacity: int, rng: random.Random):
        self.capacity = capacity
        self.rng = rng
        self.seen = 0
        self.items: list[Candidate] = []

    def add(self, candidate: Candidate) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(candidate)
            return
        index = self.rng.randrange(self.seen)
        if index < self.capacity:
            self.items[index] = candidate

    def contains(self, factor_ids: tuple[int, ...]) -> bool:
        return any(item.factor_ids == factor_ids for item in self.items)

    def force_include(self, candidate: Candidate) -> None:
        if self.contains(candidate.factor_ids):
            return
        if len(self.items) < self.capacity:
            self.items.append(candidate)
        else:
            self.items[self.rng.randrange(len(self.items))] = candidate


class PeriodicBucket:
    def __init__(
        self,
        capacity: int,
        elite_fraction: float,
        descent_fraction: float,
        random_keep_rate: float,
        rng: random.Random,
    ):
        self.capacity = capacity
        self.rng = rng
        self.seen = 0
        self.elite_size = min(capacity, round(capacity * elite_fraction))
        if elite_fraction > 0 and self.elite_size == 0:
            self.elite_size = 1
        remaining = capacity - self.elite_size
        self.descent_size = min(remaining, round(capacity * descent_fraction))
        if descent_fraction > 0 and self.descent_size == 0 and remaining > 0:
            self.descent_size = 1
        self.random_size = capacity - self.elite_size - self.descent_size
        self.random_keep_rate = random_keep_rate
        self.elite: list[Candidate] = []
        self.descent: list[Candidate] = []
        self.random: list[Candidate] = []

    @property
    def items(self) -> list[Candidate]:
        return self.elite + self.descent + self.random

    def add(self, candidate: Candidate) -> None:
        self.seen += 1
        self._add_ranked(self.elite, self.elite_size, candidate, "periodic_score")
        self._add_ranked(self.descent, self.descent_size, candidate, "descent_score")

        if self.random_size <= 0 or self.rng.random() > self.random_keep_rate:
            return
        if len(self.random) < self.random_size:
            self.random.append(candidate)
            return
        index = self.rng.randrange(self.seen)
        if index < self.random_size:
            self.random[index] = candidate

    @staticmethod
    def _add_ranked(
        items: list[Candidate],
        capacity: int,
        candidate: Candidate,
        attribute: str,
    ) -> None:
        if capacity <= 0:
            return
        if len(items) < capacity:
            items.append(candidate)
            return
        worst_index = min(range(len(items)), key=lambda index: getattr(items[index], attribute))
        if getattr(candidate, attribute) > getattr(items[worst_index], attribute):
            items[worst_index] = candidate

    def contains(self, factor_ids: tuple[int, ...]) -> bool:
        return any(item.factor_ids == factor_ids for item in self.items)

    def best(self) -> Candidate:
        return max(self.items, key=lambda item: item.periodic_score)


def paper_select_buckets(
    buckets: dict[int, UniformReservoir],
    use_best: int,
) -> tuple[list[Candidate], set[int]]:
    selected: list[Candidate] = []
    selected_projlens: set[int] = set()
    total = 0
    for projlen in sorted(buckets):
        items = buckets[projlen].items
        if total + len(items) > use_best:
            break
        selected.extend(items)
        selected_projlens.add(projlen)
        total += len(items)
    return selected, selected_projlens


def _periodic_bucket_candidates(bucket: PeriodicBucket) -> list[Candidate]:
    by_score = sorted(bucket.items, key=lambda item: item.periodic_score, reverse=True)
    by_descent = sorted(bucket.items, key=lambda item: item.descent_score, reverse=True)
    ordered: list[Candidate] = []
    seen = set()
    for index in range(max(len(by_score), len(by_descent))):
        for candidates in (by_score, by_descent):
            if index >= len(candidates):
                continue
            candidate = candidates[index]
            if candidate.factor_ids in seen:
                continue
            seen.add(candidate.factor_ids)
            ordered.append(candidate)
    return ordered


def periodic_select(
    buckets: dict[int, PeriodicBucket],
    use_best: int,
) -> list[Candidate]:
    ordered_buckets = sorted(
        buckets.items(),
        key=lambda item: (
            item[0],
            item[1].best().periodic_distance,
            -item[1].best().periodic_score,
        ),
    )
    selected: list[Candidate] = []
    seen = set()
    for _, bucket in ordered_buckets:
        for candidate in _periodic_bucket_candidates(bucket):
            if candidate.factor_ids in seen:
                continue
            seen.add(candidate.factor_ids)
            selected.append(candidate)
            if len(selected) >= use_best:
                return selected
    return selected


@dataclass
class RankCounter:
    target_value: float
    maximize: bool = True
    better: int = 0
    equal: int = 0
    total: int = 0

    def add(self, value: float) -> None:
        self.total += 1
        if value == self.target_value:
            self.equal += 1
        elif (value > self.target_value) == self.maximize:
            self.better += 1

    @property
    def best_rank(self) -> int:
        return self.better + 1

    @property
    def worst_rank(self) -> int:
        return self.better + self.equal

    @property
    def percentile(self) -> float:
        if not self.total:
            return 1.0
        return 1.0 - self.better / self.total


class CandidateSample:
    def __init__(self, capacity: int, rng: random.Random):
        self.capacity = capacity
        self.rng = rng
        self.seen = 0
        self.items: list[Candidate] = []

    def add(self, candidate: Candidate) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(candidate)
            return
        index = self.rng.randrange(self.seen)
        if index < self.capacity:
            self.items[index] = candidate
