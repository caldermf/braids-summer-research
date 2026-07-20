from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReverseState:
    residual: np.ndarray
    suffix: tuple[int, ...]
    cumulative_nll: float
    edge_nll: float
    edge_rank: int
    entropy: float
    projlen: int
    digest: str

    @property
    def depth(self) -> int:
        return len(self.suffix)

    @property
    def average_nll(self) -> float:
        return self.cumulative_nll / max(1, self.depth)

    @property
    def right_factor(self) -> int | None:
        return self.suffix[0] if self.suffix else None


@dataclass
class Bucket:
    key: int
    capacity: int
    rng: random.Random
    states: list[ReverseState] = field(default_factory=list)
    seen: int = 0

    def add(self, state: ReverseState) -> None:
        self.seen += 1
        if len(self.states) < self.capacity:
            self.states.append(state)
            return
        replacement = self.rng.randint(1, self.seen)
        if replacement <= self.capacity:
            self.states[replacement - 1] = state


class LikelihoodReservoir:
    """Uniform reservoirs in unbounded average-NLL bins."""

    def __init__(self, bucket_size: int, bin_width: float, rng: random.Random):
        if bucket_size <= 0 or bin_width <= 0:
            raise ValueError("bucket size and bin width must be positive")
        self.bucket_size = int(bucket_size)
        self.bin_width = float(bin_width)
        self.rng = rng
        self.buckets: dict[int, Bucket] = {}

    def key(self, state: ReverseState) -> int:
        return int(math.floor(max(0.0, state.average_nll) / self.bin_width))

    def add(self, state: ReverseState) -> None:
        key = self.key(state)
        self.buckets.setdefault(key, Bucket(key, self.bucket_size, self.rng)).add(state)

    def size(self) -> int:
        return sum(len(bucket.states) for bucket in self.buckets.values())

    def reset_rng(self, rng: random.Random) -> None:
        self.rng = rng
        for bucket in self.buckets.values():
            bucket.rng = rng

    def select(self, limit: int, exploit_fraction: float) -> tuple[list[ReverseState], dict]:
        all_states = [state for bucket in self.buckets.values() for state in bucket.states]
        if len(all_states) <= limit:
            return all_states, {
                "available": len(all_states), "selected": len(all_states),
                "exploit": len(all_states), "explore": 0,
            }

        exploit_limit = min(limit, int(round(limit * exploit_fraction)))
        selected: list[ReverseState] = []
        selected_ids: set[int] = set()
        for key in sorted(self.buckets):
            candidates = list(self.buckets[key].states)
            self.rng.shuffle(candidates)
            for state in candidates:
                if len(selected) >= exploit_limit:
                    break
                selected.append(state)
                selected_ids.add(id(state))
            if len(selected) >= exploit_limit:
                break

        keys = list(self.buckets)
        self.rng.shuffle(keys)
        queues: dict[int, list[ReverseState]] = {}
        for key in keys:
            queue = [s for s in self.buckets[key].states if id(s) not in selected_ids]
            self.rng.shuffle(queue)
            queues[key] = queue

        while len(selected) < limit:
            progressed = False
            for key in keys:
                if queues[key]:
                    state = queues[key].pop()
                    selected.append(state)
                    selected_ids.add(id(state))
                    progressed = True
                    if len(selected) >= limit:
                        break
            if not progressed:
                break

        return selected, {
            "available": len(all_states),
            "selected": len(selected),
            "exploit": exploit_limit,
            "explore": len(selected) - exploit_limit,
        }

    def summary(self) -> dict:
        rows = []
        for key in sorted(self.buckets):
            bucket = self.buckets[key]
            rows.append({
                "key": key,
                "low": key * self.bin_width,
                "high": (key + 1) * self.bin_width,
                "seen": bucket.seen,
                "kept": len(bucket.states),
            })
        return {"states": self.size(), "buckets": len(rows), "distribution": rows}

