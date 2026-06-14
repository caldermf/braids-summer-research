from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

import numpy as np

from .config import SearchConfig
from .models import Segment


class SuffixLSHIndex:
    """Multi-table LSH over projectively normalized suffix signatures."""

    def __init__(
        self,
        config: SearchConfig,
        suffixes: Sequence[Segment],
        signatures: np.ndarray,
        rng: random.Random,
    ):
        if len(suffixes) != len(signatures):
            raise ValueError("suffix and signature counts differ")
        self.config = config
        self.suffixes = suffixes
        self.signatures = signatures
        self.rng = rng
        coordinate_rng = np.random.default_rng(config.seed)
        self.coordinates = [
            np.sort(
                coordinate_rng.choice(
                    signatures.shape[1],
                    size=min(config.lsh_key_components, signatures.shape[1]),
                    replace=False,
                )
            )
            for _ in range(config.lsh_tables)
        ]
        self.tables: list[dict[int, list[int]]] = []
        self.by_first: dict[int, list[int]] = defaultdict(list)
        for index, suffix in enumerate(suffixes):
            self.by_first[suffix.factor_ids[0]].append(index)
        for coordinates in self.coordinates:
            table: dict[int, list[int]] = defaultdict(list)
            values = self._packed_values(signatures[:, coordinates])
            for index, (suffix, value) in enumerate(zip(suffixes, values)):
                table[self._key(suffix.factor_ids[0], int(value))].append(index)
            self.tables.append(dict(table))

    def _packed_values(self, values: np.ndarray) -> np.ndarray:
        powers = np.array(
            [self.config.p**index for index in range(values.shape[1])],
            dtype=np.int64,
        )
        return values.astype(np.int64) @ powers

    @staticmethod
    def _key(first_factor: int, packed_value: int) -> int:
        return first_factor + 32 * packed_value

    def query(
        self,
        target: np.ndarray,
        allowed_first_factors: Sequence[int],
        count: int,
    ) -> list[tuple[int, int]]:
        candidates: set[int] = set()
        for table, coordinates in zip(self.tables, self.coordinates):
            packed = int(self._packed_values(target[None, coordinates])[0])
            for first_factor in allowed_first_factors:
                candidates.update(table.get(self._key(first_factor, packed), ()))

        if len(candidates) < count:
            for first_factor in allowed_first_factors:
                source = self.by_first.get(first_factor, ())
                if not source:
                    continue
                needed = min(
                    self.config.max_lsh_candidates - len(candidates),
                    max(count, 16),
                    len(source),
                )
                if needed <= 0:
                    break
                candidates.update(self.rng.sample(list(source), needed))

        candidate_list = list(candidates)
        if len(candidate_list) > self.config.max_lsh_candidates:
            candidate_list = self.rng.sample(
                candidate_list,
                self.config.max_lsh_candidates,
            )
        if not candidate_list:
            return []
        candidate_signatures = self.signatures[np.asarray(candidate_list)]
        distances = np.count_nonzero(candidate_signatures != target[None, :], axis=1)
        order = np.argsort(distances, kind="stable")[:count]
        return [
            (candidate_list[int(position)], int(distances[int(position)]))
            for position in order
        ]

    def stats(self) -> dict:
        bucket_counts = [len(table) for table in self.tables]
        return {
            "suffixes": len(self.suffixes),
            "tables": len(self.tables),
            "key_components": self.config.lsh_key_components,
            "buckets_min": min(bucket_counts, default=0),
            "buckets_max": max(bucket_counts, default=0),
            "first_factor_classes": len(self.by_first),
        }
