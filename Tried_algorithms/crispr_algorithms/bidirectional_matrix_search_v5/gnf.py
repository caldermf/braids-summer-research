from __future__ import annotations

import random
from functools import lru_cache
from typing import Optional, Sequence

from peyl.braid_data import GNF, simple_factor_id_maps, valid_first_factor_ids, valid_suffix_factor_ids


class GNFAutomaton:
    """Legal Garside-factor transition graph used by both search directions."""

    def __init__(self, n: int = 4):
        self.n = n
        self.perm_to_id, self.id_to_perm = simple_factor_id_maps(n)
        self.factor_ids = tuple(sorted(self.id_to_perm))
        self.delta_id = self.perm_to_id[GNF.delta_perm(n)]
        if n == 2:
            self.first_ids = (self.delta_id,)
        else:
            self.first_ids = tuple(sorted(valid_first_factor_ids(n=n)))
        self.successors = {
            factor_id: tuple(sorted(valid_suffix_factor_ids(factor_id, n=n)))
            for factor_id in self.factor_ids
        }
        predecessors = {factor_id: [] for factor_id in self.factor_ids}
        for left, rights in self.successors.items():
            for right in rights:
                predecessors[right].append(left)
        self.predecessors = {
            factor_id: tuple(sorted(values))
            for factor_id, values in predecessors.items()
        }
        self.suffix_start_ids = tuple(
            factor_id for factor_id in self.first_ids if self.predecessors[factor_id]
        )

    def is_legal_prefix(self, factors: Sequence[int]) -> bool:
        return bool(factors) and factors[0] in self.first_ids and self.is_internally_legal(factors)

    def is_internally_legal(self, factors: Sequence[int]) -> bool:
        return bool(factors) and all(
            right in self.successors[left] for left, right in zip(factors, factors[1:])
        )

    def can_join(self, prefix: Sequence[int], suffix: Sequence[int]) -> bool:
        return (
            self.is_legal_prefix(prefix)
            and self.is_internally_legal(suffix)
            and suffix[0] in self.successors[prefix[-1]]
        )

    def sample_prefix(self, length: int, rng: random.Random) -> tuple[int, ...]:
        factors = [rng.choice(self.first_ids)]
        while len(factors) < length:
            factors.append(rng.choice(self.successors[factors[-1]]))
        return tuple(factors)

    def sample_suffix(self, length: int, rng: random.Random) -> tuple[int, ...]:
        factors = [rng.choice(self.suffix_start_ids)]
        while len(factors) < length:
            factors.append(rng.choice(self.successors[factors[-1]]))
        return tuple(factors)

    @lru_cache(maxsize=None)
    def can_finish(self, current: int, right: Optional[int], remaining: int) -> bool:
        if remaining < 0:
            return False
        if remaining == 0:
            return right is None or right in self.successors[current]
        return any(
            self.can_finish(next_factor, right, remaining - 1)
            for next_factor in self.successors[current]
        )

    def viable_next(
        self,
        left: Optional[int],
        right: Optional[int],
        remaining_after_choice: int,
        role: str,
    ) -> tuple[int, ...]:
        if left is None:
            candidates = self.first_ids if role == "prefix" else self.suffix_start_ids
        else:
            candidates = self.successors[left]
        return tuple(
            candidate
            for candidate in candidates
            if self.can_finish(candidate, right, remaining_after_choice)
        )

    def sample_bridge(
        self,
        left: Optional[int],
        right: Optional[int],
        length: int,
        rng: random.Random,
        role: str,
    ) -> tuple[int, ...]:
        if length <= 0:
            raise ValueError("bridge length must be positive")
        block = []
        current = left
        for offset in range(length):
            viable = self.viable_next(
                current,
                right,
                length - offset - 1,
                role,
            )
            if not viable:
                raise ValueError("no legal bridge for these boundaries")
            chosen = rng.choice(viable)
            block.append(chosen)
            current = chosen
        return tuple(block)
