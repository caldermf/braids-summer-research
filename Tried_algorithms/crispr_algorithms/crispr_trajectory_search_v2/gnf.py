from __future__ import annotations

import random
from functools import lru_cache
from typing import Iterable, Optional, Sequence

from peyl.braid_data import (
    GNF,
    simple_factor_id_maps,
    valid_first_factor_ids,
    valid_suffix_factor_ids,
)


class GNFAutomaton:
    """Cached legal-transition graph for simple Garside factors."""

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

    def is_legal(self, factor_ids: Sequence[int]) -> bool:
        if not factor_ids:
            return False
        if factor_ids[0] not in self.first_ids:
            return False
        return all(
            right in self.successors[left]
            for left, right in zip(factor_ids, factor_ids[1:])
        )

    def valid_next(self, factor_ids: Sequence[int]) -> tuple[int, ...]:
        if not factor_ids:
            return self.first_ids
        return self.successors[factor_ids[-1]]

    @lru_cache(maxsize=None)
    def can_finish(self, current: int, right: Optional[int], remaining: int) -> bool:
        """
        Return whether `remaining` additional factors can connect current to right.

        If right is not None, the final chosen factor must itself be a legal
        predecessor of right.
        """
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
    ) -> tuple[int, ...]:
        candidates = self.first_ids if left is None else self.successors[left]
        return tuple(
            candidate
            for candidate in candidates
            if self.can_finish(candidate, right, remaining_after_choice)
        )

    def sample_uniform(self, horizon: int, rng: random.Random) -> tuple[int, ...]:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        factors = [rng.choice(self.first_ids)]
        while len(factors) < horizon:
            factors.append(rng.choice(self.successors[factors[-1]]))
        return tuple(factors)

    def sample_bridge(
        self,
        left: Optional[int],
        right: Optional[int],
        length: int,
        rng: random.Random,
        chooser=None,
        absolute_start: int = 0,
        horizon: Optional[int] = None,
    ) -> tuple[int, ...]:
        """Sample a legal block between fixed optional boundary factors."""
        if length <= 0:
            raise ValueError("bridge length must be positive")
        block = []
        current = left
        for offset in range(length):
            remaining = length - offset - 1
            viable = self.viable_next(current, right, remaining)
            if not viable:
                raise ValueError("no legal GNF bridge exists for the requested boundaries")
            if chooser is None:
                chosen = rng.choice(viable)
            else:
                chosen = chooser(
                    current=current,
                    candidates=viable,
                    depth=absolute_start + offset,
                    horizon=horizon or length,
                    rng=rng,
                )
            block.append(chosen)
            current = chosen
        return tuple(block)

    def assert_legal(self, trajectories: Iterable[Sequence[int]]) -> None:
        for factor_ids in trajectories:
            if not self.is_legal(factor_ids):
                raise AssertionError(f"illegal GNF trajectory: {tuple(factor_ids)}")
