from __future__ import annotations

import random

from .config import SearchConfig
from .gnf import GNFAutomaton
from .models import Segment


class SegmentMutator:
    """Legal role-aware edits for independent prefix and suffix populations."""

    def __init__(
        self,
        config: SearchConfig,
        automaton: GNFAutomaton,
        rng: random.Random,
    ):
        self.config = config
        self.automaton = automaton
        self.rng = rng

    def _bounds(self, role: str) -> tuple[int, int]:
        if role == "prefix":
            return self.config.prefix_length_min, self.config.prefix_length_max
        return self.config.suffix_length_min, self.config.suffix_length_max

    def random_factors(self, role: str) -> tuple[int, ...]:
        minimum, maximum = self._bounds(role)
        length = self.rng.randint(minimum, maximum)
        if role == "prefix":
            return self.automaton.sample_prefix(length, self.rng)
        return self.automaton.sample_suffix(length, self.rng)

    def _is_legal(self, factors: tuple[int, ...], role: str) -> bool:
        if role == "prefix":
            return self.automaton.is_legal_prefix(factors)
        return self.automaton.is_internally_legal(factors)

    def mutate_factors(self, factors: tuple[int, ...], role: str) -> tuple[int, ...]:
        minimum, maximum = self._bounds(role)
        for _ in range(self.config.mutation_attempts):
            action_roll = self.rng.random()
            if action_roll < 0.14 and len(factors) < maximum:
                length = min(
                    self.rng.choice(self.config.length_edit_sizes),
                    maximum - len(factors),
                )
                addition = self.automaton.sample_bridge(
                    factors[-1],
                    None,
                    length,
                    self.rng,
                    role,
                )
                candidate = factors + addition
            elif action_roll < 0.26 and len(factors) > minimum:
                length = min(
                    self.rng.choice(self.config.length_edit_sizes),
                    len(factors) - minimum,
                )
                candidate = factors[:-length]
            else:
                removed = min(
                    self.rng.choice(self.config.mutation_block_sizes),
                    len(factors),
                )
                maximum_start = len(factors) - removed
                if role == "prefix":
                    start = round(self.rng.betavariate(2.5, 1.0) * maximum_start)
                else:
                    start = round(self.rng.betavariate(1.0, 2.5) * maximum_start)
                inserted = removed
                if self.rng.random() < 0.30:
                    inserted += self.rng.choice((-2, -1, 1, 2))
                inserted = max(1, inserted)
                inserted = min(
                    inserted,
                    maximum - len(factors) + removed,
                )
                if len(factors) - removed + inserted < minimum:
                    continue
                end = start + removed
                left = factors[start - 1] if start else None
                right = factors[end] if end < len(factors) else None
                try:
                    replacement = self.automaton.sample_bridge(
                        left,
                        right,
                        inserted,
                        self.rng,
                        role,
                    )
                except ValueError:
                    continue
                candidate = factors[:start] + replacement + factors[end:]
            if candidate != factors and minimum <= len(candidate) <= maximum:
                if self._is_legal(candidate, role):
                    return candidate
        return self.random_factors(role)

    def mutate(self, parent: Segment, segment_id: str) -> Segment:
        return Segment(
            factor_ids=self.mutate_factors(parent.factor_ids, parent.role),
            role=parent.role,
            segment_id=segment_id,
            origin="targeted_mutation",
            parent_id=parent.segment_id,
        )

    def local_neighbors(
        self,
        parent: Segment,
        limit: int,
        id_factory,
    ) -> list[Segment]:
        """Enumerate legal one-factor coordinate moves near the active boundary."""
        positions = list(range(parent.length))
        if parent.role == "prefix":
            positions.sort(reverse=True)
        candidates = []
        for position in positions:
            left = parent.factor_ids[position - 1] if position else None
            right = (
                parent.factor_ids[position + 1]
                if position + 1 < parent.length
                else None
            )
            viable = list(
                self.automaton.viable_next(
                    left,
                    right,
                    0,
                    parent.role,
                )
            )
            self.rng.shuffle(viable)
            for factor_id in viable:
                if factor_id == parent.factor_ids[position]:
                    continue
                factors = (
                    parent.factor_ids[:position]
                    + (factor_id,)
                    + parent.factor_ids[position + 1 :]
                )
                candidates.append(
                    Segment(
                        factor_ids=factors,
                        role=parent.role,
                        segment_id=id_factory(parent.role),
                        origin="coordinate_refinement",
                        parent_id=parent.segment_id,
                    )
                )
                if len(candidates) >= limit:
                    return candidates
        return candidates
