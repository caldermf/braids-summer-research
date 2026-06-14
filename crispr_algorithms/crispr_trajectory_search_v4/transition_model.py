from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Optional, Sequence

from .config import SearchConfig
from .gnf import GNFAutomaton
from .models import TrajectoryEvaluation


def weighted_choice(
    candidates: Sequence[int],
    weights: Sequence[float],
    rng: random.Random,
) -> int:
    total = sum(max(0.0, weight) for weight in weights)
    if total <= 0:
        return rng.choice(tuple(candidates))
    target = rng.random() * total
    cumulative = 0.0
    for candidate, weight in zip(candidates, weights):
        cumulative += max(0.0, weight)
        if cumulative >= target:
            return candidate
    return candidates[-1]


class TransitionModel:
    """Depth-conditioned Markov sampler learned from elite trajectories."""

    START = -1

    def __init__(self, config: SearchConfig, automaton: GNFAutomaton):
        self.config = config
        self.automaton = automaton
        self.probabilities: dict[tuple[int, int], dict[int, float]] = {}
        self._initialize_uniform()

    def depth_bin(self, depth: int, horizon: int) -> int:
        scaled = int(self.config.transition_depth_bins * depth / max(1, horizon))
        return min(self.config.transition_depth_bins - 1, max(0, scaled))

    def legal_candidates(self, current: Optional[int]) -> tuple[int, ...]:
        return self.automaton.first_ids if current is None else self.automaton.successors[current]

    def _initialize_uniform(self) -> None:
        previous_values = (self.START,) + self.automaton.factor_ids
        for depth_bin in range(self.config.transition_depth_bins):
            for previous in previous_values:
                current = None if previous == self.START else previous
                candidates = self.legal_candidates(current)
                if not candidates:
                    continue
                probability = 1.0 / len(candidates)
                self.probabilities[(depth_bin, previous)] = {
                    candidate: probability for candidate in candidates
                }

    def reset_uniform(self) -> None:
        """Discard learned basin bias during a true island restart."""
        self.probabilities.clear()
        self._initialize_uniform()

    def choose(
        self,
        current: Optional[int],
        candidates: Sequence[int],
        depth: int,
        horizon: int,
        rng: random.Random,
    ) -> int:
        if not candidates:
            raise ValueError("cannot choose from an empty candidate set")
        previous = self.START if current is None else current
        learned = self.probabilities[(self.depth_bin(depth, horizon), previous)]
        exploration = self.config.transition_exploration
        uniform = 1.0 / len(candidates)
        weights = [
            (1.0 - exploration) * learned.get(candidate, 0.0)
            + exploration * uniform
            for candidate in candidates
        ]
        return weighted_choice(candidates, weights, rng)

    def sample(self, horizon: int, rng: random.Random) -> tuple[int, ...]:
        factors = []
        current = None
        for depth in range(horizon):
            candidates = self.legal_candidates(current)
            chosen = self.choose(
                current=current,
                candidates=candidates,
                depth=depth,
                horizon=horizon,
                rng=rng,
            )
            factors.append(chosen)
            current = chosen
        return tuple(factors)

    def update(self, elites: Iterable[TrajectoryEvaluation]) -> None:
        counts: dict[tuple[int, int], dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        elite_list = list(elites)
        if not elite_list:
            return

        ranked = sorted(elite_list, key=lambda item: item.score, reverse=True)
        rank_weight = {
            item.trajectory.trajectory_id: 1.0 + (len(ranked) - rank) / len(ranked)
            for rank, item in enumerate(ranked)
        }
        for evaluation in elite_list:
            weight = rank_weight[evaluation.trajectory.trajectory_id]
            previous = self.START
            horizon = evaluation.trajectory.horizon
            for depth, factor_id in enumerate(evaluation.trajectory.factor_ids):
                key = (self.depth_bin(depth, horizon), previous)
                counts[key][factor_id] += weight
                previous = factor_id

        rate = self.config.transition_learning_rate
        pseudocount = self.config.transition_pseudocount
        for key, old_probabilities in self.probabilities.items():
            _, previous = key
            current = None if previous == self.START else previous
            legal = self.legal_candidates(current)
            observed = counts.get(key, {})
            total = sum(observed.get(candidate, 0.0) + pseudocount for candidate in legal)
            empirical = {
                candidate: (observed.get(candidate, 0.0) + pseudocount) / total
                for candidate in legal
            }
            updated = {
                candidate: (1.0 - rate) * old_probabilities[candidate]
                + rate * empirical[candidate]
                for candidate in legal
            }
            normalizer = sum(updated.values())
            self.probabilities[key] = {
                candidate: value / normalizer
                for candidate, value in updated.items()
            }

    def top_transitions(self, limit: int = 20) -> list[dict]:
        rows = []
        for (depth_bin, previous), probabilities in self.probabilities.items():
            for next_factor, probability in probabilities.items():
                rows.append(
                    {
                        "depth_bin": depth_bin,
                        "previous_factor": None if previous == self.START else previous,
                        "next_factor": next_factor,
                        "probability": probability,
                    }
                )
        return sorted(rows, key=lambda row: row["probability"], reverse=True)[:limit]
