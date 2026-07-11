from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .models import TrajectoryEvaluation


def island_rank(evaluation: TrajectoryEvaluation, island: str) -> tuple:
    kernel = 1 if evaluation.has_kernel else 0
    if island == "endpoint":
        return (
            kernel,
            evaluation.endpoint_advantage,
            -evaluation.normalized_mean,
            -evaluation.final_projlen,
            evaluation.novelty,
        )
    if island == "envelope":
        return (
            kernel,
            -evaluation.normalized_peak,
            -evaluation.normalized_mean,
            evaluation.endpoint_advantage,
            evaluation.novelty,
        )
    if island == "collapse":
        return (
            kernel,
            evaluation.post_turn_drop / evaluation.trajectory.horizon,
            evaluation.post_turn_descent_fraction,
            evaluation.post_turn_slope,
            -evaluation.rebound / evaluation.trajectory.horizon,
            evaluation.endpoint_advantage,
            evaluation.novelty,
        )
    if island == "suffix":
        return (
            kernel,
            -evaluation.normalized_terminal_area,
            -evaluation.rebound / evaluation.trajectory.horizon,
            evaluation.endpoint_advantage,
            evaluation.post_turn_drop / evaluation.trajectory.horizon,
            evaluation.novelty,
        )
    raise ValueError(f"unknown island: {island}")


def length_bin(evaluation: TrajectoryEvaluation, niche_width: int) -> int:
    return evaluation.trajectory.horizon // niche_width


def select_island_elites(
    evaluations: Iterable[TrajectoryEvaluation],
    island: str,
    count: int,
    niche_width: int = 4,
) -> list[TrajectoryEvaluation]:
    """Select objective leaders round-robin across active length niches."""
    groups: dict[int, list[TrajectoryEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        groups[length_bin(evaluation, niche_width)].append(evaluation)
    for group in groups.values():
        group.sort(key=lambda item: island_rank(item, island), reverse=True)

    selected: list[TrajectoryEvaluation] = []
    seen_words = set()
    seen_states = set()
    depth = 0
    ordered_bins = sorted(groups)
    while len(selected) < count:
        progress = False
        for bin_id in ordered_bins:
            group = groups[bin_id]
            while depth < len(group):
                candidate = group[depth]
                word = candidate.trajectory.factor_ids
                if word in seen_words:
                    break
                if candidate.matrix_fingerprint in seen_states and len(selected) < count // 2:
                    break
                selected.append(candidate)
                seen_words.add(word)
                seen_states.add(candidate.matrix_fingerprint)
                progress = True
                break
            if len(selected) >= count:
                return selected
        if not progress and depth >= max((len(group) for group in groups.values()), default=0):
            break
        depth += 1

    if len(selected) < count:
        ordered = sorted(
            (item for group in groups.values() for item in group),
            key=lambda item: island_rank(item, island),
            reverse=True,
        )
        for candidate in ordered:
            if candidate.trajectory.factor_ids in seen_words:
                continue
            selected.append(candidate)
            seen_words.add(candidate.trajectory.factor_ids)
            if len(selected) >= count:
                break
    return selected


@dataclass
class IslandState:
    name: str
    target_size: int
    best: TrajectoryEvaluation | None = None
    best_primary: float | None = None
    last_improvement_generation: int = 0
    restart_count: int = 0

    def primary_value(self, evaluation: TrajectoryEvaluation) -> float:
        if self.name == "endpoint":
            return evaluation.endpoint_advantage
        if self.name == "envelope":
            return -0.6 * evaluation.normalized_peak - 0.4 * evaluation.normalized_mean
        if self.name == "collapse":
            return (
                evaluation.post_turn_drop / evaluation.trajectory.horizon
                + evaluation.post_turn_descent_fraction
                + evaluation.post_turn_slope
            )
        return -evaluation.normalized_terminal_area

    def update(
        self,
        evaluations: list[TrajectoryEvaluation],
        generation: int,
        minimum: float,
    ) -> bool:
        champion = max(evaluations, key=lambda item: island_rank(item, self.name))
        value = self.primary_value(champion)
        improved = self.best_primary is None or value >= self.best_primary + minimum
        if improved:
            self.best_primary = value
            self.best = champion
            self.last_improvement_generation = generation
        elif self.best is None or island_rank(champion, self.name) > island_rank(self.best, self.name):
            self.best = champion
        return improved

    def stagnant(self, generation: int, threshold: int) -> bool:
        return generation - self.last_improvement_generation >= threshold

    def mark_restart(self, generation: int) -> None:
        self.restart_count += 1
        self.last_improvement_generation = generation
