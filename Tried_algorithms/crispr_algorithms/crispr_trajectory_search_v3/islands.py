from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .models import TrajectoryEvaluation


def island_rank(evaluation: TrajectoryEvaluation, island: str) -> tuple:
    kernel = 1 if evaluation.has_kernel else 0
    if island == "endpoint":
        return (
            kernel,
            -evaluation.final_projlen,
            -evaluation.terminal_weighted_area,
            -evaluation.rebound,
            evaluation.novelty,
        )
    if island == "collapse":
        return (
            kernel,
            evaluation.terminal_collapse,
            evaluation.terminal_slope,
            evaluation.terminal_descent_steps,
            -evaluation.rebound,
            -evaluation.final_projlen,
            evaluation.novelty,
        )
    if island == "suffix":
        return (
            kernel,
            -evaluation.terminal_weighted_area,
            -evaluation.rebound,
            -evaluation.final_projlen,
            evaluation.terminal_collapse,
            evaluation.novelty,
        )
    raise ValueError(f"unknown island: {island}")


def select_island_elites(
    evaluations: Iterable[TrajectoryEvaluation],
    island: str,
    count: int,
) -> list[TrajectoryEvaluation]:
    ordered = sorted(evaluations, key=lambda item: island_rank(item, island), reverse=True)
    selected = []
    seen_words = set()
    seen_states = set()
    for evaluation in ordered:
        word = evaluation.trajectory.factor_ids
        state = evaluation.matrix_fingerprint
        if word in seen_words:
            continue
        if state in seen_states and len(selected) < count // 2:
            continue
        selected.append(evaluation)
        seen_words.add(word)
        seen_states.add(state)
        if len(selected) >= count:
            break
    if len(selected) < count:
        for evaluation in ordered:
            if evaluation.trajectory.factor_ids in seen_words:
                continue
            selected.append(evaluation)
            seen_words.add(evaluation.trajectory.factor_ids)
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
    protected: list[TrajectoryEvaluation] = field(default_factory=list)

    def primary_value(self, evaluation: TrajectoryEvaluation) -> float:
        if self.name == "endpoint":
            return -float(evaluation.final_projlen)
        if self.name == "collapse":
            return float(evaluation.terminal_collapse)
        return -float(evaluation.terminal_weighted_area)

    def update(self, evaluations: list[TrajectoryEvaluation], generation: int, minimum: float) -> bool:
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
