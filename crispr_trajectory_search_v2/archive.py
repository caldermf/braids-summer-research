from __future__ import annotations

from typing import Iterable

from .config import SearchConfig
from .models import TrajectoryEvaluation


def _trajectory_key(evaluation: TrajectoryEvaluation) -> tuple[int, ...]:
    return evaluation.trajectory.factor_ids


class QualityDiversityArchive:
    """MAP-Elites-style archive over score, trajectory shape, and suffix pattern."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.cells: dict[tuple[int, ...], TrajectoryEvaluation] = {}
        self.champions: dict[str, TrajectoryEvaluation] = {}

    def _suffix_niche(self, evaluation: TrajectoryEvaluation) -> int:
        signature = 0
        suffix = evaluation.trajectory.factor_ids[-self.config.archive_suffix_length :]
        for factor_id in suffix:
            signature = (
                signature * 31 + factor_id
            ) % self.config.archive_transition_niches
        return signature

    def _cell_key(self, evaluation: TrajectoryEvaluation) -> tuple[int, ...]:
        return (
            evaluation.trajectory.horizon,
            evaluation.final_projlen // self.config.archive_projlen_bin_size,
            int(evaluation.terminal_collapse)
            // self.config.archive_collapse_bin_size,
            int(evaluation.rebound) // self.config.archive_rebound_bin_size,
            self._suffix_niche(evaluation),
        )

    @staticmethod
    def _quality(evaluation: TrajectoryEvaluation) -> tuple:
        return (
            evaluation.has_kernel,
            evaluation.score,
            -evaluation.final_projlen,
            evaluation.terminal_collapse,
            -evaluation.rebound,
        )

    def _update_champion(
        self,
        name: str,
        candidate: TrajectoryEvaluation,
        key,
    ) -> None:
        current = self.champions.get(name)
        if current is None or key(candidate) > key(current):
            self.champions[name] = candidate

    def update(self, evaluations: Iterable[TrajectoryEvaluation]) -> set[tuple[int, ...]]:
        for evaluation in evaluations:
            horizon = evaluation.trajectory.horizon
            self._update_champion(
                "best_score",
                evaluation,
                lambda item: (item.has_kernel, item.score),
            )
            self._update_champion(
                "lowest_final",
                evaluation,
                lambda item: (item.has_kernel, -item.final_projlen, item.score),
            )
            self._update_champion(
                "lowest_late",
                evaluation,
                lambda item: (item.has_kernel, -item.min_late_projlen, item.score),
            )
            self._update_champion(
                "best_terminal_collapse",
                evaluation,
                lambda item: (
                    item.has_kernel,
                    item.terminal_collapse,
                    item.terminal_slope,
                    -item.final_projlen,
                ),
            )
            self._update_champion(
                "lowest_rebound",
                evaluation,
                lambda item: (
                    item.has_kernel,
                    -item.rebound,
                    -item.final_projlen,
                    item.terminal_collapse,
                ),
            )
            self._update_champion(
                f"lowest_final_h{horizon}",
                evaluation,
                lambda item: (item.has_kernel, -item.final_projlen, item.score),
            )
            if evaluation.final_periodic_distance is not None:
                self._update_champion(
                    "lowest_periodic",
                    evaluation,
                    lambda item: (
                        item.has_kernel,
                        -item.final_periodic_distance,
                        -item.final_projlen,
                    ),
                )

            cell_key = self._cell_key(evaluation)
            current = self.cells.get(cell_key)
            if current is None or self._quality(evaluation) > self._quality(current):
                self.cells[cell_key] = evaluation

        if len(self.cells) > self.config.archive_size:
            retained = sorted(
                self.cells.items(),
                key=lambda item: self._quality(item[1]),
                reverse=True,
            )[: self.config.archive_size]
            self.cells = dict(retained)

        return {_trajectory_key(item) for item in self.members()}

    def members(self) -> list[TrajectoryEvaluation]:
        output = []
        seen = set()
        for evaluation in tuple(self.champions.values()) + tuple(self.cells.values()):
            key = _trajectory_key(evaluation)
            if key in seen:
                continue
            seen.add(key)
            output.append(evaluation)
        return sorted(output, key=self._quality, reverse=True)

    def summary(self) -> dict:
        return {
            "num_cells": len(self.cells),
            "num_unique_members": len(self.members()),
            "champions": {
                name: evaluation.summary()
                for name, evaluation in sorted(self.champions.items())
            },
        }
