from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

from .models import TrajectoryEvaluation


def _identity(evaluation: TrajectoryEvaluation) -> tuple[int, ...]:
    return evaluation.trajectory.factor_ids


def select_diverse_elites(
    evaluations: Iterable[TrajectoryEvaluation],
    elite_count: int,
) -> list[TrajectoryEvaluation]:
    """
    Select elites from several objective niches and every active horizon.

    This is deliberately rank-based. A single imperfect scalar score cannot
    erase trajectories that are excellent on final projlen, late collapse, or
    periodic distance.
    """
    population = list(evaluations)
    if elite_count >= len(population):
        return sorted(population, key=lambda item: item.score, reverse=True)

    rankings: list[Callable[[TrajectoryEvaluation], tuple]] = [
        lambda item: (
            item.has_kernel,
            -item.final_projlen / item.trajectory.horizon,
            item.score,
        ),
        lambda item: (
            item.has_kernel,
            -item.min_late_projlen / item.trajectory.horizon,
            item.late_drop,
            item.score,
        ),
        lambda item: (
            item.has_kernel,
            item.late_drop / item.trajectory.horizon,
            item.late_slope,
            -item.final_projlen,
        ),
        lambda item: (
            item.has_kernel,
            -(
                item.final_periodic_distance
                if item.final_periodic_distance is not None
                else float("inf")
            ),
            -item.final_projlen,
        ),
        lambda item: (item.has_kernel, item.score),
    ]

    groups = defaultdict(list)
    for evaluation in population:
        groups[evaluation.trajectory.horizon].append(evaluation)

    ordered_lists = []
    for ranking in rankings:
        ordered_lists.append(sorted(population, key=ranking, reverse=True))
        for group in groups.values():
            ordered_lists.append(sorted(group, key=ranking, reverse=True))

    selected = []
    seen = set()
    index = 0
    while len(selected) < elite_count:
        made_progress = False
        for ordered in ordered_lists:
            if index >= len(ordered):
                continue
            candidate = ordered[index]
            key = _identity(candidate)
            if key not in seen:
                seen.add(key)
                selected.append(candidate)
                made_progress = True
                if len(selected) >= elite_count:
                    break
        if not made_progress and all(index >= len(values) - 1 for values in ordered_lists):
            break
        index += 1

    if len(selected) < elite_count:
        for candidate in sorted(population, key=lambda item: item.score, reverse=True):
            key = _identity(candidate)
            if key in seen:
                continue
            seen.add(key)
            selected.append(candidate)
            if len(selected) >= elite_count:
                break

    return selected
