from __future__ import annotations

from typing import Sequence

from .models import Trajectory, TrajectoryEvaluation


def maximum_drawdown(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    running_max = float(values[0])
    best_drop = 0.0
    for value in values[1:]:
        running_max = max(running_max, float(value))
        best_drop = max(best_drop, running_max - float(value))
    return best_drop


def sustained_nonincrease_steps(values: Sequence[float]) -> int:
    best = 0
    current = 0
    for left, right in zip(values, values[1:]):
        if right <= left:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def build_evaluation(
    trajectory: Trajectory,
    projlen_history: Sequence[int],
    late_start_fraction: float,
    periodic_distance_history: Sequence[float] = (),
    kernel_depths: Sequence[int] = (),
    kernel_matches: Sequence[dict] = (),
) -> TrajectoryEvaluation:
    if not projlen_history:
        raise ValueError("projlen_history cannot be empty")

    history = tuple(int(value) for value in projlen_history)
    horizon = len(history)
    late_start = min(horizon - 1, max(0, int(horizon * late_start_fraction)))
    late_values = history[late_start:]

    final_projlen = history[-1]
    min_late_projlen = min(late_values)
    late_drop = maximum_drawdown(late_values)
    late_slope = max(
        0.0,
        float(late_values[0] - late_values[-1]) / max(1, len(late_values) - 1),
    )
    sustained = sustained_nonincrease_steps(late_values)

    normalized = [
        float(projlen) / max(1, depth)
        for depth, projlen in enumerate(history, start=1)
    ]
    best_prefix_depth = min(range(horizon), key=normalized.__getitem__) + 1

    expected_final = 2.0 * horizon
    expected_late = 2.0 * (late_start + 1)
    final_advantage = expected_final - final_projlen
    late_advantage = expected_late - min_late_projlen

    periodic_history = tuple(float(value) for value in periodic_distance_history)
    final_periodic = periodic_history[-1] if periodic_history else None
    min_periodic = min(periodic_history) if periodic_history else None

    score = 3.0 * final_advantage
    score += 2.0 * late_advantage
    score += 5.0 * late_drop
    score += 10.0 * late_slope
    score += 0.25 * sustained
    if final_periodic is not None:
        score -= 0.20 * final_periodic / max(1.0, float(horizon))
    if kernel_depths:
        score += 1_000_000.0

    objectives = (
        1.0 if kernel_depths else 0.0,
        -float(final_projlen) / horizon,
        -float(min_late_projlen) / max(1, horizon - late_start),
        late_drop / horizon,
        late_slope,
        -float(final_periodic) / horizon if final_periodic is not None else 0.0,
    )

    return TrajectoryEvaluation(
        trajectory=trajectory,
        projlen_history=history,
        periodic_distance_history=periodic_history,
        kernel_depths=tuple(int(value) for value in kernel_depths),
        kernel_matches=tuple(kernel_matches),
        final_projlen=final_projlen,
        min_projlen=min(history),
        min_late_projlen=min_late_projlen,
        best_prefix_depth=best_prefix_depth,
        late_drop=late_drop,
        late_slope=late_slope,
        sustained_drop_steps=sustained,
        final_periodic_distance=final_periodic,
        min_periodic_distance=min_periodic,
        score=score,
        objectives=objectives,
    )
