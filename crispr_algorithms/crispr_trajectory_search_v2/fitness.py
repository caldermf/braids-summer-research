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


def terminal_descent_steps(values: Sequence[float], rise_tolerance: float) -> int:
    """Count the descent-like suffix that reaches the endpoint."""
    steps = 0
    for left, right in reversed(tuple(zip(values, values[1:]))):
        if right <= left + rise_tolerance:
            steps += 1
            continue
        break
    return steps


def downward_regression_slope(values: Sequence[float]) -> float:
    """Return the positive magnitude of a downward least-squares slope."""
    if len(values) < 2:
        return 0.0
    midpoint = (len(values) - 1) / 2.0
    centered_x = [index - midpoint for index in range(len(values))]
    mean_y = sum(float(value) for value in values) / len(values)
    numerator = sum(
        x_value * (float(y_value) - mean_y)
        for x_value, y_value in zip(centered_x, values)
    )
    denominator = sum(value * value for value in centered_x)
    slope = numerator / denominator if denominator else 0.0
    return max(0.0, -slope)


def build_evaluation(
    trajectory: Trajectory,
    projlen_history: Sequence[int],
    config,
    periodic_distance_history: Sequence[float] = (),
    kernel_depths: Sequence[int] = (),
    kernel_matches: Sequence[dict] = (),
) -> TrajectoryEvaluation:
    if not projlen_history:
        raise ValueError("projlen_history cannot be empty")

    history = tuple(int(value) for value in projlen_history)
    horizon = len(history)
    late_start = min(horizon - 1, max(0, int(horizon * config.late_start_fraction)))
    late_values = history[late_start:]

    final_projlen = history[-1]
    min_late_projlen = min(late_values)
    late_drop = maximum_drawdown(late_values)
    late_slope = max(
        0.0,
        float(late_values[0] - late_values[-1]) / max(1, len(late_values) - 1),
    )
    sustained = sustained_nonincrease_steps(late_values)
    terminal_peak = max(late_values)
    terminal_collapse = max(0.0, float(terminal_peak - final_projlen))
    rebound = max(0.0, float(final_projlen - min_late_projlen))
    terminal_slope = downward_regression_slope(late_values)
    terminal_descent = terminal_descent_steps(
        late_values,
        rise_tolerance=config.terminal_rise_tolerance,
    )

    normalized = [
        float(projlen) / max(1, depth)
        for depth, projlen in enumerate(history, start=1)
    ]
    best_prefix_depth = min(range(horizon), key=normalized.__getitem__) + 1

    expected_final = 2.0 * horizon
    final_advantage = expected_final - final_projlen

    periodic_history = tuple(float(value) for value in periodic_distance_history)
    final_periodic = periodic_history[-1] if periodic_history else None
    min_periodic = min(periodic_history) if periodic_history else None

    score = config.final_advantage_weight * final_advantage
    score += config.terminal_collapse_weight * terminal_collapse
    score += config.terminal_slope_weight * terminal_slope
    score += config.terminal_descent_weight * terminal_descent
    score -= config.rebound_penalty_weight * rebound
    if final_periodic is not None:
        score -= config.periodic_distance_weight * final_periodic
    if kernel_depths:
        score += config.kernel_bonus

    objectives = (
        1.0 if kernel_depths else 0.0,
        -float(final_projlen) / horizon,
        terminal_collapse / horizon,
        terminal_slope,
        -rebound / horizon,
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
        terminal_peak_projlen=terminal_peak,
        terminal_collapse=terminal_collapse,
        rebound=rebound,
        terminal_slope=terminal_slope,
        terminal_descent_steps=terminal_descent,
        final_periodic_distance=final_periodic,
        min_periodic_distance=min_periodic,
        score=score,
        objectives=objectives,
    )
