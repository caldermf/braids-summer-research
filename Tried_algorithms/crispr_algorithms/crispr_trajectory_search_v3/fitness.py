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
    steps = 0
    for left, right in reversed(tuple(zip(values, values[1:]))):
        if right <= left + rise_tolerance:
            steps += 1
        else:
            break
    return steps


def downward_regression_slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    midpoint = (len(values) - 1) / 2.0
    centered_x = [index - midpoint for index in range(len(values))]
    mean_y = sum(float(value) for value in values) / len(values)
    denominator = sum(value * value for value in centered_x)
    if not denominator:
        return 0.0
    slope = sum(
        x_value * (float(y_value) - mean_y)
        for x_value, y_value in zip(centered_x, values)
    ) / denominator
    return max(0.0, -slope)


def weighted_terminal_area(values: Sequence[int], fraction: float) -> float:
    width = max(2, round(len(values) * fraction))
    suffix = values[-width:]
    weights = tuple(range(1, len(suffix) + 1))
    return sum(weight * value for weight, value in zip(weights, suffix)) / sum(weights)


def island_scores(
    *,
    final_projlen: int,
    terminal_collapse: float,
    terminal_slope: float,
    terminal_descent: int,
    rebound: float,
    terminal_area: float,
    kernel: bool,
    config,
) -> dict[str, float]:
    bonus = config.kernel_bonus if kernel else 0.0
    return {
        "endpoint": bonus - float(final_projlen),
        "collapse": (
            bonus
            + 8.0 * terminal_collapse
            + 20.0 * terminal_slope
            + 0.5 * terminal_descent
            - 2.0 * final_projlen
            - 8.0 * rebound
        ),
        "suffix": (
            bonus
            - 5.0 * terminal_area
            - 8.0 * rebound
            - 1.0 * final_projlen
        ),
    }


def build_evaluation(
    trajectory: Trajectory,
    projlen_history: Sequence[int],
    config,
    kernel_depths: Sequence[int] = (),
    kernel_matches: Sequence[dict] = (),
    matrix_fingerprint: str = "",
) -> TrajectoryEvaluation:
    if not projlen_history:
        raise ValueError("projlen_history cannot be empty")

    history = tuple(int(value) for value in projlen_history)
    horizon = len(history)
    late_start = min(horizon - 1, max(0, int(horizon * config.late_start_fraction)))
    late_values = history[late_start:]
    final_projlen = history[-1]
    min_late_projlen = min(late_values)
    terminal_peak = max(late_values)
    terminal_collapse = max(0.0, float(terminal_peak - final_projlen))
    rebound = max(0.0, float(final_projlen - min_late_projlen))
    terminal_slope = downward_regression_slope(late_values)
    terminal_descent = terminal_descent_steps(
        late_values,
        rise_tolerance=config.terminal_rise_tolerance,
    )
    terminal_area = weighted_terminal_area(history, config.suffix_score_fraction)
    normalized = [
        float(projlen) / max(1, depth)
        for depth, projlen in enumerate(history, start=1)
    ]
    best_prefix_depth = min(range(horizon), key=normalized.__getitem__) + 1
    scores = island_scores(
        final_projlen=final_projlen,
        terminal_collapse=terminal_collapse,
        terminal_slope=terminal_slope,
        terminal_descent=terminal_descent,
        rebound=rebound,
        terminal_area=terminal_area,
        kernel=bool(kernel_depths),
        config=config,
    )

    return TrajectoryEvaluation(
        trajectory=trajectory,
        projlen_history=history,
        kernel_depths=tuple(int(value) for value in kernel_depths),
        kernel_matches=tuple(kernel_matches),
        final_projlen=final_projlen,
        min_projlen=min(history),
        min_late_projlen=min_late_projlen,
        best_prefix_depth=best_prefix_depth,
        late_drop=maximum_drawdown(late_values),
        late_slope=max(
            0.0,
            float(late_values[0] - late_values[-1]) / max(1, len(late_values) - 1),
        ),
        sustained_drop_steps=sustained_nonincrease_steps(late_values),
        terminal_peak_projlen=terminal_peak,
        terminal_collapse=terminal_collapse,
        rebound=rebound,
        terminal_slope=terminal_slope,
        terminal_descent_steps=terminal_descent,
        terminal_weighted_area=terminal_area,
        matrix_fingerprint=matrix_fingerprint,
        island_scores=scores,
        score=scores[trajectory.island],
        objectives=(
            1.0 if kernel_depths else 0.0,
            -float(final_projlen),
            terminal_collapse,
            -terminal_area,
            -rebound,
        ),
    )
