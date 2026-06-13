from __future__ import annotations

from typing import Sequence

from .models import Trajectory, TrajectoryEvaluation


def maximum_drawdown(values: Sequence[float]) -> float:
    running_max = float(values[0])
    best = 0.0
    for value in values[1:]:
        running_max = max(running_max, float(value))
        best = max(best, running_max - float(value))
    return best


def sustained_nonincrease_steps(values: Sequence[float]) -> int:
    best = current = 0
    for left, right in zip(values, values[1:]):
        current = current + 1 if right <= left else 0
        best = max(best, current)
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
    centered = [index - midpoint for index in range(len(values))]
    mean = sum(float(value) for value in values) / len(values)
    denominator = sum(value * value for value in centered)
    if not denominator:
        return 0.0
    slope = sum(x * (float(y) - mean) for x, y in zip(centered, values)) / denominator
    return max(0.0, -slope)


def weighted_terminal_area(values: Sequence[int], fraction: float) -> float:
    width = max(2, round(len(values) * fraction))
    suffix = values[-width:]
    weights = tuple(range(1, len(suffix) + 1))
    return sum(weight * value for weight, value in zip(weights, suffix)) / sum(weights)


def best_turning_point(values: Sequence[int], config) -> tuple[int, int, float, float, float]:
    horizon = len(values)
    first = max(1, int(horizon * config.turn_min_fraction))
    last = min(horizon - 2, int(horizon * config.turn_max_fraction))
    best = None
    for index in range(first, last + 1):
        suffix = values[index:]
        drop = max(0.0, float(values[index] - suffix[-1]))
        slope = downward_regression_slope(suffix)
        descent = sum(right <= left for left, right in zip(suffix, suffix[1:]))
        consistency = descent / max(1, len(suffix) - 1)
        candidate = (
            drop,
            consistency,
            slope,
            -index,
            index + 1,
            int(values[index]),
        )
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return horizon, int(values[-1]), 0.0, 0.0, 0.0
    drop, consistency, slope, _, depth, projlen = best
    return depth, projlen, drop, slope, consistency


def island_scores(metrics: dict, kernel: bool, config) -> dict[str, float]:
    bonus = config.kernel_bonus if kernel else 0.0
    return {
        "endpoint": bonus + metrics["endpoint_advantage"] - 0.10 * metrics["normalized_mean"],
        "envelope": (
            bonus
            - 0.60 * metrics["normalized_peak"]
            - 0.40 * metrics["normalized_mean"]
            + 0.10 * metrics["endpoint_advantage"]
        ),
        "collapse": (
            bonus
            + metrics["post_turn_drop"] / metrics["horizon"]
            + metrics["post_turn_slope"]
            + metrics["post_turn_descent_fraction"]
            - metrics["rebound"] / metrics["horizon"]
            - 0.25 * metrics["normalized_final"]
        ),
        "suffix": (
            bonus
            - metrics["normalized_terminal_area"]
            - metrics["rebound"] / metrics["horizon"]
            + 0.25 * metrics["endpoint_advantage"]
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
    late = history[late_start:]
    final = history[-1]
    peak = max(history)
    mean = sum(history) / horizon
    min_late = min(late)
    terminal_peak = max(late)
    collapse = max(0.0, float(terminal_peak - final))
    rebound = max(0.0, float(final - min_late))
    terminal_area = weighted_terminal_area(history, config.suffix_score_fraction)
    turn_depth, turn_projlen, post_drop, post_slope, post_consistency = best_turning_point(
        history,
        config,
    )
    normalized = [float(value) / depth for depth, value in enumerate(history, start=1)]
    best_prefix_depth = min(range(horizon), key=normalized.__getitem__) + 1
    metrics = {
        "horizon": horizon,
        "normalized_final": final / horizon,
        "normalized_peak": peak / horizon,
        "normalized_mean": mean / horizon,
        "normalized_terminal_area": terminal_area / horizon,
        "endpoint_advantage": 2.0 - final / horizon,
        "post_turn_drop": post_drop,
        "post_turn_slope": post_slope,
        "post_turn_descent_fraction": post_consistency,
        "rebound": rebound,
    }
    scores = island_scores(metrics, bool(kernel_depths), config)
    return TrajectoryEvaluation(
        trajectory=trajectory,
        projlen_history=history,
        kernel_depths=tuple(int(value) for value in kernel_depths),
        kernel_matches=tuple(kernel_matches),
        final_projlen=final,
        min_projlen=min(history),
        min_late_projlen=min_late,
        best_prefix_depth=best_prefix_depth,
        peak_projlen=peak,
        mean_projlen=mean,
        normalized_final=metrics["normalized_final"],
        normalized_peak=metrics["normalized_peak"],
        normalized_mean=metrics["normalized_mean"],
        endpoint_advantage=metrics["endpoint_advantage"],
        late_drop=maximum_drawdown(late),
        late_slope=max(0.0, float(late[0] - late[-1]) / max(1, len(late) - 1)),
        sustained_drop_steps=sustained_nonincrease_steps(late),
        terminal_peak_projlen=terminal_peak,
        terminal_collapse=collapse,
        rebound=rebound,
        terminal_slope=downward_regression_slope(late),
        terminal_descent_steps=terminal_descent_steps(
            late,
            config.terminal_rise_tolerance,
        ),
        terminal_weighted_area=terminal_area,
        normalized_terminal_area=metrics["normalized_terminal_area"],
        turning_point_depth=turn_depth,
        turning_point_projlen=turn_projlen,
        post_turn_drop=post_drop,
        post_turn_slope=post_slope,
        post_turn_descent_fraction=post_consistency,
        matrix_fingerprint=matrix_fingerprint,
        island_scores=scores,
        score=scores[trajectory.island],
        objectives=(
            1.0 if kernel_depths else 0.0,
            metrics["endpoint_advantage"],
            -metrics["normalized_peak"],
            post_drop / horizon,
            -metrics["normalized_terminal_area"],
        ),
    )
