from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median


@dataclass(frozen=True)
class DownturnConfig:
    """Conservative detector for the sustained terminal projlen descent."""

    min_depth: int = 20
    smoothing_window: int = 3
    trend_window: int = 8
    min_drop: float = 4.0
    max_slope: float = -0.35
    min_negative_fraction: float = 0.50
    confirmation_steps: int = 2
    extra_depths: int = 4

    def validate(self) -> None:
        if self.min_depth < 1:
            raise ValueError("min_depth must be positive")
        if self.smoothing_window < 1 or self.trend_window < 2:
            raise ValueError("downturn windows are too short")
        if self.min_drop <= 0 or self.max_slope >= 0:
            raise ValueError("downturn drop must be positive and slope negative")
        if not 0.0 <= self.min_negative_fraction <= 1.0:
            raise ValueError("min_negative_fraction must lie in [0, 1]")
        if self.confirmation_steps < 1 or self.extra_depths < 0:
            raise ValueError("confirmation_steps and extra_depths are invalid")


def _linear_slope(values: list[float]) -> float:
    center = (len(values) - 1) / 2.0
    denominator = sum((index - center) ** 2 for index in range(len(values)))
    return sum(
        (index - center) * (value - sum(values) / len(values))
        for index, value in enumerate(values)
    ) / denominator


class DownturnMonitor:
    """Track a noisy projlen trajectory and decide when to hand off."""

    def __init__(self, config: DownturnConfig):
        config.validate()
        self.config = config
        self.depths: list[int] = []
        self.values: list[float] = []
        self.smoothed: list[float] = []
        self.candidate_streak = 0
        self.confirmed_depth: int | None = None
        self.handoff_depth: int | None = None

    def observe(self, depth: int, projlen: int) -> dict:
        if self.depths and depth <= self.depths[-1]:
            raise ValueError("downturn observations must have increasing depths")
        self.depths.append(int(depth))
        self.values.append(float(projlen))
        width = min(self.config.smoothing_window, len(self.values))
        self.smoothed.append(float(median(self.values[-width:])))

        trend = self.smoothed[-self.config.trend_window :]
        peak = max(self.smoothed)
        drop = peak - self.smoothed[-1]
        slope = _linear_slope(trend) if len(trend) == self.config.trend_window else 0.0
        deltas = [right - left for left, right in zip(trend, trend[1:])]
        negative_fraction = (
            sum(delta < 0 for delta in deltas) / len(deltas) if deltas else 0.0
        )
        candidate = (
            self.confirmed_depth is None
            and depth >= self.config.min_depth
            and len(trend) == self.config.trend_window
            and drop >= self.config.min_drop
            and slope <= self.config.max_slope
            and negative_fraction >= self.config.min_negative_fraction
        )
        if self.confirmed_depth is None:
            self.candidate_streak = self.candidate_streak + 1 if candidate else 0
            if self.candidate_streak >= self.config.confirmation_steps:
                self.confirmed_depth = int(depth)
                self.handoff_depth = int(depth + self.config.extra_depths)

        return {
            "raw_projlen": int(projlen),
            "smoothed_projlen": self.smoothed[-1],
            "historical_smoothed_peak": peak,
            "drop_from_peak": drop,
            "trend_slope": slope,
            "negative_step_fraction": negative_fraction,
            "downturn_candidate": candidate,
            "candidate_streak": self.candidate_streak,
            "confirmed_depth": self.confirmed_depth,
            "planned_handoff_depth": self.handoff_depth,
            "should_handoff": self.handoff_depth is not None and depth >= self.handoff_depth,
        }

    def metadata(self) -> dict:
        return {
            "config": asdict(self.config),
            "confirmed_depth": self.confirmed_depth,
            "planned_handoff_depth": self.handoff_depth,
        }
