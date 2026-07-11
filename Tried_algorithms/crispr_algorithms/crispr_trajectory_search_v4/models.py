from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class MutationRecord:
    island: str
    action: str
    mode: str
    start: int
    removed_length: int
    inserted_length: int
    location_bin: int

    @property
    def block_length(self) -> int:
        return max(self.removed_length, self.inserted_length)


@dataclass
class Trajectory:
    factor_ids: Tuple[int, ...]
    island: str = "endpoint"
    origin: str = "random"
    parent_id: Optional[str] = None
    parent_score: Optional[float] = None
    mutation_records: Tuple[MutationRecord, ...] = ()
    trajectory_id: str = ""
    protected_until_generation: int = -1

    @property
    def horizon(self) -> int:
        return len(self.factor_ids)


@dataclass
class TrajectoryEvaluation:
    trajectory: Trajectory
    projlen_history: Tuple[int, ...]
    kernel_depths: Tuple[int, ...] = ()
    kernel_matches: Tuple[Dict[str, Any], ...] = ()

    final_projlen: int = 0
    min_projlen: int = 0
    min_late_projlen: int = 0
    best_prefix_depth: int = 0
    peak_projlen: int = 0
    mean_projlen: float = 0.0
    normalized_final: float = 0.0
    normalized_peak: float = 0.0
    normalized_mean: float = 0.0
    endpoint_advantage: float = 0.0
    late_drop: float = 0.0
    late_slope: float = 0.0
    sustained_drop_steps: int = 0
    terminal_peak_projlen: int = 0
    terminal_collapse: float = 0.0
    rebound: float = 0.0
    terminal_slope: float = 0.0
    terminal_descent_steps: int = 0
    terminal_weighted_area: float = 0.0
    normalized_terminal_area: float = 0.0
    turning_point_depth: int = 0
    turning_point_projlen: int = 0
    post_turn_drop: float = 0.0
    post_turn_slope: float = 0.0
    post_turn_descent_fraction: float = 0.0
    matrix_fingerprint: str = ""
    novelty: float = 0.0
    island_scores: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    objectives: Tuple[float, ...] = ()

    @property
    def has_kernel(self) -> bool:
        return bool(self.kernel_depths)

    def score_for(self, island: Optional[str] = None) -> float:
        return self.island_scores[island or self.trajectory.island]

    def summary(self) -> dict:
        return {
            "trajectory_id": self.trajectory.trajectory_id,
            "island": self.trajectory.island,
            "origin": self.trajectory.origin,
            "parent_id": self.trajectory.parent_id,
            "horizon": self.trajectory.horizon,
            "score": self.score,
            "island_scores": dict(self.island_scores),
            "final_projlen": self.final_projlen,
            "min_projlen": self.min_projlen,
            "min_late_projlen": self.min_late_projlen,
            "best_prefix_depth": self.best_prefix_depth,
            "peak_projlen": self.peak_projlen,
            "mean_projlen": self.mean_projlen,
            "normalized_final": self.normalized_final,
            "normalized_peak": self.normalized_peak,
            "normalized_mean": self.normalized_mean,
            "endpoint_advantage": self.endpoint_advantage,
            "late_drop": self.late_drop,
            "terminal_collapse": self.terminal_collapse,
            "rebound": self.rebound,
            "terminal_slope": self.terminal_slope,
            "terminal_descent_steps": self.terminal_descent_steps,
            "terminal_weighted_area": self.terminal_weighted_area,
            "normalized_terminal_area": self.normalized_terminal_area,
            "turning_point_depth": self.turning_point_depth,
            "turning_point_projlen": self.turning_point_projlen,
            "post_turn_drop": self.post_turn_drop,
            "post_turn_slope": self.post_turn_slope,
            "post_turn_descent_fraction": self.post_turn_descent_fraction,
            "matrix_fingerprint": self.matrix_fingerprint,
            "novelty": self.novelty,
            "kernel_depths": list(self.kernel_depths),
            "factor_ids": list(self.trajectory.factor_ids),
        }
