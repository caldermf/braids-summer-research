from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class MutationRecord:
    mode: str
    start: int
    block_length: int
    location_bin: int


@dataclass
class Trajectory:
    factor_ids: Tuple[int, ...]
    origin: str = "random"
    parent_id: Optional[str] = None
    parent_score: Optional[float] = None
    mutation_records: Tuple[MutationRecord, ...] = ()
    trajectory_id: str = ""

    @property
    def horizon(self) -> int:
        return len(self.factor_ids)


@dataclass
class TrajectoryEvaluation:
    trajectory: Trajectory
    projlen_history: Tuple[int, ...]
    periodic_distance_history: Tuple[float, ...] = ()
    kernel_depths: Tuple[int, ...] = ()
    kernel_matches: Tuple[Dict[str, Any], ...] = ()

    final_projlen: int = 0
    min_projlen: int = 0
    min_late_projlen: int = 0
    best_prefix_depth: int = 0
    late_drop: float = 0.0
    late_slope: float = 0.0
    sustained_drop_steps: int = 0
    final_periodic_distance: Optional[float] = None
    min_periodic_distance: Optional[float] = None
    score: float = 0.0
    objectives: Tuple[float, ...] = ()

    @property
    def has_kernel(self) -> bool:
        return bool(self.kernel_depths)

    def summary(self) -> dict:
        return {
            "trajectory_id": self.trajectory.trajectory_id,
            "origin": self.trajectory.origin,
            "horizon": self.trajectory.horizon,
            "score": self.score,
            "final_projlen": self.final_projlen,
            "min_projlen": self.min_projlen,
            "min_late_projlen": self.min_late_projlen,
            "best_prefix_depth": self.best_prefix_depth,
            "late_drop": self.late_drop,
            "late_slope": self.late_slope,
            "sustained_drop_steps": self.sustained_drop_steps,
            "final_periodic_distance": self.final_periodic_distance,
            "min_periodic_distance": self.min_periodic_distance,
            "kernel_depths": list(self.kernel_depths),
            "factor_ids": list(self.trajectory.factor_ids),
        }


@dataclass
class GenerationResult:
    generation: int
    evaluations: list[TrajectoryEvaluation]
    elites: list[TrajectoryEvaluation] = field(default_factory=list)
    kernel_hits: list[TrajectoryEvaluation] = field(default_factory=list)
