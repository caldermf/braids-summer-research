from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class MutationRecord:
    island: str
    mode: str
    start: int
    block_length: int
    location_bin: int


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
    late_drop: float = 0.0
    late_slope: float = 0.0
    sustained_drop_steps: int = 0
    terminal_peak_projlen: int = 0
    terminal_collapse: float = 0.0
    rebound: float = 0.0
    terminal_slope: float = 0.0
    terminal_descent_steps: int = 0
    terminal_weighted_area: float = 0.0
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
            "late_drop": self.late_drop,
            "late_slope": self.late_slope,
            "sustained_drop_steps": self.sustained_drop_steps,
            "terminal_peak_projlen": self.terminal_peak_projlen,
            "terminal_collapse": self.terminal_collapse,
            "rebound": self.rebound,
            "terminal_slope": self.terminal_slope,
            "terminal_descent_steps": self.terminal_descent_steps,
            "terminal_weighted_area": self.terminal_weighted_area,
            "matrix_fingerprint": self.matrix_fingerprint,
            "novelty": self.novelty,
            "kernel_depths": list(self.kernel_depths),
            "factor_ids": list(self.trajectory.factor_ids),
        }


@dataclass
class GenerationResult:
    generation: int
    evaluations: list[TrajectoryEvaluation]
    island_elites: Dict[str, list[TrajectoryEvaluation]] = field(default_factory=dict)
    kernel_hits: list[TrajectoryEvaluation] = field(default_factory=list)
