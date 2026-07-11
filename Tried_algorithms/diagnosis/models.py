from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KernelCase:
    name: str
    factor_ids: tuple[int, ...]
    source: str = "embedded"


@dataclass(frozen=True)
class AuditConfig:
    p: int = 5
    n: int = 4
    max_depth: int | None = None
    bootstrap_depth: int = 5
    bucket_size: int = 15_000
    use_best: int = 30_000
    baseline_samples: int = 512
    periodic_bucket_size: int = 3_000
    periodic_use_best: int = 50_000
    periodic_elite_fraction: float = 0.35
    periodic_descent_fraction: float = 0.25
    periodic_random_keep_rate: float = 1.0
    mcts_beam_width: int = 64
    breakout_weight: float = 0.5
    breakout_depth_power: float = 1.0
    crispr_sample_size: int = 5_000
    crispr_population_size: int = 50_000
    seed: int = 3
    output_dir: Path = Path("results/diagnosis")
    render_plots: bool = True

    def validate(self) -> None:
        if self.p <= 1:
            raise ValueError("p must exceed 1")
        if self.n < 2:
            raise ValueError("n must be at least 2")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.bootstrap_depth < 0:
            raise ValueError("bootstrap_depth cannot be negative")
        for name, value in (
            ("bucket_size", self.bucket_size),
            ("use_best", self.use_best),
            ("baseline_samples", self.baseline_samples),
            ("periodic_bucket_size", self.periodic_bucket_size),
            ("periodic_use_best", self.periodic_use_best),
            ("mcts_beam_width", self.mcts_beam_width),
            ("crispr_sample_size", self.crispr_sample_size),
            ("crispr_population_size", self.crispr_population_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        fractions = (
            self.periodic_elite_fraction,
            self.periodic_descent_fraction,
            self.periodic_random_keep_rate,
        )
        if any(value < 0.0 or value > 1.0 for value in fractions):
            raise ValueError("periodic retention fractions must lie in [0, 1]")
        if self.periodic_elite_fraction + self.periodic_descent_fraction > 1.0:
            raise ValueError("periodic elite and descent fractions cannot exceed 1")
        if self.breakout_weight < 0:
            raise ValueError("breakout_weight cannot be negative")
        if self.breakout_depth_power <= 0:
            raise ValueError("breakout_depth_power must be positive")


@dataclass
class Candidate:
    factor_ids: tuple[int, ...]
    matrix: Any
    projlen_history: tuple[int, ...]
    periodic_distance_history: tuple[float, ...]
    typical_projlen: float
    surprise: float
    surprise_z: float
    periodic_distance: float
    periodic_score: float
    descent_score: float
    mcts_value: float
    breakout_value: float
    kernel_match: dict

    @property
    def depth(self) -> int:
        return len(self.factor_ids)

    @property
    def projlen(self) -> int:
        return self.projlen_history[-1] if self.projlen_history else 0

    @property
    def last_factor_id(self) -> int | None:
        return self.factor_ids[-1] if self.factor_ids else None
