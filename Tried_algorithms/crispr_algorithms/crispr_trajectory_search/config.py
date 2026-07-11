from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for one evolutionary trajectory-search run."""

    p: int = 3
    n: int = 4
    horizons: Tuple[int, ...] = (30, 35, 40)
    population_size: int = 5000
    generations: int = 30
    elite_fraction: float = 0.05
    carry_elites_fraction: float = 0.05
    learned_sample_fraction: float = 0.15
    random_sample_fraction: float = 0.05
    two_mutation_fraction: float = 0.10

    targeted_location_fraction: float = 0.50
    random_location_fraction: float = 0.30
    adaptive_location_fraction: float = 0.20
    mutation_block_sizes: Tuple[int, ...] = (1, 3, 5, 8, 12)
    mutation_attempts: int = 20
    location_bins: int = 8

    transition_depth_bins: int = 8
    transition_learning_rate: float = 0.20
    transition_pseudocount: float = 0.25
    transition_exploration: float = 0.05

    late_start_fraction: float = 0.55
    periodic_distance: bool = False
    backend: str = "cpu"
    device: str = "cuda"
    required_cuda_partition: str = "scavenge_gpu"
    eval_batch_size: int = 5000
    seed_trajectory_json: Optional[str] = None
    seed_known_example: Optional[str] = None
    seed_population_fraction: float = 0.0
    seed_corruption_fraction: float = 0.20

    seed: int = 1
    output_dir: str = "results/crispr_trajectory_search"
    stop_at_kernel: bool = True
    max_kernel_hits: int = 20

    def validate(self) -> None:
        if self.p <= 1:
            raise ValueError("p must be at least 2")
        if self.n < 2:
            raise ValueError("n must be at least 2")
        if not self.horizons or any(value <= 0 for value in self.horizons):
            raise ValueError("horizons must contain positive lengths")
        if self.population_size <= 0 or self.generations <= 0:
            raise ValueError("population_size and generations must be positive")
        if not (0.0 < self.elite_fraction <= 1.0):
            raise ValueError("elite_fraction must be in (0, 1]")
        fraction_fields = (
            self.carry_elites_fraction,
            self.learned_sample_fraction,
            self.random_sample_fraction,
            self.two_mutation_fraction,
            self.targeted_location_fraction,
            self.random_location_fraction,
            self.adaptive_location_fraction,
            self.transition_learning_rate,
            self.transition_exploration,
            self.late_start_fraction,
            self.seed_population_fraction,
            self.seed_corruption_fraction,
        )
        if any(value < 0.0 or value > 1.0 for value in fraction_fields):
            raise ValueError("fractional configuration values must be in [0, 1]")
        location_total = (
            self.targeted_location_fraction
            + self.random_location_fraction
            + self.adaptive_location_fraction
        )
        if abs(location_total - 1.0) > 1e-9:
            raise ValueError("mutation location fractions must sum to 1")
        generation_reserved = (
            self.carry_elites_fraction
            + self.learned_sample_fraction
            + self.random_sample_fraction
        )
        if generation_reserved >= 1.0:
            raise ValueError("elite, learned, and random fractions must sum to less than 1")
        if not self.mutation_block_sizes or any(value <= 0 for value in self.mutation_block_sizes):
            raise ValueError("mutation_block_sizes must contain positive values")
        if self.backend not in {"cpu", "torch"}:
            raise ValueError("backend must be 'cpu' or 'torch'")
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        if self.seed_trajectory_json and self.seed_known_example:
            raise ValueError("choose either seed_trajectory_json or seed_known_example")
        if self.seed_known_example == "p5_length54" and self.p != 5:
            raise ValueError("the p5_length54 calibration seed requires p=5")

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    @property
    def elite_count(self) -> int:
        return max(1, round(self.population_size * self.elite_fraction))

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["horizons"] = list(self.horizons)
        payload["mutation_block_sizes"] = list(self.mutation_block_sizes)
        return payload
