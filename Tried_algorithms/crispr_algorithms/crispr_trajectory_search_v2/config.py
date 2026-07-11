from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for one quality-diversity trajectory-search run."""

    p: int = 3
    n: int = 4
    horizons: Tuple[int, ...] = (30, 35, 40)
    population_size: int = 5000
    generations: int = 30
    elite_fraction: float = 0.05
    archive_fraction: float = 0.10
    local_mutation_fraction: float = 0.50
    escape_mutation_fraction: float = 0.20
    crossover_fraction: float = 0.10
    random_sample_fraction: float = 0.10

    local_mutation_block_sizes: Tuple[int, ...] = (1, 3)
    escape_mutation_block_sizes: Tuple[int, ...] = (5, 8, 12, 16)
    mutation_attempts: int = 20
    location_bins: int = 8
    mutation_statistics_decay: float = 0.95
    mutation_success_bonus: float = 1.0
    mutation_archive_bonus: float = 2.0
    terminal_location_bias: float = 3.0

    archive_size: int = 5000
    archive_projlen_bin_size: int = 5
    archive_collapse_bin_size: int = 3
    archive_rebound_bin_size: int = 3
    archive_transition_niches: int = 32
    archive_suffix_length: int = 8

    stagnation_generations: int = 10
    stagnation_escape_boost: float = 0.15
    stagnation_random_boost: float = 0.05
    terminal_rise_tolerance: float = 0.0

    transition_depth_bins: int = 8
    transition_learning_rate: float = 0.20
    transition_pseudocount: float = 0.25
    transition_exploration: float = 0.05

    late_start_fraction: float = 0.55
    final_advantage_weight: float = 6.0
    terminal_collapse_weight: float = 8.0
    terminal_slope_weight: float = 20.0
    terminal_descent_weight: float = 0.5
    rebound_penalty_weight: float = 8.0
    periodic_distance_weight: float = 1.0
    kernel_bonus: float = 1_000_000.0
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
    output_dir: str = "results/crispr_trajectory_search_v2"
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
            self.archive_fraction,
            self.local_mutation_fraction,
            self.escape_mutation_fraction,
            self.crossover_fraction,
            self.random_sample_fraction,
            self.transition_learning_rate,
            self.transition_exploration,
            self.late_start_fraction,
            self.seed_population_fraction,
            self.seed_corruption_fraction,
        )
        if any(value < 0.0 or value > 1.0 for value in fraction_fields):
            raise ValueError("fractional configuration values must be in [0, 1]")
        generation_total = (
            self.archive_fraction
            + self.local_mutation_fraction
            + self.escape_mutation_fraction
            + self.crossover_fraction
            + self.random_sample_fraction
        )
        if abs(generation_total - 1.0) > 1e-9:
            raise ValueError("generation fractions must sum to 1")
        for label, values in (
            ("local_mutation_block_sizes", self.local_mutation_block_sizes),
            ("escape_mutation_block_sizes", self.escape_mutation_block_sizes),
        ):
            if not values or any(value <= 0 for value in values):
                raise ValueError(f"{label} must contain positive values")
        if self.archive_size <= 0 or self.archive_transition_niches <= 0:
            raise ValueError("archive sizes must be positive")
        if min(
            self.archive_projlen_bin_size,
            self.archive_collapse_bin_size,
            self.archive_rebound_bin_size,
            self.archive_suffix_length,
            self.stagnation_generations,
        ) <= 0:
            raise ValueError("archive bins, suffix length, and stagnation threshold must be positive")
        if not 0.0 < self.mutation_statistics_decay <= 1.0:
            raise ValueError("mutation_statistics_decay must be in (0, 1]")
        if self.stagnation_escape_boost + self.stagnation_random_boost > self.local_mutation_fraction:
            raise ValueError("stagnation boosts cannot exceed the local mutation fraction")
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
        payload["local_mutation_block_sizes"] = list(self.local_mutation_block_sizes)
        payload["escape_mutation_block_sizes"] = list(self.escape_mutation_block_sizes)
        return payload
