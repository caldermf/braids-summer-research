from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


ISLAND_NAMES = ("endpoint", "collapse", "suffix")


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for the V3 island evolution plus MCTS finisher."""

    p: int = 5
    n: int = 4
    horizons: Tuple[int, ...] = (54,)
    population_size: int = 50_000
    generations: int = 60
    island_fractions: Tuple[float, ...] = (0.40, 0.30, 0.30)
    elite_fraction: float = 0.05
    carry_fraction: float = 0.10
    random_fraction: float = 0.10
    crossover_fraction: float = 0.08

    offspring_per_parent: int = 4
    offspring_survivors_per_parent: int = 1
    mutation_attempts: int = 24
    endpoint_block_sizes: Tuple[int, ...] = (1, 3, 5, 8, 12)
    collapse_block_sizes: Tuple[int, ...] = (5, 8, 12, 16, 20)
    suffix_block_sizes: Tuple[int, ...] = (8, 12, 16, 20, 24)
    stagnation_block_sizes: Tuple[int, ...] = (16, 20, 24, 32)
    terminal_location_bias: float = 5.0
    mutation_statistics_decay: float = 0.95
    transition_depth_bins: int = 8
    transition_learning_rate: float = 0.20
    transition_pseudocount: float = 0.25
    transition_exploration: float = 0.05

    migration_interval: int = 5
    migration_fraction: float = 0.03
    migration_protection_generations: int = 1
    stagnation_generations: int = 10
    stagnation_min_improvement: float = 1.0
    restart_preserve_fraction: float = 0.08
    restart_large_mutation_fraction: float = 0.42
    restart_random_fraction: float = 0.20

    late_start_fraction: float = 0.55
    suffix_score_fraction: float = 0.45
    terminal_rise_tolerance: float = 0.0
    evaluation_cache_size: int = 250_000
    novelty_archive_size: int = 250_000
    finishing_queue_size: int = 2_048
    finishing_projlen_threshold: int = 24
    kernel_bonus: float = 1_000_000.0

    mcts_enabled: bool = True
    mcts_interval: int = 5
    mcts_top_fraction: float = 0.25
    mcts_seed_count: int = 96
    mcts_simulations_per_seed: int = 64
    mcts_max_depth: int = 10
    mcts_branching_factor: int = 4
    mcts_exploration: float = 1.4
    mcts_block_sizes: Tuple[int, ...] = (5, 8, 12, 16, 20, 24)

    backend: str = "cpu"
    device: str = "cuda"
    required_cuda_partition: str = "scavenge_gpu"
    eval_batch_size: int = 10_000
    seed_trajectory_json: Optional[str] = None
    seed_known_example: Optional[str] = None
    seed_population_fraction: float = 0.0
    seed_corruption_fraction: float = 0.20

    seed: int = 1
    output_dir: str = "results/crispr_trajectory_search_v3"
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
        if len(self.island_fractions) != len(ISLAND_NAMES):
            raise ValueError("island_fractions must have one value per island")
        if abs(sum(self.island_fractions) - 1.0) > 1e-9:
            raise ValueError("island_fractions must sum to 1")
        fractions = (
            *self.island_fractions,
            self.elite_fraction,
            self.carry_fraction,
            self.random_fraction,
            self.crossover_fraction,
            self.migration_fraction,
            self.restart_preserve_fraction,
            self.restart_large_mutation_fraction,
            self.restart_random_fraction,
            self.late_start_fraction,
            self.suffix_score_fraction,
            self.mcts_top_fraction,
            self.seed_population_fraction,
            self.seed_corruption_fraction,
            self.transition_learning_rate,
            self.transition_exploration,
        )
        if any(value < 0.0 or value > 1.0 for value in fractions):
            raise ValueError("fractional values must be in [0, 1]")
        if self.carry_fraction + self.random_fraction + self.crossover_fraction >= 1.0:
            raise ValueError("carry, random, and crossover fractions leave no mutation budget")
        if (
            self.restart_preserve_fraction
            + self.restart_large_mutation_fraction
            + self.restart_random_fraction
            > 1.0
        ):
            raise ValueError("restart fractions cannot exceed 1")
        if self.offspring_per_parent <= 0 or self.offspring_survivors_per_parent <= 0:
            raise ValueError("offspring counts must be positive")
        if self.offspring_survivors_per_parent > self.offspring_per_parent:
            raise ValueError("cannot retain more offspring than are produced")
        integer_fields = (
            self.mutation_attempts,
            self.transition_depth_bins,
            self.migration_interval,
            self.stagnation_generations,
            self.finishing_queue_size,
            self.evaluation_cache_size,
            self.novelty_archive_size,
            self.mcts_interval,
            self.mcts_seed_count,
            self.mcts_simulations_per_seed,
            self.mcts_max_depth,
            self.mcts_branching_factor,
            self.eval_batch_size,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("count and interval settings must be positive")
        for label, values in (
            ("endpoint_block_sizes", self.endpoint_block_sizes),
            ("collapse_block_sizes", self.collapse_block_sizes),
            ("suffix_block_sizes", self.suffix_block_sizes),
            ("stagnation_block_sizes", self.stagnation_block_sizes),
            ("mcts_block_sizes", self.mcts_block_sizes),
        ):
            if not values or any(value <= 0 for value in values):
                raise ValueError(f"{label} must contain positive lengths")
        if not 0.0 < self.mutation_statistics_decay <= 1.0:
            raise ValueError("mutation_statistics_decay must be in (0, 1]")
        if self.backend not in {"cpu", "torch"}:
            raise ValueError("backend must be 'cpu' or 'torch'")
        if self.seed_trajectory_json and self.seed_known_example:
            raise ValueError("choose either seed_trajectory_json or seed_known_example")
        if self.seed_known_example == "p5_length54" and self.p != 5:
            raise ValueError("the p5_length54 calibration seed requires p=5")

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    @property
    def island_sizes(self) -> dict[str, int]:
        sizes = {
            name: round(self.population_size * fraction)
            for name, fraction in zip(ISLAND_NAMES[:-1], self.island_fractions[:-1])
        }
        sizes[ISLAND_NAMES[-1]] = self.population_size - sum(sizes.values())
        return sizes

    def block_sizes_for(self, island: str, stagnant: bool = False) -> Tuple[int, ...]:
        base = {
            "endpoint": self.endpoint_block_sizes,
            "collapse": self.collapse_block_sizes,
            "suffix": self.suffix_block_sizes,
        }[island]
        if not stagnant:
            return base
        return tuple(dict.fromkeys(base + self.stagnation_block_sizes))

    def to_json(self) -> dict:
        payload = asdict(self)
        for key, value in tuple(payload.items()):
            if isinstance(value, tuple):
                payload[key] = list(value)
        payload["island_sizes"] = self.island_sizes
        return payload
