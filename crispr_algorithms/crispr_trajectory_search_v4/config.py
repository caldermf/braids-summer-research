from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


ISLAND_NAMES = ("endpoint", "envelope", "collapse", "suffix")


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for variable-length V4 island evolution and MCTS."""

    p: int = 5
    n: int = 4
    min_horizon: int = 36
    initial_max_horizon: int = 72
    hard_max_horizon: int = 120
    horizon_expand_step: int = 8
    horizon_expand_interval: int = 5
    horizon_boundary_margin: int = 4
    horizon_boundary_elite_fraction: float = 0.15
    length_niche_width: int = 4

    population_size: int = 50_000
    min_generations: int = 60
    max_generations: int = 200
    global_stagnation_generations: int = 20
    island_fractions: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    elite_fraction: float = 0.05
    carry_fraction: float = 0.08
    random_fraction: float = 0.12
    crossover_fraction: float = 0.08

    offspring_per_parent: int = 6
    offspring_survivors_per_parent: int = 1
    mutation_attempts: int = 30
    endpoint_block_sizes: Tuple[int, ...] = (1, 3, 5, 8, 12, 16)
    envelope_block_sizes: Tuple[int, ...] = (3, 5, 8, 12, 16, 24)
    collapse_block_sizes: Tuple[int, ...] = (5, 8, 12, 16, 24, 32)
    suffix_block_sizes: Tuple[int, ...] = (8, 12, 16, 24, 32, 40)
    stagnation_block_sizes: Tuple[int, ...] = (24, 32, 40, 48)
    length_edit_sizes: Tuple[int, ...] = (1, 2, 3, 5, 8)
    structural_mutation_fraction: float = 0.30
    post_turn_rewrite_fraction: float = 0.25
    terminal_location_bias: float = 4.0
    mutation_statistics_decay: float = 0.95
    transition_depth_bins: int = 8
    transition_learning_rate: float = 0.20
    transition_pseudocount: float = 0.25
    transition_exploration: float = 0.08

    migration_interval: int = 10
    migration_fraction: float = 0.005
    migration_protection_generations: int = 1
    stagnation_generations: int = 10
    stagnation_min_improvement: float = 0.005
    restart_preserve_fraction: float = 0.05
    restart_large_mutation_fraction: float = 0.45
    restart_random_fraction: float = 0.30

    late_start_fraction: float = 0.55
    suffix_score_fraction: float = 0.45
    turn_min_fraction: float = 0.30
    turn_max_fraction: float = 0.85
    terminal_rise_tolerance: float = 0.0
    evaluation_cache_size: int = 250_000
    novelty_archive_size: int = 250_000
    finishing_queue_size_per_island: int = 512
    kernel_bonus: float = 1_000_000.0

    mcts_enabled: bool = True
    mcts_interval: int = 5
    mcts_top_fraction: float = 0.25
    mcts_seed_count: int = 96
    mcts_simulations_per_seed: int = 64
    mcts_max_depth: int = 12
    mcts_branching_factor: int = 4
    mcts_exploration: float = 1.4
    mcts_block_sizes: Tuple[int, ...] = (8, 12, 16, 24, 32, 40)
    mcts_length_edit_fraction: float = 0.25

    backend: str = "cpu"
    device: str = "cuda"
    required_cuda_partition: str = "scavenge_gpu"
    eval_batch_size: int = 10_000
    seed_trajectory_json: Optional[str] = None
    seed_known_example: Optional[str] = None
    seed_population_fraction: float = 0.0
    seed_corruption_fraction: float = 0.20

    seed: int = 1
    output_dir: str = "results/crispr_trajectory_search_v4"
    stop_at_kernel: bool = True
    max_kernel_hits: int = 20

    def validate(self) -> None:
        if self.p <= 1 or self.n < 2:
            raise ValueError("p must exceed 1 and n must be at least 2")
        if not 1 <= self.min_horizon <= self.initial_max_horizon <= self.hard_max_horizon:
            raise ValueError("horizon bounds must satisfy min <= initial max <= hard max")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if not 1 <= self.min_generations <= self.max_generations:
            raise ValueError("generation bounds must satisfy 1 <= min <= max")
        if len(self.island_fractions) != len(ISLAND_NAMES):
            raise ValueError("island_fractions must have one value per island")
        if abs(sum(self.island_fractions) - 1.0) > 1e-9:
            raise ValueError("island_fractions must sum to 1")
        fractions = (
            *self.island_fractions,
            self.horizon_boundary_elite_fraction,
            self.elite_fraction,
            self.carry_fraction,
            self.random_fraction,
            self.crossover_fraction,
            self.structural_mutation_fraction,
            self.post_turn_rewrite_fraction,
            self.migration_fraction,
            self.restart_preserve_fraction,
            self.restart_large_mutation_fraction,
            self.restart_random_fraction,
            self.late_start_fraction,
            self.suffix_score_fraction,
            self.turn_min_fraction,
            self.turn_max_fraction,
            self.mcts_top_fraction,
            self.mcts_length_edit_fraction,
            self.seed_population_fraction,
            self.seed_corruption_fraction,
            self.transition_learning_rate,
            self.transition_exploration,
        )
        if any(value < 0.0 or value > 1.0 for value in fractions):
            raise ValueError("fractional values must be in [0, 1]")
        if self.turn_min_fraction >= self.turn_max_fraction:
            raise ValueError("turn_min_fraction must be below turn_max_fraction")
        if self.carry_fraction + self.random_fraction + self.crossover_fraction >= 1.0:
            raise ValueError("generation fractions leave no mutation budget")
        if (
            self.restart_preserve_fraction
            + self.restart_large_mutation_fraction
            + self.restart_random_fraction
            > 1.0
        ):
            raise ValueError("restart fractions cannot exceed 1")
        integer_fields = (
            self.horizon_expand_step,
            self.horizon_expand_interval,
            self.horizon_boundary_margin,
            self.length_niche_width,
            self.global_stagnation_generations,
            self.offspring_per_parent,
            self.offspring_survivors_per_parent,
            self.mutation_attempts,
            self.transition_depth_bins,
            self.migration_interval,
            self.stagnation_generations,
            self.evaluation_cache_size,
            self.novelty_archive_size,
            self.finishing_queue_size_per_island,
            self.mcts_interval,
            self.mcts_seed_count,
            self.mcts_simulations_per_seed,
            self.mcts_max_depth,
            self.mcts_branching_factor,
            self.eval_batch_size,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("count and interval settings must be positive")
        if self.offspring_survivors_per_parent > self.offspring_per_parent:
            raise ValueError("cannot retain more offspring than are produced")
        for label, values in (
            ("endpoint_block_sizes", self.endpoint_block_sizes),
            ("envelope_block_sizes", self.envelope_block_sizes),
            ("collapse_block_sizes", self.collapse_block_sizes),
            ("suffix_block_sizes", self.suffix_block_sizes),
            ("stagnation_block_sizes", self.stagnation_block_sizes),
            ("length_edit_sizes", self.length_edit_sizes),
            ("mcts_block_sizes", self.mcts_block_sizes),
        ):
            if not values or any(value <= 0 for value in values):
                raise ValueError(f"{label} must contain positive lengths")
        if self.backend not in {"cpu", "torch"}:
            raise ValueError("backend must be 'cpu' or 'torch'")
        if self.seed_trajectory_json and self.seed_known_example:
            raise ValueError("choose either seed_trajectory_json or seed_known_example")
        if self.seed_known_example == "p5_length54" and self.p != 5:
            raise ValueError("the p5_length54 calibration seed requires p=5")

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
            "envelope": self.envelope_block_sizes,
            "collapse": self.collapse_block_sizes,
            "suffix": self.suffix_block_sizes,
        }[island]
        return tuple(dict.fromkeys(base + self.stagnation_block_sizes)) if stagnant else base

    def to_json(self) -> dict:
        payload = asdict(self)
        for key, value in tuple(payload.items()):
            if isinstance(value, tuple):
                payload[key] = list(value)
        payload["island_sizes"] = self.island_sizes
        return payload
