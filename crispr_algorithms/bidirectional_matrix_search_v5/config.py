from __future__ import annotations

from dataclasses import asdict, dataclass


TARGET_TYPES = ("identity", "delta")


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for bidirectional prefix/suffix matrix matching."""

    p: int = 5
    n: int = 4
    prefix_count: int = 12_000
    suffix_count: int = 60_000
    generations: int = 80
    prefix_length_min: int = 18
    prefix_length_max: int = 48
    suffix_length_min: int = 10
    suffix_length_max: int = 36

    field_points: int = 8
    lsh_tables: int = 16
    lsh_key_components: int = 4
    max_lsh_candidates: int = 1_024
    join_candidates_per_prefix: int = 4
    elite_pairs: int = 1_000
    algebra_elite_fraction: float = 0.50
    length_niche_width: int = 4
    carry_fraction: float = 0.10
    random_fraction: float = 0.15
    refinement_pairs: int = 128
    refinement_trials: int = 16
    mutation_attempts: int = 30
    mutation_block_sizes: tuple[int, ...] = (1, 3, 5, 8, 12, 16)
    length_edit_sizes: tuple[int, ...] = (1, 2, 3, 5)

    signature_batch_size: int = 20_000
    exact_batch_size: int = 10_000
    backend: str = "torch"
    device: str = "cuda"
    required_cuda_partition: str = "scavenge_gpu"

    seed: int = 1
    output_dir: str = "results/bidirectional_v5"
    stop_at_kernel: bool = True
    max_kernel_hits: int = 20
    resume_latest: bool = True

    def validate(self) -> None:
        if self.p <= 2 or self.n != 4:
            raise ValueError("V5 currently requires an odd prime p and n=4")
        if self.prefix_count <= 0 or self.suffix_count <= 0:
            raise ValueError("prefix_count and suffix_count must be positive")
        if self.generations <= 0:
            raise ValueError("generations must be positive")
        if not 1 <= self.prefix_length_min <= self.prefix_length_max:
            raise ValueError("invalid prefix length range")
        if not 1 <= self.suffix_length_min <= self.suffix_length_max:
            raise ValueError("invalid suffix length range")
        if self.field_points <= 0 or self.field_points > self.p * (self.p - 1):
            raise ValueError("field_points exceeds available non-base GF(p^2) points")
        if self.lsh_tables <= 0 or self.lsh_key_components <= 0:
            raise ValueError("LSH settings must be positive")
        if self.join_candidates_per_prefix <= 0 or self.elite_pairs <= 0:
            raise ValueError("join and elite counts must be positive")
        if not 0.0 <= self.algebra_elite_fraction <= 1.0:
            raise ValueError("algebra_elite_fraction must lie in [0, 1]")
        if self.length_niche_width <= 0:
            raise ValueError("length_niche_width must be positive")
        if not 0.0 <= self.carry_fraction < 1.0:
            raise ValueError("carry_fraction must lie in [0, 1)")
        if not 0.0 <= self.random_fraction < 1.0:
            raise ValueError("random_fraction must lie in [0, 1)")
        if self.carry_fraction + self.random_fraction >= 1.0:
            raise ValueError("carry and random fractions leave no mutation budget")
        if self.refinement_pairs < 0 or self.refinement_trials < 0:
            raise ValueError("refinement settings cannot be negative")
        if self.backend not in {"cpu", "torch"}:
            raise ValueError("backend must be cpu or torch")
        if any(value <= 0 for value in self.mutation_block_sizes + self.length_edit_sizes):
            raise ValueError("mutation sizes must be positive")

    def to_dict(self) -> dict:
        return asdict(self)
