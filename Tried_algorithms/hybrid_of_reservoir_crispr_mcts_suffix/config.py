from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class BackboneConfig:
    n: int = 4
    r: int = 1
    p: int = 5
    bootstrap_depth: int = 5
    target_depth: int = 35
    step_size: int = 1
    bucket_size: int = 15_000
    use_best: int = 30_000
    seed: int = 3


@dataclass(frozen=True)
class CrisprConfig:
    pool_size: int = 30_000
    population_per_island: int = 7_500
    generations: int = 40
    offspring_multiplier: int = 3
    elite_fraction: float = 0.20
    parent_fraction: float = 0.35
    append_fraction: float = 0.35
    backend: str = "cpu"
    device: str = "cpu"
    seed: int = 104


@dataclass(frozen=True)
class ReservoirMCTSConfig:
    pool_size: int = 30_000
    database_limit: int = 100_000
    iterations: int = 80
    selected_nodes_per_iteration: int = 8
    children_per_node: int = 0
    playout_bucket_size: int = 1_500
    playout_use_best: int = 6_000
    exploration_floor: float = 0.08
    seed: int = 205


@dataclass(frozen=True)
class SuffixLookupConfig:
    prefix_pool_size: int = 30_000
    suffixes_per_length: int = 50_000
    field_points: int = 8
    lsh_tables: int = 16
    lsh_key_components: int = 4
    max_lsh_candidates: int = 2_048
    joins_per_prefix: int = 6
    exact_candidates_per_depth: int = 25_000
    backend: str = "cpu"
    device: str = "cpu"
    seed: int = 306


@dataclass(frozen=True)
class HybridConfig:
    output_dir: str = "results/hybrid_reservoir_crispr_mcts_suffix"
    max_depth: int = 45
    stop_at_kernel: bool = True
    backbone: BackboneConfig = BackboneConfig()
    crispr: CrisprConfig = CrisprConfig()
    reservoir_mcts: ReservoirMCTSConfig = ReservoirMCTSConfig()
    suffix_lookup: SuffixLookupConfig = SuffixLookupConfig()

    @property
    def author_repo(self) -> Path:
        return (
            Path(__file__).resolve().parent
            / "third_party"
            / "braids_project"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def profile_config(name: str, output_dir: str | None = None) -> HybridConfig:
    base = HybridConfig()
    if name == "cluster":
        config = base
    elif name == "laptop":
        config = replace(
            base,
            crispr=replace(
                base.crispr,
                pool_size=2_000,
                population_per_island=600,
                generations=20,
                offspring_multiplier=2,
            ),
            reservoir_mcts=replace(
                base.reservoir_mcts,
                pool_size=1_500,
                database_limit=20_000,
                iterations=30,
                selected_nodes_per_iteration=3,
                playout_bucket_size=300,
                playout_use_best=1_200,
            ),
            suffix_lookup=replace(
                base.suffix_lookup,
                prefix_pool_size=2_000,
                suffixes_per_length=8_000,
                exact_candidates_per_depth=4_000,
            ),
        )
    elif name == "smoke":
        config = replace(
            base,
            max_depth=5,
            backbone=replace(
                base.backbone,
                bootstrap_depth=2,
                target_depth=3,
                bucket_size=50,
                use_best=100,
            ),
            crispr=replace(
                base.crispr,
                pool_size=24,
                population_per_island=8,
                generations=2,
                offspring_multiplier=1,
            ),
            reservoir_mcts=replace(
                base.reservoir_mcts,
                pool_size=20,
                database_limit=200,
                iterations=2,
                selected_nodes_per_iteration=1,
                children_per_node=2,
                playout_bucket_size=10,
                playout_use_best=30,
            ),
            suffix_lookup=replace(
                base.suffix_lookup,
                prefix_pool_size=20,
                suffixes_per_length=40,
                field_points=4,
                lsh_tables=4,
                max_lsh_candidates=40,
                joins_per_prefix=2,
                exact_candidates_per_depth=40,
            ),
        )
    else:
        raise ValueError(f"unknown profile {name!r}")
    return replace(config, output_dir=output_dir or config.output_dir)
