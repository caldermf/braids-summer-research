from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class ReservoirConfig:
    n: int = 4
    r: int = 1
    p: int = 5
    bootstrap_depth: int = 5
    target_depth: int = 60
    step_size: int = 1
    bucket_size: int = 15_000
    use_best: int = 30_000
    seed: int = 3
    stop_at_author_projlen_one: bool = True


@dataclass(frozen=True)
class CrisprConfig:
    pool_size: int = 30_000
    population_per_island: int = 7_500
    generations: int = 60
    offspring_multiplier: int = 3
    elite_fraction: float = 0.20
    parent_fraction: float = 0.35
    append_fraction: float = 0.45
    boundary_fraction: float = 0.30
    boundary_margin: int = 1
    backend: str = "cpu"
    device: str = "cpu"
    seed: int = 104


@dataclass(frozen=True)
class HybridConfig:
    output_dir: str = "results/hybrid_crispr_reservoir_p5"
    crispr_max_depth: int = 80
    stop_at_kernel: bool = True
    reservoir: ReservoirConfig = ReservoirConfig()
    crispr: CrisprConfig = CrisprConfig()

    @property
    def author_repo(self) -> Path:
        package_dir = Path(__file__).resolve().parent
        candidates = (
            package_dir / "third_party" / "braids_project",
            package_dir.parent
            / "hybrid_of_reservoir_crispr_mcts_suffix"
            / "third_party"
            / "braids_project",
            package_dir.parent.parent / "braids-project",
        )
        for candidate in candidates:
            if (candidate / "peyl" / "braidsearch.py").is_file():
                return candidate
        return candidates[0]

    def to_dict(self) -> dict:
        return asdict(self)


def profile_config(name: str, output_dir: str | None = None) -> HybridConfig:
    base = HybridConfig()
    if name == "cluster":
        config = base
    elif name == "laptop":
        config = replace(
            base,
            reservoir=replace(
                base.reservoir,
                target_depth=45,
                bucket_size=3_000,
                use_best=10_000,
            ),
            crispr_max_depth=55,
            crispr=replace(
                base.crispr,
                pool_size=2_000,
                population_per_island=500,
                generations=20,
                offspring_multiplier=2,
            ),
        )
    elif name == "smoke":
        config = replace(
            base,
            reservoir=replace(
                base.reservoir,
                bootstrap_depth=2,
                target_depth=3,
                bucket_size=50,
                use_best=100,
            ),
            crispr_max_depth=5,
            crispr=replace(
                base.crispr,
                pool_size=24,
                population_per_island=6,
                generations=2,
                offspring_multiplier=1,
                boundary_fraction=0.25,
            ),
        )
    else:
        raise ValueError(f"unknown profile {name!r}")
    return replace(config, output_dir=output_dir or config.output_dir)
