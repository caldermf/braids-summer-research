from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .checkpoint import (
    candidate_summary,
    load_checkpoint,
    verify_author_kernel_candidates,
)
from .config import HybridConfig, profile_config
from .crispr_repair import run_crispr_repair
from .io_utils import write_json
from .reservoir import run_author_reservoir


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=("smoke", "laptop", "cluster"),
        default="laptop",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--author-repo")
    parser.add_argument(
        "--author-python",
        help="Python >=3.10 with NumPy and pandas for the paper Tracker worker",
    )
    parser.add_argument("--p", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--r", type=int)
    parser.add_argument("--reservoir-depth", type=int)
    parser.add_argument("--crispr-max-depth", type=int)
    parser.add_argument("--bucket-size", type=int)
    parser.add_argument("--use-best", type=int)
    parser.add_argument("--reservoir-seed", type=int)
    parser.add_argument("--crispr-seed", type=int)
    parser.add_argument("--crispr-pool-size", type=int)
    parser.add_argument("--population-per-island", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--backend", choices=("cpu", "torch"))
    parser.add_argument("--device")
    parser.add_argument("--continue-after-projlen-one", action="store_true")
    parser.add_argument("--no-stop-at-kernel", action="store_true")
    parser.add_argument(
        "--force-crispr",
        action="store_true",
        help="run repair even if exact reservoir verification already found a kernel",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper-exact reservoir broadly, then conditionally repair "
            "its final low-projlen pool with CRISPR."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("reservoir", "verify", "crispr", "all"):
        child = subparsers.add_parser(command)
        _add_common(child)
    return parser


def _config(args) -> HybridConfig:
    config = profile_config(args.profile, args.output_dir)
    reservoir_overrides = {}
    for argument, field in (
        ("p", "p"),
        ("n", "n"),
        ("r", "r"),
        ("reservoir_depth", "target_depth"),
        ("bucket_size", "bucket_size"),
        ("use_best", "use_best"),
        ("reservoir_seed", "seed"),
    ):
        value = getattr(args, argument)
        if value is not None:
            reservoir_overrides[field] = value
    if args.continue_after_projlen_one:
        reservoir_overrides["stop_at_author_projlen_one"] = False
    if reservoir_overrides:
        config = replace(
            config,
            reservoir=replace(config.reservoir, **reservoir_overrides),
        )

    crispr_overrides = {}
    for argument, field in (
        ("crispr_seed", "seed"),
        ("crispr_pool_size", "pool_size"),
        ("population_per_island", "population_per_island"),
        ("generations", "generations"),
        ("backend", "backend"),
        ("device", "device"),
    ):
        value = getattr(args, argument)
        if value is not None:
            crispr_overrides[field] = value
    if crispr_overrides:
        config = replace(config, crispr=replace(config.crispr, **crispr_overrides))
    if args.crispr_max_depth is not None:
        config = replace(config, crispr_max_depth=args.crispr_max_depth)
    if args.no_stop_at_kernel:
        config = replace(config, stop_at_kernel=False)

    reservoir = config.reservoir
    if not _is_prime(reservoir.p):
        raise ValueError("p must be prime")
    if not 3 <= reservoir.n <= 7:
        raise ValueError("n must lie between 3 and 7")
    if not 0 <= 2 * reservoir.r <= reservoir.n:
        raise ValueError("r must satisfy 0 <= 2r <= n")
    if reservoir.bootstrap_depth > reservoir.target_depth:
        raise ValueError("bootstrap depth cannot exceed reservoir depth")
    if reservoir.bucket_size <= 0 or reservoir.use_best <= 0:
        raise ValueError("bucket-size and use-best must be positive")
    if args.command in {"crispr", "all"}:
        if reservoir.n != 4:
            raise ValueError("CRISPR repair currently requires n=4")
        if config.crispr_max_depth <= reservoir.target_depth and not args.checkpoint:
            raise ValueError("crispr-max-depth must exceed reservoir-depth")
    return config


def _checkpoint_path(args, config: HybridConfig, output: Path) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint).resolve()
    return (
        output
        / f"paper_reservoir_depth_{config.reservoir.target_depth:03d}.json.gz"
    )


def _ensure_reservoir(args, config: HybridConfig, output: Path) -> Path:
    checkpoint = _checkpoint_path(args, config, output)
    if checkpoint.exists():
        return checkpoint
    author_repo = (
        Path(args.author_repo).resolve() if args.author_repo else config.author_repo
    )
    return run_author_reservoir(
        config.reservoir,
        author_repo,
        checkpoint,
        python_executable=args.author_python,
    )


def _inspect_checkpoint(checkpoint: Path, config: HybridConfig, output: Path):
    metadata, candidates, suspected = load_checkpoint(checkpoint)
    if metadata["p"] != config.reservoir.p or metadata["n"] != config.reservoir.n:
        raise ValueError("checkpoint p/n does not match the selected configuration")
    verification = verify_author_kernel_candidates(
        metadata,
        candidates,
        suspected,
    )
    reservoir_summary = {
        "checkpoint": str(checkpoint),
        "metadata": metadata,
        "frontier": candidate_summary(candidates),
        "exact_verification": verification,
    }
    write_json(output / "reservoir_summary.json", reservoir_summary)
    return metadata, candidates, verification


def _run_conditional_crispr(
    args,
    config: HybridConfig,
    metadata: dict,
    candidates: list,
    verification: dict,
    output: Path,
) -> tuple[str, dict | None]:
    if verification["kernel_hits"] and not args.force_crispr:
        skipped = {
            "status": "skipped",
            "reason": "exact reservoir kernel already found",
            "reservoir_kernel_hits": verification["kernel_hits"],
        }
        write_json(output / "crispr" / "result.json", skipped)
        return "reservoir_kernel_found", skipped

    base_depth = int(metadata["actual_depth"])
    if config.crispr_max_depth <= base_depth:
        raise ValueError(
            f"crispr-max-depth {config.crispr_max_depth} must exceed "
            f"checkpoint depth {base_depth}"
        )
    result = run_crispr_repair(
        candidates,
        config.crispr,
        p=int(metadata["p"]),
        n=int(metadata["n"]),
        base_depth=base_depth,
        max_depth=config.crispr_max_depth,
        output_dir=output / "crispr",
        stop_at_kernel=config.stop_at_kernel,
    )
    status = "crispr_kernel_found" if result["kernel_hits"] else "no_kernel_found"
    return status, result


def main() -> None:
    args = build_parser().parse_args()
    config = _config(args)
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"config_{args.command}.json", config.to_dict())

    checkpoint = _ensure_reservoir(args, config, output)
    if args.command == "reservoir":
        print(checkpoint)
        return

    metadata, candidates, verification = _inspect_checkpoint(
        checkpoint,
        config,
        output,
    )
    if args.command == "verify":
        print(output / "reservoir_summary.json")
        return

    status, crispr_result = _run_conditional_crispr(
        args,
        config,
        metadata,
        candidates,
        verification,
        output,
    )
    summary = {
        "status": status,
        "checkpoint": str(checkpoint),
        "reservoir_depth": metadata["actual_depth"],
        "reservoir_frontier": candidate_summary(candidates),
        "reservoir_kernel_hits": verification["kernel_hits"],
        "crispr_kernel_hits": (
            crispr_result.get("kernel_hits", []) if crispr_result else []
        ),
    }
    summary_path = write_json(output / "summary.json", summary)
    print(summary_path)


if __name__ == "__main__":
    main()
