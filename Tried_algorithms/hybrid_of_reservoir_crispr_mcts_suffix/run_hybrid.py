from __future__ import annotations

import argparse
import traceback
from dataclasses import replace
from pathlib import Path

from .backbone import run_author_backbone
from .candidates import (
    candidate_summary,
    load_author_checkpoint,
    select_diverse_candidates,
    write_branch_pool,
)
from .config import HybridConfig, profile_config
from .crispr_branch import run_crispr_branch
from .io_utils import write_json
from .reservoir_mcts_branch import run_reservoir_mcts_branch
from .suffix_lookup_branch import run_suffix_lookup_branch


BRANCHES = ("crispr", "reservoir-mcts", "suffix-lookup")


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
    parser.add_argument("--backend", choices=("cpu", "torch"))
    parser.add_argument("--device")
    parser.add_argument("--p", type=int, help="prime modulus (default: 5)")
    parser.add_argument("--n", type=int, help="braid index (default: 4)")
    parser.add_argument("--r", type=int, help="paper Jones-summand parameter (default: 1)")
    parser.add_argument("--backbone-depth", type=int, help="paper frontier depth")
    parser.add_argument("--max-depth", type=int, help="maximum branch depth")
    parser.add_argument("--backbone-seed", type=int)
    parser.add_argument("--bucket-size", type=int)
    parser.add_argument("--use-best", type=int)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--no-stop-at-kernel", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-exact reservoir through depth 35, followed by independent "
            "CRISPR, reservoir-MCTS, and suffix-lookup searches."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("backbone", "prepare", "all"):
        child = subparsers.add_parser(command)
        _add_common(child)
    branch = subparsers.add_parser("branch")
    _add_common(branch)
    branch.add_argument("name", choices=BRANCHES)
    return parser


def _config(args) -> HybridConfig:
    config = profile_config(args.profile, args.output_dir)
    backbone_overrides = {}
    for argument, field in (
        ("p", "p"),
        ("n", "n"),
        ("r", "r"),
        ("backbone_depth", "target_depth"),
        ("backbone_seed", "seed"),
        ("bucket_size", "bucket_size"),
        ("use_best", "use_best"),
    ):
        value = getattr(args, argument)
        if value is not None:
            backbone_overrides[field] = value
    if backbone_overrides:
        config = replace(
            config,
            backbone=replace(config.backbone, **backbone_overrides),
        )
    if args.max_depth is not None:
        config = replace(config, max_depth=args.max_depth)
    if config.backbone.p <= 2 or not _is_prime(config.backbone.p):
        raise ValueError("the hybrid suffix branch requires an odd prime p")
    if not 3 <= config.backbone.n <= 7:
        raise ValueError("n must lie between 3 and 7")
    if not 0 <= 2 * config.backbone.r <= config.backbone.n:
        raise ValueError("r must satisfy 0 <= 2r <= n")
    if args.command != "backbone" and config.backbone.n != 4:
        raise ValueError("the three hybrid branches currently require n=4")
    if config.backbone.bootstrap_depth > config.backbone.target_depth:
        raise ValueError("bootstrap depth cannot exceed backbone depth")
    if config.backbone.bucket_size <= 0 or config.backbone.use_best <= 0:
        raise ValueError("bucket-size and use-best must be positive")
    if config.max_depth <= config.backbone.target_depth:
        raise ValueError("max-depth must be greater than backbone-depth")
    if args.backend or args.device:
        backend = args.backend or config.crispr.backend
        device = args.device or config.crispr.device
        config = replace(
            config,
            crispr=replace(config.crispr, backend=backend, device=device),
            suffix_lookup=replace(
                config.suffix_lookup,
                backend=backend,
                device=device,
            ),
        )
    if args.no_stop_at_kernel:
        config = replace(config, stop_at_kernel=False)
    return config


def _checkpoint_path(args, config: HybridConfig, output: Path) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint).resolve()
    return output / f"paper_frontier_depth_{config.backbone.target_depth:03d}.json.gz"


def _ensure_backbone(args, config: HybridConfig, output: Path) -> Path:
    checkpoint = _checkpoint_path(args, config, output)
    if checkpoint.exists():
        return checkpoint
    author_repo = Path(args.author_repo).resolve() if args.author_repo else config.author_repo
    return run_author_backbone(
        config.backbone,
        author_repo,
        checkpoint,
        python_executable=args.author_python,
    )


def _prepare_pools(
    checkpoint: Path,
    config: HybridConfig,
    output: Path,
) -> tuple[dict, list]:
    metadata, candidates = load_author_checkpoint(checkpoint)
    if metadata["p"] != config.backbone.p or metadata["n"] != config.backbone.n:
        raise ValueError("checkpoint p/n does not match the selected profile")
    write_json(
        output / "frontier_summary.json",
        {
            "checkpoint": str(checkpoint),
            "metadata": metadata,
            "summary": candidate_summary(candidates),
        },
    )
    branch_specs = {
        "crispr": (config.crispr.pool_size, config.crispr.seed),
        "reservoir-mcts": (
            config.reservoir_mcts.pool_size,
            config.reservoir_mcts.seed,
        ),
        "suffix-lookup": (
            config.suffix_lookup.prefix_pool_size,
            config.suffix_lookup.seed,
        ),
    }
    for name, (limit, seed) in branch_specs.items():
        pool = select_diverse_candidates(candidates, limit, seed)
        write_branch_pool(
            output / "pools" / f"{name}.json.gz",
            checkpoint,
            metadata,
            pool,
        )
    return metadata, candidates


def _run_one(
    name: str,
    config: HybridConfig,
    metadata: dict,
    candidates: list,
    output: Path,
) -> dict:
    common = {
        "p": metadata["p"],
        "n": metadata["n"],
        "max_depth": config.max_depth,
        "output_dir": output / name,
        "stop_at_kernel": config.stop_at_kernel,
    }
    if name == "crispr":
        return run_crispr_branch(
            candidates,
            config.crispr,
            base_depth=metadata["target_depth"],
            **common,
        )
    if name == "reservoir-mcts":
        return run_reservoir_mcts_branch(
            candidates,
            config.reservoir_mcts,
            **common,
        )
    if name == "suffix-lookup":
        return run_suffix_lookup_branch(
            candidates,
            config.suffix_lookup,
            base_depth=metadata["target_depth"],
            **common,
        )
    raise ValueError(f"unknown branch {name!r}")


def main() -> None:
    args = build_parser().parse_args()
    config = _config(args)
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_name = (
        f"config_{args.name}.json" if args.command == "branch" else f"config_{args.command}.json"
    )
    write_json(output / config_name, config.to_dict())

    if args.command == "backbone":
        checkpoint = _ensure_backbone(args, config, output)
        print(checkpoint)
        return

    checkpoint = _ensure_backbone(args, config, output)
    if args.command == "prepare":
        _prepare_pools(checkpoint, config, output)
        print(output / "pools")
        return

    if args.command == "branch":
        metadata, candidates = load_author_checkpoint(checkpoint)
        if metadata["p"] != config.backbone.p or metadata["n"] != config.backbone.n:
            raise ValueError("checkpoint p/n does not match the selected profile")
    else:
        metadata, candidates = _prepare_pools(checkpoint, config, output)

    names = BRANCHES if args.command == "all" else (args.name,)
    results = {}
    failures = {}
    for name in names:
        try:
            results[name] = _run_one(name, config, metadata, candidates, output)
        except Exception as exc:
            failures[name] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(output / name / "failure.json", failures[name])
            if not args.keep_going:
                raise
    summary = {
        "checkpoint": str(checkpoint),
        "frontier": candidate_summary(candidates),
        "branches": {
            name: {
                "kernel_hits": len(result.get("kernel_hits", [])),
                "best_count": len(result.get("best", [])),
            }
            for name, result in results.items()
        },
        "failures": failures,
    }
    summary_path = (
        output / args.name / "summary.json"
        if args.command == "branch"
        else output / "summary.json"
    )
    write_json(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
