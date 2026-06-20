from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .dataset import generate_mutation_dataset
from .repair import run_guided_repair
from .training import train_geometry_model


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _add_evaluator(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("cpu", "torch"), default="cpu")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-batch-size", type=int, default=10_000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper reservoir plus transformer-guided legal CRISPR repair."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    reservoir = commands.add_parser(
        "reservoir",
        help="run the paper reservoir to a fixed depth or an adaptive downturn handoff",
    )
    reservoir.add_argument("--output", required=True)
    reservoir.add_argument("--author-repo", default=str(PACKAGE_ROOT / "third_party" / "braids_project"))
    reservoir.add_argument("--author-python", default=sys.executable)
    reservoir.add_argument("--p", type=int, default=5)
    reservoir.add_argument("--n", type=int, default=4)
    reservoir.add_argument("--r", type=int, default=1)
    reservoir.add_argument("--bootstrap-depth", type=int, default=5)
    reservoir.add_argument("--target-depth", type=int, default=60)
    reservoir.add_argument("--step-size", type=int, default=1)
    reservoir.add_argument("--bucket-size", type=int, default=15_000)
    reservoir.add_argument("--use-best", type=int, default=30_000)
    reservoir.add_argument("--seed", type=int, default=1)
    reservoir.add_argument("--continue-after-projlen-one", action="store_true")
    reservoir.add_argument("--adaptive-downturn", action="store_true")
    reservoir.add_argument("--downturn-min-depth", type=int, default=20)
    reservoir.add_argument("--downturn-smoothing-window", type=int, default=3)
    reservoir.add_argument("--downturn-trend-window", type=int, default=8)
    reservoir.add_argument("--downturn-min-drop", type=float, default=4.0)
    reservoir.add_argument("--downturn-max-slope", type=float, default=-0.35)
    reservoir.add_argument("--downturn-min-negative-fraction", type=float, default=0.50)
    reservoir.add_argument("--downturn-confirmation-steps", type=int, default=2)
    reservoir.add_argument("--handoff-extra-depths", type=int, default=4)

    dataset = commands.add_parser("dataset", help="generate exact variable-edit labels")
    dataset.add_argument("--checkpoint", action="append", required=True)
    dataset.add_argument("--output-dir", required=True)
    dataset.add_argument("--parents-limit", type=int, default=5_000)
    dataset.add_argument("--actions-per-parent", type=int, default=16)
    dataset.add_argument("--replacements-per-action", type=int, default=4)
    dataset.add_argument("--max-delete", type=int, default=16)
    dataset.add_argument("--max-insert", type=int, default=16)
    dataset.add_argument("--max-net-delta", type=int, default=3)
    dataset.add_argument("--min-length", type=int)
    dataset.add_argument("--max-length", type=int)
    dataset.add_argument("--baseline-samples-per-length", type=int, default=2_048)
    dataset.add_argument("--augmented-parent-fraction", type=float, default=0.25)
    dataset.add_argument("--target-top-k", type=int, default=2)
    dataset.add_argument("--allow-unconfirmed-handoff", action="store_true")
    dataset.add_argument("--seed", type=int, default=1)
    _add_evaluator(dataset)

    train = commands.add_parser("train", help="train one prime-specific geometry transformer")
    train.add_argument("--dataset", required=True)
    train.add_argument("--dataset-summary", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--validation-fraction", type=float, default=0.15)
    train.add_argument("--target-temperature", type=float, default=0.20)
    train.add_argument("--d-model", type=int, default=128)
    train.add_argument("--nhead", type=int, default=4)
    train.add_argument("--num-layers", type=int, default=4)
    train.add_argument("--dim-feedforward", type=int, default=512)
    train.add_argument("--dropout", type=float, default=0.10)
    train.add_argument("--device", default="cpu")
    train.add_argument("--seed", type=int, default=1)

    repair = commands.add_parser("repair", help="run guided or matched-random repair")
    repair.add_argument("--checkpoint", action="append", required=True)
    repair.add_argument("--baseline", required=True)
    repair.add_argument("--model")
    repair.add_argument("--mode", choices=("guided", "random"), default="guided")
    repair.add_argument("--output-dir", required=True)
    repair.add_argument("--population-size", type=int, default=512)
    repair.add_argument("--generations", type=int, default=40)
    repair.add_argument("--actions-per-parent", type=int, default=4)
    repair.add_argument("--replacements-per-action", type=int, default=4)
    repair.add_argument("--exploration-fraction", type=float, default=0.15)
    repair.add_argument("--geometry-candidates-per-parent", type=int, default=1_024)
    repair.add_argument("--stagnation-generations", type=int, default=15)
    repair.add_argument("--restart-fraction", type=float, default=0.25)
    repair.add_argument("--no-stop-at-kernel", action="store_true")
    repair.add_argument("--seed", type=int, default=1)
    _add_evaluator(repair)
    return parser


def _run_reservoir(args) -> dict:
    command = [
        str(args.author_python),
        str(PACKAGE_ROOT / "author_reservoir_worker.py"),
        "--author-repo",
        str(Path(args.author_repo).resolve()),
        "--output",
        str(Path(args.output).resolve()),
        "--p",
        str(args.p),
        "--n",
        str(args.n),
        "--r",
        str(args.r),
        "--bootstrap-depth",
        str(args.bootstrap_depth),
        "--target-depth",
        str(args.target_depth),
        "--step-size",
        str(args.step_size),
        "--bucket-size",
        str(args.bucket_size),
        "--use-best",
        str(args.use_best),
        "--seed",
        str(args.seed),
    ]
    if args.continue_after_projlen_one:
        command.append("--continue-after-projlen-one")
    if args.adaptive_downturn:
        command.extend(
            [
                "--adaptive-downturn",
                "--downturn-min-depth",
                str(args.downturn_min_depth),
                "--downturn-smoothing-window",
                str(args.downturn_smoothing_window),
                "--downturn-trend-window",
                str(args.downturn_trend_window),
                "--downturn-min-drop",
                str(args.downturn_min_drop),
                "--downturn-max-slope",
                str(args.downturn_max_slope),
                "--downturn-min-negative-fraction",
                str(args.downturn_min_negative_fraction),
                "--downturn-confirmation-steps",
                str(args.downturn_confirmation_steps),
                "--handoff-extra-depths",
                str(args.handoff_extra_depths),
            ]
        )
    subprocess.run(command, check=True)
    return {"checkpoint": str(Path(args.output).resolve())}


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "reservoir":
        result = _run_reservoir(args)
    elif args.command == "dataset":
        result = generate_mutation_dataset(
            checkpoints=args.checkpoint,
            output_dir=args.output_dir,
            parents_limit=args.parents_limit,
            actions_per_parent=args.actions_per_parent,
            replacements_per_action=args.replacements_per_action,
            max_delete=args.max_delete,
            max_insert=args.max_insert,
            max_net_delta=args.max_net_delta,
            min_length=args.min_length,
            max_length=args.max_length,
            baseline_samples_per_length=args.baseline_samples_per_length,
            augmented_parent_fraction=args.augmented_parent_fraction,
            target_top_k=args.target_top_k,
            allow_unconfirmed_handoff=args.allow_unconfirmed_handoff,
            backend=args.backend,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            seed=args.seed,
        )
    elif args.command == "train":
        result = train_geometry_model(
            dataset_path=args.dataset,
            dataset_summary_path=args.dataset_summary,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            validation_fraction=args.validation_fraction,
            target_temperature=args.target_temperature,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            device=args.device,
            seed=args.seed,
        )
    else:
        result = run_guided_repair(
            checkpoints=args.checkpoint,
            baseline_path=args.baseline,
            model_path=args.model,
            mode=args.mode,
            output_dir=args.output_dir,
            population_size=args.population_size,
            generations=args.generations,
            actions_per_parent=args.actions_per_parent,
            replacements_per_action=args.replacements_per_action,
            exploration_fraction=args.exploration_fraction,
            geometry_candidates_per_parent=args.geometry_candidates_per_parent,
            stagnation_generations=args.stagnation_generations,
            restart_fraction=args.restart_fraction,
            backend=args.backend,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            stop_at_kernel=not args.no_stop_at_kernel,
            seed=args.seed,
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
