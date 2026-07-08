from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .core import BraidEnvironment, parse_int_list, word_digest, write_json
from .frontier import iter_frontier_cache
from .ledger import RunLedger
from .search import candidate_rank, compact_candidate_row, parse_completion_targets


def _parse_lengths(value: str, *, default: Sequence[int]) -> tuple[int, ...]:
    if not value.strip():
        return tuple(int(x) for x in default)
    return tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))


def _target_score(metrics_by_target: dict[str, dict]) -> tuple[int, int, int]:
    best_target_defect = min(int(metrics["target_defect"]) for metrics in metrics_by_target.values())
    best_projlen = min(int(metrics["projlen"]) for metrics in metrics_by_target.values())
    best_identity_defect = min(int(metrics["identity_defect"]) for metrics in metrics_by_target.values())
    return best_target_defect, best_identity_defect, best_projlen


def _choose_action(
    *,
    env: BraidEnvironment,
    rng: random.Random,
    factors: tuple[int, ...],
    image: np.ndarray,
    targets: tuple[str, ...],
    mode: str,
    action_samples: int,
    temperature: float,
    epsilon: float,
) -> tuple[int, np.ndarray, int]:
    legal = list(env.legal_next(factors))
    if not legal:
        raise ValueError("dead GNF state with no legal successors")
    if mode == "random":
        action = int(rng.choice(legal))
        return action, env.exact_append(image, action), 1

    if action_samples > 0 and action_samples < len(legal):
        candidates = rng.sample(legal, action_samples)
    else:
        candidates = legal

    scored: list[tuple[tuple[int, int, int], int, np.ndarray]] = []
    for action in candidates:
        child_image = env.exact_append(image, int(action))
        metrics_by_target = {
            target: env.exact_target_metrics(child_image, target)
            for target in targets
        }
        scored.append((_target_score(metrics_by_target), int(action), child_image))

    if mode == "greedy":
        _, action, child_image = min(scored, key=lambda item: item[0])
        return action, child_image, len(scored)

    if mode != "softmin":
        raise ValueError("--growth-mode must be random, greedy, or softmin")
    if epsilon > 0.0 and rng.random() < epsilon:
        _, action, child_image = rng.choice(scored)
        return action, child_image, len(scored)

    temp = max(float(temperature), 1e-6)
    raw_scores = [float(score[0]) for score, _, _ in scored]
    best = min(raw_scores)
    weights = [math.exp(-(score - best) / temp) for score in raw_scores]
    total = sum(weights)
    threshold = rng.random() * total
    acc = 0.0
    for weight, (_, action, child_image) in zip(weights, scored):
        acc += weight
        if acc >= threshold:
            return action, child_image, len(scored)
    _, action, child_image = scored[-1]
    return action, child_image, len(scored)


def run_frontier_growth(args: argparse.Namespace) -> dict:
    start_time = time.time()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    ledger = RunLedger(output_dir=output_dir)
    t_values = parse_int_list(args.t_values, default=tuple(range(1, args.p)))
    targets = parse_completion_targets(args.completion_targets)
    check_lengths = _parse_lengths(args.check_lengths, default=(args.target_length,))

    env = BraidEnvironment(
        author_repo=Path(args.author_repo),
        n=args.n,
        r=args.r,
        p=args.p,
        t_values=t_values,
    )
    config = vars(args).copy()
    config["t_values"] = list(t_values)
    config["completion_targets"] = list(targets)
    config["check_lengths"] = list(check_lengths)
    config["representation"] = env.representation_label
    write_json(output_dir / "config.json", config)
    no_oracle_summary = {
        "mode": "direct_frontier_growth",
        "uses_collision_oracle": False,
        "uses_suffix_bank": False,
        "frontier_path": args.frontier_path,
        "frontier_shard_by": args.frontier_shard_by,
        "frontier_shard_count": args.frontier_shard_count,
        "frontier_shard_index": args.frontier_shard_index,
    }
    write_json(output_dir / "oracle_summary.json", no_oracle_summary)
    write_json(output_dir / "growth_summary.json", no_oracle_summary)

    best_prefix_candidate: dict | None = None
    best_target_candidate: dict | None = None
    best_scalar_identity_candidate: dict | None = None
    best_projlen: int | None = None
    best_identity_defect: int | None = None
    best_target_defect: int | None = None
    frontier_loaded = 0
    rollouts_started = 0
    completed_rollouts = 0
    exact_evaluations = 0
    symbolic_factor_multiplications = 0
    scalar_identity_candidates = 0
    target_match_candidates = 0
    checked_states = 0
    last_progress = start_time

    per_length_counts: dict[int, dict[str, int]] = {
        int(length): {"checked": 0, "target_matches": 0, "scalar_identities": 0}
        for length in check_lengths
    }

    def observe_candidate(
        *,
        factors: tuple[int, ...],
        image: np.ndarray,
        label: str,
        frontier_record_id: int,
        rollout_index: int,
    ) -> None:
        nonlocal best_target_candidate
        nonlocal best_scalar_identity_candidate
        nonlocal best_projlen
        nonlocal best_identity_defect
        nonlocal best_target_defect
        nonlocal scalar_identity_candidates
        nonlocal target_match_candidates
        nonlocal checked_states

        metrics = env.exact_target_metrics(image, label)
        checked_states += 1
        per_length_counts[len(factors)]["checked"] += 1
        best_projlen = int(metrics["projlen"]) if best_projlen is None else min(best_projlen, int(metrics["projlen"]))
        best_identity_defect = (
            int(metrics["identity_defect"])
            if best_identity_defect is None
            else min(best_identity_defect, int(metrics["identity_defect"]))
        )
        best_target_defect = (
            int(metrics["target_defect"])
            if best_target_defect is None
            else min(best_target_defect, int(metrics["target_defect"]))
        )
        row = compact_candidate_row(
            kind="direct_growth_target_check",
            factors=factors,
            metrics=metrics,
            extra={
                "target_label": label,
                "frontier_length": args.frontier_length,
                "frontier_record_id": frontier_record_id,
                "rollout_index": rollout_index,
                "matrix_digest": env.exact_digest(image),
            },
        )
        if best_target_candidate is None or (
            int(metrics["target_defect"]),
            int(metrics["identity_defect"]),
            int(metrics["projlen"]),
            int(row["length"]),
        ) < (
            int(best_target_candidate["metrics"]["target_defect"]),
            int(best_target_candidate["metrics"]["identity_defect"]),
            int(best_target_candidate["metrics"]["projlen"]),
            int(best_target_candidate["length"]),
        ):
            best_target_candidate = row
        if metrics.get("scalar_identity"):
            scalar_identity_candidates += 1
            per_length_counts[len(factors)]["scalar_identities"] += 1
            best_scalar_identity_candidate = row
            ledger.candidate(row)
            print(json.dumps({"phase": "exact_scalar_identity", **row}, sort_keys=True), flush=True)
        if metrics.get("target_match"):
            target_match_candidates += 1
            per_length_counts[len(factors)]["target_matches"] += 1
            ledger.candidate(row)
            print(json.dumps({"phase": "exact_target_match", **row}, sort_keys=True), flush=True)

    frontier_iter = iter_frontier_cache(
        env=env,
        path=Path(args.frontier_path),
        shard_count=args.frontier_shard_count,
        shard_index=args.frontier_shard_index,
        shard_by=args.frontier_shard_by,
        max_records=args.frontier_max_records,
    )
    for record in frontier_iter:
        if record.length != args.frontier_length:
            raise ValueError(
                f"frontier record length {record.length} does not match --frontier-length {args.frontier_length}"
            )
        frontier_loaded += 1
        base_exact = env.exact_evaluate(record.factors)
        exact_evaluations += 1
        symbolic_factor_multiplications += len(record.factors)
        base_metrics = env.exact_metrics(base_exact)
        prefix_row = compact_candidate_row(kind="frontier_prefix", factors=record.factors, metrics=base_metrics)
        if best_prefix_candidate is None or candidate_rank(prefix_row) < candidate_rank(best_prefix_candidate):
            best_prefix_candidate = prefix_row

        for rollout_index in range(args.rollouts_per_frontier):
            rollouts_started += 1
            factors = record.factors
            image = base_exact
            for total_length in range(args.frontier_length + 1, args.target_length + 1):
                action, image, action_evaluations = _choose_action(
                    env=env,
                    rng=rng,
                    factors=factors,
                    image=image,
                    targets=targets,
                    mode=args.growth_mode,
                    action_samples=args.action_samples,
                    temperature=args.temperature,
                    epsilon=args.epsilon,
                )
                factors = factors + (action,)
                exact_evaluations += action_evaluations
                symbolic_factor_multiplications += action_evaluations
                if total_length in check_lengths:
                    for target in targets:
                        observe_candidate(
                            factors=factors,
                            image=image,
                            label=target,
                            frontier_record_id=record.record_id,
                            rollout_index=rollout_index,
                        )
                    if args.stop_after_scalar_identity and scalar_identity_candidates:
                        break
                if args.stop_after_scalar_identity and scalar_identity_candidates:
                    break
            completed_rollouts += 1
            if args.stop_after_scalar_identity and scalar_identity_candidates:
                break
        now = time.time()
        if now - last_progress >= args.progress_interval_seconds:
            progress = {
                "phase": "frontier_growth_progress",
                "frontier_loaded": frontier_loaded,
                "rollouts_started": rollouts_started,
                "completed_rollouts": completed_rollouts,
                "checked_states": checked_states,
                "exact_evaluations": exact_evaluations,
                "best_projlen": best_projlen,
                "best_identity_defect": best_identity_defect,
                "best_target_defect": best_target_defect,
                "scalar_identity_candidates": scalar_identity_candidates,
                "target_match_candidates": target_match_candidates,
                "elapsed_seconds": round(now - start_time, 2),
            }
            ledger.progress(progress)
            print(json.dumps(progress, sort_keys=True), flush=True)
            last_progress = now
        if args.stop_after_scalar_identity and scalar_identity_candidates:
            break

    elapsed = time.time() - start_time
    length_range = {
        "frontier_length": args.frontier_length,
        "target_length": args.target_length,
        "check_lengths": list(check_lengths),
        "rollouts_per_frontier": args.rollouts_per_frontier,
    }
    summary = {
        "format": "braidzero-frontier-growth-summary-v1",
        "status": "clean",
        "method": "braidzero_exhaustive_frontier_direct_growth",
        "prime": args.p,
        "representation": env.representation_label,
        "seed": args.seed,
        "length_range": length_range,
        "t_values": list(t_values),
        "completion_targets": list(targets),
        "frontier": {
            "frontier_path": args.frontier_path,
            "frontier_loaded": frontier_loaded,
            "frontier_shard_by": args.frontier_shard_by,
            "frontier_shard_count": args.frontier_shard_count,
            "frontier_shard_index": args.frontier_shard_index,
            "frontier_max_records": args.frontier_max_records,
        },
        "search": {
            "elapsed_seconds": round(elapsed, 2),
            "rollouts_started": rollouts_started,
            "completed_rollouts": completed_rollouts,
            "exact_evaluations": exact_evaluations,
            "symbolic_factor_multiplications": symbolic_factor_multiplications,
            "checked_states": checked_states,
            "per_length_counts": per_length_counts,
            "best_projlen": best_projlen,
            "best_identity_defect": best_identity_defect,
            "best_target_defect": best_target_defect,
            "best_prefix_candidate": best_prefix_candidate,
            "best_target_candidate": best_target_candidate,
            "best_scalar_identity_candidate": best_scalar_identity_candidate,
            "scalar_identity_candidates": scalar_identity_candidates,
            "target_match_candidates": target_match_candidates,
            "verified_kernel_quotients": scalar_identity_candidates,
        },
    }
    ledger_row = {
        "prime": args.p,
        "representation": env.representation_label,
        "seed": args.seed,
        "method": summary["method"],
        "length_range": length_range,
        "number_exact_evaluations": exact_evaluations,
        "best_projlen": best_projlen,
        "best_identity_defect": best_identity_defect,
        "best_scalar_identity_candidate": best_scalar_identity_candidate,
        "number_exact_collisions": 0,
        "number_verified_kernel_quotients": scalar_identity_candidates,
        "verifier_version": env.verifier_version,
        "status": "clean",
    }
    ledger.finalize(summary=summary, ledger_row=ledger_row)
    print(json.dumps({"phase": "done", **summary["search"]}, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Directly grow exact words from an exhaustive BraidZero frontier; no collision oracle."
    )
    parser.add_argument("--author-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--t-values", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--frontier-path", required=True)
    parser.add_argument("--frontier-length", type=int, default=8)
    parser.add_argument("--frontier-shard-count", type=int, default=1)
    parser.add_argument("--frontier-shard-index", type=int, default=0)
    parser.add_argument("--frontier-shard-by", choices=["record", "key", "none"], default="record")
    parser.add_argument("--frontier-max-records", type=int, default=0)
    parser.add_argument("--target-length", type=int, default=66)
    parser.add_argument("--check-lengths", default="54,63,65,66")
    parser.add_argument("--rollouts-per-frontier", type=int, default=1)
    parser.add_argument("--growth-mode", choices=["random", "greedy", "softmin"], default="random")
    parser.add_argument("--action-samples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=25.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--completion-targets", default="identity,delta")
    parser.add_argument("--stop-after-scalar-identity", action="store_true")
    parser.add_argument("--progress-interval-seconds", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_frontier_growth(args)


if __name__ == "__main__":
    main()
