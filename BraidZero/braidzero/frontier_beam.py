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
from .search import SearchState, compact_candidate_row, parse_completion_targets


VALID_HEURISTICS = {
    "target",
    "identity",
    "projlen",
    "scalar_shape",
    "terms",
    "random",
    "identity_target",
    "delta_target",
}


def _parse_lengths(value: str, *, default: Sequence[int]) -> tuple[int, ...]:
    if not value.strip():
        return tuple(int(x) for x in default)
    return tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))


def _parse_heuristics(value: str) -> tuple[str, ...]:
    heuristics = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not heuristics:
        return ("target", "identity", "projlen", "scalar_shape", "random")
    unknown = [heuristic for heuristic in heuristics if heuristic not in VALID_HEURISTICS]
    if unknown:
        raise ValueError(f"unknown heuristic(s): {unknown}; valid heuristics are {sorted(VALID_HEURISTICS)}")
    return tuple(dict.fromkeys(heuristics))


def _gumbel(rng: random.Random) -> float:
    u = min(max(rng.random(), 1e-12), 1.0 - 1e-12)
    return -math.log(-math.log(u))


def _heuristic_cost(metrics: dict, heuristic: str, *, identity_weight: float, projlen_weight: float) -> float:
    target_defect = float(metrics["target_defect"])
    identity_defect = float(metrics["identity_defect"])
    projlen = float(metrics["projlen"])
    if heuristic == "target":
        return target_defect + identity_weight * identity_defect + projlen_weight * projlen
    if heuristic == "identity":
        return identity_defect + 0.10 * target_defect + projlen_weight * projlen
    if heuristic == "projlen":
        return projlen + 0.25 * identity_defect + 0.05 * target_defect
    if heuristic == "scalar_shape":
        return (
            float(metrics.get("off_diagonal_terms", 0))
            + float(metrics.get("diagonal_mismatch_terms", 0))
            + float(metrics.get("scalar_extra_degrees", 0))
            + 0.05 * target_defect
            + 0.05 * projlen
        )
    if heuristic == "terms":
        return float(metrics.get("nonzero_terms", 0)) + 0.10 * target_defect + 0.05 * projlen
    if heuristic == "identity_target":
        return float(metrics.get("identity_target_defect", target_defect)) + identity_weight * identity_defect + projlen_weight * projlen
    if heuristic == "delta_target":
        return float(metrics.get("delta_target_defect", target_defect)) + identity_weight * identity_defect + projlen_weight * projlen
    if heuristic == "random":
        return 0.0
    raise ValueError(f"unknown heuristic: {heuristic}")


def _priority(
    metrics: dict,
    *,
    heuristic: str,
    rng: random.Random,
    selection_temperature: float,
    identity_weight: float,
    projlen_weight: float,
) -> float:
    if heuristic == "random":
        return rng.random()
    cost = _heuristic_cost(metrics, heuristic, identity_weight=identity_weight, projlen_weight=projlen_weight)
    if selection_temperature <= 0.0:
        return -cost
    return -(cost / selection_temperature) + _gumbel(rng)


def _best_target_metrics(env: BraidEnvironment, image: np.ndarray, targets: tuple[str, ...]) -> tuple[dict, dict[str, dict]]:
    by_target = {target: env.exact_target_metrics(image, target) for target in targets}
    best_label, best_metrics = min(
        by_target.items(),
        key=lambda item: (
            int(item[1]["target_defect"]),
            int(item[1]["identity_defect"]),
            int(item[1]["projlen"]),
            item[0],
        ),
    )
    target_defects = {target: int(metrics["target_defect"]) for target, metrics in by_target.items()}
    return {
        **best_metrics,
        "best_target_label": best_label,
        "target_defects": target_defects,
        "identity_target_defect": int(by_target["identity"]["target_defect"]) if "identity" in by_target else int(best_metrics["target_defect"]),
        "delta_target_defect": int(by_target["delta"]["target_defect"]) if "delta" in by_target else int(best_metrics["target_defect"]),
    }, by_target


def _candidate_sort_tuple(row: dict) -> tuple[int, int, int, int]:
    metrics = row["metrics"]
    return (
        int(metrics["target_defect"]),
        int(metrics["identity_defect"]),
        int(metrics["projlen"]),
        int(row["length"]),
    )


def _bucket_key(state: SearchState, *, target_defect_bin: int, projlen_bin: int) -> tuple[int, str, int, int]:
    target_defect_bin = max(1, int(target_defect_bin))
    projlen_bin = max(1, int(projlen_bin))
    return (
        int(state.factors[-1]),
        str(state.metrics.get("best_target_label", state.metrics.get("target_label", ""))),
        int(state.metrics["target_defect"]) // target_defect_bin,
        int(state.metrics["projlen"]) // projlen_bin,
    )


def _select_population(
    states: list[SearchState],
    *,
    beam_size: int,
    per_finite_key_cap: int,
    diversity_bucket_cap: int,
    target_defect_bin: int,
    projlen_bin: int,
) -> list[SearchState]:
    if len(states) <= beam_size and per_finite_key_cap <= 0 and diversity_bucket_cap <= 0:
        return states

    states.sort(key=lambda state: state.score, reverse=True)
    selected: list[SearchState] = []
    selected_ids: set[int] = set()
    finite_counts: dict[tuple[int, ...], int] = {}
    bucket_counts: dict[tuple[int, str, int, int], int] = {}

    def allowed(state: SearchState, *, enforce_bucket: bool) -> bool:
        if per_finite_key_cap > 0 and finite_counts.get(state.finite_key, 0) >= per_finite_key_cap:
            return False
        if enforce_bucket and diversity_bucket_cap > 0:
            bucket = _bucket_key(state, target_defect_bin=target_defect_bin, projlen_bin=projlen_bin)
            if bucket_counts.get(bucket, 0) >= diversity_bucket_cap:
                return False
        return True

    def add(state: SearchState) -> None:
        selected.append(state)
        selected_ids.add(id(state))
        finite_counts[state.finite_key] = finite_counts.get(state.finite_key, 0) + 1
        bucket = _bucket_key(state, target_defect_bin=target_defect_bin, projlen_bin=projlen_bin)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    for state in states:
        if allowed(state, enforce_bucket=True):
            add(state)
            if len(selected) >= beam_size:
                return selected

    for state in states:
        if id(state) in selected_ids:
            continue
        if allowed(state, enforce_bucket=False):
            add(state)
            if len(selected) >= beam_size:
                break
    return selected


def run_frontier_beam(args: argparse.Namespace) -> dict:
    start_time = time.time()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    ledger = RunLedger(output_dir=output_dir)
    t_values = parse_int_list(args.t_values, default=tuple(range(1, args.p)))
    targets = parse_completion_targets(args.completion_targets)
    check_lengths = _parse_lengths(args.check_lengths, default=(args.target_length,))
    heuristics = _parse_heuristics(args.heuristics)

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
    config["heuristics"] = list(heuristics)
    config["representation"] = env.representation_label
    write_json(output_dir / "config.json", config)
    write_json(
        output_dir / "oracle_summary.json",
        {
            "mode": "frontier_population_beam",
            "uses_collision_oracle": False,
            "uses_suffix_bank": False,
            "frontier_path": args.frontier_path,
            "frontier_shard_by": args.frontier_shard_by,
            "frontier_shard_count": args.frontier_shard_count,
            "frontier_shard_index": args.frontier_shard_index,
        },
    )

    beams: dict[str, list[SearchState]] = {heuristic: [] for heuristic in heuristics}
    initial_beam_size = args.initial_beam_size if args.initial_beam_size > 0 else args.beam_size
    buffer_limit = max(args.beam_buffer_min, initial_beam_size * max(1, args.beam_buffer_factor))
    step_buffer_limit = max(args.beam_buffer_min, args.beam_size * max(1, args.beam_buffer_factor))

    frontier_loaded = 0
    exact_evaluations = 0
    symbolic_factor_multiplications = 0
    expanded_states = 0
    checked_states = 0
    scalar_identity_candidates = 0
    target_match_candidates = 0
    best_prefix_candidate: dict | None = None
    best_target_candidate: dict | None = None
    best_scalar_identity_candidate: dict | None = None
    best_projlen: int | None = None
    best_identity_defect: int | None = None
    best_target_defect: int | None = None
    last_progress = start_time
    seen_target_checks: set[tuple[str, str]] = set()
    per_length_counts: dict[int, dict[str, int]] = {
        int(length): {"checked": 0, "target_matches": 0, "scalar_identities": 0}
        for length in check_lengths
    }

    def update_best_from_metrics(*, factors: tuple[int, ...], metrics: dict, kind: str, extra: dict | None = None) -> None:
        nonlocal best_projlen, best_identity_defect, best_target_defect, best_target_candidate, best_prefix_candidate
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
        row = compact_candidate_row(kind=kind, factors=factors, metrics=metrics, extra=extra)
        if kind == "frontier_prefix":
            if best_prefix_candidate is None or (
                int(metrics["identity_defect"]),
                int(metrics["projlen"]),
                int(row["length"]),
            ) < (
                int(best_prefix_candidate["metrics"]["identity_defect"]),
                int(best_prefix_candidate["metrics"]["projlen"]),
                int(best_prefix_candidate["length"]),
            ):
                best_prefix_candidate = row
        if best_target_candidate is None or _candidate_sort_tuple(row) < _candidate_sort_tuple(best_target_candidate):
            best_target_candidate = row

    def check_target_hits(
        *,
        factors: tuple[int, ...],
        image: np.ndarray,
        by_target: dict[str, dict],
        source: str,
    ) -> None:
        nonlocal checked_states, scalar_identity_candidates, target_match_candidates, best_scalar_identity_candidate
        digest = word_digest(0, factors)
        for target_label, metrics in by_target.items():
            check_key = (target_label, digest)
            if check_key in seen_target_checks:
                continue
            seen_target_checks.add(check_key)
            checked_states += 1
            per_length_counts[len(factors)]["checked"] += 1
            row = compact_candidate_row(
                kind="frontier_beam_target_check",
                factors=factors,
                metrics=metrics,
                extra={
                    "target_label": target_label,
                    "source": source,
                    "frontier_length": args.frontier_length,
                    "matrix_digest": env.exact_digest(image),
                },
            )
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

    frontier_start = time.time()
    for record in iter_frontier_cache(
        env=env,
        path=Path(args.frontier_path),
        shard_count=args.frontier_shard_count,
        shard_index=args.frontier_shard_index,
        shard_by=args.frontier_shard_by,
        max_records=args.frontier_max_records,
    ):
        if record.length != args.frontier_length:
            raise ValueError(
                f"frontier record length {record.length} does not match --frontier-length {args.frontier_length}"
            )
        frontier_loaded += 1
        exact = env.exact_evaluate(record.factors)
        finite = env.finite_evaluate(record.factors)
        metrics, by_target = _best_target_metrics(env, exact, targets)
        exact_evaluations += 1
        symbolic_factor_multiplications += len(record.factors)
        update_best_from_metrics(
            factors=record.factors,
            metrics=metrics,
            kind="frontier_prefix",
            extra={"frontier_record_id": record.record_id},
        )
        finite_key = env.finite_key(finite)
        for heuristic in heuristics:
            beams[heuristic].append(
                SearchState(
                    factors=record.factors,
                    finite=finite,
                    finite_key=finite_key,
                    exact=exact,
                    metrics=metrics,
                    score=_priority(
                        metrics,
                        heuristic=heuristic,
                        rng=rng,
                        selection_temperature=args.selection_temperature,
                        identity_weight=args.identity_weight,
                        projlen_weight=args.projlen_weight,
                    ),
                )
            )
            if len(beams[heuristic]) >= buffer_limit:
                beams[heuristic] = _select_population(
                    beams[heuristic],
                    beam_size=initial_beam_size,
                    per_finite_key_cap=args.per_finite_key_cap,
                    diversity_bucket_cap=args.diversity_bucket_cap,
                    target_defect_bin=args.target_defect_bin,
                    projlen_bin=args.projlen_bin,
                )
        now = time.time()
        if now - last_progress >= args.progress_interval_seconds:
            progress = {
                "phase": "frontier_load",
                "frontier_loaded": frontier_loaded,
                "heuristic_beam_sizes": {heuristic: len(items) for heuristic, items in beams.items()},
                "total_beam_size": sum(len(items) for items in beams.values()),
                "exact_evaluations": exact_evaluations,
                "best_projlen": best_projlen,
                "best_identity_defect": best_identity_defect,
                "best_target_defect": best_target_defect,
                "elapsed_seconds": round(now - start_time, 2),
            }
            ledger.progress(progress)
            print(json.dumps(progress, sort_keys=True), flush=True)
            last_progress = now
    for heuristic in heuristics:
        beams[heuristic] = _select_population(
            beams[heuristic],
            beam_size=initial_beam_size,
            per_finite_key_cap=args.per_finite_key_cap,
            diversity_bucket_cap=args.diversity_bucket_cap,
            target_defect_bin=args.target_defect_bin,
            projlen_bin=args.projlen_bin,
        )
    frontier_summary = {
        "frontier_path": args.frontier_path,
        "frontier_loaded": frontier_loaded,
        "frontier_shard_by": args.frontier_shard_by,
        "frontier_shard_count": args.frontier_shard_count,
        "frontier_shard_index": args.frontier_shard_index,
        "frontier_max_records": args.frontier_max_records,
        "heuristics": list(heuristics),
        "initial_population_size_per_heuristic": {heuristic: len(items) for heuristic, items in beams.items()},
        "initial_population_size_total": sum(len(items) for items in beams.values()),
        "elapsed_seconds": round(time.time() - frontier_start, 2),
    }
    write_json(output_dir / "frontier_summary.json", frontier_summary)
    print(json.dumps({"phase": "frontier_population_ready", **frontier_summary}, sort_keys=True), flush=True)

    for total_length in range(args.frontier_length + 1, args.target_length + 1):
        next_beams: dict[str, list[SearchState]] = {heuristic: [] for heuristic in heuristics}
        depth_expansions = 0
        depth_checked = 0
        depth_target_matches = 0
        depth_scalar_identities = 0
        for heuristic in heuristics:
            for state in beams[heuristic]:
                legal = list(env.legal_next(state.factors))
                if args.max_actions_per_state > 0 and args.max_actions_per_state < len(legal):
                    legal = rng.sample(legal, args.max_actions_per_state)
                for action in legal:
                    child_factors = state.factors + (int(action),)
                    child_exact = env.exact_append(state.exact, int(action))
                    child_finite = env.finite_append(state.finite, int(action))
                    metrics, by_target = _best_target_metrics(env, child_exact, targets)
                    exact_evaluations += 1
                    symbolic_factor_multiplications += 1
                    expanded_states += 1
                    depth_expansions += 1
                    update_best_from_metrics(
                        factors=child_factors,
                        metrics=metrics,
                        kind="frontier_beam_prefix",
                        extra={"total_length": total_length, "heuristic": heuristic},
                    )
                    if total_length in check_lengths:
                        before_checked = checked_states
                        before_matches = target_match_candidates
                        before_scalar = scalar_identity_candidates
                        check_target_hits(
                            factors=child_factors,
                            image=child_exact,
                            by_target=by_target,
                            source=f"expanded_child:{heuristic}",
                        )
                        depth_checked += checked_states - before_checked
                        depth_target_matches += target_match_candidates - before_matches
                        depth_scalar_identities += scalar_identity_candidates - before_scalar
                    next_beams[heuristic].append(
                        SearchState(
                            factors=child_factors,
                            finite=child_finite,
                            finite_key=env.finite_key(child_finite),
                            exact=child_exact,
                            metrics=metrics,
                            score=_priority(
                                metrics,
                                heuristic=heuristic,
                                rng=rng,
                                selection_temperature=args.selection_temperature,
                                identity_weight=args.identity_weight,
                                projlen_weight=args.projlen_weight,
                            ),
                        )
                    )
                    if len(next_beams[heuristic]) >= step_buffer_limit:
                        next_beams[heuristic] = _select_population(
                            next_beams[heuristic],
                            beam_size=args.beam_size,
                            per_finite_key_cap=args.per_finite_key_cap,
                            diversity_bucket_cap=args.diversity_bucket_cap,
                            target_defect_bin=args.target_defect_bin,
                            projlen_bin=args.projlen_bin,
                        )
                    if args.stop_after_scalar_identity and scalar_identity_candidates:
                        break
                if args.stop_after_scalar_identity and scalar_identity_candidates:
                    break
            next_beams[heuristic] = _select_population(
                next_beams[heuristic],
                beam_size=args.beam_size,
                per_finite_key_cap=args.per_finite_key_cap,
                diversity_bucket_cap=args.diversity_bucket_cap,
                target_defect_bin=args.target_defect_bin,
                projlen_bin=args.projlen_bin,
            )
            if args.stop_after_scalar_identity and scalar_identity_candidates:
                break
        beams = next_beams
        all_live_states = [state for states in beams.values() for state in states]
        beam_best = min((int(state.metrics["projlen"]) for state in all_live_states), default=None)
        beam_best_defect = min((int(state.metrics["identity_defect"]) for state in all_live_states), default=None)
        beam_best_target_defect = min((int(state.metrics["target_defect"]) for state in all_live_states), default=None)
        progress = {
            "phase": "depth_done",
            "total_length": total_length,
            "heuristic_beam_sizes": {heuristic: len(items) for heuristic, items in beams.items()},
            "total_beam_size": sum(len(items) for items in beams.values()),
            "depth_expansions": depth_expansions,
            "expanded_states": expanded_states,
            "depth_checked_states": depth_checked,
            "checked_states": checked_states,
            "best_projlen": best_projlen,
            "best_identity_defect": best_identity_defect,
            "best_target_defect": best_target_defect,
            "beam_best_projlen": beam_best,
            "beam_best_identity_defect": beam_best_defect,
            "beam_best_target_defect": beam_best_target_defect,
            "depth_scalar_identities": depth_scalar_identities,
            "depth_target_matches": depth_target_matches,
            "scalar_identity_candidates": scalar_identity_candidates,
            "target_match_candidates": target_match_candidates,
            "exact_evaluations": exact_evaluations,
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        ledger.progress(progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
        if not all_live_states:
            break
        if args.stop_after_scalar_identity and scalar_identity_candidates:
            break

    elapsed = time.time() - start_time
    length_range = {
        "frontier_length": args.frontier_length,
        "target_length": args.target_length,
        "check_lengths": list(check_lengths),
        "initial_beam_size": initial_beam_size,
        "beam_size": args.beam_size,
        "heuristics": list(heuristics),
    }
    summary = {
        "format": "braidzero-frontier-beam-summary-v1",
        "status": "clean",
        "method": "braidzero_exhaustive_frontier_population_beam",
        "prime": args.p,
        "representation": env.representation_label,
        "seed": args.seed,
        "length_range": length_range,
        "t_values": list(t_values),
        "completion_targets": list(targets),
        "frontier": frontier_summary,
        "search": {
            "elapsed_seconds": round(elapsed, 2),
            "frontier_loaded": frontier_loaded,
            "expanded_states": expanded_states,
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
            "final_beam_size_per_heuristic": {heuristic: len(items) for heuristic, items in beams.items()},
            "final_beam_size_total": sum(len(items) for items in beams.values()),
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
        description="Population beam growth from an exhaustive BraidZero frontier; no collision oracle."
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
    parser.add_argument(
        "--heuristics",
        default="target,identity,projlen,scalar_shape,random",
        help=(
            "Comma-separated ensemble beams. Valid: target, identity, projlen, "
            "scalar_shape, terms, random, identity_target, delta_target."
        ),
    )
    parser.add_argument("--initial-beam-size", type=int, default=0)
    parser.add_argument("--beam-size", type=int, default=50_000)
    parser.add_argument("--beam-buffer-factor", type=int, default=4)
    parser.add_argument("--beam-buffer-min", type=int, default=100_000)
    parser.add_argument("--max-actions-per-state", type=int, default=0)
    parser.add_argument("--per-finite-key-cap", type=int, default=8)
    parser.add_argument("--diversity-bucket-cap", type=int, default=64)
    parser.add_argument("--target-defect-bin", type=int, default=16)
    parser.add_argument("--projlen-bin", type=int, default=8)
    parser.add_argument("--selection-temperature", type=float, default=25.0)
    parser.add_argument("--identity-weight", type=float, default=0.25)
    parser.add_argument("--projlen-weight", type=float, default=0.05)
    parser.add_argument("--completion-targets", default="identity,delta")
    parser.add_argument("--stop-after-scalar-identity", action="store_true")
    parser.add_argument("--progress-interval-seconds", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_frontier_beam(args)


if __name__ == "__main__":
    main()
