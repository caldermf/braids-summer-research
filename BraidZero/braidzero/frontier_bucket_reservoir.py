from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
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


def _scalar_shape_score(metrics: dict) -> int:
    return (
        int(metrics.get("off_diagonal_terms", 0))
        + int(metrics.get("diagonal_mismatch_terms", 0))
        + int(metrics.get("scalar_extra_degrees", 0))
    )


def _bucket_key(metrics: dict, heuristic: str, *, length: int) -> tuple:
    if heuristic == "target":
        return (int(length), int(metrics["target_defect"]), str(metrics.get("best_target_label", metrics.get("target_label", ""))))
    if heuristic == "identity":
        return (int(length), int(metrics["identity_defect"]))
    if heuristic == "projlen":
        return (int(length), int(metrics["projlen"]))
    if heuristic == "scalar_shape":
        return (int(length), _scalar_shape_score(metrics))
    if heuristic == "terms":
        return (int(length), int(metrics.get("nonzero_terms", 0)))
    if heuristic == "identity_target":
        return (int(length), int(metrics.get("identity_target_defect", metrics["target_defect"])))
    if heuristic == "delta_target":
        return (int(length), int(metrics.get("delta_target_defect", metrics["target_defect"])))
    if heuristic == "random":
        return (int(length), 0)
    raise ValueError(f"unknown heuristic: {heuristic}")


def _bucket_sort_key(key: tuple) -> tuple:
    return key


def _jsonable_key(key: tuple) -> list:
    return [int(item) if isinstance(item, (int, np.integer)) else str(item) for item in key]


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
        "identity_target_defect": int(by_target["identity"]["target_defect"])
        if "identity" in by_target
        else int(best_metrics["target_defect"]),
        "delta_target_defect": int(by_target["delta"]["target_defect"])
        if "delta" in by_target
        else int(best_metrics["target_defect"]),
    }, by_target


def _candidate_sort_tuple(row: dict) -> tuple[int, int, int, int]:
    metrics = row["metrics"]
    return (
        int(metrics["target_defect"]),
        int(metrics["identity_defect"]),
        int(metrics["projlen"]),
        int(row["length"]),
    )


@dataclass
class ReservoirBucket:
    key: tuple
    capacity: int
    rng: random.Random
    states: list[SearchState] = field(default_factory=list)
    seen: int = 0

    def insert(self, state: SearchState) -> None:
        self.seen += 1
        if len(self.states) < self.capacity:
            self.states.append(state)
            return
        replacement = self.rng.randint(1, self.seen)
        if replacement <= self.capacity:
            self.states[replacement - 1] = state


class ReservoirPopulation:
    def __init__(
        self,
        *,
        heuristic: str,
        length: int,
        bucket_size: int,
        random_bucket_size: int,
        rng: random.Random,
    ):
        self.heuristic = heuristic
        self.length = int(length)
        self.bucket_size = int(bucket_size)
        self.random_bucket_size = int(random_bucket_size)
        self.rng = rng
        self.buckets: dict[tuple, ReservoirBucket] = {}

    def _capacity_for(self, key: tuple) -> int:
        if self.heuristic == "random" and self.random_bucket_size > 0:
            return self.random_bucket_size
        return self.bucket_size

    def insert(self, state: SearchState) -> None:
        key = _bucket_key(state.metrics, self.heuristic, length=self.length)
        bucket = self.buckets.get(key)
        if bucket is None:
            bucket = ReservoirBucket(key=key, capacity=self._capacity_for(key), rng=self.rng)
            self.buckets[key] = bucket
        bucket.insert(state)

    def size(self) -> int:
        return sum(len(bucket.states) for bucket in self.buckets.values())

    def bucket_count(self) -> int:
        return len(self.buckets)

    def bucket_summary(self, *, limit: int = 8) -> dict:
        ordered = sorted(self.buckets.values(), key=lambda bucket: _bucket_sort_key(bucket.key))
        return {
            "buckets": len(ordered),
            "states": self.size(),
            "best_keys": [
                {
                    "key": _jsonable_key(bucket.key),
                    "states": len(bucket.states),
                    "seen": bucket.seen,
                }
                for bucket in ordered[:limit]
            ],
        }

    def select_parents(self, *, use_best: int) -> tuple[list[SearchState], dict]:
        ordered = sorted(self.buckets.values(), key=lambda bucket: _bucket_sort_key(bucket.key))
        selected: list[SearchState] = []
        selected_bucket_rows: list[dict] = []
        truncated_bucket: dict | None = None
        limit = int(use_best)

        for bucket in ordered:
            if limit > 0 and len(selected) >= limit:
                break
            remaining = limit - len(selected) if limit > 0 else len(bucket.states)
            take_count = len(bucket.states) if limit <= 0 else min(len(bucket.states), remaining)
            if take_count <= 0:
                break
            if take_count == len(bucket.states):
                chosen = list(bucket.states)
                truncated = False
            else:
                chosen = self.rng.sample(bucket.states, take_count)
                truncated = True
            selected.extend(chosen)
            row = {
                "key": _jsonable_key(bucket.key),
                "bucket_states": len(bucket.states),
                "bucket_seen": bucket.seen,
                "selected": take_count,
                "truncated": truncated,
            }
            selected_bucket_rows.append(row)
            if truncated:
                truncated_bucket = row
                break

        return selected, {
            "heuristic": self.heuristic,
            "length": self.length,
            "population_states": self.size(),
            "population_buckets": len(ordered),
            "selected_states": len(selected),
            "selected_buckets": len(selected_bucket_rows),
            "first_selected_bucket": selected_bucket_rows[0] if selected_bucket_rows else None,
            "last_selected_bucket": selected_bucket_rows[-1] if selected_bucket_rows else None,
            "truncated_bucket": truncated_bucket,
        }


def _make_populations(
    *,
    heuristics: tuple[str, ...],
    length: int,
    bucket_size: int,
    random_bucket_size: int,
    use_best: int,
    rng: random.Random,
) -> dict[str, ReservoirPopulation]:
    actual_random_bucket_size = random_bucket_size if random_bucket_size > 0 else max(1, use_best)
    return {
        heuristic: ReservoirPopulation(
            heuristic=heuristic,
            length=length,
            bucket_size=bucket_size,
            random_bucket_size=actual_random_bucket_size,
            rng=rng,
        )
        for heuristic in heuristics
    }


def run_frontier_bucket_reservoir(args: argparse.Namespace) -> dict:
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
            "mode": "frontier_bucketed_reservoir_ensemble",
            "uses_collision_oracle": False,
            "uses_suffix_bank": False,
            "frontier_path": args.frontier_path,
            "frontier_shard_by": args.frontier_shard_by,
            "frontier_shard_count": args.frontier_shard_count,
            "frontier_shard_index": args.frontier_shard_index,
            "bucket_size": args.bucket_size,
            "use_best": args.use_best,
        },
    )

    populations = _make_populations(
        heuristics=heuristics,
        length=args.frontier_length,
        bucket_size=args.bucket_size,
        random_bucket_size=args.random_bucket_size,
        use_best=args.use_best,
        rng=rng,
    )

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
        prefix_rank = (int(metrics["identity_defect"]), int(metrics["projlen"]), int(row["length"]))
        if best_prefix_candidate is None or prefix_rank < (
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
    ) -> tuple[int, int, int]:
        nonlocal checked_states, scalar_identity_candidates, target_match_candidates, best_scalar_identity_candidate
        digest = word_digest(0, factors)
        local_checked = 0
        local_scalar = 0
        local_target = 0
        per_length_counts.setdefault(len(factors), {"checked": 0, "target_matches": 0, "scalar_identities": 0})
        for target_label, metrics in by_target.items():
            check_key = (target_label, digest)
            if check_key in seen_target_checks:
                continue
            seen_target_checks.add(check_key)
            checked_states += 1
            local_checked += 1
            per_length_counts[len(factors)]["checked"] += 1
            row = compact_candidate_row(
                kind="frontier_bucket_reservoir_target_check",
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
                local_scalar += 1
                per_length_counts[len(factors)]["scalar_identities"] += 1
                best_scalar_identity_candidate = row
                ledger.candidate(row)
                print(json.dumps({"phase": "exact_scalar_identity", **row}, sort_keys=True), flush=True)
            if metrics.get("target_match"):
                target_match_candidates += 1
                local_target += 1
                per_length_counts[len(factors)]["target_matches"] += 1
                ledger.candidate(row)
                print(json.dumps({"phase": "exact_target_match", **row}, sort_keys=True), flush=True)
        return local_checked, local_scalar, local_target

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
            kind="frontier_bucket_reservoir_prefix",
            extra={"frontier_record_id": record.record_id, "total_length": args.frontier_length},
        )
        if record.length in check_lengths:
            check_target_hits(
                factors=record.factors,
                image=exact,
                by_target=by_target,
                source="frontier",
            )
        finite_key = env.finite_key(finite)
        state = SearchState(
            factors=record.factors,
            finite=finite,
            finite_key=finite_key,
            exact=exact,
            metrics=metrics,
            score=0.0,
        )
        for population in populations.values():
            population.insert(state)
        now = time.time()
        if now - last_progress >= args.progress_interval_seconds:
            progress = {
                "phase": "frontier_load",
                "frontier_loaded": frontier_loaded,
                "heuristic_population_sizes": {
                    heuristic: population.size() for heuristic, population in populations.items()
                },
                "heuristic_bucket_counts": {
                    heuristic: population.bucket_count() for heuristic, population in populations.items()
                },
                "total_population_size": sum(population.size() for population in populations.values()),
                "exact_evaluations": exact_evaluations,
                "best_projlen": best_projlen,
                "best_identity_defect": best_identity_defect,
                "best_target_defect": best_target_defect,
                "elapsed_seconds": round(now - start_time, 2),
            }
            ledger.progress(progress)
            print(json.dumps(progress, sort_keys=True), flush=True)
            last_progress = now

    frontier_summary = {
        "frontier_path": args.frontier_path,
        "frontier_loaded": frontier_loaded,
        "frontier_shard_by": args.frontier_shard_by,
        "frontier_shard_count": args.frontier_shard_count,
        "frontier_shard_index": args.frontier_shard_index,
        "frontier_max_records": args.frontier_max_records,
        "heuristics": list(heuristics),
        "bucket_size": args.bucket_size,
        "use_best": args.use_best,
        "population_size_per_heuristic": {heuristic: population.size() for heuristic, population in populations.items()},
        "bucket_count_per_heuristic": {heuristic: population.bucket_count() for heuristic, population in populations.items()},
        "bucket_summaries": {heuristic: population.bucket_summary() for heuristic, population in populations.items()},
        "elapsed_seconds": round(time.time() - frontier_start, 2),
    }
    write_json(output_dir / "frontier_summary.json", frontier_summary)
    print(json.dumps({"phase": "frontier_bucket_reservoir_ready", **frontier_summary}, sort_keys=True), flush=True)

    for total_length in range(args.frontier_length + 1, args.target_length + 1):
        next_populations = _make_populations(
            heuristics=heuristics,
            length=total_length,
            bucket_size=args.bucket_size,
            random_bucket_size=args.random_bucket_size,
            use_best=args.use_best,
            rng=rng,
        )
        depth_expansions = 0
        depth_checked = 0
        depth_target_matches = 0
        depth_scalar_identities = 0
        depth_selected: dict[str, dict] = {}

        for heuristic in heuristics:
            parents, selection_summary = populations[heuristic].select_parents(use_best=args.use_best)
            depth_selected[heuristic] = selection_summary
            for state in parents:
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
                        kind="frontier_bucket_reservoir_prefix",
                        extra={"total_length": total_length, "heuristic": heuristic},
                    )
                    if total_length in check_lengths:
                        local_checked, local_scalar, local_target = check_target_hits(
                            factors=child_factors,
                            image=child_exact,
                            by_target=by_target,
                            source=f"expanded_child:{heuristic}",
                        )
                        depth_checked += local_checked
                        depth_scalar_identities += local_scalar
                        depth_target_matches += local_target
                    next_populations[heuristic].insert(
                        SearchState(
                            factors=child_factors,
                            finite=child_finite,
                            finite_key=env.finite_key(child_finite),
                            exact=child_exact,
                            metrics=metrics,
                            score=0.0,
                        )
                    )
                    now = time.time()
                    if now - last_progress >= args.progress_interval_seconds:
                        progress = {
                            "phase": "depth_expand",
                            "total_length": total_length,
                            "heuristic": heuristic,
                            "depth_expansions": depth_expansions,
                            "expanded_states": expanded_states,
                            "depth_checked_states": depth_checked,
                            "checked_states": checked_states,
                            "next_population_size_for_heuristic": next_populations[heuristic].size(),
                            "next_bucket_count_for_heuristic": next_populations[heuristic].bucket_count(),
                            "best_projlen": best_projlen,
                            "best_identity_defect": best_identity_defect,
                            "best_target_defect": best_target_defect,
                            "scalar_identity_candidates": scalar_identity_candidates,
                            "target_match_candidates": target_match_candidates,
                            "exact_evaluations": exact_evaluations,
                            "elapsed_seconds": round(now - start_time, 2),
                        }
                        ledger.progress(progress)
                        print(json.dumps(progress, sort_keys=True), flush=True)
                        last_progress = now
                    if args.stop_after_scalar_identity and scalar_identity_candidates:
                        break
                if args.stop_after_scalar_identity and scalar_identity_candidates:
                    break
            if args.stop_after_scalar_identity and scalar_identity_candidates:
                break

        populations = next_populations
        progress = {
            "phase": "depth_done",
            "total_length": total_length,
            "heuristic_population_sizes": {
                heuristic: population.size() for heuristic, population in populations.items()
            },
            "heuristic_bucket_counts": {
                heuristic: population.bucket_count() for heuristic, population in populations.items()
            },
            "selected_parent_summaries": depth_selected,
            "bucket_summaries": {heuristic: population.bucket_summary(limit=4) for heuristic, population in populations.items()},
            "total_population_size": sum(population.size() for population in populations.values()),
            "depth_expansions": depth_expansions,
            "expanded_states": expanded_states,
            "depth_checked_states": depth_checked,
            "checked_states": checked_states,
            "best_projlen": best_projlen,
            "best_identity_defect": best_identity_defect,
            "best_target_defect": best_target_defect,
            "depth_scalar_identities": depth_scalar_identities,
            "depth_target_matches": depth_target_matches,
            "scalar_identity_candidates": scalar_identity_candidates,
            "target_match_candidates": target_match_candidates,
            "exact_evaluations": exact_evaluations,
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        ledger.progress(progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
        if sum(population.size() for population in populations.values()) == 0:
            break
        if args.stop_after_scalar_identity and scalar_identity_candidates:
            break

    elapsed = time.time() - start_time
    length_range = {
        "frontier_length": args.frontier_length,
        "target_length": args.target_length,
        "check_lengths": list(check_lengths),
        "bucket_size": args.bucket_size,
        "use_best": args.use_best,
        "heuristics": list(heuristics),
    }
    summary = {
        "format": "braidzero-frontier-bucket-reservoir-summary-v1",
        "status": "clean",
        "method": "braidzero_exhaustive_frontier_bucketed_reservoir_ensemble",
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
            "final_population_size_per_heuristic": {
                heuristic: population.size() for heuristic, population in populations.items()
            },
            "final_bucket_count_per_heuristic": {
                heuristic: population.bucket_count() for heuristic, population in populations.items()
            },
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
        description="Bucketed-reservoir ensemble growth from an exhaustive BraidZero frontier."
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
            "Comma-separated independent reservoir populations. Valid: target, identity, "
            "projlen, scalar_shape, terms, random, identity_target, delta_target."
        ),
    )
    parser.add_argument("--bucket-size", type=int, default=3000)
    parser.add_argument("--use-best", type=int, default=50_000)
    parser.add_argument(
        "--random-bucket-size",
        type=int,
        default=0,
        help="Capacity for the random heuristic's single bucket; default is --use-best.",
    )
    parser.add_argument("--max-actions-per-state", type=int, default=0)
    parser.add_argument("--completion-targets", default="identity,delta")
    parser.add_argument("--stop-after-scalar-identity", action="store_true")
    parser.add_argument("--progress-interval-seconds", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bucket_size <= 0:
        raise ValueError("--bucket-size must be positive")
    if args.use_best <= 0:
        raise ValueError("--use-best must be positive")
    run_frontier_bucket_reservoir(args)


if __name__ == "__main__":
    main()
