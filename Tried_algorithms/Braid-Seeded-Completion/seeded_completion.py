#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_GPT_ROOT = REPO_ROOT / "Braid-Matrix-GPT"
DEFAULT_BRAID_GPT_ROOT = REPO_ROOT / "Braid-GPT"
DEFAULT_AUTHOR_REPO = REPO_ROOT / "hybrid_of_reservoir_crispr_mcts_suffix" / "third_party" / "braids_project"


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_matrix_gpt(matrix_gpt_root: Path):
    return load_module("matrix_gpt_runtime_for_seeded_completion", matrix_gpt_root / "matrix_gpt.py")


def read_json(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_int_list(value: str) -> tuple[int, ...]:
    output = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not output:
        raise ValueError("expected at least one integer")
    return output


def parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    allowed = {"right", "left", "both"}
    bad = set(modes) - allowed
    if bad:
        raise ValueError(f"unknown modes: {sorted(bad)}")
    if not modes:
        raise ValueError("expected at least one mode")
    return modes


def row_factor_power(row: dict) -> tuple[int, tuple[int, ...]] | None:
    for key in ("factor_ids", "final_factors", "powered_factors", "factors"):
        if key not in row:
            continue
        try:
            factors = tuple(int(value) for value in row[key])
            power = int(row.get("power", row.get("final_power", row.get("powered_power", 0))))
        except (TypeError, ValueError):
            return None
        return power, factors
    return None


def walk_factor_rows(obj) -> Iterable[dict]:
    if isinstance(obj, dict):
        parsed = row_factor_power(obj)
        if parsed is not None:
            power, factors = parsed
            yield {
                "power": power,
                "factor_ids": list(factors),
                "metrics": obj.get("metrics", obj.get("exact_metrics", {})),
                "objective": obj.get("objective"),
                "source": obj.get("source"),
                "rank": obj.get("rank"),
            }
        for value in obj.values():
            yield from walk_factor_rows(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_factor_rows(value)


def metric_projlen(metrics: dict) -> int:
    return int(metrics.get("projlen", metrics.get("projective_width", 10**9)))


@dataclass(frozen=True)
class SeedCandidate:
    seed_id: int
    power: int
    factors: tuple[int, ...]
    source: str
    source_rank: int | None
    metrics: dict
    objective: float | None


def rank_seed_tuple(row: SeedCandidate) -> tuple[float, int, int, int]:
    objective = float(row.objective) if row.objective is not None else float(row.metrics.get("identity_defect", 10**9))
    return (
        objective,
        int(row.metrics.get("identity_defect", 10**9)),
        metric_projlen(row.metrics),
        len(row.factors),
    )


def load_seed_candidates(
    paths: Sequence[str],
    *,
    automaton,
    min_length: int,
    max_length: int,
    limit: int,
    reject_illegal: bool,
) -> list[SeedCandidate]:
    rows: list[SeedCandidate] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            print(json.dumps({"phase": "missing_seed_source", "path": str(path)}), flush=True)
            continue
        if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"):
            iterator = iter_jsonl(path)
        else:
            iterator = walk_factor_rows(read_json(path))
        for row in iterator:
            parsed = row_factor_power(row)
            if parsed is None:
                continue
            power, factors = parsed
            if len(factors) < min_length or len(factors) > max_length:
                continue
            if reject_illegal and not automaton.is_legal(factors):
                continue
            key = (int(power) % 2, factors)
            if key in seen:
                continue
            seen.add(key)
            objective = row.get("objective")
            source_rank = row.get("rank")
            try:
                source_rank_int = int(source_rank) if source_rank is not None else None
            except (TypeError, ValueError):
                source_rank_int = None
            rows.append(
                SeedCandidate(
                    seed_id=len(rows),
                    power=int(power),
                    factors=factors,
                    source=str(row.get("source", path)),
                    source_rank=source_rank_int,
                    metrics=dict(row.get("metrics", row.get("exact_metrics", {})) or {}),
                    objective=float(objective) if objective is not None else None,
                )
            )
    rows.sort(key=rank_seed_tuple)
    return rows[:limit] if limit else rows


def make_evaluator(args: argparse.Namespace, mgpt, bgpt):
    return mgpt.MatrixEvaluator(
        bgpt=bgpt,
        author_repo=Path(args.author_repo),
        p=args.p,
        n=args.n,
        r=args.r,
        matrix_max_degree=args.matrix_max_degree,
        identity_weight=args.identity_weight,
        projlen_weight=args.projlen_weight,
        identity_density_weight=args.identity_density_weight,
        projlen_density_weight=args.projlen_density_weight,
        degeneracy_weight=args.degeneracy_weight,
        min_length=args.min_score_length,
        kernel_bonus=args.kernel_bonus,
    )


def sample_bridge_or_none(
    automaton,
    left: int | None,
    right: int | None,
    length: int,
    rng: random.Random,
    *,
    attempts: int,
) -> tuple[int, ...] | None:
    for _ in range(attempts):
        try:
            return tuple(int(x) for x in automaton.sample_bridge(left, right, length, rng))
        except ValueError:
            continue
    return None


def add_completion(
    proposals: list[dict],
    seen: set[tuple[int, ...]],
    *,
    automaton,
    core: SeedCandidate,
    mode: str,
    left_factors: tuple[int, ...],
    right_factors: tuple[int, ...],
    min_final_length: int,
    max_final_length: int,
) -> None:
    child = left_factors + core.factors + right_factors
    if len(child) < min_final_length or len(child) > max_final_length:
        return
    if child == core.factors or child in seen:
        return
    if not automaton.is_legal(child):
        return
    seen.add(child)
    proposals.append(
        {
            "core_id": core.seed_id,
            "core_power": core.power,
            "core_factors": list(core.factors),
            "core_length": len(core.factors),
            "core_source": core.source,
            "core_source_rank": core.source_rank,
            "core_metrics": core.metrics,
            "core_objective": core.objective,
            "mode": mode,
            "left_factors": list(left_factors),
            "right_factors": list(right_factors),
            "factor_ids": list(child),
            "length": len(child),
        }
    )


def generate_completions(
    *,
    automaton,
    core: SeedCandidate,
    rng: random.Random,
    modes: Sequence[str],
    right_lengths: Sequence[int],
    left_lengths: Sequence[int],
    right_samples_per_length: int,
    left_samples_per_length: int,
    both_pairs_per_core: int,
    bridge_attempts: int,
    min_final_length: int,
    max_final_length: int,
) -> list[dict]:
    if not core.factors:
        return []
    proposals: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    first = int(core.factors[0])
    last = int(core.factors[-1])

    if "right" in modes:
        for length in right_lengths:
            for _ in range(right_samples_per_length):
                suffix = sample_bridge_or_none(automaton, last, None, int(length), rng, attempts=bridge_attempts)
                if suffix is not None:
                    add_completion(
                        proposals,
                        seen,
                        automaton=automaton,
                        core=core,
                        mode="right",
                        left_factors=(),
                        right_factors=suffix,
                        min_final_length=min_final_length,
                        max_final_length=max_final_length,
                    )

    if "left" in modes:
        for length in left_lengths:
            for _ in range(left_samples_per_length):
                prefix = sample_bridge_or_none(automaton, None, first, int(length), rng, attempts=bridge_attempts)
                if prefix is not None:
                    add_completion(
                        proposals,
                        seen,
                        automaton=automaton,
                        core=core,
                        mode="left",
                        left_factors=prefix,
                        right_factors=(),
                        min_final_length=min_final_length,
                        max_final_length=max_final_length,
                    )

    if "both" in modes:
        for _ in range(both_pairs_per_core):
            left_length = int(rng.choice(tuple(left_lengths)))
            right_length = int(rng.choice(tuple(right_lengths)))
            prefix = sample_bridge_or_none(automaton, None, first, left_length, rng, attempts=bridge_attempts)
            suffix = sample_bridge_or_none(automaton, last, None, right_length, rng, attempts=bridge_attempts)
            if prefix is not None and suffix is not None:
                add_completion(
                    proposals,
                    seen,
                    automaton=automaton,
                    core=core,
                    mode="both",
                    left_factors=prefix,
                    right_factors=suffix,
                    min_final_length=min_final_length,
                    max_final_length=max_final_length,
                )

    return proposals


def record_key(row: dict) -> tuple[int, tuple[int, ...]]:
    return int(row.get("power", 0)) % 2, tuple(int(x) for x in row["factor_ids"])


def row_score(row: dict) -> tuple[float, int, int, int]:
    metrics = row.get("metrics", {})
    return (
        float(row.get("objective", 10**18)),
        int(metrics.get("identity_defect", 10**9)),
        metric_projlen(metrics),
        int(row.get("length", len(row.get("factor_ids", ())))),
    )


def identity_score(row: dict) -> tuple[int, float, int, int]:
    metrics = row.get("metrics", {})
    return (
        int(metrics.get("identity_defect", 10**9)),
        float(row.get("objective", 10**18)),
        metric_projlen(metrics),
        int(row.get("length", len(row.get("factor_ids", ())))),
    )


def projlen_score(row: dict) -> tuple[int, int, float, int]:
    metrics = row.get("metrics", {})
    return (
        metric_projlen(metrics),
        int(metrics.get("identity_defect", 10**9)),
        float(row.get("objective", 10**18)),
        int(row.get("length", len(row.get("factor_ids", ())))),
    )


def merge_unique(target: list[dict], selected: Iterable[dict], seen: set[tuple[int, tuple[int, ...]]]) -> None:
    for row in selected:
        key = record_key(row)
        if key in seen:
            continue
        seen.add(key)
        target.append(row)


def trim_survivors(
    rows: Sequence[dict],
    *,
    rng: random.Random,
    keep_best: int,
    keep_identity: int,
    keep_projlen: int,
    keep_random: int,
) -> list[dict]:
    if not rows:
        return []
    selected: list[dict] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    merge_unique(selected, sorted(rows, key=row_score)[:keep_best], seen)
    merge_unique(selected, sorted(rows, key=identity_score)[:keep_identity], seen)
    merge_unique(selected, sorted(rows, key=projlen_score)[:keep_projlen], seen)
    if keep_random > 0:
        pool = [row for row in rows if record_key(row) not in seen]
        if len(pool) > keep_random:
            pool = rng.sample(pool, keep_random)
        merge_unique(selected, pool, seen)
    selected.sort(key=row_score)
    return selected


def result_row(proposal: dict, evaluated) -> dict:
    metrics = dict(evaluated.metrics)
    core_metrics = proposal.get("core_metrics") or {}
    identity_delta = None
    projlen_delta = None
    if "identity_defect" in core_metrics:
        identity_delta = int(metrics.get("identity_defect", 10**9)) - int(core_metrics.get("identity_defect", 10**9))
    if "projlen" in core_metrics or "projective_width" in core_metrics:
        projlen_delta = metric_projlen(metrics) - metric_projlen(core_metrics)
    return {
        "power": int(evaluated.power),
        "factor_ids": list(evaluated.factors),
        "length": len(evaluated.factors),
        "metrics": metrics,
        "objective": float(evaluated.objective),
        "matrix_width": int(evaluated.matrix_width),
        "mode": proposal["mode"],
        "left_factors": proposal["left_factors"],
        "right_factors": proposal["right_factors"],
        "core_id": int(proposal["core_id"]),
        "core_power": int(proposal["core_power"]),
        "core_factors": proposal["core_factors"],
        "core_length": int(proposal["core_length"]),
        "core_source": proposal["core_source"],
        "core_source_rank": proposal["core_source_rank"],
        "core_metrics": core_metrics,
        "core_objective": proposal["core_objective"],
        "identity_delta": identity_delta,
        "projlen_delta": projlen_delta,
    }


def objective_metadata(args: argparse.Namespace) -> dict:
    return {
        "identity_weight": args.identity_weight,
        "projlen_weight": args.projlen_weight,
        "identity_density_weight": args.identity_density_weight,
        "projlen_density_weight": args.projlen_density_weight,
        "degeneracy_weight": args.degeneracy_weight,
        "min_score_length": args.min_score_length,
        "kernel_bonus": args.kernel_bonus,
    }


def run(args: argparse.Namespace) -> None:
    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    completions_path = output_dir / "completions.jsonl"
    kernel_path = output_dir / "kernel_hits.jsonl"
    for path in (progress_path, completions_path, kernel_path):
        path.write_text("", encoding="utf-8")

    mgpt = load_matrix_gpt(Path(args.matrix_gpt_root))
    bgpt = mgpt.load_braid_gpt_module(Path(args.braid_gpt_root))
    automaton = bgpt.GNFAutomaton(args.n)
    evaluator = make_evaluator(args, mgpt, bgpt)
    rng = random.Random(args.seed)

    modes = parse_modes(args.modes)
    right_lengths = parse_int_list(args.right_lengths)
    left_lengths = parse_int_list(args.left_lengths)
    seeds = load_seed_candidates(
        args.seed_source,
        automaton=automaton,
        min_length=args.min_core_length,
        max_length=args.max_core_length,
        limit=args.candidate_limit,
        reject_illegal=not args.allow_illegal_cores,
    )
    if not seeds:
        raise RuntimeError("No seed candidates loaded")

    metadata = {
        "format": "braid-seeded-completion-v1",
        "p": args.p,
        "n": args.n,
        "r": args.r,
        "seed": args.seed,
        "seed_sources": args.seed_source,
        "selected_cores": len(seeds),
        "candidate_limit": args.candidate_limit,
        "min_core_length": args.min_core_length,
        "max_core_length": args.max_core_length,
        "min_final_length": args.min_final_length,
        "max_final_length": args.max_final_length,
        "modes": list(modes),
        "right_lengths": list(right_lengths),
        "left_lengths": list(left_lengths),
        "right_samples_per_length": args.right_samples_per_length,
        "left_samples_per_length": args.left_samples_per_length,
        "both_pairs_per_core": args.both_pairs_per_core,
        "keep_best": args.keep_best,
        "keep_identity": args.keep_identity,
        "keep_projlen": args.keep_projlen,
        "keep_random": args.keep_random,
        "objective": objective_metadata(args),
    }
    print(json.dumps({"phase": "setup", **metadata}, sort_keys=True), flush=True)
    append_jsonl(progress_path, [{"phase": "setup", **metadata}])

    survivors: list[dict] = []
    kernel_hits: list[dict] = []
    generated = 0
    evaluated_count = 0
    mode_counts: dict[str, int] = {"right": 0, "left": 0, "both": 0}
    trim_limit = max(1000, 3 * (args.keep_best + args.keep_identity + args.keep_projlen + args.keep_random))

    for index, core in enumerate(seeds, start=1):
        proposals = generate_completions(
            automaton=automaton,
            core=core,
            rng=rng,
            modes=modes,
            right_lengths=right_lengths,
            left_lengths=left_lengths,
            right_samples_per_length=args.right_samples_per_length,
            left_samples_per_length=args.left_samples_per_length,
            both_pairs_per_core=args.both_pairs_per_core,
            bridge_attempts=args.bridge_attempts,
            min_final_length=args.min_final_length,
            max_final_length=args.max_final_length,
        )
        generated += len(proposals)
        for proposal in proposals:
            mode_counts[proposal["mode"]] += 1
        if proposals:
            evaluated = evaluator.evaluate_batch(
                [(core.power, tuple(int(x) for x in proposal["factor_ids"])) for proposal in proposals],
                batch_size=args.eval_batch_size,
            )
            rows = [result_row(proposal, item) for proposal, item in zip(proposals, evaluated)]
            evaluated_count += len(rows)
            survivors.extend(rows)
            for row in rows:
                if row["metrics"].get("scalar_identity"):
                    kernel_hits.append(row)
            if len(survivors) > trim_limit:
                survivors = trim_survivors(
                    survivors,
                    rng=rng,
                    keep_best=args.keep_best,
                    keep_identity=args.keep_identity,
                    keep_projlen=args.keep_projlen,
                    keep_random=args.keep_random,
                )
            if len(kernel_hits) > args.kernel_limit:
                kernel_hits = sorted(kernel_hits, key=row_score)[: args.kernel_limit]

        if index % args.trim_every == 0:
            survivors = trim_survivors(
                survivors,
                rng=rng,
                keep_best=args.keep_best,
                keep_identity=args.keep_identity,
                keep_projlen=args.keep_projlen,
                keep_random=args.keep_random,
            )
        if index % args.progress_every == 0 or index == len(seeds):
            best = min(survivors, key=row_score) if survivors else None
            progress = {
                "phase": "completion",
                "cores_done": index,
                "cores_total": len(seeds),
                "generated": generated,
                "evaluated": evaluated_count,
                "survivors_kept": len(survivors),
                "kernel_hits": len(kernel_hits),
                "mode_counts": mode_counts,
                "elapsed_seconds": round(time.time() - start, 2),
            }
            if best is not None:
                progress["best_objective"] = best["objective"]
                progress["best_identity_defect"] = best["metrics"].get("identity_defect")
                progress["best_projlen"] = best["metrics"].get("projlen")
                progress["best_length"] = best["length"]
            print(json.dumps(progress, sort_keys=True), flush=True)
            append_jsonl(progress_path, [progress])

    survivors = trim_survivors(
        survivors,
        rng=rng,
        keep_best=args.keep_best,
        keep_identity=args.keep_identity,
        keep_projlen=args.keep_projlen,
        keep_random=args.keep_random,
    )
    kernel_hits = sorted(kernel_hits, key=row_score)[: args.kernel_limit]
    append_jsonl(completions_path, survivors)
    append_jsonl(kernel_path, kernel_hits)

    summary = {
        **metadata,
        "elapsed_seconds": round(time.time() - start, 2),
        "generated": generated,
        "evaluated": evaluated_count,
        "mode_counts": mode_counts,
        "survivors_kept": len(survivors),
        "kernel_hits": len(kernel_hits),
        "usable_kernel_hits": len(kernel_hits),
        "best": survivors[:50],
        "best_by_identity_defect": sorted(survivors, key=identity_score)[:50],
        "best_by_projlen": sorted(survivors, key=projlen_score)[:50],
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"phase": "done", **{k: v for k, v in summary.items() if k != "best"}}, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seeded left/right/both completion search for braid candidates.")
    parser.add_argument("--matrix-gpt-root", default=str(DEFAULT_MATRIX_GPT_ROOT))
    parser.add_argument("--braid-gpt-root", default=str(DEFAULT_BRAID_GPT_ROOT))
    parser.add_argument("--author-repo", default=str(DEFAULT_AUTHOR_REPO))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-source", action="append", default=[])
    parser.add_argument("--p", type=int, default=7)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--min-core-length", type=int, default=50)
    parser.add_argument("--max-core-length", type=int, default=160)
    parser.add_argument("--min-final-length", type=int, default=50)
    parser.add_argument("--max-final-length", type=int, default=220)
    parser.add_argument("--matrix-max-degree", type=int, default=256)
    parser.add_argument("--modes", default="right,left,both")
    parser.add_argument("--right-lengths", default="1,2,3,4,5,6")
    parser.add_argument("--left-lengths", default="1,2,3,4")
    parser.add_argument("--right-samples-per-length", type=int, default=1)
    parser.add_argument("--left-samples-per-length", type=int, default=1)
    parser.add_argument("--both-pairs-per-core", type=int, default=2)
    parser.add_argument("--bridge-attempts", type=int, default=80)
    parser.add_argument("--eval-batch-size", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--trim-every", type=int, default=25)
    parser.add_argument("--keep-best", type=int, default=2000)
    parser.add_argument("--keep-identity", type=int, default=1000)
    parser.add_argument("--keep-projlen", type=int, default=1000)
    parser.add_argument("--keep-random", type=int, default=1000)
    parser.add_argument("--kernel-limit", type=int, default=200)
    parser.add_argument("--allow-illegal-cores", action="store_true")
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--projlen-weight", type=float, default=0.25)
    parser.add_argument("--identity-density-weight", type=float, default=8.0)
    parser.add_argument("--projlen-density-weight", type=float, default=4.0)
    parser.add_argument("--degeneracy-weight", type=float, default=1.0)
    parser.add_argument("--min-score-length", type=int, default=45)
    parser.add_argument("--kernel-bonus", type=float, default=10000.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
