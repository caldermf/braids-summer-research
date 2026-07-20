from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from braidzero.core import BraidEnvironment, parse_int_list, sha256_file

from .algebra import ReverseAlgebra, Verification
from .io import append_jsonl, atomic_json, atomic_torch
from .model import LastFactorOracle
from .reservoir import LikelihoodReservoir, ReverseState


SCHEMA_VERSION = 1
METHOD = "transformer_reverse_likelihood_reservoir_v1"
VERIFIER_VERSION = "transformer-reverse-reservoir exact-projective-peyl-v1"


def compact_verification(verification: Verification) -> dict:
    return {
        "target_match": verification.target_match,
        "quotient_nontrivial": verification.quotient_nontrivial,
        "quotient_kernel": verification.quotient_kernel,
        "quotient_power": verification.quotient_power,
        "quotient_factors": list(verification.quotient_factors),
        "metrics": verification.metrics,
    }


def state_row(state: ReverseState) -> dict:
    return {
        "suffix": list(state.suffix),
        "depth": state.depth,
        "cumulative_nll": state.cumulative_nll,
        "average_nll": state.average_nll,
        "edge_nll": state.edge_nll,
        "edge_rank": state.edge_rank,
        "entropy": state.entropy,
        "projlen": state.projlen,
        "residual_digest": state.digest,
    }


def config_signature(args: argparse.Namespace, oracle: LastFactorOracle) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "n": args.n,
        "r": args.r,
        "p": args.p,
        "seed": args.seed,
        "target_power": args.target_power,
        "target_length": args.target_length,
        "bucket_size": args.bucket_size,
        "use_best": args.use_best,
        "nll_bin_width": args.nll_bin_width,
        "exploit_fraction": args.exploit_fraction,
        "checkpoint_checksum": oracle.metadata["checkpoint_checksum"],
        "calibration_checksum": oracle.metadata["calibration_checksum"],
    }


def print_banner(args, env, oracle, algebra) -> None:
    print("=" * 68)
    print("TRANSFORMER-GUIDED REVERSE RESERVOIR")
    print("=" * 68)
    print(f"Representation: {env.representation_label}")
    print(f"Target: rho(Delta^{args.target_power}) ({algebra.target_label} projective class)")
    print(f"Requested proper-factor length: {args.target_length}")
    print(f"Reservoir bucket size: {args.bucket_size}")
    print(f"Selected states per depth: {args.use_best}")
    print(f"Average-NLL bin width: {args.nll_bin_width}")
    print(f"Exploit fraction: {args.exploit_fraction}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"Model: {oracle.metadata['checkpoint']}")
    print(f"Model checksum: {oracle.metadata['checkpoint_checksum']}")
    print(f"Temperature: {oracle.temperature}")
    print("Every expansion uses exact projective Laurent-polynomial arithmetic.", flush=True)


def candidate_row(
    *, kind: str, state: ReverseState, verification: Verification, target_power: int
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "target_power": target_power,
        **state_row(state),
        "verification": compact_verification(verification),
        "status": "clean" if (
            verification.target_match
            and verification.quotient_nontrivial
            and verification.quotient_kernel
        ) else "malformed",
    }


def run(args: argparse.Namespace) -> dict:
    if not 0 < args.exploit_fraction <= 1:
        raise ValueError("exploit-fraction must be in (0,1]")
    if args.target_length <= 0:
        raise ValueError("target-length must be positive")

    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    checkpoint_path = output / "checkpoint.pt"
    progress_path = output / "progress.jsonl"
    candidates_path = output / "candidates.jsonl"
    collisions_path = output / "collisions.jsonl"
    atomic_json(status_path, {
        "schema_version": SCHEMA_VERSION,
        "status": "truncated",
        "reason": "reverse search has not completed",
    })

    t_values = parse_int_list(args.t_values, default=tuple(range(1, args.p)))
    env = BraidEnvironment(
        author_repo=args.author_repo, n=args.n, r=args.r, p=args.p, t_values=t_values
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    oracle = LastFactorOracle(args.checkpoint, args.calibration, env, device)
    algebra = ReverseAlgebra(env, oracle.proper_factor_ids, args.target_power)
    signature = config_signature(args, oracle)
    config = {
        **signature,
        "author_repo": str(args.author_repo.resolve()),
        "output_dir": str(output),
        "checkpoint": str(args.checkpoint.resolve()),
        "calibration": str(args.calibration.resolve()),
        "t_values": list(t_values),
        "inference_batch_size": args.inference_batch_size,
        "detect_collisions": args.detect_collisions,
        "device": args.device,
        "representation": env.representation_label,
        "target_label": algebra.target_label,
        "verifier_version": VERIFIER_VERSION,
    }
    atomic_json(output / "config.json", config)
    print_banner(args, env, oracle, algebra)

    population = LikelihoodReservoir(args.bucket_size, args.nll_bin_width, rng)
    root = ReverseState(
        residual=np.asarray(algebra.target),
        suffix=(), cumulative_nll=0.0, edge_nll=0.0, edge_rank=0,
        entropy=0.0, projlen=algebra.projlen(algebra.target),
        digest=env.exact_digest(algebra.target),
    )
    population.add(root)
    completed_depth = 0
    model_evaluations = exact_removals = generated_states = 0
    target_candidates = verified_kernels = collision_count = 0
    best_projlen_by_remaining_length: dict[int, int] = {
        args.target_length: root.projlen
    }
    candidate_seen: set[tuple[int, ...]] = set()
    collision_seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    if checkpoint_path.exists() and not args.no_resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["signature"] != signature:
            raise RuntimeError("existing checkpoint configuration does not match this run")
        population = saved["population"]
        rng.setstate(saved["rng_state"])
        population.reset_rng(rng)
        completed_depth = int(saved["completed_depth"])
        model_evaluations = int(saved["model_evaluations"])
        exact_removals = int(saved["exact_removals"])
        generated_states = int(saved["generated_states"])
        target_candidates = int(saved["target_candidates"])
        verified_kernels = int(saved["verified_kernels"])
        collision_count = int(saved["collision_count"])
        best_projlen_by_remaining_length = {
            int(k): int(v) for k, v in saved["best_projlen_by_remaining_length"].items()
        }
        candidate_seen = {tuple(x) for x in saved.get("candidate_seen", [])}
        collision_seen = {
            (tuple(left), tuple(right))
            for left, right in saved.get("collision_seen", [])
        }
        print(f"Resumed after completed reverse depth {completed_depth}.", flush=True)

    for depth in range(completed_depth + 1, args.target_length + 1):
        level_started = time.time()
        remaining_before = args.target_length - (depth - 1)
        selected, selection = population.select(args.use_best, args.exploit_fraction)
        print("=" * 68)
        print(
            f"Reverse depth {depth}/{args.target_length}: "
            f"remaining length {remaining_before} -> {remaining_before - 1}"
        )
        print(
            f"Population={population.size():,}; selected={len(selected):,}; "
            f"buckets={len(population.buckets)}"
        )

        logits = oracle.logits(
            [state.residual for state in selected], args.inference_batch_size
        )
        model_evaluations += len(selected)
        destination = LikelihoodReservoir(args.bucket_size, args.nll_bin_width, rng)
        residual_owners: dict[str, ReverseState] = {}
        level_generated = 0
        level_best_projlen: int | None = None
        level_rank_counts: dict[int, int] = {}

        for parent, parent_logits in zip(selected, logits):
            legal = algebra.legal_predecessors(parent.right_factor)
            log_probs, ranks, entropy = oracle.legal_distribution(parent_logits, legal)
            for factor_id in legal:
                residual = algebra.remove(parent.residual, factor_id)
                exact_removals += 1
                level_generated += 1
                generated_states += 1
                edge_nll = -log_probs[factor_id]
                suffix = (int(factor_id),) + parent.suffix
                projlen = algebra.projlen(residual)
                digest = env.exact_digest(residual)
                state = ReverseState(
                    residual=residual,
                    suffix=suffix,
                    cumulative_nll=parent.cumulative_nll + edge_nll,
                    edge_nll=edge_nll,
                    edge_rank=ranks[factor_id],
                    entropy=entropy,
                    projlen=projlen,
                    digest=digest,
                )
                destination.add(state)
                level_best_projlen = (
                    projlen if level_best_projlen is None else min(level_best_projlen, projlen)
                )
                level_rank_counts[state.edge_rank] = level_rank_counts.get(state.edge_rank, 0) + 1

                if args.detect_collisions:
                    previous = residual_owners.get(digest)
                    if previous is None:
                        residual_owners[digest] = state
                    elif previous.suffix != state.suffix and algebra.same_projective_matrix(
                        previous.residual, state.residual
                    ):
                        pair = tuple(sorted((previous.suffix, state.suffix)))
                        if pair not in collision_seen:
                            collision_seen.add(pair)
                            verification = algebra.verify_collision(*pair)
                            if verification.quotient_nontrivial and verification.quotient_kernel:
                                collision_count += 1
                                append_jsonl(collisions_path, {
                                    "schema_version": SCHEMA_VERSION,
                                    "kind": "equal_reverse_residual",
                                    "depth": depth,
                                    "left_suffix": list(pair[0]),
                                    "right_suffix": list(pair[1]),
                                    "residual_digest": digest,
                                    "verification": compact_verification(verification),
                                    "status": "clean",
                                })
                                print(
                                    f"  VERIFIED RESIDUAL COLLISION at depth {depth}", flush=True
                                )

                if depth == args.target_length and algebra.is_identity(residual):
                    target_candidates += 1
                    if suffix not in candidate_seen:
                        candidate_seen.add(suffix)
                        verification = algebra.verify_target_preimage(suffix)
                        row = candidate_row(
                            kind="complete_target_preimage", state=state,
                            verification=verification, target_power=args.target_power,
                        )
                        append_jsonl(candidates_path, row)
                        if row["status"] == "clean":
                            verified_kernels += 1
                            print(
                                f"  VERIFIED KERNEL PREIMAGE: length={len(suffix)} "
                                f"average_nll={state.average_nll:.6f}", flush=True
                            )

        remaining_after = args.target_length - depth
        if level_best_projlen is not None:
            best_projlen_by_remaining_length[remaining_after] = level_best_projlen
        pop_summary = destination.summary()
        average_keys = sorted(destination.buckets)
        elapsed = time.time() - level_started
        row = {
            "schema_version": SCHEMA_VERSION,
            "depth": depth,
            "remaining_length": remaining_after,
            "selected_parents": len(selected),
            "selection": selection,
            "generated_states": level_generated,
            "population_states": destination.size(),
            "population_buckets": len(destination.buckets),
            "average_nll_range": None if not average_keys else [
                average_keys[0] * args.nll_bin_width,
                (average_keys[-1] + 1) * args.nll_bin_width,
            ],
            "best_projlen": level_best_projlen,
            "edge_rank_counts": level_rank_counts,
            "model_evaluations_total": model_evaluations,
            "exact_removals_total": exact_removals,
            "target_candidates_total": target_candidates,
            "verified_kernels_total": verified_kernels,
            "verified_collisions_total": collision_count,
            "elapsed_seconds": elapsed,
            "population_summary": pop_summary,
        }
        append_jsonl(progress_path, row)
        print(
            f"Generated={level_generated:,}; kept={destination.size():,}; "
            f"NLL bins={len(destination.buckets)}; best projlen={level_best_projlen}; "
            f"time={elapsed:.2f}s"
        )
        print(
            f"Verified target preimages={verified_kernels}; "
            f"verified residual collisions={collision_count}", flush=True
        )

        population = destination
        completed_depth = depth
        atomic_torch(checkpoint_path, {
            "schema_version": SCHEMA_VERSION,
            "signature": signature,
            "completed_depth": completed_depth,
            "population": population,
            "rng_state": rng.getstate(),
            "model_evaluations": model_evaluations,
            "exact_removals": exact_removals,
            "generated_states": generated_states,
            "target_candidates": target_candidates,
            "verified_kernels": verified_kernels,
            "collision_count": collision_count,
            "best_projlen_by_remaining_length": best_projlen_by_remaining_length,
            "candidate_seen": [list(x) for x in candidate_seen],
            "collision_seen": [[list(a), list(b)] for a, b in collision_seen],
        })
        atomic_json(status_path, {
            "schema_version": SCHEMA_VERSION,
            "status": "truncated",
            "reason": "reverse search has not completed",
            "completed_depth": completed_depth,
            "target_length": args.target_length,
            "verified_kernels": verified_kernels,
        })

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "clean",
        "method": METHOD,
        "prime": args.p,
        "representation": env.representation_label,
        "seed": args.seed,
        "target_power": args.target_power,
        "target_label": algebra.target_label,
        "length_range": [args.target_length, args.target_length],
        "model_config": oracle.metadata,
        "search_config": {
            "bucket_size": args.bucket_size,
            "use_best": args.use_best,
            "nll_bin_width": args.nll_bin_width,
            "exploit_fraction": args.exploit_fraction,
            "detect_collisions": args.detect_collisions,
        },
        "exact_evaluations": exact_removals,
        "model_evaluations": model_evaluations,
        "generated_states": generated_states,
        "best_projlen_by_remaining_length": best_projlen_by_remaining_length,
        "target_candidates": target_candidates,
        "verified_kernels": verified_kernels,
        "verified_residual_collisions": collision_count,
        "artifact_path": str(checkpoint_path),
        "artifact_checksum": sha256_file(checkpoint_path),
        "verifier_version": VERIFIER_VERSION,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(status_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Transformer-guided reverse paper-style reservoir search"
    )
    p.add_argument("--author-repo", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--r", type=int, default=1)
    p.add_argument("--p", type=int, default=5)
    p.add_argument("--target-power", type=int, default=0)
    p.add_argument("--target-length", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--bucket-size", type=int, default=3000)
    p.add_argument("--use-best", type=int, default=50000)
    p.add_argument("--nll-bin-width", type=float, default=0.25)
    p.add_argument("--exploit-fraction", type=float, default=0.60)
    p.add_argument("--inference-batch-size", type=int, default=256)
    p.add_argument("--t-values", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--detect-collisions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-resume", action="store_true")
    return p


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
