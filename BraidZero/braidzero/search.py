from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .core import BraidEnvironment, parse_int_list, word_digest, write_json
from .ledger import RunLedger
from .oracle import ShadowOracle, ShadowRecord


@dataclass
class SearchState:
    factors: tuple[int, ...]
    finite: tuple[tuple[int, ...], ...]
    finite_key: tuple[int, ...]
    exact: np.ndarray
    metrics: dict
    score: float


def model_features(
    *,
    length: int,
    projlen: int,
    identity_defect: int,
    scalar_suffix_hits: int = 0,
    collision_hits: int = 0,
    bank_length: int = 0,
) -> list[float]:
    return [
        length / 512.0,
        projlen / 256.0,
        identity_defect / 512.0,
        math.log1p(scalar_suffix_hits) / 10.0,
        math.log1p(collision_hits) / 10.0,
        bank_length / 256.0,
    ]


def choose_actions(
    *,
    state: SearchState,
    legal: Sequence[int],
    args: argparse.Namespace,
    rng: random.Random,
    policy_model,
    policy_device,
) -> list[int]:
    legal = list(int(x) for x in legal)
    if not legal:
        return []
    if policy_model is None:
        if args.max_actions_per_state <= 0 or args.max_actions_per_state >= len(legal):
            return legal
        rng.shuffle(legal)
        return legal[: args.max_actions_per_state]

    from .model import score_legal_actions

    features = model_features(
        length=len(state.factors),
        projlen=int(state.metrics["projlen"]),
        identity_defect=int(state.metrics["identity_defect"]),
        bank_length=args.bank_length,
    )
    ranked = score_legal_actions(
        policy_model,
        factors=state.factors,
        features=features,
        legal_actions=legal,
        device=policy_device,
    )
    chosen = [action for action, _ in ranked[: args.model_top_k]]
    if args.model_random_extra > 0:
        remaining = [action for action in legal if action not in set(chosen)]
        rng.shuffle(remaining)
        chosen.extend(remaining[: args.model_random_extra])
    if args.include_all_legal_with_model:
        for action in legal:
            if action not in set(chosen):
                chosen.append(action)
    return chosen


def state_priority(*, metrics: dict, scalar_suffix_hits: int, collision_hits: int) -> float:
    finite_hits = scalar_suffix_hits + collision_hits
    hit_bonus = 200.0 if finite_hits else 0.0
    return (
        hit_bonus
        + 12.0 * math.log1p(finite_hits)
        - 1.5 * float(metrics["identity_defect"])
        - 0.75 * float(metrics["projlen"])
    )


def select_beam(states: list[SearchState], *, beam_size: int, per_finite_key_cap: int) -> list[SearchState]:
    states.sort(key=lambda item: item.score, reverse=True)
    selected: list[SearchState] = []
    counts: dict[tuple[int, ...], int] = {}
    for state in states:
        if per_finite_key_cap > 0:
            count = counts.get(state.finite_key, 0)
            if count >= per_finite_key_cap:
                continue
            counts[state.finite_key] = count + 1
        selected.append(state)
        if len(selected) >= beam_size:
            break
    return selected


def compact_candidate_row(*, kind: str, factors: Sequence[int], metrics: dict, extra: dict | None = None) -> dict:
    return {
        "kind": kind,
        "power": 0,
        "length": len(factors),
        "factor_ids": [int(x) for x in factors],
        "word_digest": word_digest(0, factors),
        "metrics": metrics,
        **(extra or {}),
    }


def run_search(args: argparse.Namespace) -> dict:
    start_time = time.time()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    ledger = RunLedger(output_dir=output_dir)
    t_values = parse_int_list(args.t_values, default=tuple(range(1, args.p)))

    env = BraidEnvironment(
        author_repo=Path(args.author_repo),
        n=args.n,
        r=args.r,
        p=args.p,
        t_values=t_values,
    )
    config = vars(args).copy()
    config["t_values"] = list(t_values)
    config["representation"] = env.representation_label
    write_json(output_dir / "config.json", config)

    oracle = ShadowOracle.build(
        env=env,
        bank_length=args.bank_length,
        mode=args.bank_mode,
        samples=args.bank_samples,
        seed=args.seed + 104729,
        max_exhaustive=args.max_exhaustive_bank,
        max_records_per_key=args.max_bank_records_per_key,
        progress_interval_seconds=args.progress_interval_seconds,
    )
    write_json(output_dir / "oracle_summary.json", oracle.metadata)
    print(json.dumps({"phase": "oracle_ready", **oracle.metadata}, sort_keys=True), flush=True)

    policy_model = None
    policy_device = "cpu"
    if args.policy_checkpoint:
        from .model import load_checkpoint
        import torch

        policy_device = args.policy_device
        if policy_device == "cuda" and not torch.cuda.is_available():
            policy_device = "cpu"
        policy_model = load_checkpoint(Path(args.policy_checkpoint), map_location=policy_device).to(policy_device)
        print(
            json.dumps(
                {
                    "phase": "policy_loaded",
                    "checkpoint": args.policy_checkpoint,
                    "device": policy_device,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    initial_metrics = env.exact_metrics(env.identity_exact)
    beam = [
        SearchState(
            factors=(),
            finite=env.identity_finite,
            finite_key=env.finite_key(env.identity_finite),
            exact=env.identity_exact,
            metrics=initial_metrics,
            score=0.0,
        )
    ]

    exact_partner_cache: dict[int, tuple[np.ndarray, str, dict]] = {}
    seen_complete_by_digest: dict[str, tuple[int, ...]] = {}
    best_scalar_identity_candidate: dict | None = None
    best_projlen = int(initial_metrics["projlen"])
    best_identity_defect = int(initial_metrics["identity_defect"])
    exact_evaluations = 0
    symbolic_factor_multiplications = 0
    finite_collision_pairs = 0
    finite_scalar_completion_pairs = 0
    exact_collisions = 0
    verified_kernel_quotients = 0
    scalar_identity_candidates = 0
    expanded_states = 0
    last_progress = start_time

    def get_partner_exact(record: ShadowRecord) -> tuple[np.ndarray, str, dict]:
        nonlocal exact_evaluations, symbolic_factor_multiplications
        cached = exact_partner_cache.get(record.record_id)
        if cached is not None:
            return cached
        image = env.exact_evaluate(record.factors)
        symbolic_factor_multiplications += len(record.factors)
        exact_evaluations += 1
        digest = env.exact_digest(image)
        metrics = env.exact_metrics(image)
        exact_partner_cache[record.record_id] = (image, digest, metrics)
        return image, digest, metrics

    for depth in range(1, args.prefix_length + 1):
        next_states: list[SearchState] = []
        depth_expansions = 0
        depth_finite_collision_pairs = 0
        depth_finite_scalar_pairs = 0
        depth_exact_collisions = 0
        depth_scalar_identities = 0

        for state in beam:
            legal = env.legal_next(state.factors)
            actions = choose_actions(
                state=state,
                legal=legal,
                args=args,
                rng=rng,
                policy_model=policy_model,
                policy_device=policy_device,
            )
            for action in actions:
                parent_factors = state.factors
                child_factors = parent_factors + (int(action),)
                child_finite = env.finite_append(state.finite, int(action))
                child_key = env.finite_key(child_finite)
                child_exact = env.exact_append(state.exact, int(action))
                exact_evaluations += 1
                symbolic_factor_multiplications += 1
                expanded_states += 1
                depth_expansions += 1
                child_metrics = env.exact_metrics(child_exact)
                best_projlen = min(best_projlen, int(child_metrics["projlen"]))
                best_identity_defect = min(best_identity_defect, int(child_metrics["identity_defect"]))

                collision_hits, collision_records = oracle.collision_partners(
                    child_key,
                    limit=args.max_collision_partners_per_prefix,
                )
                scalar_hits, scalar_records = oracle.scalar_suffixes(
                    child_finite,
                    legal_first=env.legal_next(child_factors),
                    limit=args.max_scalar_suffixes_per_prefix,
                )
                finite_collision_pairs += collision_hits
                finite_scalar_completion_pairs += scalar_hits
                depth_finite_collision_pairs += collision_hits
                depth_finite_scalar_pairs += scalar_hits

                score = state_priority(
                    metrics=child_metrics,
                    scalar_suffix_hits=scalar_hits,
                    collision_hits=collision_hits,
                )
                next_states.append(
                    SearchState(
                        factors=child_factors,
                        finite=child_finite,
                        finite_key=child_key,
                        exact=child_exact,
                        metrics=child_metrics,
                        score=score,
                    )
                )

                ledger.training_example(
                    {
                        "parent_factors": list(parent_factors),
                        "action": int(action),
                        "child_factors": list(child_factors),
                        "parent_projlen": int(state.metrics["projlen"]),
                        "parent_identity_defect": int(state.metrics["identity_defect"]),
                        "child_projlen": int(child_metrics["projlen"]),
                        "child_identity_defect": int(child_metrics["identity_defect"]),
                        "scalar_suffix_hits": int(scalar_hits),
                        "collision_hits": int(collision_hits),
                        "bank_length": int(args.bank_length),
                        "depth": depth,
                        "exact_scalar_identity": False,
                        "exact_collision": False,
                    }
                )

                child_digest = env.exact_digest(child_exact)
                for partner in collision_records:
                    if tuple(partner.factors) == child_factors:
                        continue
                    _, partner_digest, partner_metrics = get_partner_exact(partner)
                    if partner_digest != child_digest:
                        continue
                    exact_collisions += 1
                    verified_kernel_quotients += 1
                    depth_exact_collisions += 1
                    row = {
                        "kind": "exact_matrix_collision",
                        "p": args.p,
                        "representation": env.representation_label,
                        "prefix_length": len(child_factors),
                        "partner_length": len(partner.factors),
                        "u_factor_ids": list(child_factors),
                        "v_factor_ids": list(partner.factors),
                        "u_digest": word_digest(0, child_factors),
                        "v_digest": word_digest(0, partner.factors),
                        "matrix_digest": child_digest,
                        "u_metrics": child_metrics,
                        "v_metrics": partner_metrics,
                        "quotient_certificate": "u*v^{-1} is nontrivial because u and v are distinct positive GNF normal forms with the same exact projective matrix",
                    }
                    ledger.collision(row)
                    print(json.dumps({"phase": "exact_collision", **row}, sort_keys=True), flush=True)
                    if args.stop_after_verified_kernel:
                        break

                for suffix in scalar_records:
                    full_factors = child_factors + suffix.factors
                    if not env.is_legal(full_factors):
                        continue
                    full_exact = env.exact_append_sequence(child_exact, suffix.factors)
                    exact_evaluations += 1
                    symbolic_factor_multiplications += len(suffix.factors)
                    full_metrics = env.exact_metrics(full_exact)
                    best_projlen = min(best_projlen, int(full_metrics["projlen"]))
                    best_identity_defect = min(best_identity_defect, int(full_metrics["identity_defect"]))
                    full_digest = env.exact_digest(full_exact)
                    finite_product = env.finite_mul(child_finite, env.finite_evaluate(suffix.factors))
                    row = compact_candidate_row(
                        kind="finite_scalar_completion",
                        factors=full_factors,
                        metrics=full_metrics,
                        extra={
                            "prefix_length": len(child_factors),
                            "suffix_record_id": suffix.record_id,
                            "suffix_factor_ids": list(suffix.factors),
                            "finite_scalar_at_t_values": env.finite_scalar_flags(finite_product),
                            "matrix_digest": full_digest,
                        },
                    )
                    ledger.candidate(row)
                    if full_metrics["scalar_identity"]:
                        scalar_identity_candidates += 1
                        depth_scalar_identities += 1
                        best_scalar_identity_candidate = row
                        print(json.dumps({"phase": "exact_scalar_identity", **row}, sort_keys=True), flush=True)
                    prior = seen_complete_by_digest.get(full_digest)
                    if prior is not None and prior != full_factors:
                        exact_collisions += 1
                        verified_kernel_quotients += 1
                        depth_exact_collisions += 1
                        ledger.collision(
                            {
                                "kind": "exact_completion_collision",
                                "p": args.p,
                                "representation": env.representation_label,
                                "u_factor_ids": list(prior),
                                "v_factor_ids": list(full_factors),
                                "u_digest": word_digest(0, prior),
                                "v_digest": word_digest(0, full_factors),
                                "matrix_digest": full_digest,
                                "v_metrics": full_metrics,
                                "quotient_certificate": "u*v^{-1} is nontrivial because u and v are distinct positive GNF normal forms with the same exact projective matrix",
                            }
                        )
                    else:
                        seen_complete_by_digest[full_digest] = tuple(full_factors)

                if args.stop_after_verified_kernel and verified_kernel_quotients:
                    break
                if args.stop_after_scalar_identity and scalar_identity_candidates:
                    break
            if args.stop_after_verified_kernel and verified_kernel_quotients:
                break
            if args.stop_after_scalar_identity and scalar_identity_candidates:
                break

        beam = select_beam(
            next_states,
            beam_size=args.beam_size,
            per_finite_key_cap=args.per_finite_key_cap,
        )
        beam_best = min((int(state.metrics["projlen"]) for state in beam), default=None)
        beam_best_defect = min((int(state.metrics["identity_defect"]) for state in beam), default=None)
        progress = {
            "phase": "depth_done",
            "depth": depth,
            "beam_size": len(beam),
            "depth_expansions": depth_expansions,
            "expanded_states": expanded_states,
            "best_projlen": best_projlen,
            "best_identity_defect": best_identity_defect,
            "beam_best_projlen": beam_best,
            "beam_best_identity_defect": beam_best_defect,
            "finite_collision_pairs": finite_collision_pairs,
            "finite_scalar_completion_pairs": finite_scalar_completion_pairs,
            "depth_finite_collision_pairs": depth_finite_collision_pairs,
            "depth_finite_scalar_completion_pairs": depth_finite_scalar_pairs,
            "exact_collisions": exact_collisions,
            "verified_kernel_quotients": verified_kernel_quotients,
            "scalar_identity_candidates": scalar_identity_candidates,
            "depth_exact_collisions": depth_exact_collisions,
            "depth_scalar_identities": depth_scalar_identities,
            "exact_evaluations": exact_evaluations,
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        ledger.progress(progress)
        print(json.dumps(progress, sort_keys=True), flush=True)

        now = time.time()
        if now - last_progress >= args.progress_interval_seconds:
            last_progress = now
        if not beam:
            break
        if args.stop_after_verified_kernel and verified_kernel_quotients:
            break
        if args.stop_after_scalar_identity and scalar_identity_candidates:
            break

    elapsed = time.time() - start_time
    length_range = {
        "prefix_length": [1, args.prefix_length],
        "bank_length": args.bank_length,
        "direct_completion_length": [1 + args.bank_length, args.prefix_length + args.bank_length],
        "collision_pair_lengths": [args.bank_length, args.prefix_length],
    }
    summary = {
        "format": "braidzero-summary-v1",
        "status": "clean",
        "method": "braidzero_finite_shadow_collision_completion_search",
        "prime": args.p,
        "representation": env.representation_label,
        "seed": args.seed,
        "length_range": length_range,
        "t_values": list(t_values),
        "oracle": oracle.metadata,
        "search": {
            "elapsed_seconds": round(elapsed, 2),
            "expanded_states": expanded_states,
            "exact_evaluations": exact_evaluations,
            "symbolic_factor_multiplications": symbolic_factor_multiplications,
            "best_projlen": best_projlen,
            "best_identity_defect": best_identity_defect,
            "best_scalar_identity_candidate": best_scalar_identity_candidate,
            "finite_collision_pairs": finite_collision_pairs,
            "finite_scalar_completion_pairs": finite_scalar_completion_pairs,
            "exact_collisions": exact_collisions,
            "verified_kernel_quotients": verified_kernel_quotients,
            "scalar_identity_candidates": scalar_identity_candidates,
            "final_beam_size": len(beam),
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
        "number_exact_collisions": exact_collisions,
        "number_verified_kernel_quotients": verified_kernel_quotients,
        "verifier_version": env.verifier_version,
        "status": "clean",
    }
    ledger.finalize(summary=summary, ledger_row=ledger_row)
    print(json.dumps({"phase": "done", **summary["search"]}, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "BraidZero finite-shadow guided search: build a finite image bank, then run "
            "forward exact GNF search with collision and scalar-completion oracles."
        )
    )
    parser.add_argument("--author-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--p", type=int, default=7)
    parser.add_argument("--t-values", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bank-length", type=int, default=17)
    parser.add_argument("--bank-mode", choices=["auto", "exhaustive", "random"], default="random")
    parser.add_argument("--bank-samples", type=int, default=250_000)
    parser.add_argument("--max-exhaustive-bank", type=int, default=2_000_000)
    parser.add_argument("--max-bank-records-per-key", type=int, default=256)
    parser.add_argument("--prefix-length", type=int, default=24)
    parser.add_argument("--beam-size", type=int, default=25_000)
    parser.add_argument("--per-finite-key-cap", type=int, default=8)
    parser.add_argument("--max-actions-per-state", type=int, default=0)
    parser.add_argument("--max-collision-partners-per-prefix", type=int, default=8)
    parser.add_argument("--max-scalar-suffixes-per-prefix", type=int, default=8)
    parser.add_argument("--policy-checkpoint", default="")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--model-top-k", type=int, default=8)
    parser.add_argument("--model-random-extra", type=int, default=2)
    parser.add_argument("--include-all-legal-with-model", action="store_true")
    parser.add_argument("--stop-after-verified-kernel", action="store_true")
    parser.add_argument("--stop-after-scalar-identity", action="store_true")
    parser.add_argument("--progress-interval-seconds", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
