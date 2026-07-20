from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from braidzero.core import BraidEnvironment, parse_int_list
from last_factor_confusion.positive_control_v3 import KNOWN_LENGTH54

from .io import atomic_json
from .model import LastFactorOracle


def load_words(kernel_db: Path) -> dict[str, tuple[int, ...]]:
    database = json.loads(kernel_db.read_text())
    words = {"validated_length54": tuple(KNOWN_LENGTH54)}
    for index, entry in enumerate(database["primes"]["5"]["elements"].values()):
        words[f"kernel_db_{index}"] = tuple(int(x) for x in entry["word"])
    return words


def replay_word(name: str, factors: tuple[int, ...], env, oracle, batch_size: int) -> dict:
    if not env.is_legal(factors):
        raise ValueError(f"{name} is not a legal positive Garside normal form")
    images = []
    image = env.identity_exact
    for factor_id in factors:
        image = env.exact_append(image, factor_id)
        images.append(image)
    logits = oracle.logits(images, batch_size)

    rows = []
    cumulative_nll = 0.0
    for prefix_length in range(len(factors), 0, -1):
        true_factor = factors[prefix_length - 1]
        right_factor = factors[prefix_length] if prefix_length < len(factors) else None
        if right_factor is None:
            legal = oracle.proper_factor_ids
        else:
            legal = tuple(
                factor_id for factor_id in oracle.proper_factor_ids
                if bool(env.nf_table.is_normalised[factor_id][right_factor])
            )
        log_probs, ranks, entropy = oracle.legal_distribution(
            logits[prefix_length - 1], legal
        )
        nll = -log_probs[true_factor]
        cumulative_nll += nll
        rows.append({
            "prefix_length": prefix_length,
            "distance_from_target": len(factors) - prefix_length,
            "true_factor_id": true_factor,
            "true_factor_class": oracle.factor_to_class[true_factor],
            "true_probability": math.exp(log_probs[true_factor]),
            "true_nll": nll,
            "true_rank": ranks[true_factor],
            "legal_factors": len(legal),
            "entropy": entropy,
            "cumulative_reverse_nll": cumulative_nll,
        })

    final = images[-1]
    identity = bool(env.exact_target_metrics(final, "identity")["target_match"])
    delta = bool(env.exact_target_metrics(final, "delta")["target_match"])
    ranks = [row["true_rank"] for row in rows]
    return {
        "kernel_id": name,
        "length": len(factors),
        "terminal_type": "identity" if identity else "delta" if delta else "other",
        "top_k_recall": {
            str(k): sum(rank <= k for rank in ranks) / len(ranks)
            for k in (1, 3, 5, 10, 22)
        },
        "worst_true_rank": max(ranks),
        "mean_true_rank": sum(ranks) / len(ranks),
        "cumulative_reverse_nll": cumulative_nll,
        "mean_reverse_nll": cumulative_nll / len(ranks),
        "steps": rows,
        "status": "clean" if (identity or delta) else "malformed",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Replay known p=5 kernels through reverse decoding")
    p.add_argument("--author-repo", type=Path, required=True)
    p.add_argument("--kernel-db", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--only", action="append", default=[])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--t-values", default="")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    env = BraidEnvironment(
        author_repo=args.author_repo, n=4, r=1, p=5,
        t_values=parse_int_list(args.t_values, default=(1, 2, 3, 4)),
    )
    oracle = LastFactorOracle(args.checkpoint, args.calibration, env, device)
    words = load_words(args.kernel_db)
    if args.only:
        missing = sorted(set(args.only) - set(words))
        if missing:
            raise ValueError(f"unknown kernel ids: {missing}")
        words = {key: words[key] for key in args.only}

    results = [
        replay_word(name, factors, env, oracle, args.batch_size)
        for name, factors in words.items()
    ]
    payload = {
        "schema_version": 1,
        "status": "clean" if all(row["status"] == "clean" for row in results) else "malformed",
        "model": oracle.metadata,
        "kernels": results,
    }
    atomic_json(args.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "output": str(args.output.resolve()),
        "kernels": [
            {
                "kernel_id": row["kernel_id"],
                "length": row["length"],
                "terminal_type": row["terminal_type"],
                "top_k_recall": row["top_k_recall"],
                "worst_true_rank": row["worst_true_rank"],
                "mean_reverse_nll": row["mean_reverse_nll"],
            }
            for row in results
        ],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

