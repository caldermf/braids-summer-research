#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .config import SearchConfig
from .exact_evaluator import CPUExactEvaluator, TorchExactEvaluator
from .field_sketch import ExtensionFieldSketch
from .gnf import GNFAutomaton
from .known_examples import (
    KNOWN_P3_LENGTH24_FACTOR_IDS,
    KNOWN_P5_LENGTH54_FACTOR_IDS,
)
from .models import Segment
from .operators import SegmentMutator
from .suffix_index import SuffixLSHIndex


def config(backend: str, device: str, p: int = 5) -> SearchConfig:
    return SearchConfig(
        p=p,
        n=4,
        prefix_count=32,
        suffix_count=128,
        generations=1,
        prefix_length_min=12,
        prefix_length_max=32,
        suffix_length_min=8,
        suffix_length_max=28,
        field_points=min(8, p * (p - 1)),
        lsh_tables=12,
        lsh_key_components=4,
        max_lsh_candidates=128,
        join_candidates_per_prefix=4,
        elite_pairs=16,
        refinement_pairs=4,
        refinement_trials=4,
        signature_batch_size=64,
        exact_batch_size=64,
        backend=backend,
        device=device,
        resume_latest=False,
    )


def main() -> None:
    if os.environ.get("SLURM_JOB_PARTITION") != "scavenge_gpu":
        raise SystemExit("V5 validation must run in scavenge_gpu.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this scavenge_gpu job.")

    rng = random.Random(20260614)
    automaton = GNFAutomaton(4)
    prefixes = [
        Segment(automaton.sample_prefix(rng.randint(12, 24), rng), "prefix", f"p{i}")
        for i in range(32)
    ]
    suffixes = [
        Segment(automaton.sample_suffix(rng.randint(8, 24), rng), "suffix", f"s{i}")
        for i in range(128)
    ]
    cpu_sketch = ExtensionFieldSketch(config("cpu", "cpu"))
    gpu_sketch = ExtensionFieldSketch(config("torch", "cuda"))
    cpu_suffix = cpu_sketch.suffix_signatures(suffixes)
    gpu_suffix = gpu_sketch.suffix_signatures(suffixes)
    if not np.array_equal(cpu_suffix, gpu_suffix):
        raise AssertionError("CPU and CUDA suffix signatures differ")
    cpu_targets = cpu_sketch.prefix_target_signatures(prefixes)
    gpu_targets = gpu_sketch.prefix_target_signatures(prefixes)
    for target_type in ("identity", "delta"):
        if not np.array_equal(cpu_targets[target_type], gpu_targets[target_type]):
            raise AssertionError(f"CPU and CUDA {target_type} targets differ")

    p3_cpu_sketch = ExtensionFieldSketch(config("cpu", "cpu", p=3))
    p3_gpu_sketch = ExtensionFieldSketch(config("torch", "cuda", p=3))
    p3_prefix = Segment(KNOWN_P3_LENGTH24_FACTOR_IDS[:12], "prefix", "known-p3-prefix")
    p3_suffix = Segment(KNOWN_P3_LENGTH24_FACTOR_IDS[12:], "suffix", "known-p3-suffix")
    p3_cpu_target = p3_cpu_sketch.prefix_target_signatures([p3_prefix])["identity"]
    p3_gpu_target = p3_gpu_sketch.prefix_target_signatures([p3_prefix])["identity"]
    p3_cpu_signature = p3_cpu_sketch.suffix_signatures([p3_suffix])
    p3_gpu_signature = p3_gpu_sketch.suffix_signatures([p3_suffix])
    if not np.array_equal(p3_cpu_target, p3_gpu_target):
        raise AssertionError("CPU and CUDA p=3 identity targets differ")
    if not np.array_equal(p3_cpu_signature, p3_gpu_signature):
        raise AssertionError("CPU and CUDA p=3 suffix signatures differ")
    p3_distance = int(
        p3_gpu_sketch.distance(p3_gpu_target, p3_gpu_signature)[0]
    )
    if p3_distance != 0:
        raise AssertionError("known p=3 suffix did not match P^-1")

    split = 27
    known_prefix = Segment(
        KNOWN_P5_LENGTH54_FACTOR_IDS[:split],
        "prefix",
        "known-prefix",
    )
    known_suffix = Segment(
        KNOWN_P5_LENGTH54_FACTOR_IDS[split:],
        "suffix",
        "known-suffix",
    )
    known_target = gpu_sketch.prefix_target_signatures([known_prefix])
    known_signature = gpu_sketch.suffix_signatures([known_suffix])
    delta_distance = int(
        gpu_sketch.distance(known_target["delta"], known_signature)[0]
    )
    if delta_distance != 0:
        raise AssertionError("known p=5 suffix did not match P^-1 Delta")

    mixed_suffixes = suffixes + [known_suffix]
    mixed_signatures = gpu_sketch.suffix_signatures(mixed_suffixes)
    index = SuffixLSHIndex(
        config("torch", "cuda"),
        mixed_suffixes,
        mixed_signatures,
        rng,
    )
    matches = index.query(
        known_target["delta"][0],
        automaton.successors[known_prefix.factor_ids[-1]],
        4,
    )
    if not matches or matches[0] != (len(mixed_suffixes) - 1, 0):
        raise AssertionError("LSH did not recover the exact known suffix")

    cpu_exact = CPUExactEvaluator(config("cpu", "cpu"))
    gpu_exact = TorchExactEvaluator(config("torch", "cuda"))
    words = [
        prefix.factor_ids + suffix.factor_ids
        for prefix, suffix in zip(prefixes[:16], suffixes[:16])
        if automaton.can_join(prefix.factor_ids, suffix.factor_ids)
    ]
    words.append(KNOWN_P5_LENGTH54_FACTOR_IDS)
    cpu_results = cpu_exact.evaluate(words)
    gpu_results = gpu_exact.evaluate(words)
    if [item.projlen_history for item in cpu_results] != [
        item.projlen_history for item in gpu_results
    ]:
        raise AssertionError("CPU and CUDA exact projlen histories differ")
    if not gpu_results[-1].has_kernel:
        raise AssertionError("CUDA exact evaluator missed the known p=5 kernel")
    p3_cpu_exact = CPUExactEvaluator(config("cpu", "cpu", p=3))
    p3_gpu_exact = TorchExactEvaluator(config("torch", "cuda", p=3))
    if not p3_cpu_exact.evaluate_one(KNOWN_P3_LENGTH24_FACTOR_IDS).has_kernel:
        raise AssertionError("CPU exact evaluator missed the known p=3 kernel")
    if not p3_gpu_exact.evaluate([KNOWN_P3_LENGTH24_FACTOR_IDS])[0].has_kernel:
        raise AssertionError("CUDA exact evaluator missed the known p=3 kernel")

    mutator = SegmentMutator(config("cpu", "cpu"), automaton, rng)
    legal_mutations = 0
    for role, parent in (("prefix", known_prefix), ("suffix", known_suffix)):
        for index_value in range(250):
            child = mutator.mutate(parent, f"{role}-{index_value}")
            legal = (
                automaton.is_legal_prefix(child.factor_ids)
                if role == "prefix"
                else automaton.is_internally_legal(child.factor_ids)
            )
            if not legal:
                raise AssertionError(f"illegal {role} mutation")
            legal_mutations += 1

    marker = Path(
        os.environ.get(
            "BIDIRECTIONAL_V5_VALIDATION_MARKER",
            "results/bidirectional_v5_validation/scavenge_gpu_v5_validated.json",
        )
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "passed",
        "algorithm": "bidirectional_matrix_search_v5",
        "partition": os.environ["SLURM_JOB_PARTITION"],
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "field": "GF(5^2)",
        "field_points": 8,
        "cpu_cuda_signatures_compared": len(prefixes) + len(suffixes),
        "known_p5_inverse_delta_distance": delta_distance,
        "known_p3_inverse_identity_distance": p3_distance,
        "known_p5_suffix_retrieved_by_lsh": True,
        "known_p3_kernel_exactly_verified": True,
        "known_p5_kernel_exactly_verified": True,
        "legal_mutations_checked": legal_mutations,
    }
    marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Validation marker: {marker}", flush=True)


if __name__ == "__main__":
    main()
