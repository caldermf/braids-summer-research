#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
for path in (REPO_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

from crispr_transformer.edits import apply_geometry, valid_geometries
from crispr_transformer.exact import CPUExactEvaluator, TorchExactEvaluator, require_compatible_cuda
from crispr_transformer.gnf import GNFAutomaton
from crispr_transformer.io_utils import write_json
from structural_experiments.audit import known_p5_factor_ids
from structural_experiments.commutator_exact import (
    CPUCommutatorEvaluator,
    TorchCommutatorEvaluator,
)
from structural_experiments.datta import analyze_factor_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", required=True)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    require_compatible_cuda(torch)

    rng = random.Random(args.seed)
    automaton = GNFAutomaton(4)
    words = [automaton.sample_uniform(rng.randint(3, 8), rng) for _ in range(16)]

    ordinary_cpu = CPUExactEvaluator(p=5, n=4).evaluate(words)
    ordinary_gpu = TorchExactEvaluator(
        p=5, n=4, device="cuda", batch_size=64
    ).evaluate(words)
    if [item.projlen_history for item in ordinary_cpu] != [
        item.projlen_history for item in ordinary_gpu
    ]:
        raise AssertionError("ordinary CPU/GPU evaluator mismatch")

    comm_cpu = CPUCommutatorEvaluator(
        p=5, n=4, generator_index=1
    ).evaluate(words)
    comm_gpu = TorchCommutatorEvaluator(
        p=5,
        n=4,
        generator_index=1,
        device="cuda",
        batch_size=64,
        max_length=12,
    ).evaluate(words)
    if [item.projlen_history for item in comm_cpu] != [
        item.projlen_history for item in comm_gpu
    ]:
        raise AssertionError("commutator CPU/GPU evaluator mismatch")

    known = known_p5_factor_ids()
    known_evaluation = CPUExactEvaluator(p=5, n=4).evaluate_one(known)
    known_datta = analyze_factor_ids(known)
    if not known_evaluation.has_kernel or known_evaluation.final_projlen != 0:
        raise AssertionError("known p=5 kernel was not verified")
    if len(known_datta.defects) == 0:
        raise AssertionError("known p=5 trajectory has no Datta defects")

    parent = automaton.sample_uniform(20, rng)
    checked = 0
    for geometry in valid_geometries(
        len(parent),
        min_length=17,
        max_length=23,
        max_delete=6,
        max_insert=6,
        max_net_delta=3,
    ):
        try:
            child = apply_geometry(parent, geometry, automaton, rng)
        except (ValueError, RuntimeError):
            continue
        if not automaton.is_legal(child):
            raise AssertionError("edit produced an illegal GNF word")
        checked += 1
        if checked == 100:
            break
    if checked != 100:
        raise AssertionError("could not validate 100 legal edits")

    payload = {
        "status": "passed",
        "algorithm": "structural-kernel-experiments-v1",
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "ordinary_cpu_gpu_words": len(words),
        "commutator_cpu_gpu_words": len(words),
        "commutator_generator": 1,
        "known_p5_kernel_verified": True,
        "known_p5_datta_defects": len(known_datta.defects),
        "legal_edits_checked": checked,
    }
    marker = write_json(args.marker, payload)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Validation marker: {marker}", flush=True)


if __name__ == "__main__":
    main()
