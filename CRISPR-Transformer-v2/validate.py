#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
for path in (REPO_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

from crispr_transformer.edits import apply_geometry, valid_geometries
from crispr_transformer.downturn import DownturnConfig, DownturnMonitor
from crispr_transformer.exact import (
    CPUExactEvaluator,
    TorchExactEvaluator,
    require_compatible_cuda,
)
from crispr_transformer.gnf import GNFAutomaton
from crispr_transformer.io_utils import write_json
from crispr_transformer.model import GeometryTransformer, ModelConfig
from crispr_transformer.percentiles import LengthPercentiles


KNOWN_P5_LENGTH54 = (
    7, 7, 10, 13, 4, 13, 4, 2, 13, 20, 13, 20, 13, 10, 2, 13, 4, 13,
    4, 13, 7, 21, 20, 13, 20, 13, 10, 16, 16, 2, 13, 4, 13, 4, 13, 21,
    20, 13, 20, 13, 21, 10, 13, 4, 13, 4, 2, 16, 13, 11, 13, 11, 13, 21,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", required=True)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    require_compatible_cuda(torch)

    rng = random.Random(args.seed)
    automaton = GNFAutomaton(4)
    compared = 32
    for p in (3, 5, 7):
        words = [automaton.sample_uniform(rng.randint(8, 24), rng) for _ in range(compared)]
        cpu = CPUExactEvaluator(p=p, n=4).evaluate(words)
        gpu = TorchExactEvaluator(p=p, n=4, device="cuda", batch_size=256).evaluate(words)
        if [item.projlen_history for item in cpu] != [item.projlen_history for item in gpu]:
            raise AssertionError(f"CPU/GPU evaluator mismatch for p={p}")

    parent = automaton.sample_uniform(40, rng)
    geometries = valid_geometries(
        len(parent),
        min_length=37,
        max_length=43,
        max_delete=16,
        max_insert=16,
        max_net_delta=3,
    )
    checked = 0
    rng.shuffle(geometries)
    for geometry in geometries:
        try:
            child = apply_geometry(parent, geometry, automaton, rng)
        except (ValueError, RuntimeError):
            continue
        if not automaton.is_legal(child):
            raise AssertionError("variable-length edit produced illegal GNF")
        if len(child) != len(parent) + geometry.length_delta:
            raise AssertionError("variable-length edit produced the wrong length")
        checked += 1
        if checked == 250:
            break
    if checked != 250:
        raise AssertionError("could not produce 250 nontrivial legal edits")

    known = CPUExactEvaluator(p=5, n=4).evaluate_one(KNOWN_P5_LENGTH54)
    if not known.has_kernel:
        raise AssertionError("known p=5 kernel was not verified")

    calibrated = LengthPercentiles.from_samples(
        p=7,
        n=4,
        samples={10: [8, 10, 12, 14], 11: [8, 10, 12, 14] * 10},
        effective_sample_size=4,
    )
    if calibrated.quality(10, 10) != calibrated.quality(11, 10):
        raise AssertionError("percentile quality depends on per-length sample count")

    downturn = DownturnMonitor(
        DownturnConfig(min_depth=48, confirmation_steps=2, extra_depths=2)
    )
    for depth, projlen in enumerate(
        [23, 23, 23, 22, 21, 20, 21, 20, 19, 17, 15, 13, 11],
        start=48,
    ):
        downturn_status = downturn.observe(depth, projlen)
    if not downturn_status["should_handoff"]:
        raise AssertionError("known p=5-style terminal downturn was not detected")

    config = ModelConfig(p=5, max_length=64)
    model = GeometryTransformer(config).cuda().eval()
    tokens = torch.tensor([[value + 1 for value in parent]], device="cuda")
    histories = torch.tensor(
        [CPUExactEvaluator(p=5, n=4).evaluate_one(parent).projlen_history],
        dtype=torch.float32,
        device="cuda",
    )
    lengths = torch.tensor([len(parent)], device="cuda")
    actions = torch.tensor(
        [
            (item.start, item.delete_length, item.insert_length)
            for item in geometries[:128]
        ],
        device="cuda",
    )
    parents = torch.zeros(len(actions), dtype=torch.long, device="cuda")
    with torch.no_grad():
        scores = model(tokens, histories, lengths, parents, actions)
    if scores.shape != (len(actions),) or not torch.isfinite(scores).all():
        raise AssertionError("geometry transformer produced invalid scores")

    payload = {
        "status": "passed",
        "algorithm": "CRISPR-Transformer-v2",
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "random_trajectories_compared_per_prime": compared,
        "primes_compared": [3, 5, 7],
        "legal_variable_edits_checked": checked,
        "all_integer_delete_lengths": [1, 16],
        "all_integer_insert_lengths": [1, 16],
        "known_p5_kernel_verified": True,
        "model_forward_verified": True,
        "calibrated_percentiles_verified": True,
        "adaptive_downturn_handoff_verified": True,
    }
    marker = write_json(args.marker, payload)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Validation marker: {marker}", flush=True)


if __name__ == "__main__":
    main()
