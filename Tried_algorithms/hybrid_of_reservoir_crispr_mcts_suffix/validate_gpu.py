from __future__ import annotations

import json
import os
import random
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is required for hybrid GPU validation") from exc

try:
    from crispr_algorithms.bidirectional_matrix_search_v5.config import (
        SearchConfig as SuffixSearchConfig,
    )
    from crispr_algorithms.bidirectional_matrix_search_v5.exact_evaluator import (
        CPUExactEvaluator,
        TorchExactEvaluator,
    )
    from crispr_algorithms.bidirectional_matrix_search_v5.field_sketch import (
        ExtensionFieldSketch,
    )
    from crispr_algorithms.bidirectional_matrix_search_v5.models import Segment
    from crispr_algorithms.crispr_trajectory_search_v4.config import (
        SearchConfig as CrisprSearchConfig,
    )
    from crispr_algorithms.crispr_trajectory_search_v4.evaluators import (
        CPUTrajectoryEvaluator,
        TorchTrajectoryEvaluator,
    )
    from crispr_algorithms.crispr_trajectory_search_v4.gnf import GNFAutomaton
    from crispr_algorithms.crispr_trajectory_search_v4.models import Trajectory
except ModuleNotFoundError:
    from bidirectional_matrix_search_v5.config import SearchConfig as SuffixSearchConfig
    from bidirectional_matrix_search_v5.exact_evaluator import (
        CPUExactEvaluator,
        TorchExactEvaluator,
    )
    from bidirectional_matrix_search_v5.field_sketch import ExtensionFieldSketch
    from bidirectional_matrix_search_v5.models import Segment
    from crispr_trajectory_search_v4.config import SearchConfig as CrisprSearchConfig
    from crispr_trajectory_search_v4.evaluators import (
        CPUTrajectoryEvaluator,
        TorchTrajectoryEvaluator,
    )
    from crispr_trajectory_search_v4.gnf import GNFAutomaton
    from crispr_trajectory_search_v4.models import Trajectory


def validate_crispr(words: list[tuple[int, ...]]) -> None:
    base = CrisprSearchConfig(
        p=5,
        n=4,
        min_horizon=3,
        initial_max_horizon=8,
        hard_max_horizon=8,
        population_size=len(words),
        min_generations=1,
        max_generations=1,
        eval_batch_size=7,
        mcts_enabled=False,
    )
    trajectories = [
        Trajectory(word, island="endpoint", trajectory_id=f"validation-{index}")
        for index, word in enumerate(words)
    ]
    cpu = CPUTrajectoryEvaluator(base).evaluate(trajectories)
    gpu = TorchTrajectoryEvaluator(
        replace(base, backend="torch", device="cuda")
    ).evaluate(trajectories)
    for cpu_item, gpu_item in zip(cpu, gpu):
        if cpu_item.projlen_history != gpu_item.projlen_history:
            raise AssertionError("CRISPR CPU/CUDA projective-length mismatch")


def validate_suffix(words: list[tuple[int, ...]]) -> None:
    base = SuffixSearchConfig(
        p=5,
        n=4,
        prefix_count=len(words),
        suffix_count=len(words),
        generations=1,
        prefix_length_min=3,
        prefix_length_max=8,
        suffix_length_min=1,
        suffix_length_max=8,
        field_points=8,
        signature_batch_size=7,
        exact_batch_size=7,
        backend="cpu",
        device="cpu",
    )
    cpu = CPUExactEvaluator(base).evaluate(words)
    gpu_config = replace(base, backend="torch", device="cuda")
    gpu = TorchExactEvaluator(gpu_config).evaluate(words)
    for cpu_item, gpu_item in zip(cpu, gpu):
        if cpu_item.projlen_history != gpu_item.projlen_history:
            raise AssertionError("suffix exact CPU/CUDA projective-length mismatch")

    segments = [
        Segment(word, role="suffix", segment_id=f"validation-{index}")
        for index, word in enumerate(words)
    ]
    cpu_signatures = ExtensionFieldSketch(base).suffix_signatures(segments)
    gpu_signatures = ExtensionFieldSketch(gpu_config).suffix_signatures(segments)
    if not np.array_equal(cpu_signatures, gpu_signatures):
        raise AssertionError("GF(p^2) suffix signatures differ between CPU and CUDA")


def main() -> None:
    if os.environ.get("SLURM_JOB_PARTITION") != "scavenge_gpu":
        raise SystemExit("hybrid GPU validation must run on scavenge_gpu")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this scavenge_gpu allocation")

    automaton = GNFAutomaton(4)
    rng = random.Random(20260614)
    words = [automaton.sample_uniform(rng.randint(3, 8), rng) for _ in range(32)]
    validate_crispr(words)
    validate_suffix(words)

    marker = Path(
        os.environ.get(
            "HYBRID_VALIDATION_MARKER",
            "results/hybrid_validation/scavenge_gpu_validated.json",
        )
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "passed",
        "algorithm": "hybrid_of_reservoir_crispr_mcts_suffix",
        "partition": os.environ["SLURM_JOB_PARTITION"],
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "random_legal_words_compared": len(words),
        "crispr_cpu_cuda_match": True,
        "suffix_exact_cpu_cuda_match": True,
        "suffix_gf_p2_cpu_cuda_match": True,
    }
    marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
