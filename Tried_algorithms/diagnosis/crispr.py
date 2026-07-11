from __future__ import annotations

import math

from crispr_trajectory_search_v4.config import ISLAND_NAMES, SearchConfig
from crispr_trajectory_search_v4.fitness import build_evaluation
from crispr_trajectory_search_v4.islands import island_rank
from crispr_trajectory_search_v4.models import Trajectory

from .models import AuditConfig, Candidate


def _evaluation(candidate: Candidate, config: SearchConfig):
    trajectory = Trajectory(
        factor_ids=candidate.factor_ids,
        island="endpoint",
        origin="diagnosis",
    )
    kernel_depths = (candidate.depth,) if candidate.kernel_match.get("matches") else ()
    kernel_matches = (candidate.kernel_match,) if kernel_depths else ()
    return build_evaluation(
        trajectory=trajectory,
        projlen_history=candidate.projlen_history,
        config=config,
        kernel_depths=kernel_depths,
        kernel_matches=kernel_matches,
    )


def sampled_crispr_ranks(
    target: Candidate,
    sample: list[Candidate],
    audit_config: AuditConfig,
) -> dict:
    config = SearchConfig(
        p=audit_config.p,
        n=audit_config.n,
        population_size=audit_config.crispr_population_size,
        mcts_enabled=False,
    )
    target_evaluation = _evaluation(target, config)
    sample_evaluations = [_evaluation(candidate, config) for candidate in sample]
    result = {
        "crispr_sample_size": len(sample_evaluations),
    }
    for island in ISLAND_NAMES:
        target_rank = island_rank(target_evaluation, island)
        better = sum(
            island_rank(evaluation, island) > target_rank
            for evaluation in sample_evaluations
        )
        equal = sum(
            island_rank(evaluation, island) == target_rank
            for evaluation in sample_evaluations
        )
        sample_size = len(sample_evaluations)
        estimated_rank = (
            1 + math.ceil((better / sample_size) * audit_config.crispr_population_size)
            if sample_size
            else 1
        )
        island_capacity = config.island_sizes[island]
        result[f"crispr_{island}_sample_best_rank"] = better + 1
        result[f"crispr_{island}_sample_worst_rank"] = better + equal
        result[f"crispr_{island}_sample_percentile"] = (
            1.0 - better / sample_size if sample_size else 1.0
        )
        result[f"crispr_{island}_estimated_population_rank"] = estimated_rank
        result[f"crispr_{island}_island_capacity"] = island_capacity
        result[f"crispr_{island}_proxy_selected"] = estimated_rank <= island_capacity
    return result
