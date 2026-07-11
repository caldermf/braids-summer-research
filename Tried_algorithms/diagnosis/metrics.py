from __future__ import annotations

import math
import random
from dataclasses import dataclass

from monte_carlo_algorithms.periodic_frontier_reservoir_search import (
    projective_identity_distance,
    projective_target_distance,
    recent_downward_slope,
)
from peyl.braid_data import (
    GNF,
    append_factor_to_burau_matrix,
    delta_burau_matrix,
    identity_burau_matrix,
    polynomial_matrix_projlen,
    projective_kernel_match,
    simple_factor_burau_table,
    simple_factor_id_maps,
    valid_first_factor_ids,
    valid_suffix_factor_ids,
)

from .models import AuditConfig, Candidate


@dataclass
class DepthBaseline:
    max_depth: int

    def __post_init__(self) -> None:
        self.counts = [0] * (self.max_depth + 1)
        self.means = [0.0] * (self.max_depth + 1)
        self.m2 = [0.0] * (self.max_depth + 1)

    def add(self, depth: int, value: int) -> None:
        self.counts[depth] += 1
        count = self.counts[depth]
        delta = float(value) - self.means[depth]
        self.means[depth] += delta / count
        self.m2[depth] += delta * (float(value) - self.means[depth])

    def mean(self, depth: int) -> float:
        return self.means[depth] if self.counts[depth] else 2.0 * depth

    def std(self, depth: int) -> float:
        if self.counts[depth] < 2:
            return 1.0
        return max(1.0, math.sqrt(self.m2[depth] / (self.counts[depth] - 1)))

    def surprise(self, depth: int, value: int) -> float:
        return self.mean(depth) - float(value)

    def surprise_z(self, depth: int, value: int) -> float:
        return self.surprise(depth, value) / self.std(depth)

    def rows(self) -> list[dict]:
        return [
            {
                "depth": depth,
                "samples": self.counts[depth],
                "mean_projlen": self.mean(depth),
                "std_projlen": self.std(depth),
            }
            for depth in range(1, self.max_depth + 1)
        ]


def legal_actions(factor_ids: tuple[int, ...], n: int, delta_factor_id: int) -> list[int]:
    if n == 2:
        return [delta_factor_id]
    if not factor_ids:
        return valid_first_factor_ids(n=n)
    return valid_suffix_factor_ids(factor_ids[-1], n=n)


def estimate_baseline(
    config: AuditConfig,
    max_depth: int,
    simple_table: dict,
    rng: random.Random,
) -> DepthBaseline:
    baseline = DepthBaseline(max_depth)
    delta_factor_id = simple_factor_id_maps(config.n)[0][GNF.delta_perm(config.n)]
    for _ in range(config.baseline_samples):
        factors: tuple[int, ...] = ()
        matrix = identity_burau_matrix(p=config.p, n=config.n)
        for depth in range(1, max_depth + 1):
            action = rng.choice(legal_actions(factors, config.n, delta_factor_id))
            factors += (action,)
            matrix = append_factor_to_burau_matrix(
                current_matrix=matrix,
                factor_id=action,
                simple_table=simple_table,
                p=config.p,
            )
            baseline.add(depth, polynomial_matrix_projlen(matrix))
    return baseline


def root_candidate(config: AuditConfig) -> Candidate:
    return Candidate(
        factor_ids=(),
        matrix=identity_burau_matrix(p=config.p, n=config.n),
        projlen_history=(),
        periodic_distance_history=(),
        typical_projlen=0.0,
        surprise=0.0,
        surprise_z=0.0,
        periodic_distance=0.0,
        periodic_score=0.0,
        descent_score=0.0,
        mcts_value=0.0,
        breakout_value=0.0,
        kernel_match={"matches": False},
    )


def make_child(
    parent: Candidate,
    action: int,
    config: AuditConfig,
    max_depth: int,
    baseline: DepthBaseline,
    simple_table: dict,
    delta_target,
) -> Candidate:
    factors = parent.factor_ids + (int(action),)
    matrix = append_factor_to_burau_matrix(
        current_matrix=parent.matrix,
        factor_id=int(action),
        simple_table=simple_table,
        p=config.p,
    )
    depth = len(factors)
    projlen = polynomial_matrix_projlen(matrix)
    typical = baseline.mean(depth)
    surprise = baseline.surprise(depth, projlen)
    surprise_z = baseline.surprise_z(depth, projlen)
    history = parent.projlen_history + (projlen,)

    identity_distance = projective_identity_distance(matrix, p=config.p, n=config.n)
    delta_distance = projective_target_distance(
        matrix,
        delta_target,
        p=config.p,
        n=config.n,
    )
    periodic_distance = min(identity_distance, delta_distance)
    periodic_history = parent.periodic_distance_history + (periodic_distance,)

    recent_drop = max(0.0, float(parent.projlen - projlen)) if parent.depth else 0.0
    periodic_drop = (
        max(0.0, float(parent.periodic_distance - periodic_distance))
        if parent.depth
        else 0.0
    )
    slope = recent_downward_slope([float(value) for value in history], 8)
    periodic_slope = recent_downward_slope(periodic_history, 8)
    frontier_closeness = 1.0 / (1.0 + periodic_distance)
    low_projlen_advantage = max(0.0, 1.0 - projlen / max(1.0, typical))
    periodic_distance_norm = periodic_distance / max(1.0, float(depth))
    descent_multiplier = 2.0 if depth >= 35 else 1.0
    kernel_match = projective_kernel_match(matrix, p=config.p, n=config.n)

    periodic_score = surprise_z
    periodic_score += 0.1 * surprise / max(1.0, float(depth))
    periodic_score += 0.25 * low_projlen_advantage
    periodic_score += descent_multiplier * 0.25 * recent_drop
    periodic_score += descent_multiplier * 0.75 * slope
    periodic_score += 4.0 * frontier_closeness
    periodic_score -= 0.25 * periodic_distance_norm
    periodic_score += descent_multiplier * 0.8 * periodic_drop / 10.0
    periodic_score += descent_multiplier * periodic_slope / 10.0

    descent_score = -float(projlen) + 0.5 * surprise_z
    descent_score += descent_multiplier * (2.0 * recent_drop + 4.0 * slope)
    descent_score += descent_multiplier * (0.2 * periodic_drop + 0.4 * periodic_slope)
    descent_score -= 0.05 * periodic_distance_norm

    mcts_value = surprise_z
    mcts_value += 0.01 * surprise / max(1.0, float(depth))
    mcts_value += 0.001 * depth / max_depth

    prior_best_z = max(
        (
            baseline.surprise_z(index, value)
            for index, value in enumerate(parent.projlen_history, start=1)
        ),
        default=surprise_z,
    )
    breakout = max(0.0, surprise_z - prior_best_z)
    depth_weight = (depth / max_depth) ** config.breakout_depth_power
    breakout_value = depth_weight * surprise_z + config.breakout_weight * breakout
    breakout_value += 0.01 * surprise / max(1.0, float(depth))
    breakout_value += 0.001 * depth / max_depth

    if kernel_match.get("matches"):
        periodic_score += 1_000.0
        descent_score += 1_000.0
        mcts_value += 1_000.0
        breakout_value += 1_000.0

    return Candidate(
        factor_ids=factors,
        matrix=matrix,
        projlen_history=history,
        periodic_distance_history=periodic_history,
        typical_projlen=typical,
        surprise=surprise,
        surprise_z=surprise_z,
        periodic_distance=periodic_distance,
        periodic_score=periodic_score,
        descent_score=descent_score,
        mcts_value=mcts_value,
        breakout_value=breakout_value,
        kernel_match=kernel_match,
    )


def build_target_trajectory(
    factor_ids: tuple[int, ...],
    config: AuditConfig,
    max_depth: int,
    baseline: DepthBaseline,
    simple_table: dict,
) -> list[Candidate]:
    delta_target = delta_burau_matrix(p=config.p, n=config.n)
    parent = root_candidate(config)
    targets = []
    for action in factor_ids[:max_depth]:
        parent = make_child(
            parent,
            action,
            config=config,
            max_depth=max_depth,
            baseline=baseline,
            simple_table=simple_table,
            delta_target=delta_target,
        )
        targets.append(parent)
    return targets
