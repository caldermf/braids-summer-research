from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Callable

from .caches import EvaluationCache, MatrixStateTranspositionTable, SeenWordCache
from .config import SearchConfig
from .islands import island_rank
from .models import Trajectory, TrajectoryEvaluation
from .mutation import StructuralMutationPlanner


@dataclass
class MCTSNode:
    evaluation: TrajectoryEvaluation
    parent: "MCTSNode | None" = None
    depth: int = 0
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class StructuralMCTSFinisher:
    """Batched MCTS over suffix rewrites and legal length-changing actions."""

    def __init__(
        self,
        config: SearchConfig,
        evaluator,
        evaluation_cache: EvaluationCache,
        seen_words: SeenWordCache,
        planners: dict[str, StructuralMutationPlanner],
        rng: random.Random,
        assign_id: Callable[[Trajectory, int], Trajectory],
    ):
        self.config = config
        self.evaluator = evaluator
        self.evaluation_cache = evaluation_cache
        self.seen_words = seen_words
        self.planners = planners
        self.rng = rng
        self.assign_id = assign_id
        self.transpositions = MatrixStateTranspositionTable()
        self.total_runs = 0
        self.total_simulations = 0
        self.generated = 0
        self.duplicate_rejections = 0
        self.action_counts: dict[str, int] = {}

    def _select(self, root: MCTSNode) -> MCTSNode:
        node = root
        while (
            node.depth < self.config.mcts_max_depth
            and len(node.children) >= self.config.mcts_branching_factor
        ):
            log_parent = math.log(max(2, node.visits))
            node = max(
                node.children,
                key=lambda child: child.mean_value
                + self.config.mcts_exploration
                * math.sqrt(log_parent / max(1, child.visits)),
            )
        return node

    def _choose_action(self, horizon: int, active_max_horizon: int) -> str:
        if self.rng.random() < self.config.mcts_length_edit_fraction:
            actions = []
            if horizon < active_max_horizon:
                actions.extend(("append", "insert"))
            if horizon > self.config.min_horizon:
                actions.extend(("truncate", "delete"))
            if actions:
                return self.rng.choice(actions)
        if self.rng.random() < 0.55:
            return "post_turn"
        return "replace"

    def _expand(
        self,
        node: MCTSNode,
        island: str,
        generation: int,
        active_max_horizon: int,
    ) -> Trajectory | None:
        planner = self.planners[island]
        for _ in range(self.config.mutation_attempts):
            action = self._choose_action(node.evaluation.trajectory.horizon, active_max_horizon)
            block_length = (
                self.rng.choice(self.config.mcts_block_sizes)
                if action == "replace"
                else None
            )
            factors, record = planner.mutate_once(
                node.evaluation,
                active_max_horizon=active_max_horizon,
                stagnant=True,
                force_large=node.depth >= self.config.mcts_max_depth // 2,
                use_learned=node.depth < self.config.mcts_max_depth // 2,
                allow_length_change=True,
                action=action,
                block_length=block_length,
            )
            if factors == node.evaluation.trajectory.factor_ids:
                continue
            if not self.seen_words.add(factors):
                self.duplicate_rejections += 1
                continue
            self.generated += 1
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
            return self.assign_id(
                Trajectory(
                    factor_ids=factors,
                    island=island,
                    origin=f"mcts_{action}",
                    parent_id=node.evaluation.trajectory.trajectory_id,
                    parent_score=node.evaluation.score_for(island),
                    mutation_records=(record,),
                ),
                generation,
            )
        return None

    @staticmethod
    def _backpropagate(node: MCTSNode, value: float) -> None:
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.value_sum += value
            current = current.parent

    def run(
        self,
        seeds_by_island: dict[str, list[TrajectoryEvaluation]],
        generation: int,
        active_max_horizon: int,
    ) -> list[TrajectoryEvaluation]:
        roots: list[tuple[str, MCTSNode, float]] = []
        for island, seeds in seeds_by_island.items():
            scores = [seed.score_for(island) for seed in seeds]
            scale = max(0.05, statistics.pstdev(scores) if len(scores) > 1 else 0.05)
            for seed in seeds:
                root = MCTSNode(seed)
                roots.append((island, root, scale))
                self.transpositions.add(seed, self.config.mcts_max_depth)
        if not roots:
            return []

        self.total_runs += 1
        best_by_root = {id(root): root.evaluation for _, root, _ in roots}
        kernel_hits: list[TrajectoryEvaluation] = []

        for _ in range(self.config.mcts_simulations_per_seed):
            pending: list[tuple[str, MCTSNode, MCTSNode, float, Trajectory]] = []
            for island, root, scale in roots:
                selected = self._select(root)
                if selected.depth >= self.config.mcts_max_depth:
                    selected = root
                trajectory = self._expand(
                    selected,
                    island,
                    generation,
                    active_max_horizon,
                )
                if trajectory is not None:
                    pending.append((island, root, selected, scale, trajectory))
            if not pending:
                break

            evaluations = self.evaluation_cache.evaluate(
                self.evaluator,
                [trajectory for _, _, _, _, trajectory in pending],
            )
            self.total_simulations += len(evaluations)
            for (island, root, parent, scale, _), evaluation in zip(pending, evaluations):
                child = MCTSNode(evaluation=evaluation, parent=parent, depth=parent.depth + 1)
                parent.children.append(child)
                remaining = self.config.mcts_max_depth - child.depth
                self.transpositions.add(evaluation, remaining)
                root_score = root.evaluation.score_for(island)
                normalized_gain = (evaluation.score_for(island) - root_score) / scale
                value = normalized_gain + 0.05 * evaluation.novelty
                self._backpropagate(child, value)
                current_best = best_by_root[id(root)]
                if island_rank(evaluation, island) > island_rank(current_best, island):
                    best_by_root[id(root)] = evaluation
                if evaluation.has_kernel:
                    kernel_hits.append(evaluation)
            if kernel_hits and self.config.stop_at_kernel:
                break

        improvements = []
        for island, root, _ in roots:
            best = best_by_root[id(root)]
            if island_rank(best, island) > island_rank(root.evaluation, island):
                improvements.append(best)
        return kernel_hits + improvements

    def stats(self) -> dict:
        return {
            "runs": self.total_runs,
            "simulations": self.total_simulations,
            "generated": self.generated,
            "duplicate_rejections": self.duplicate_rejections,
            "actions": dict(sorted(self.action_counts.items())),
            "transpositions": self.transpositions.stats(),
        }


SuffixMCTSFinisher = StructuralMCTSFinisher
