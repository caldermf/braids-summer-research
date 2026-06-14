from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

from .caches import EvaluationCache, MatrixStateTranspositionTable, SeenWordCache
from .config import SearchConfig
from .islands import island_rank
from .models import Trajectory, TrajectoryEvaluation
from .mutation import AdaptiveSuffixMutationPlanner


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


class SuffixMCTSFinisher:
    """Selective batched MCTS that rewrites terminal GNF blocks."""

    def __init__(
        self,
        config: SearchConfig,
        evaluator,
        evaluation_cache: EvaluationCache,
        seen_words: SeenWordCache,
        planners: dict[str, AdaptiveSuffixMutationPlanner],
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

    def _expand(
        self,
        node: MCTSNode,
        island: str,
        generation: int,
    ) -> Trajectory | None:
        planner = self.planners[island]
        horizon = node.evaluation.trajectory.horizon
        block_lengths = tuple(
            min(horizon, value) for value in self.config.mcts_block_sizes
        )
        for _ in range(self.config.mutation_attempts):
            block_length = self.rng.choice(block_lengths)
            factors, record = planner.mutate_once(
                node.evaluation,
                stagnant=True,
                force_large=node.depth >= self.config.mcts_max_depth // 2,
                block_length=block_length,
            )
            if factors == node.evaluation.trajectory.factor_ids:
                continue
            if not self.seen_words.add(factors):
                self.duplicate_rejections += 1
                continue
            self.generated += 1
            return self.assign_id(
                Trajectory(
                    factor_ids=factors,
                    island=island,
                    origin="mcts_suffix",
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
    ) -> list[TrajectoryEvaluation]:
        roots: list[tuple[str, MCTSNode]] = []
        for island, seeds in seeds_by_island.items():
            for seed in seeds:
                roots.append((island, MCTSNode(seed)))
                self.transpositions.add(seed, self.config.mcts_max_depth)
        if not roots:
            return []

        self.total_runs += 1
        best_by_root = {
            id(root): root.evaluation
            for _, root in roots
        }
        all_kernel_hits: list[TrajectoryEvaluation] = []

        for _ in range(self.config.mcts_simulations_per_seed):
            pending: list[tuple[str, MCTSNode, Trajectory]] = []
            for island, root in roots:
                selected = self._select(root)
                if selected.depth >= self.config.mcts_max_depth:
                    selected = root
                trajectory = self._expand(selected, island, generation)
                if trajectory is not None:
                    pending.append((island, selected, trajectory))
            if not pending:
                break

            evaluations = self.evaluation_cache.evaluate(
                self.evaluator,
                [trajectory for _, _, trajectory in pending],
            )
            self.total_simulations += len(evaluations)
            for (island, parent, _), evaluation in zip(pending, evaluations):
                child = MCTSNode(
                    evaluation=evaluation,
                    parent=parent,
                    depth=parent.depth + 1,
                )
                parent.children.append(child)
                remaining = self.config.mcts_max_depth - child.depth
                self.transpositions.add(evaluation, remaining)
                value = evaluation.score_for(island) + 0.25 * evaluation.novelty
                self._backpropagate(child, value)
                root = child
                while root.parent is not None:
                    root = root.parent
                current_best = best_by_root[id(root)]
                if island_rank(evaluation, island) > island_rank(current_best, island):
                    best_by_root[id(root)] = evaluation
                if evaluation.has_kernel:
                    all_kernel_hits.append(evaluation)

            if all_kernel_hits and self.config.stop_at_kernel:
                break

        improvements = []
        for island, root in roots:
            best = best_by_root[id(root)]
            if island_rank(best, island) > island_rank(root.evaluation, island):
                improvements.append(best)
        return all_kernel_hits + improvements

    def stats(self) -> dict:
        return {
            "runs": self.total_runs,
            "simulations": self.total_simulations,
            "generated": self.generated,
            "duplicate_rejections": self.duplicate_rejections,
            "transpositions": self.transpositions.stats(),
        }
