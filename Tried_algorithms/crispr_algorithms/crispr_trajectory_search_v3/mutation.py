from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Sequence

from .config import SearchConfig
from .gnf import GNFAutomaton
from .models import MutationRecord, Trajectory, TrajectoryEvaluation
from .transition_model import TransitionModel, weighted_choice


class AdaptiveSuffixMutationPlanner:
    """Island-specific legal block rewriting with adaptive suffix expansion."""

    def __init__(
        self,
        config: SearchConfig,
        island: str,
        automaton: GNFAutomaton,
        transition_model: TransitionModel,
        rng: random.Random,
    ):
        self.config = config
        self.island = island
        self.automaton = automaton
        self.transition_model = transition_model
        self.rng = rng
        self.stats = defaultdict(
            lambda: {"attempts": 0.0, "successes": 0.0, "improvement_sum": 0.0}
        )

    def reset_statistics(self) -> None:
        self.stats.clear()

    def decay_statistics(self) -> None:
        for values in self.stats.values():
            for key in values:
                values[key] *= self.config.mutation_statistics_decay

    def _adaptive_weight(self, block_length: int) -> float:
        values = self.stats[block_length]
        attempts = values["attempts"]
        success = (values["successes"] + 1.0) / (attempts + 2.0)
        improvement = values["improvement_sum"] / max(1.0, attempts)
        total = sum(item["attempts"] for item in self.stats.values())
        exploration = math.sqrt(math.log(total + 2.0) / (attempts + 1.0))
        return math.exp(max(-5.0, min(5.0, 2.0 * success + 0.05 * improvement + exploration)))

    def choose_block_length(self, horizon: int, stagnant: bool, force_large: bool = False) -> int:
        candidates = tuple(
            dict.fromkeys(min(horizon, value) for value in self.config.block_sizes_for(self.island, stagnant))
        )
        if force_large:
            candidates = tuple(value for value in candidates if value >= max(candidates) // 2)
        return weighted_choice(
            candidates,
            [self._adaptive_weight(value) for value in candidates],
            self.rng,
        )

    def choose_start(
        self,
        evaluation: TrajectoryEvaluation,
        block_length: int,
        force_large: bool = False,
    ) -> tuple[int, str]:
        horizon = evaluation.trajectory.horizon
        maximum = horizon - block_length
        if maximum <= 0:
            return 0, "full"
        if force_large and self.rng.random() < 0.35:
            return self.rng.randint(0, maximum), "restart_global"

        terminal_start = max(0, horizon - max(block_length, round(horizon * 0.45)))
        starts = list(range(terminal_start, maximum + 1))
        weights = []
        history = evaluation.projlen_history
        for start in starts:
            end = start + block_length
            weight = 1.0 + self.config.terminal_location_bias * (end / horizon) ** 3
            for position in range(max(1, start), min(horizon, end + 1)):
                weight += 2.0 * max(0, history[position] - history[position - 1])
            if end == horizon:
                weight *= 2.0
            weights.append(weight)
        return weighted_choice(starts, weights, self.rng), "adaptive_suffix"

    def mutate_once(
        self,
        parent: TrajectoryEvaluation,
        *,
        stagnant: bool = False,
        force_large: bool = False,
        factor_ids: Sequence[int] | None = None,
        block_length: int | None = None,
    ) -> tuple[tuple[int, ...], MutationRecord]:
        original = tuple(factor_ids or parent.trajectory.factor_ids)
        horizon = len(original)
        chosen_length = block_length or self.choose_block_length(
            horizon,
            stagnant=stagnant,
            force_large=force_large,
        )
        chosen_length = min(chosen_length, horizon)
        start, mode = self.choose_start(parent, chosen_length, force_large=force_large)
        end = start + chosen_length
        left = original[start - 1] if start else None
        right = original[end] if end < horizon else None

        for _ in range(self.config.mutation_attempts):
            replacement = self.automaton.sample_bridge(
                left=left,
                right=right,
                length=chosen_length,
                rng=self.rng,
                chooser=self.transition_model.choose,
                absolute_start=start,
                horizon=horizon,
            )
            candidate = original[:start] + replacement + original[end:]
            if candidate != original:
                break
        else:
            candidate = original

        record = MutationRecord(
            island=self.island,
            mode=mode,
            start=start,
            block_length=chosen_length,
            location_bin=min(7, int(8 * start / max(1, horizon))),
        )
        return candidate, record

    def make_child(
        self,
        parent: TrajectoryEvaluation,
        *,
        stagnant: bool = False,
        force_large: bool = False,
        two_mutations: bool = False,
    ) -> Trajectory:
        factors, first = self.mutate_once(
            parent,
            stagnant=stagnant,
            force_large=force_large,
        )
        records = [first]
        if two_mutations:
            factors, second = self.mutate_once(
                parent,
                stagnant=stagnant,
                force_large=force_large,
                factor_ids=factors,
            )
            records.append(second)
        return Trajectory(
            factor_ids=factors,
            island=self.island,
            origin="restart_mutation" if force_large else "adaptive_mutation",
            parent_id=parent.trajectory.trajectory_id,
            parent_score=parent.score_for(self.island),
            mutation_records=tuple(records),
        )

    def observe(self, evaluation: TrajectoryEvaluation) -> None:
        parent_score = evaluation.trajectory.parent_score
        if parent_score is None:
            return
        improvement = evaluation.score_for(self.island) - parent_score
        for record in evaluation.trajectory.mutation_records:
            values = self.stats[record.block_length]
            values["attempts"] += 1.0
            values["successes"] += float(improvement > 0.0)
            values["improvement_sum"] += max(0.0, improvement)

    def stats_json(self) -> dict:
        return {
            str(length): {
                **{name: round(value, 6) for name, value in values.items()},
                "success_rate": round(
                    values["successes"] / max(1.0, values["attempts"]),
                    6,
                ),
            }
            for length, values in sorted(self.stats.items())
        }


MutationPlanner = AdaptiveSuffixMutationPlanner
