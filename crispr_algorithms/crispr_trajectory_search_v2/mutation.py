from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Sequence

from .config import SearchConfig
from .gnf import GNFAutomaton
from .models import MutationRecord, Trajectory, TrajectoryEvaluation
from .transition_model import TransitionModel, weighted_choice


class MutationPlanner:
    """Learn successful local-refinement and basin-escape GNF edits."""

    def __init__(
        self,
        config: SearchConfig,
        automaton: GNFAutomaton,
        transition_model: TransitionModel,
        rng: random.Random,
    ):
        self.config = config
        self.automaton = automaton
        self.transition_model = transition_model
        self.rng = rng
        self.location_stats = defaultdict(
            lambda: {
                "attempts": 0.0,
                "successes": 0.0,
                "archive_survivals": 0.0,
                "improvement_sum": 0.0,
            }
        )
        self.block_stats = defaultdict(
            lambda: {
                "attempts": 0.0,
                "successes": 0.0,
                "archive_survivals": 0.0,
                "improvement_sum": 0.0,
            }
        )

    def decay_statistics(self) -> None:
        decay = self.config.mutation_statistics_decay
        for table in (self.location_stats, self.block_stats):
            for values in table.values():
                for key in values:
                    values[key] *= decay

    def _block_candidates(self, lane: str, horizon: int) -> tuple[int, ...]:
        configured = (
            self.config.local_mutation_block_sizes
            if lane == "local"
            else self.config.escape_mutation_block_sizes
        )
        return tuple(dict.fromkeys(min(length, horizon) for length in configured))

    def _adaptive_weight(self, stats: dict, total_attempts: float) -> float:
        attempts = stats["attempts"]
        success_rate = (
            stats["successes"]
            + self.config.mutation_archive_bonus * stats["archive_survivals"]
            + 1.0
        ) / (attempts + 2.0)
        mean_improvement = stats["improvement_sum"] / max(1.0, attempts)
        exploration = math.sqrt(math.log(total_attempts + 2.0) / (attempts + 1.0))
        value = (
            self.config.mutation_success_bonus * success_rate
            + 0.10 * mean_improvement
            + exploration
        )
        return math.exp(max(-5.0, min(5.0, value)))

    def _choose_block_length(self, lane: str, horizon: int) -> int:
        candidates = self._block_candidates(lane, horizon)
        total_attempts = sum(
            self.block_stats[(lane, length)]["attempts"] for length in candidates
        )
        weights = [
            self._adaptive_weight(
                self.block_stats[(lane, length)],
                total_attempts,
            )
            for length in candidates
        ]
        return weighted_choice(candidates, weights, self.rng)

    def _location_bin(self, start: int, horizon: int) -> int:
        return min(
            self.config.location_bins - 1,
            int(self.config.location_bins * start / max(1, horizon)),
        )

    def _targeted_start_weights(
        self,
        evaluation: TrajectoryEvaluation,
        block_length: int,
        lane: str,
    ) -> tuple[list[int], list[float]]:
        history = evaluation.projlen_history
        horizon = len(history)
        starts = list(range(0, horizon - block_length + 1))
        late_start = int(horizon * self.config.late_start_fraction)
        weights = []
        for start in starts:
            end = start + block_length
            local_weight = 1.0
            for position in range(max(1, start), min(horizon, end + 1)):
                increase = max(0, history[position] - history[position - 1])
                stagnation = history[position] == history[position - 1]
                local_weight += 2.0 * increase + float(stagnation)
            if lane == "local" and end >= late_start:
                local_weight *= self.config.terminal_location_bias
            weights.append(local_weight)
        return starts, weights

    def _choose_start(
        self,
        evaluation: TrajectoryEvaluation,
        block_length: int,
        lane: str,
    ) -> tuple[int, str]:
        horizon = evaluation.trajectory.horizon
        maximum = horizon - block_length
        if maximum <= 0:
            return 0, "full"

        if lane == "escape" and self.rng.random() < 0.65:
            return self.rng.randint(0, maximum), "random"

        if lane == "local" or self.rng.random() < 0.50:
            starts, weights = self._targeted_start_weights(
                evaluation,
                block_length,
                lane,
            )
            return weighted_choice(starts, weights, self.rng), "targeted"

        bins = tuple(range(self.config.location_bins))
        total_attempts = sum(
            self.location_stats[(lane, location_bin)]["attempts"]
            for location_bin in bins
        )
        weights = [
            self._adaptive_weight(
                self.location_stats[(lane, location_bin)],
                total_attempts,
            )
            for location_bin in bins
        ]
        chosen_bin = weighted_choice(bins, weights, self.rng)
        low = math.floor(chosen_bin * (maximum + 1) / self.config.location_bins)
        high = math.ceil(
            (chosen_bin + 1) * (maximum + 1) / self.config.location_bins
        ) - 1
        start = self.rng.randint(max(0, low), min(maximum, max(low, high)))
        return start, "adaptive"

    def mutate_once(
        self,
        evaluation: TrajectoryEvaluation,
        lane: str,
        factor_ids: Sequence[int] | None = None,
    ) -> tuple[tuple[int, ...], MutationRecord]:
        if lane not in {"local", "escape"}:
            raise ValueError("mutation lane must be 'local' or 'escape'")

        original = tuple(factor_ids or evaluation.trajectory.factor_ids)
        horizon = len(original)
        block_length = self._choose_block_length(lane, horizon)
        start, mode = self._choose_start(evaluation, block_length, lane)
        end = start + block_length
        left = original[start - 1] if start > 0 else None
        right = original[end] if end < horizon else None

        for _ in range(self.config.mutation_attempts):
            replacement = self.automaton.sample_bridge(
                left=left,
                right=right,
                length=block_length,
                rng=self.rng,
                chooser=self.transition_model.choose,
                absolute_start=start,
                horizon=horizon,
            )
            mutated = original[:start] + replacement + original[end:]
            if mutated != original:
                record = MutationRecord(
                    lane=lane,
                    mode=mode,
                    start=start,
                    block_length=block_length,
                    location_bin=self._location_bin(start, horizon),
                )
                return mutated, record

        replacement = self.automaton.sample_bridge(
            left=left,
            right=right,
            length=block_length,
            rng=self.rng,
            absolute_start=start,
            horizon=horizon,
        )
        record = MutationRecord(
            lane=lane,
            mode=mode,
            start=start,
            block_length=block_length,
            location_bin=self._location_bin(start, horizon),
        )
        return original[:start] + replacement + original[end:], record

    def make_child(
        self,
        parent: TrajectoryEvaluation,
        lane: str,
        two_mutations: bool = False,
    ) -> Trajectory:
        factor_ids, first = self.mutate_once(parent, lane=lane)
        records = [first]
        if two_mutations:
            factor_ids, second = self.mutate_once(
                parent,
                lane=lane,
                factor_ids=factor_ids,
            )
            records.append(second)
        return Trajectory(
            factor_ids=factor_ids,
            origin=f"{lane}_mutation",
            parent_id=parent.trajectory.trajectory_id,
            parent_score=parent.score,
            mutation_records=tuple(records),
        )

    def observe(
        self,
        evaluation: TrajectoryEvaluation,
        archive_survivor: bool,
    ) -> None:
        trajectory = evaluation.trajectory
        if trajectory.parent_score is None or not trajectory.mutation_records:
            return

        improvement = max(0.0, evaluation.score - trajectory.parent_score)
        successful = improvement > 0.0
        for record in trajectory.mutation_records:
            keys = (
                (self.location_stats, (record.lane, record.location_bin)),
                (self.block_stats, (record.lane, record.block_length)),
            )
            for table, key in keys:
                stats = table[key]
                stats["attempts"] += 1.0
                stats["successes"] += float(successful)
                stats["archive_survivals"] += float(archive_survivor)
                stats["improvement_sum"] += improvement

    def stats_json(self) -> dict:
        def serialize(table):
            output = {}
            for key, values in sorted(table.items()):
                lane, value = key
                attempts = values["attempts"]
                output[f"{lane}:{value}"] = {
                    **{name: round(number, 6) for name, number in values.items()},
                    "success_rate": round(
                        values["successes"] / max(1.0, attempts),
                        6,
                    ),
                    "archive_survival_rate": round(
                        values["archive_survivals"] / max(1.0, attempts),
                        6,
                    ),
                    "mean_positive_improvement": round(
                        values["improvement_sum"] / max(1.0, attempts),
                        6,
                    ),
                }
            return output

        return {
            "locations": serialize(self.location_stats),
            "block_lengths": serialize(self.block_stats),
        }
