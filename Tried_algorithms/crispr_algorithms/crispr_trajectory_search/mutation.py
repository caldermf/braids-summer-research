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
    """Choose and learn useful GNF block-edit locations and sizes."""

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
        self.location_stats = defaultdict(lambda: [0, 0.0])
        self.block_stats = defaultdict(lambda: [0, 0.0])

    def _choose_mode(self) -> str:
        value = self.rng.random()
        targeted_cutoff = self.config.targeted_location_fraction
        random_cutoff = targeted_cutoff + self.config.random_location_fraction
        if value < targeted_cutoff:
            return "targeted"
        if value < random_cutoff:
            return "random"
        return "adaptive"

    def _choose_block_length(self, horizon: int) -> int:
        candidates = [
            min(length, horizon)
            for length in self.config.mutation_block_sizes
        ]
        candidates = tuple(dict.fromkeys(candidates))
        weights = []
        total_attempts = sum(self.block_stats[length][0] for length in candidates) + 1
        for length in candidates:
            attempts, reward = self.block_stats[length]
            mean = reward / attempts if attempts else 0.0
            exploration = math.sqrt(math.log(total_attempts + 1) / (attempts + 1))
            weights.append(math.exp(max(-5.0, min(5.0, mean + exploration))))
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
                stagnation = 1 if history[position] == history[position - 1] else 0
                local_weight += 2.0 * increase + stagnation
            if end >= late_start:
                local_weight *= 2.0
            weights.append(local_weight)
        return starts, weights

    def _choose_start(
        self,
        evaluation: TrajectoryEvaluation,
        block_length: int,
        mode: str,
    ) -> int:
        horizon = evaluation.trajectory.horizon
        maximum = horizon - block_length
        if maximum <= 0:
            return 0
        if mode == "random":
            return self.rng.randint(0, maximum)
        if mode == "targeted":
            starts, weights = self._targeted_start_weights(evaluation, block_length)
            return weighted_choice(starts, weights, self.rng)

        bins = tuple(range(self.config.location_bins))
        total_attempts = sum(self.location_stats[location_bin][0] for location_bin in bins) + 1
        weights = []
        for location_bin in bins:
            attempts, reward = self.location_stats[location_bin]
            mean = reward / attempts if attempts else 0.0
            exploration = math.sqrt(math.log(total_attempts + 1) / (attempts + 1))
            weights.append(math.exp(max(-5.0, min(5.0, mean + exploration))))
        chosen_bin = weighted_choice(bins, weights, self.rng)
        low = math.floor(chosen_bin * (maximum + 1) / self.config.location_bins)
        high = math.ceil((chosen_bin + 1) * (maximum + 1) / self.config.location_bins) - 1
        return self.rng.randint(max(0, low), min(maximum, max(low, high)))

    def mutate_once(
        self,
        evaluation: TrajectoryEvaluation,
        factor_ids: Sequence[int] | None = None,
    ) -> tuple[tuple[int, ...], MutationRecord]:
        original = tuple(factor_ids or evaluation.trajectory.factor_ids)
        horizon = len(original)
        block_length = self._choose_block_length(horizon)
        mode = self._choose_mode()
        start = self._choose_start(evaluation, block_length, mode)
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
                    mode=mode,
                    start=start,
                    block_length=block_length,
                    location_bin=self._location_bin(start, horizon),
                )
                return mutated, record

        # The original bridge is always legal, so failure here only means the
        # sampler repeatedly returned it. Force a uniform retry.
        replacement = self.automaton.sample_bridge(
            left=left,
            right=right,
            length=block_length,
            rng=self.rng,
            absolute_start=start,
            horizon=horizon,
        )
        record = MutationRecord(
            mode=mode,
            start=start,
            block_length=block_length,
            location_bin=self._location_bin(start, horizon),
        )
        return original[:start] + replacement + original[end:], record

    def make_child(
        self,
        parent: TrajectoryEvaluation,
        two_mutations: bool,
    ) -> Trajectory:
        factor_ids, first = self.mutate_once(parent)
        records = [first]
        if two_mutations:
            factor_ids, second = self.mutate_once(parent, factor_ids=factor_ids)
            records.append(second)
        return Trajectory(
            factor_ids=factor_ids,
            origin="mutation",
            parent_id=parent.trajectory.trajectory_id,
            parent_score=parent.score,
            mutation_records=tuple(records),
        )

    def observe(self, trajectory: Trajectory, child_score: float) -> None:
        if trajectory.parent_score is None:
            return
        reward = max(-5.0, min(5.0, (child_score - trajectory.parent_score) / 10.0))
        for record in trajectory.mutation_records:
            self.location_stats[record.location_bin][0] += 1
            self.location_stats[record.location_bin][1] += reward
            self.block_stats[record.block_length][0] += 1
            self.block_stats[record.block_length][1] += reward

    def stats_json(self) -> dict:
        return {
            "locations": {
                str(key): {"attempts": value[0], "reward_sum": value[1]}
                for key, value in sorted(self.location_stats.items())
            },
            "block_lengths": {
                str(key): {"attempts": value[0], "reward_sum": value[1]}
                for key, value in sorted(self.block_stats.items())
            },
        }
