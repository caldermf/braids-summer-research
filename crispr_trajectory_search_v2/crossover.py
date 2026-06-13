from __future__ import annotations

import random

from .config import SearchConfig
from .gnf import GNFAutomaton
from .models import Trajectory, TrajectoryEvaluation
from .transition_model import TransitionModel


class SuffixCrossover:
    """Graft a donor suffix onto a recipient through a legal GNF bridge."""

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

    def make_child(
        self,
        recipient: TrajectoryEvaluation,
        donor: TrajectoryEvaluation,
    ) -> Trajectory:
        if recipient.trajectory.horizon != donor.trajectory.horizon:
            raise ValueError("crossover parents must have the same horizon")

        left_word = recipient.trajectory.factor_ids
        right_word = donor.trajectory.factor_ids
        horizon = len(left_word)
        late_start = max(1, int(horizon * self.config.late_start_fraction))
        bridge_choices = tuple(
            length
            for length in self.config.local_mutation_block_sizes
            + self.config.escape_mutation_block_sizes
            if length < horizon
        )

        for _ in range(self.config.mutation_attempts):
            bridge_length = min(self.rng.choice(bridge_choices), horizon - 1)
            start_max = horizon - bridge_length
            start = self.rng.randint(min(late_start, start_max), start_max)
            end = start + bridge_length
            left = left_word[start - 1] if start else None
            right = right_word[end] if end < horizon else None
            try:
                bridge = self.automaton.sample_bridge(
                    left=left,
                    right=right,
                    length=bridge_length,
                    rng=self.rng,
                    chooser=self.transition_model.choose,
                    absolute_start=start,
                    horizon=horizon,
                )
            except ValueError:
                continue
            factors = left_word[:start] + bridge + right_word[end:]
            if factors != left_word and factors != right_word:
                return Trajectory(
                    factor_ids=factors,
                    origin="crossover",
                    parent_id=(
                        f"{recipient.trajectory.trajectory_id},"
                        f"{donor.trajectory.trajectory_id}"
                    ),
                    parent_score=max(recipient.score, donor.score),
                )

        return Trajectory(
            factor_ids=left_word,
            origin="crossover_fallback",
            parent_id=recipient.trajectory.trajectory_id,
            parent_score=recipient.score,
        )
