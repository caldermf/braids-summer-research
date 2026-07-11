from __future__ import annotations

import random

from .config import SearchConfig
from .gnf import GNFAutomaton
from .models import Trajectory, TrajectoryEvaluation
from .transition_model import TransitionModel


class SuffixCrossover:
    """Graft a donor suffix onto a recipient through a legal variable bridge."""

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
        active_max_horizon: int,
    ) -> Trajectory:
        left_word = recipient.trajectory.factor_ids
        right_word = donor.trajectory.factor_ids
        bridge_choices = tuple(
            value
            for value in dict.fromkeys(
                self.config.endpoint_block_sizes
                + self.config.envelope_block_sizes
                + self.config.collapse_block_sizes
                + self.config.suffix_block_sizes
            )
            if value <= 8
        ) or (1, 3, 5)

        for _ in range(self.config.mutation_attempts):
            recipient_cut = self.rng.randint(
                max(1, int(0.25 * len(left_word))),
                max(1, int(0.85 * len(left_word))),
            )
            donor_cut = self.rng.randint(
                max(0, int(0.25 * len(right_word))),
                max(0, int(0.85 * len(right_word))),
            )
            donor_suffix = right_word[donor_cut:]
            bridge_length = self.rng.choice(bridge_choices)
            result_length = recipient_cut + bridge_length + len(donor_suffix)
            if not self.config.min_horizon <= result_length <= active_max_horizon:
                continue
            left = left_word[recipient_cut - 1]
            right = donor_suffix[0] if donor_suffix else None
            try:
                bridge = self.automaton.sample_bridge(
                    left=left,
                    right=right,
                    length=bridge_length,
                    rng=self.rng,
                    chooser=self.transition_model.choose,
                    absolute_start=recipient_cut,
                    horizon=result_length,
                )
            except ValueError:
                continue
            factors = left_word[:recipient_cut] + bridge + donor_suffix
            if (
                factors != left_word
                and factors != right_word
                and self.automaton.is_legal(factors)
            ):
                island = recipient.trajectory.island
                return Trajectory(
                    factor_ids=factors,
                    island=island,
                    origin="variable_crossover",
                    parent_id=(
                        f"{recipient.trajectory.trajectory_id},"
                        f"{donor.trajectory.trajectory_id}"
                    ),
                    parent_score=max(
                        recipient.score_for(island),
                        donor.score_for(island),
                    ),
                )

        island = recipient.trajectory.island
        return Trajectory(
            factor_ids=left_word,
            island=island,
            origin="crossover_fallback",
            parent_id=recipient.trajectory.trajectory_id,
            parent_score=recipient.score_for(island),
        )
