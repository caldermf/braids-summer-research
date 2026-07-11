from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Sequence

from .config import SearchConfig
from .gnf import GNFAutomaton
from .models import MutationRecord, Trajectory, TrajectoryEvaluation
from .transition_model import TransitionModel, weighted_choice


class StructuralMutationPlanner:
    """Legal replacement and length-changing edits over complete GNF words."""

    ACTIONS = ("replace", "post_turn", "append", "truncate", "insert", "delete")

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

    def _adaptive_weight(self, action: str, size: int) -> float:
        values = self.stats[(action, size)]
        attempts = values["attempts"]
        success = (values["successes"] + 1.0) / (attempts + 2.0)
        improvement = values["improvement_sum"] / max(1.0, attempts)
        total = sum(item["attempts"] for item in self.stats.values())
        exploration = math.sqrt(math.log(total + 2.0) / (attempts + 1.0))
        return math.exp(max(-5.0, min(5.0, 2.0 * success + 0.05 * improvement + exploration)))

    def choose_block_length(self, horizon: int, stagnant: bool, action: str) -> int:
        configured = (
            self.config.length_edit_sizes
            if action in {"append", "truncate", "insert", "delete"}
            else self.config.block_sizes_for(self.island, stagnant)
        )
        candidates = tuple(dict.fromkeys(max(1, min(horizon, value)) for value in configured))
        return weighted_choice(
            candidates,
            [self._adaptive_weight(action, value) for value in candidates],
            self.rng,
        )

    def choose_action(
        self,
        horizon: int,
        active_max_horizon: int,
        stagnant: bool,
        force_large: bool,
        allow_length_change: bool,
    ) -> str:
        structural = self.config.structural_mutation_fraction if allow_length_change else 0.0
        post_turn = self.config.post_turn_rewrite_fraction
        weights = {
            "replace": max(0.05, 1.0 - structural - post_turn),
            "post_turn": post_turn,
            "append": structural * 0.24 if horizon < active_max_horizon else 0.0,
            "truncate": structural * 0.20 if horizon > self.config.min_horizon else 0.0,
            "insert": structural * 0.32 if horizon < active_max_horizon else 0.0,
            "delete": structural * 0.24 if horizon > self.config.min_horizon else 0.0,
        }
        if self.island == "envelope":
            weights["replace"] *= 1.3
            weights["delete"] *= 1.4
            weights["append"] *= 0.5
        elif self.island == "collapse":
            weights["post_turn"] *= 1.5
        elif self.island == "suffix":
            weights["post_turn"] *= 1.4
            weights["append"] *= 1.3
        if stagnant or force_large:
            weights["replace"] *= 0.7
            weights["post_turn"] *= 1.6
            weights["insert"] *= 1.4
            weights["delete"] *= 1.3
        actions = tuple(weights)
        return weighted_choice(actions, tuple(weights[action] for action in actions), self.rng)

    def _history_value(
        self,
        evaluation: TrajectoryEvaluation,
        position: int,
        horizon: int,
    ) -> int:
        source = evaluation.projlen_history
        mapped = min(len(source) - 1, int(position * len(source) / max(1, horizon)))
        return source[mapped]

    def choose_start(
        self,
        evaluation: TrajectoryEvaluation,
        horizon: int,
        removed_length: int,
        force_large: bool,
    ) -> tuple[int, str]:
        maximum = max(0, horizon - removed_length)
        if maximum == 0:
            return 0, "full"
        if force_large and self.rng.random() < 0.35:
            return self.rng.randint(0, maximum), "global"
        starts = list(range(maximum + 1))
        weights = []
        turn = round(
            (evaluation.turning_point_depth - 1)
            * horizon
            / max(1, evaluation.trajectory.horizon)
        )
        for start in starts:
            end = start + removed_length
            center = start + removed_length / 2
            if self.island == "envelope":
                local = max(
                    self._history_value(evaluation, position, horizon)
                    for position in range(start, max(start + 1, end))
                )
                weight = 1.0 + local
            elif self.island == "collapse":
                distance = abs(center - turn)
                weight = 1.0 + 8.0 / (1.0 + distance)
            elif self.island == "suffix":
                weight = 1.0 + self.config.terminal_location_bias * (end / horizon) ** 3
            else:
                weight = 1.0 + 2.0 * (end / horizon) ** 2
            if end == horizon:
                weight *= 1.5
            weights.append(weight)
        return weighted_choice(starts, weights, self.rng), "objective_targeted"

    def _sample_bridge(
        self,
        left: int | None,
        right: int | None,
        length: int,
        start: int,
        horizon: int,
        use_learned: bool,
    ) -> tuple[int, ...]:
        return self.automaton.sample_bridge(
            left=left,
            right=right,
            length=length,
            rng=self.rng,
            chooser=self.transition_model.choose if use_learned else None,
            absolute_start=start,
            horizon=horizon,
        )

    def _record(
        self,
        action: str,
        mode: str,
        start: int,
        removed: int,
        inserted: int,
        horizon: int,
    ) -> MutationRecord:
        return MutationRecord(
            island=self.island,
            action=action,
            mode=mode,
            start=start,
            removed_length=removed,
            inserted_length=inserted,
            location_bin=min(7, int(8 * start / max(1, horizon))),
        )

    def _replace(
        self,
        original: tuple[int, ...],
        evaluation: TrajectoryEvaluation,
        removed: int,
        inserted: int,
        force_large: bool,
        use_learned: bool,
    ) -> tuple[tuple[int, ...], MutationRecord]:
        horizon = len(original)
        removed = min(removed, horizon)
        start, mode = self.choose_start(evaluation, horizon, removed, force_large)
        end = start + removed
        left = original[start - 1] if start else None
        right = original[end] if end < horizon else None
        replacement = self._sample_bridge(
            left,
            right,
            inserted,
            start,
            horizon - removed + inserted,
            use_learned,
        )
        return (
            original[:start] + replacement + original[end:],
            self._record("replace", mode, start, removed, inserted, horizon),
        )

    def _append(
        self,
        original: tuple[int, ...],
        length: int,
        use_learned: bool,
    ) -> tuple[tuple[int, ...], MutationRecord]:
        addition = self._sample_bridge(
            original[-1],
            None,
            length,
            len(original),
            len(original) + length,
            use_learned,
        )
        return (
            original + addition,
            self._record("append", "terminal", len(original), 0, length, len(original)),
        )

    def _truncate(
        self,
        original: tuple[int, ...],
        length: int,
    ) -> tuple[tuple[int, ...], MutationRecord]:
        length = min(length, len(original) - self.config.min_horizon)
        start = len(original) - length
        return (
            original[:start],
            self._record("truncate", "terminal", start, length, 0, len(original)),
        )

    def _insert(
        self,
        original: tuple[int, ...],
        length: int,
        use_learned: bool,
    ) -> tuple[tuple[int, ...], MutationRecord]:
        start = self.rng.randint(0, len(original))
        left = original[start - 1] if start else None
        right = original[start] if start < len(original) else None
        block = self._sample_bridge(
            left,
            right,
            length,
            start,
            len(original) + length,
            use_learned,
        )
        return (
            original[:start] + block + original[start:],
            self._record("insert", "structural", start, 0, length, len(original)),
        )

    def _delete(
        self,
        original: tuple[int, ...],
        length: int,
    ) -> tuple[tuple[int, ...], MutationRecord] | None:
        maximum = len(original) - self.config.min_horizon
        length = min(length, maximum)
        if length <= 0:
            return None
        starts = list(range(0, len(original) - length + 1))
        self.rng.shuffle(starts)
        for start in starts:
            end = start + length
            if start == 0 and end < len(original):
                legal = original[end] in self.automaton.first_ids
            elif end == len(original):
                legal = start > 0
            else:
                legal = original[end] in self.automaton.successors[original[start - 1]]
            if not legal:
                continue
            return (
                original[:start] + original[end:],
                self._record("delete", "structural", start, length, 0, len(original)),
            )
        return None

    def _post_turn(
        self,
        original: tuple[int, ...],
        evaluation: TrajectoryEvaluation,
        active_max_horizon: int,
        force_large: bool,
        use_learned: bool,
    ) -> tuple[tuple[int, ...], MutationRecord]:
        mapped_turn = max(
            1,
            round(
                evaluation.turning_point_depth
                * len(original)
                / max(1, evaluation.trajectory.horizon)
            ),
        )
        start = min(len(original) - 1, mapped_turn)
        removed = len(original) - start
        delta_limit = min(8 if force_large else 3, active_max_horizon - start)
        low = max(1, removed - delta_limit)
        high = max(low, min(active_max_horizon - start, removed + delta_limit))
        inserted = self.rng.randint(low, high)
        suffix = self._sample_bridge(
            original[start - 1],
            None,
            inserted,
            start,
            start + inserted,
            use_learned,
        )
        return (
            original[:start] + suffix,
            self._record("post_turn", "post_turn_suffix", start, removed, inserted, len(original)),
        )

    def mutate_once(
        self,
        parent: TrajectoryEvaluation,
        *,
        active_max_horizon: int,
        stagnant: bool = False,
        force_large: bool = False,
        use_learned: bool = True,
        allow_length_change: bool = True,
        factor_ids: Sequence[int] | None = None,
        action: str | None = None,
        block_length: int | None = None,
    ) -> tuple[tuple[int, ...], MutationRecord]:
        original = tuple(factor_ids or parent.trajectory.factor_ids)
        for _ in range(self.config.mutation_attempts):
            chosen_action = action or self.choose_action(
                len(original),
                active_max_horizon,
                stagnant,
                force_large,
                allow_length_change,
            )
            size = block_length or self.choose_block_length(
                len(original),
                stagnant,
                chosen_action,
            )
            try:
                if chosen_action == "append" and len(original) < active_max_horizon:
                    size = min(size, active_max_horizon - len(original))
                    candidate, record = self._append(original, size, use_learned)
                elif chosen_action == "truncate" and len(original) > self.config.min_horizon:
                    candidate, record = self._truncate(original, size)
                elif chosen_action == "insert" and len(original) < active_max_horizon:
                    size = min(size, active_max_horizon - len(original))
                    candidate, record = self._insert(original, size, use_learned)
                elif chosen_action == "delete" and len(original) > self.config.min_horizon:
                    result = self._delete(original, size)
                    if result is None:
                        continue
                    candidate, record = result
                elif chosen_action == "post_turn":
                    candidate, record = self._post_turn(
                        original,
                        parent,
                        active_max_horizon,
                        force_large,
                        use_learned,
                    )
                else:
                    removed = min(size, len(original))
                    inserted = removed
                    if allow_length_change and self.rng.random() < self.config.structural_mutation_fraction:
                        delta = self.rng.choice((-2, -1, 1, 2))
                        inserted = max(1, min(removed + delta, active_max_horizon - len(original) + removed))
                    candidate, record = self._replace(
                        original,
                        parent,
                        removed,
                        inserted,
                        force_large,
                        use_learned,
                    )
            except (ValueError, IndexError):
                continue
            if (
                candidate != original
                and self.config.min_horizon <= len(candidate) <= active_max_horizon
                and self.automaton.is_legal(candidate)
            ):
                return candidate, record
        fallback, record = self._replace(
            original,
            parent,
            min(3, len(original)),
            min(3, len(original)),
            False,
            False,
        )
        return fallback, record

    def make_child(
        self,
        parent: TrajectoryEvaluation,
        *,
        active_max_horizon: int,
        stagnant: bool = False,
        force_large: bool = False,
        use_learned: bool = True,
        two_mutations: bool = False,
    ) -> Trajectory:
        factors, first = self.mutate_once(
            parent,
            active_max_horizon=active_max_horizon,
            stagnant=stagnant,
            force_large=force_large,
            use_learned=use_learned,
        )
        records = [first]
        if two_mutations:
            factors, second = self.mutate_once(
                parent,
                active_max_horizon=active_max_horizon,
                stagnant=stagnant,
                force_large=force_large,
                use_learned=use_learned,
                factor_ids=factors,
            )
            records.append(second)
        return Trajectory(
            factor_ids=factors,
            island=self.island,
            origin="restart_mutation" if force_large else "structural_mutation",
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
            values = self.stats[(record.action, record.block_length)]
            values["attempts"] += 1.0
            values["successes"] += float(improvement > 0.0)
            values["improvement_sum"] += max(0.0, improvement)

    def stats_json(self) -> dict:
        return {
            f"{action}:{size}": {
                **{name: round(value, 6) for name, value in values.items()},
                "success_rate": round(values["successes"] / max(1.0, values["attempts"]), 6),
            }
            for (action, size), values in sorted(self.stats.items())
        }


AdaptiveSuffixMutationPlanner = StructuralMutationPlanner
MutationPlanner = StructuralMutationPlanner
