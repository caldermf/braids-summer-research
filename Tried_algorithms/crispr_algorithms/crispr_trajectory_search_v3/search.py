from __future__ import annotations

import gzip
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

from peyl.braid_data import (
    factor_ids_to_artin_word,
    factor_ids_to_perms,
    simple_factor_id_maps,
)

from .caches import EvaluationCache, FinishingQueue, MatrixNoveltyArchive, SeenWordCache
from .config import ISLAND_NAMES, SearchConfig
from .crossover import SuffixCrossover
from .evaluators import make_evaluator
from .gnf import GNFAutomaton
from .islands import IslandState, island_rank, select_island_elites
from .known_examples import known_example
from .mcts import SuffixMCTSFinisher
from .models import Trajectory, TrajectoryEvaluation
from .mutation import AdaptiveSuffixMutationPlanner
from .transition_model import TransitionModel


class IslandTrajectorySearch:
    """Three-objective island evolution with a shared suffix MCTS finisher."""

    def __init__(self, config: SearchConfig):
        config.validate()
        self.config = config
        self.rng = random.Random(config.seed)
        self.automaton = GNFAutomaton(n=config.n)
        self.evaluator = make_evaluator(config)
        self.evaluation_cache = EvaluationCache(config.evaluation_cache_size)
        self.seen_words = SeenWordCache()
        self.novelty_archive = MatrixNoveltyArchive(config.novelty_archive_size)
        self.finishing_queue = FinishingQueue(
            config.finishing_queue_size,
            config.finishing_projlen_threshold,
        )
        self.transitions = {
            island: TransitionModel(config, self.automaton)
            for island in ISLAND_NAMES
        }
        self.planners = {
            island: AdaptiveSuffixMutationPlanner(
                config,
                island,
                self.automaton,
                self.transitions[island],
                self.rng,
            )
            for island in ISLAND_NAMES
        }
        self.crossovers = {
            island: SuffixCrossover(
                config,
                self.automaton,
                self.transitions[island],
                self.rng,
            )
            for island in ISLAND_NAMES
        }
        self.islands = {
            island: IslandState(island, config.island_sizes[island])
            for island in ISLAND_NAMES
        }
        self.run_dir = self._create_run_directory()
        self.generation_log = self.run_dir / "generations.jsonl"
        self.lineage_log = self.run_dir / "lineage.jsonl"
        self.kernel_hits: list[TrajectoryEvaluation] = []
        self.next_trajectory_number = 0
        self.mcts = SuffixMCTSFinisher(
            config,
            self.evaluator,
            self.evaluation_cache,
            self.seen_words,
            self.planners,
            self.rng,
            self._assign_id,
        )
        self._write_json("config.json", config.to_json())

    def _create_run_directory(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base = Path(self.config.output_dir)
        candidate = base / f"crispr_v3_{timestamp}_seed{self.config.seed}"
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = base / f"crispr_v3_{timestamp}_seed{self.config.seed}_{suffix}"
        candidate.mkdir(parents=True)
        return candidate

    def _write_json(self, filename: str, payload) -> None:
        with (self.run_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _assign_id(self, trajectory: Trajectory, generation: int) -> Trajectory:
        trajectory.trajectory_id = f"g{generation:04d}_t{self.next_trajectory_number:09d}"
        self.next_trajectory_number += 1
        return trajectory

    def _sample_horizon(self) -> int:
        return self.rng.choice(self.config.horizons)

    def _random_trajectory(
        self,
        island: str,
        generation: int,
        *,
        learned: bool = False,
        register: bool = True,
    ) -> Trajectory:
        for _ in range(1_000):
            horizon = self._sample_horizon()
            factors = (
                self.transitions[island].sample(horizon, self.rng)
                if learned
                else self.automaton.sample_uniform(horizon, self.rng)
            )
            if not register or self.seen_words.add(factors):
                return self._assign_id(
                    Trajectory(
                        factor_ids=factors,
                        island=island,
                        origin="learned_random" if learned else "random",
                    ),
                    generation,
                )
        raise RuntimeError("failed to sample an unseen legal trajectory")

    def _load_seed_factors(self) -> Optional[tuple[int, ...]]:
        if self.config.seed_known_example:
            factors = known_example(self.config.seed_known_example)
        elif self.config.seed_trajectory_json:
            with Path(self.config.seed_trajectory_json).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if "factor_ids" in payload:
                factors = tuple(int(value) for value in payload["factor_ids"])
            elif "gnf_factors" in payload:
                perm_to_id, _ = simple_factor_id_maps(self.config.n)
                factors = tuple(perm_to_id[tuple(perm)] for perm in payload["gnf_factors"])
            else:
                raise ValueError("seed JSON must contain factor_ids or gnf_factors")
        else:
            return None
        if not self.automaton.is_legal(factors):
            raise ValueError("seed trajectory is not legal under the GNF automaton")
        return tuple(factors)

    def _corrupt_seed(self, factors: tuple[int, ...]) -> tuple[int, ...]:
        target = max(1, round(len(factors) * self.config.seed_corruption_fraction))
        current = factors
        edited = 0
        attempts = 0
        while edited < target and attempts < 10 * target:
            attempts += 1
            length = min(self.rng.choice(self.config.suffix_block_sizes), len(current), target)
            start = self.rng.randint(0, len(current) - length)
            end = start + length
            replacement = self.automaton.sample_bridge(
                left=current[start - 1] if start else None,
                right=current[end] if end < len(current) else None,
                length=length,
                rng=self.rng,
                absolute_start=start,
                horizon=len(current),
            )
            candidate = current[:start] + replacement + current[end:]
            if candidate != current:
                current = candidate
                edited += length
        return current

    def initial_populations(self) -> dict[str, list[Trajectory]]:
        seed = self._load_seed_factors()
        populations = {island: [] for island in ISLAND_NAMES}
        for island in ISLAND_NAMES:
            target = self.islands[island].target_size
            seed_count = (
                round(target * self.config.seed_population_fraction)
                if seed is not None and len(seed) in self.config.horizons
                else 0
            )
            while len(populations[island]) < seed_count:
                factors = self._corrupt_seed(seed)
                if not self.seen_words.add(factors):
                    continue
                populations[island].append(
                    self._assign_id(
                        Trajectory(
                            factor_ids=factors,
                            island=island,
                            origin="seed_corruption",
                        ),
                        0,
                    )
                )
            while len(populations[island]) < target:
                populations[island].append(self._random_trajectory(island, 0))
        return populations

    def _evaluate_populations(
        self,
        populations: dict[str, list[Trajectory]],
    ) -> dict[str, list[TrajectoryEvaluation]]:
        trajectories = [
            trajectory
            for island in ISLAND_NAMES
            for trajectory in populations[island]
        ]
        evaluations = self.evaluation_cache.evaluate(self.evaluator, trajectories)
        self.novelty_archive.assign(evaluations)
        grouped = {island: [] for island in ISLAND_NAMES}
        for evaluation in evaluations:
            evaluation.score = evaluation.score_for(evaluation.trajectory.island)
            grouped[evaluation.trajectory.island].append(evaluation)
        return grouped

    def _update_hits(self, evaluations: Iterable[TrajectoryEvaluation]) -> list[TrajectoryEvaluation]:
        new_hits = []
        known = {item.trajectory.factor_ids for item in self.kernel_hits}
        for evaluation in evaluations:
            factors = evaluation.trajectory.factor_ids
            if not evaluation.has_kernel or factors in known:
                continue
            known.add(factors)
            new_hits.append(evaluation)
            if len(self.kernel_hits) < self.config.max_kernel_hits:
                self.kernel_hits.append(evaluation)
        return new_hits

    @staticmethod
    def _as_island(evaluation: TrajectoryEvaluation, island: str) -> TrajectoryEvaluation:
        trajectory = replace(evaluation.trajectory, island=island)
        return replace(
            evaluation,
            trajectory=trajectory,
            score=evaluation.score_for(island),
            island_scores=dict(evaluation.island_scores),
        )

    def _mcts_seeds(
        self,
        grouped: dict[str, list[TrajectoryEvaluation]],
    ) -> dict[str, list[TrajectoryEvaluation]]:
        per_island = max(1, self.config.mcts_seed_count // len(ISLAND_NAMES))
        queue = self.finishing_queue.members()
        output = {}
        for island in ISLAND_NAMES:
            ordered = sorted(
                grouped[island],
                key=lambda item: island_rank(item, island),
                reverse=True,
            )
            eligible_count = max(1, round(len(ordered) * self.config.mcts_top_fraction))
            candidates = ordered[:eligible_count] + [
                self._as_island(item, island) for item in queue
            ]
            selected = []
            states = set()
            words = set()
            for candidate in candidates:
                factors = candidate.trajectory.factor_ids
                if factors in words:
                    continue
                if candidate.matrix_fingerprint in states and len(selected) < per_island // 2:
                    continue
                selected.append(candidate)
                words.add(factors)
                states.add(candidate.matrix_fingerprint)
                if len(selected) >= per_island:
                    break
            output[island] = selected
        return output

    def _run_mcts_if_due(
        self,
        grouped: dict[str, list[TrajectoryEvaluation]],
        generation: int,
        stagnant: dict[str, bool],
    ) -> list[TrajectoryEvaluation]:
        if not self.config.mcts_enabled:
            return []
        due = generation > 0 and generation % self.config.mcts_interval == 0
        if not due and not any(stagnant.values()):
            return []
        improvements = self.mcts.run(self._mcts_seeds(grouped), generation)
        if improvements:
            self.novelty_archive.assign(improvements)
            for evaluation in improvements:
                evaluation.score = evaluation.score_for(evaluation.trajectory.island)
                grouped[evaluation.trajectory.island].append(evaluation)
        return improvements

    def _parent_pool(
        self,
        evaluations: list[TrajectoryEvaluation],
        island: str,
    ) -> list[TrajectoryEvaluation]:
        count = max(2, round(len(evaluations) * 0.25))
        pool = select_island_elites(evaluations, island, count)
        queue = [self._as_island(item, island) for item in self.finishing_queue.members()]
        seen = {item.trajectory.factor_ids for item in pool}
        pool.extend(item for item in queue if item.trajectory.factor_ids not in seen)
        return pool

    def _migration(
        self,
        grouped: dict[str, list[TrajectoryEvaluation]],
        generation: int,
    ) -> dict[str, list[Trajectory]]:
        incoming = {island: [] for island in ISLAND_NAMES}
        if generation == 0 or generation % self.config.migration_interval:
            return incoming
        for index, source in enumerate(ISLAND_NAMES):
            destination = ISLAND_NAMES[(index + 1) % len(ISLAND_NAMES)]
            count = max(1, round(self.islands[source].target_size * self.config.migration_fraction))
            migrants = select_island_elites(grouped[source], source, count)
            for evaluation in migrants:
                incoming[destination].append(
                    self._assign_id(
                        Trajectory(
                            factor_ids=evaluation.trajectory.factor_ids,
                            island=destination,
                            origin=f"migration_from_{source}",
                            parent_id=evaluation.trajectory.trajectory_id,
                            parent_score=evaluation.score_for(destination),
                            protected_until_generation=(
                                generation + self.config.migration_protection_generations
                            ),
                        ),
                        generation + 1,
                    )
                )
        return incoming

    def _register_candidate(self, trajectory: Trajectory, generation: int) -> Trajectory | None:
        if not self.seen_words.add(trajectory.factor_ids):
            return None
        return self._assign_id(trajectory, generation)

    def _mutation_groups(
        self,
        island: str,
        generation: int,
        parents: list[TrajectoryEvaluation],
        slots: int,
        stagnant: bool,
        force_large: bool,
    ) -> list[list[Trajectory]]:
        if slots <= 0:
            return []
        survivors = self.config.offspring_survivors_per_parent
        group_count = math.ceil(slots / survivors)
        groups = []
        for _ in range(group_count):
            parent = self.rng.choice(parents)
            group = []
            attempts = 0
            while len(group) < self.config.offspring_per_parent and attempts < 10 * self.config.offspring_per_parent:
                attempts += 1
                child = self.planners[island].make_child(
                    parent,
                    stagnant=stagnant,
                    force_large=force_large,
                    two_mutations=force_large and self.rng.random() < 0.35,
                )
                registered = self._register_candidate(child, generation)
                if registered is not None:
                    group.append(registered)
            if group:
                groups.append(group)
        return groups

    def _crossover_candidates(
        self,
        island: str,
        generation: int,
        parents: list[TrajectoryEvaluation],
        count: int,
    ) -> list[Trajectory]:
        by_horizon = defaultdict(list)
        for parent in parents:
            by_horizon[parent.trajectory.horizon].append(parent)
        viable = [group for group in by_horizon.values() if len(group) >= 2]
        output = []
        attempts = 0
        while len(output) < count and viable and attempts < 20 * max(1, count):
            attempts += 1
            recipient, donor = self.rng.sample(self.rng.choice(viable), 2)
            child = self.crossovers[island].make_child(recipient, donor)
            registered = self._register_candidate(child, generation)
            if registered is not None:
                output.append(registered)
        return output

    def _deduplicated_append(
        self,
        target: list[Trajectory],
        candidates: Iterable[Trajectory],
        seen: set[tuple[int, ...]],
        limit: int,
    ) -> None:
        for candidate in candidates:
            if len(target) >= limit or candidate.factor_ids in seen:
                continue
            target.append(candidate)
            seen.add(candidate.factor_ids)

    def _next_island_population(
        self,
        island: str,
        generation: int,
        evaluations: list[TrajectoryEvaluation],
        incoming: list[Trajectory],
        stagnant: bool,
    ) -> list[Trajectory]:
        target = self.islands[island].target_size
        parents = self._parent_pool(evaluations, island)
        restarting = stagnant
        if restarting:
            preserve_count = round(target * self.config.restart_preserve_fraction)
            random_count = round(target * self.config.restart_random_fraction)
            large_slots = round(target * self.config.restart_large_mutation_fraction)
            crossover_count = round(target * self.config.crossover_fraction)
            mutation_slots = target - preserve_count - random_count - large_slots - crossover_count
            self.planners[island].reset_statistics()
            self.islands[island].mark_restart(generation)
        else:
            preserve_count = round(target * self.config.carry_fraction)
            random_count = round(target * self.config.random_fraction)
            crossover_count = round(target * self.config.crossover_fraction)
            large_slots = 0
            mutation_slots = target - preserve_count - random_count - crossover_count

        carries = [
            self._assign_id(
                Trajectory(
                    factor_ids=item.trajectory.factor_ids,
                    island=island,
                    origin="elite_carry",
                    parent_id=item.trajectory.trajectory_id,
                    parent_score=item.score_for(island),
                ),
                generation,
            )
            for item in select_island_elites(evaluations, island, preserve_count)
        ]
        mutation_groups = self._mutation_groups(
            island,
            generation,
            parents,
            mutation_slots,
            stagnant=stagnant,
            force_large=False,
        )
        large_groups = self._mutation_groups(
            island,
            generation,
            parents,
            large_slots,
            stagnant=True,
            force_large=True,
        )
        crossovers = self._crossover_candidates(
            island,
            generation,
            parents,
            crossover_count,
        )
        randoms = [
            self._random_trajectory(
                island,
                generation,
                learned=self.rng.random() < 0.25 and not restarting,
            )
            for _ in range(random_count)
        ]

        flat_mutations = [child for group in mutation_groups + large_groups for child in group]
        generated = flat_mutations + crossovers + randoms
        generated_evaluations = self.evaluation_cache.evaluate(self.evaluator, generated)
        self.novelty_archive.assign(generated_evaluations)
        by_id = {
            item.trajectory.trajectory_id: item
            for item in generated_evaluations
        }
        for evaluation in generated_evaluations:
            evaluation.score = evaluation.score_for(island)
            if evaluation.trajectory.mutation_records:
                self.planners[island].observe(evaluation)

        mutation_winners = []
        remaining_mutation_slots = mutation_slots + large_slots
        for group in mutation_groups + large_groups:
            ranked = sorted(
                (by_id[child.trajectory_id] for child in group),
                key=lambda item: island_rank(item, island),
                reverse=True,
            )
            mutation_winners.extend(
                item.trajectory
                for item in ranked[: self.config.offspring_survivors_per_parent]
            )
            if len(mutation_winners) >= remaining_mutation_slots:
                mutation_winners = mutation_winners[:remaining_mutation_slots]
                break

        population: list[Trajectory] = []
        seen: set[tuple[int, ...]] = set()
        protected = [
            item
            for item in incoming
            if item.protected_until_generation >= generation
        ]
        self._deduplicated_append(population, protected, seen, target)
        self._deduplicated_append(population, carries, seen, target)
        self._deduplicated_append(population, mutation_winners, seen, target)

        ranked_other = sorted(
            (by_id[item.trajectory_id] for item in crossovers + randoms),
            key=lambda item: island_rank(item, island),
            reverse=True,
        )
        self._deduplicated_append(
            population,
            (item.trajectory for item in ranked_other),
            seen,
            target,
        )

        while len(population) < target:
            candidate = self._random_trajectory(island, generation)
            if candidate.factor_ids not in seen:
                population.append(candidate)
                seen.add(candidate.factor_ids)

        self.planners[island].decay_statistics()
        return population

    def _write_lineage(self, populations: dict[str, list[Trajectory]], generation: int) -> None:
        with self.lineage_log.open("a", encoding="utf-8") as handle:
            for island in ISLAND_NAMES:
                for trajectory in populations[island]:
                    if trajectory.origin == "random":
                        continue
                    handle.write(
                        json.dumps(
                            {
                                "generation": generation,
                                "trajectory_id": trajectory.trajectory_id,
                                "island": island,
                                "origin": trajectory.origin,
                                "parent_id": trajectory.parent_id,
                                "mutation_records": [
                                    record.__dict__
                                    for record in trajectory.mutation_records
                                ],
                            }
                        )
                        + "\n"
                    )

    def _evaluation_json(self, evaluation: TrajectoryEvaluation) -> dict:
        payload = evaluation.summary()
        payload.update(
            {
                "gnf_factors": [
                    list(perm)
                    for perm in factor_ids_to_perms(
                        evaluation.trajectory.factor_ids,
                        n=self.config.n,
                    )
                ],
                "artin_word": factor_ids_to_artin_word(
                    evaluation.trajectory.factor_ids,
                    n=self.config.n,
                ),
                "projlen_history": list(evaluation.projlen_history),
                "kernel_matches": list(evaluation.kernel_matches),
            }
        )
        return payload

    def _save_checkpoint(
        self,
        generation: int,
        populations: dict[str, list[Trajectory]],
    ) -> None:
        payload = {
            "next_generation": generation,
            "populations": {
                island: [
                    {
                        "factor_ids": list(item.factor_ids),
                        "trajectory_id": item.trajectory_id,
                        "origin": item.origin,
                        "parent_id": item.parent_id,
                        "protected_until_generation": item.protected_until_generation,
                    }
                    for item in populations[island]
                ]
                for island in ISLAND_NAMES
            },
        }
        with gzip.open(self.run_dir / "checkpoint.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def _save_progress(self) -> None:
        for island, state in self.islands.items():
            if state.best is not None:
                self._write_json(
                    f"best_{island}_candidate.json",
                    self._evaluation_json(state.best),
                )
        self._write_json(
            "kernel_hits.json",
            [self._evaluation_json(item) for item in self.kernel_hits],
        )
        self._write_json(
            "mutation_stats.json",
            {
                island: self.planners[island].stats_json()
                for island in ISLAND_NAMES
            },
        )
        self._write_json(
            "transition_models.json",
            {
                island: self.transitions[island].top_transitions(100)
                for island in ISLAND_NAMES
            },
        )
        self._write_json(
            "cache_stats.json",
            {
                "evaluation": self.evaluation_cache.stats(),
                "seen_words": self.seen_words.stats(),
                "matrix_novelty": self.novelty_archive.stats(),
                "finishing_queue": self.finishing_queue.stats(),
            },
        )
        self._write_json("mcts_stats.json", self.mcts.stats())

    def _generation_summary(
        self,
        generation: int,
        grouped: dict[str, list[TrajectoryEvaluation]],
        stagnant: dict[str, bool],
        new_hits: list[TrajectoryEvaluation],
        mcts_results: list[TrajectoryEvaluation],
        elapsed: float,
    ) -> dict:
        combined = [item for island in ISLAND_NAMES for item in grouped[island]]
        return {
            "generation": generation,
            "population_size": self.config.population_size,
            "evaluated_candidates": len(combined),
            "unique_trajectories": len({item.trajectory.factor_ids for item in combined}),
            "kernel_hits_this_generation": len(new_hits),
            "kernel_hits_total": len(self.kernel_hits),
            "mcts_results": len(mcts_results),
            "islands": {
                island: {
                    "size": len(grouped[island]),
                    "stagnant": stagnant[island],
                    "restart_count": self.islands[island].restart_count,
                    "best_final_projlen": min(item.final_projlen for item in grouped[island]),
                    "best_terminal_collapse": max(
                        item.terminal_collapse for item in grouped[island]
                    ),
                    "best_terminal_weighted_area": min(
                        item.terminal_weighted_area for item in grouped[island]
                    ),
                    "matrix_states": len(
                        {item.matrix_fingerprint for item in grouped[island]}
                    ),
                    "origins": dict(
                        Counter(item.trajectory.origin for item in grouped[island])
                    ),
                }
                for island in ISLAND_NAMES
            },
            "finishing_queue_size": len(self.finishing_queue.members()),
            "cache": self.evaluation_cache.stats(),
            "mcts": self.mcts.stats(),
            "elapsed_sec": round(elapsed, 4),
        }

    def run(self) -> dict:
        start = time.perf_counter()
        populations = self.initial_populations()
        completed_generations = 0

        for generation in range(self.config.generations):
            generation_start = time.perf_counter()
            grouped = self._evaluate_populations(populations)
            combined = [item for island in ISLAND_NAMES for item in grouped[island]]
            self.finishing_queue.update(combined)
            new_hits = self._update_hits(combined)

            stagnant = {}
            elites = {}
            for island in ISLAND_NAMES:
                state = self.islands[island]
                state.update(
                    grouped[island],
                    generation,
                    self.config.stagnation_min_improvement,
                )
                stagnant[island] = state.stagnant(
                    generation,
                    self.config.stagnation_generations,
                )
                elite_count = max(
                    1,
                    round(state.target_size * self.config.elite_fraction),
                )
                elites[island] = select_island_elites(
                    grouped[island],
                    island,
                    elite_count,
                )
                self.transitions[island].update(elites[island])

            mcts_results = self._run_mcts_if_due(grouped, generation, stagnant)
            if mcts_results:
                self.finishing_queue.update(mcts_results)
                mcts_hits = self._update_hits(mcts_results)
                new_hits.extend(mcts_hits)
                for island in ISLAND_NAMES:
                    self.islands[island].update(
                        grouped[island],
                        generation,
                        self.config.stagnation_min_improvement,
                    )
                    stagnant[island] = self.islands[island].stagnant(
                        generation,
                        self.config.stagnation_generations,
                    )

            summary = self._generation_summary(
                generation,
                grouped,
                stagnant,
                new_hits,
                mcts_results,
                time.perf_counter() - generation_start,
            )
            with self.generation_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary) + "\n")
            print(json.dumps(summary), flush=True)
            self._save_progress()
            completed_generations = generation + 1

            if new_hits and self.config.stop_at_kernel:
                break
            if generation + 1 >= self.config.generations:
                break

            incoming = self._migration(grouped, generation)
            populations = {
                island: self._next_island_population(
                    island,
                    generation + 1,
                    grouped[island],
                    incoming[island],
                    stagnant[island],
                )
                for island in ISLAND_NAMES
            }
            self._write_lineage(populations, generation + 1)
            self._save_checkpoint(generation + 1, populations)

        champions = {
            island: state.best.summary() if state.best is not None else None
            for island, state in self.islands.items()
        }
        final_summary = {
            "run_dir": str(self.run_dir),
            "algorithm": "crispr_trajectory_search_v3",
            "p": self.config.p,
            "n": self.config.n,
            "backend": self.config.backend,
            "device": self.config.device,
            "completed_generations": completed_generations,
            "population_size": self.config.population_size,
            "num_kernel_hits": len(self.kernel_hits),
            "champions": champions,
            "cache": self.evaluation_cache.stats(),
            "mcts": self.mcts.stats(),
            "elapsed_sec": round(time.perf_counter() - start, 4),
        }
        self._write_json("summary.json", final_summary)
        self._save_progress()
        return final_summary


EvolutionaryTrajectorySearch = IslandTrajectorySearch
