from __future__ import annotations

import gzip
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

from peyl.braid_data import (
    factor_ids_to_artin_word,
    factor_ids_to_perms,
    simple_factor_id_maps,
)

from .archive import QualityDiversityArchive
from .config import SearchConfig
from .crossover import SuffixCrossover
from .evaluators import make_evaluator
from .gnf import GNFAutomaton
from .known_examples import known_example
from .models import GenerationResult, Trajectory, TrajectoryEvaluation
from .mutation import MutationPlanner
from .selection import select_diverse_elites
from .transition_model import TransitionModel


class EvolutionaryTrajectorySearch:
    """Quality-diversity evolutionary search over complete legal GNF words."""

    def __init__(self, config: SearchConfig):
        config.validate()
        self.config = config
        self.rng = random.Random(config.seed)
        self.automaton = GNFAutomaton(n=config.n)
        self.transition_model = TransitionModel(config, self.automaton)
        self.mutation_planner = MutationPlanner(
            config,
            self.automaton,
            self.transition_model,
            self.rng,
        )
        self.crossover = SuffixCrossover(
            config,
            self.automaton,
            self.transition_model,
            self.rng,
        )
        self.archive = QualityDiversityArchive(config)
        self.evaluator = make_evaluator(config)
        self.run_dir = self._create_run_directory()
        self.generation_log = self.run_dir / "generations.jsonl"
        self.kernel_hits: list[TrajectoryEvaluation] = []
        self.best: Optional[TrajectoryEvaluation] = None
        self.next_trajectory_number = 0
        self.best_final_seen = float("inf")
        self.last_final_improvement_generation = 0

        self._write_json("config.json", config.to_json())

    def _create_run_directory(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base = Path(self.config.output_dir)
        candidate = base / f"crispr_v2_{timestamp}_seed{self.config.seed}"
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = base / f"crispr_v2_{timestamp}_seed{self.config.seed}_{suffix}"
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

    def _random_trajectory(self, generation: int) -> Trajectory:
        horizon = self._sample_horizon()
        return self._assign_id(
            Trajectory(
                factor_ids=self.automaton.sample_uniform(horizon, self.rng),
                origin="random",
            ),
            generation,
        )

    def _load_seed_factors(self) -> Optional[tuple[int, ...]]:
        if self.config.seed_known_example:
            factors = known_example(self.config.seed_known_example)
            if not self.automaton.is_legal(factors):
                raise ValueError("built-in seed is not legal under the current GNF automaton")
            return factors
        if not self.config.seed_trajectory_json:
            return None
        with Path(self.config.seed_trajectory_json).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "factor_ids" in payload:
            factors = tuple(int(value) for value in payload["factor_ids"])
        elif "gnf_factors" in payload:
            perm_to_id, _ = simple_factor_id_maps(self.config.n)
            factors = tuple(perm_to_id[tuple(perm)] for perm in payload["gnf_factors"])
        else:
            raise ValueError("seed JSON must contain factor_ids or gnf_factors")
        if not self.automaton.is_legal(factors):
            raise ValueError("seed trajectory is not legal under the current GNF automaton")
        return factors

    def _corrupt_seed(self, seed: tuple[int, ...]) -> tuple[int, ...]:
        factors = seed
        target_edits = max(1, round(len(seed) * self.config.seed_corruption_fraction))
        block_sizes = (
            self.config.local_mutation_block_sizes
            + self.config.escape_mutation_block_sizes
        )
        edited = 0
        attempts = 0
        while edited < target_edits and attempts < target_edits * 10:
            attempts += 1
            block_length = min(
                self.rng.choice(block_sizes),
                len(seed),
                target_edits - edited,
            )
            start = self.rng.randint(0, len(seed) - block_length)
            end = start + block_length
            left = factors[start - 1] if start else None
            right = factors[end] if end < len(factors) else None
            replacement = self.automaton.sample_bridge(
                left=left,
                right=right,
                length=block_length,
                rng=self.rng,
                absolute_start=start,
                horizon=len(seed),
            )
            candidate = factors[:start] + replacement + factors[end:]
            if candidate == factors:
                continue
            factors = candidate
            edited += block_length
        return factors

    def initial_population(self) -> list[Trajectory]:
        population = []
        seed_factors = self._load_seed_factors()
        seed_count = (
            round(self.config.population_size * self.config.seed_population_fraction)
            if seed_factors is not None
            else 0
        )
        for _ in range(seed_count):
            population.append(
                self._assign_id(
                    Trajectory(
                        factor_ids=self._corrupt_seed(seed_factors),
                        origin="seed_corruption",
                    ),
                    generation=0,
                )
            )
        while len(population) < self.config.population_size:
            population.append(self._random_trajectory(generation=0))
        return population

    def _deduplicated_fill(
        self,
        candidates: Iterable[Trajectory],
        population: list[Trajectory],
        seen: set[tuple[int, ...]],
    ) -> None:
        for candidate in candidates:
            if len(population) >= self.config.population_size:
                return
            if candidate.factor_ids in seen:
                continue
            seen.add(candidate.factor_ids)
            population.append(candidate)

    def _population_counts(self, stagnant: bool) -> dict[str, int]:
        target = self.config.population_size
        fractions = {
            "archive": self.config.archive_fraction,
            "local": self.config.local_mutation_fraction,
            "escape": self.config.escape_mutation_fraction,
            "crossover": self.config.crossover_fraction,
            "random": self.config.random_sample_fraction,
        }
        if stagnant:
            fractions["local"] -= (
                self.config.stagnation_escape_boost
                + self.config.stagnation_random_boost
            )
            fractions["escape"] += self.config.stagnation_escape_boost
            fractions["random"] += self.config.stagnation_random_boost

        counts = {
            name: round(target * fraction)
            for name, fraction in fractions.items()
            if name != "local"
        }
        counts["local"] = target - sum(counts.values())
        return counts

    def _archive_carries(
        self,
        generation: int,
        parent_pool: list[TrajectoryEvaluation],
        count: int,
    ) -> list[Trajectory]:
        if not parent_pool:
            return []
        shuffled = list(parent_pool)
        self.rng.shuffle(shuffled)
        return [
            self._assign_id(
                Trajectory(
                    factor_ids=evaluation.trajectory.factor_ids,
                    origin="archive_carry",
                    parent_id=evaluation.trajectory.trajectory_id,
                    parent_score=evaluation.score,
                ),
                generation,
            )
            for evaluation in shuffled[:count]
        ]

    def _mutation_children(
        self,
        generation: int,
        parent_pool: list[TrajectoryEvaluation],
        count: int,
        lane: str,
    ) -> list[Trajectory]:
        if not parent_pool:
            return []
        output = []
        for _ in range(count):
            parent = self.rng.choice(parent_pool)
            child = self.mutation_planner.make_child(
                parent,
                lane=lane,
                two_mutations=lane == "escape" and self.rng.random() < 0.20,
            )
            output.append(self._assign_id(child, generation))
        return output

    def _crossover_children(
        self,
        generation: int,
        parent_pool: list[TrajectoryEvaluation],
        count: int,
    ) -> list[Trajectory]:
        by_horizon = defaultdict(list)
        for evaluation in parent_pool:
            by_horizon[evaluation.trajectory.horizon].append(evaluation)
        viable_groups = [group for group in by_horizon.values() if len(group) >= 2]
        if not viable_groups:
            return []

        output = []
        for _ in range(count):
            group = self.rng.choice(viable_groups)
            recipient, donor = self.rng.sample(group, 2)
            output.append(
                self._assign_id(
                    self.crossover.make_child(recipient, donor),
                    generation,
                )
            )
        return output

    def next_population(
        self,
        generation: int,
        elites: list[TrajectoryEvaluation],
        stagnant: bool,
    ) -> list[Trajectory]:
        parent_pool = self.archive.members()
        if not parent_pool:
            parent_pool = elites
        counts = self._population_counts(stagnant)
        population: list[Trajectory] = []
        seen: set[tuple[int, ...]] = set()

        candidate_groups = (
            self._archive_carries(
                generation,
                parent_pool,
                counts["archive"],
            ),
            self._mutation_children(
                generation,
                parent_pool,
                counts["local"],
                lane="local",
            ),
            self._mutation_children(
                generation,
                parent_pool,
                counts["escape"],
                lane="escape",
            ),
            self._crossover_children(
                generation,
                parent_pool,
                counts["crossover"],
            ),
            [
                self._random_trajectory(generation)
                for _ in range(counts["random"])
            ],
        )
        for candidates in candidate_groups:
            self._deduplicated_fill(candidates, population, seen)

        attempts = 0
        while len(population) < self.config.population_size:
            attempts += 1
            if parent_pool and attempts % 3:
                lane = "escape" if stagnant and attempts % 2 else "local"
                parent = self.rng.choice(parent_pool)
                candidate = self._assign_id(
                    self.mutation_planner.make_child(parent, lane=lane),
                    generation,
                )
            else:
                candidate = self._random_trajectory(generation)
            if candidate.factor_ids in seen:
                continue
            seen.add(candidate.factor_ids)
            population.append(candidate)

        return population

    def _update_hits(self, evaluations: list[TrajectoryEvaluation]) -> None:
        for evaluation in evaluations:
            if evaluation.has_kernel and len(self.kernel_hits) < self.config.max_kernel_hits:
                self.kernel_hits.append(evaluation)

    def _update_stagnation(self, generation: int, evaluations) -> bool:
        generation_best = min(item.final_projlen for item in evaluations)
        if generation_best < self.best_final_seen:
            self.best_final_seen = generation_best
            self.last_final_improvement_generation = generation
        return (
            generation - self.last_final_improvement_generation
            >= self.config.stagnation_generations
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
                "periodic_distance_history": list(evaluation.periodic_distance_history),
                "kernel_matches": list(evaluation.kernel_matches),
            }
        )
        return payload

    def _generation_summary(
        self,
        result: GenerationResult,
        elapsed: float,
        stagnant: bool,
    ) -> dict:
        evaluations = result.evaluations
        origins = Counter(item.trajectory.origin for item in evaluations)
        origin_best = defaultdict(lambda: float("-inf"))
        for item in evaluations:
            origin_best[item.trajectory.origin] = max(
                origin_best[item.trajectory.origin],
                item.score,
            )
        return {
            "generation": result.generation,
            "population_size": len(evaluations),
            "unique_trajectories": len(
                {item.trajectory.factor_ids for item in evaluations}
            ),
            "elite_count": len(result.elites),
            "archive_size": len(self.archive.members()),
            "stagnant": stagnant,
            "generations_since_final_improvement": (
                result.generation - self.last_final_improvement_generation
            ),
            "kernel_hits_this_generation": len(result.kernel_hits),
            "kernel_hits_total": len(self.kernel_hits),
            "best_score": max(item.score for item in evaluations),
            "best_final_projlen": min(item.final_projlen for item in evaluations),
            "best_late_projlen": min(item.min_late_projlen for item in evaluations),
            "largest_terminal_collapse": max(
                item.terminal_collapse for item in evaluations
            ),
            "lowest_rebound": min(item.rebound for item in evaluations),
            "best_periodic_distance": min(
                (
                    item.final_periodic_distance
                    for item in evaluations
                    if item.final_periodic_distance is not None
                ),
                default=None,
            ),
            "origins": dict(origins),
            "best_score_by_origin": dict(origin_best),
            "next_generation_counts": self._population_counts(stagnant),
            "elapsed_sec": round(elapsed, 4),
        }

    def _save_checkpoint(
        self,
        generation: int,
        population: list[Trajectory],
    ) -> None:
        payload = {
            "next_generation": generation,
            "population": [
                {
                    "factor_ids": list(item.factor_ids),
                    "origin": item.origin,
                    "parent_id": item.parent_id,
                    "parent_score": item.parent_score,
                    "trajectory_id": item.trajectory_id,
                }
                for item in population
            ],
        }
        with gzip.open(
            self.run_dir / "checkpoint.json.gz",
            "wt",
            encoding="utf-8",
        ) as handle:
            json.dump(payload, handle)

    def _save_progress(self, transition_limit: int = 100) -> None:
        champion_files = {
            "best_score": "best_candidate.json",
            "lowest_final": "lowest_final_candidate.json",
            "lowest_late": "lowest_late_candidate.json",
            "best_terminal_collapse": "best_terminal_collapse_candidate.json",
            "lowest_rebound": "lowest_rebound_candidate.json",
            "lowest_periodic": "lowest_periodic_candidate.json",
        }
        for name, filename in champion_files.items():
            champion = self.archive.champions.get(name)
            if champion is not None:
                self._write_json(filename, self._evaluation_json(champion))
        self._write_json(
            "kernel_hits.json",
            [self._evaluation_json(item) for item in self.kernel_hits],
        )
        self._write_json("archive.json", self.archive.summary())
        self._write_json(
            "transition_model.json",
            self.transition_model.top_transitions(limit=transition_limit),
        )
        self._write_json("mutation_stats.json", self.mutation_planner.stats_json())

    def run(self) -> dict:
        start = time.perf_counter()
        population = self.initial_population()
        completed_generations = 0

        for generation in range(self.config.generations):
            generation_start = time.perf_counter()
            evaluations = self.evaluator.evaluate(population)
            previous_hit_count = len(self.kernel_hits)
            self._update_hits(evaluations)
            archive_members = self.archive.update(evaluations)
            self.best = self.archive.champions.get("best_score")
            elites = select_diverse_elites(evaluations, self.config.elite_count)
            elite_keys = {item.trajectory.factor_ids for item in elites}
            for evaluation in evaluations:
                self.mutation_planner.observe(
                    evaluation,
                    archive_survivor=(
                        evaluation.trajectory.factor_ids in archive_members
                        or evaluation.trajectory.factor_ids in elite_keys
                    ),
                )
            self.mutation_planner.decay_statistics()
            self.transition_model.update(elites)
            stagnant = self._update_stagnation(generation, evaluations)
            result = GenerationResult(
                generation=generation,
                evaluations=evaluations,
                elites=elites,
                kernel_hits=self.kernel_hits[previous_hit_count:],
            )
            summary = self._generation_summary(
                result,
                elapsed=time.perf_counter() - generation_start,
                stagnant=stagnant,
            )
            with self.generation_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary) + "\n")
            print(json.dumps(summary), flush=True)
            self._save_progress()
            completed_generations = generation + 1

            if result.kernel_hits and self.config.stop_at_kernel:
                break
            if generation + 1 >= self.config.generations:
                break

            population = self.next_population(
                generation + 1,
                elites,
                stagnant=stagnant,
            )
            self._save_checkpoint(generation + 1, population)

        final_summary = {
            "run_dir": str(self.run_dir),
            "p": self.config.p,
            "n": self.config.n,
            "backend": self.config.backend,
            "device": self.config.device,
            "completed_generations": completed_generations,
            "population_size": self.config.population_size,
            "num_kernel_hits": len(self.kernel_hits),
            "archive_size": len(self.archive.members()),
            "best": self.best.summary() if self.best is not None else None,
            "champions": {
                name: evaluation.summary()
                for name, evaluation in sorted(self.archive.champions.items())
                if not name.startswith("lowest_final_h")
            },
            "elapsed_sec": round(time.perf_counter() - start, 4),
        }
        self._write_json("summary.json", final_summary)
        self._save_progress()
        return final_summary
