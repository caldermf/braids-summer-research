from __future__ import annotations

import gzip
import json
import pickle
import random
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import TARGET_TYPES, SearchConfig
from .exact_evaluator import make_exact_evaluator
from .field_sketch import ExtensionFieldSketch
from .gnf import GNFAutomaton
from .models import JoinCandidate, JoinEvaluation, Segment
from .operators import SegmentMutator
from .suffix_index import SuffixLSHIndex


class BidirectionalMatrixSearch:
    """Meet prefixes with reusable suffixes near inverse Burau targets."""

    def __init__(self, config: SearchConfig):
        config.validate()
        self.config = config
        self.rng = random.Random(config.seed)
        self.automaton = GNFAutomaton(config.n)
        self.sketch = ExtensionFieldSketch(config)
        self.exact = make_exact_evaluator(config)
        self.mutator = SegmentMutator(config, self.automaton, self.rng)
        self.segment_counter = 0
        self.stop_requested = False
        self.best: JoinEvaluation | None = None
        self.best_algebraic: JoinEvaluation | None = None
        self.kernel_hits: list[dict] = []
        self.output_root = Path(config.output_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.pointer_path = self.output_root / "latest_checkpoint.txt"
        self.run_dir: Path
        self.start_generation = 0
        self.prefixes: list[Segment]
        self.suffixes: list[Segment]
        if config.resume_latest and self.pointer_path.exists():
            self._load_checkpoint(Path(self.pointer_path.read_text().strip()))
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.output_root / f"bidirectional_{stamp}_seed{config.seed}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.prefixes = self._random_population("prefix", config.prefix_count)
            self.suffixes = self._random_population("suffix", config.suffix_count)
        self.generation_log = self.run_dir / "generations.jsonl"
        signal.signal(signal.SIGUSR1, self._request_stop)

    def _request_stop(self, _signum, _frame) -> None:
        self.stop_requested = True

    def _next_id(self, role: str, generation: int) -> str:
        self.segment_counter += 1
        return f"g{generation:04d}_{role[0]}{self.segment_counter:09d}"

    def _random_segment(self, role: str, generation: int = 0) -> Segment:
        return Segment(
            factor_ids=self.mutator.random_factors(role),
            role=role,
            segment_id=self._next_id(role, generation),
            origin="uniform_random",
        )

    def _random_population(self, role: str, count: int) -> list[Segment]:
        population = []
        seen = set()
        while len(population) < count:
            segment = self._random_segment(role)
            if segment.factor_ids in seen:
                continue
            seen.add(segment.factor_ids)
            population.append(segment)
        return population

    def _retrieve_candidates(
        self,
        prefix_targets: dict[str, np.ndarray],
        suffix_signatures: np.ndarray,
        suffix_index: SuffixLSHIndex,
    ) -> list[JoinCandidate]:
        candidates = []
        for prefix_index, prefix in enumerate(self.prefixes):
            allowed = self.automaton.successors[prefix.factor_ids[-1]]
            per_prefix: dict[int, JoinCandidate] = {}
            for target_type in TARGET_TYPES:
                matches = suffix_index.query(
                    prefix_targets[target_type][prefix_index],
                    allowed,
                    self.config.join_candidates_per_prefix,
                )
                for suffix_index_value, distance in matches:
                    candidate = JoinCandidate(
                        prefix_index=prefix_index,
                        suffix_index=suffix_index_value,
                        target_type=target_type,
                        sketch_distance=distance,
                    )
                    previous = per_prefix.get(suffix_index_value)
                    if previous is None or candidate.sketch_distance < previous.sketch_distance:
                        per_prefix[suffix_index_value] = candidate
            candidates.extend(
                sorted(
                    per_prefix.values(),
                    key=lambda item: item.sketch_distance,
                )[: self.config.join_candidates_per_prefix]
            )
        return candidates

    def _evaluate_joins(self, candidates: Sequence[JoinCandidate]) -> list[JoinEvaluation]:
        unique: dict[tuple[int, ...], JoinCandidate] = {}
        for candidate in candidates:
            prefix = self.prefixes[candidate.prefix_index]
            suffix = self.suffixes[candidate.suffix_index]
            if not self.automaton.can_join(prefix.factor_ids, suffix.factor_ids):
                continue
            factors = prefix.factor_ids + suffix.factor_ids
            previous = unique.get(factors)
            if previous is None or candidate.sketch_distance < previous.sketch_distance:
                unique[factors] = candidate
        words = list(unique)
        results = self.exact.evaluate(words)
        evaluations = []
        for factors, result in zip(words, results):
            candidate = unique[factors]
            evaluations.append(
                JoinEvaluation(
                    candidate=candidate,
                    prefix=self.prefixes[candidate.prefix_index],
                    suffix=self.suffixes[candidate.suffix_index],
                    word=result,
                )
            )
        evaluations.sort(key=JoinEvaluation.rank, reverse=True)
        return evaluations

    def _targeted_refinement(
        self,
        elites: Sequence[JoinEvaluation],
        prefix_targets: dict[str, np.ndarray],
        suffix_signatures: np.ndarray,
        generation: int,
    ) -> tuple[list[Segment], list[Segment]]:
        if not self.config.refinement_pairs or not self.config.refinement_trials:
            return [], []
        chosen = elites[: self.config.refinement_pairs]
        suffix_trials: list[Segment] = []
        suffix_owners: list[JoinEvaluation] = []
        prefix_trials: list[Segment] = []
        prefix_owners: list[JoinEvaluation] = []
        for elite in chosen:
            suffix_neighbors = self.mutator.local_neighbors(
                elite.suffix,
                self.config.refinement_trials,
                lambda role: self._next_id(role, generation),
            )
            prefix_neighbors = self.mutator.local_neighbors(
                elite.prefix,
                self.config.refinement_trials,
                lambda role: self._next_id(role, generation),
            )
            for suffix in suffix_neighbors:
                if self.automaton.can_join(elite.prefix.factor_ids, suffix.factor_ids):
                    suffix_trials.append(suffix)
                    suffix_owners.append(elite)
            for prefix in prefix_neighbors:
                if self.automaton.can_join(prefix.factor_ids, elite.suffix.factor_ids):
                    prefix_trials.append(prefix)
                    prefix_owners.append(elite)

            random_trials = max(2, self.config.refinement_trials // 4)
            for _ in range(random_trials):
                suffix = self.mutator.mutate(
                    elite.suffix,
                    self._next_id("suffix", generation),
                )
                if self.automaton.can_join(elite.prefix.factor_ids, suffix.factor_ids):
                    suffix_trials.append(suffix)
                    suffix_owners.append(elite)
                prefix = self.mutator.mutate(
                    elite.prefix,
                    self._next_id("prefix", generation),
                )
                if self.automaton.can_join(prefix.factor_ids, elite.suffix.factor_ids):
                    prefix_trials.append(prefix)
                    prefix_owners.append(elite)

        refined_suffixes = []
        if suffix_trials:
            trial_signatures = self.sketch.suffix_signatures(suffix_trials)
            best_by_pair: dict[tuple[int, int], tuple[int, Segment]] = {}
            for trial, owner, signature in zip(
                suffix_trials,
                suffix_owners,
                trial_signatures,
            ):
                target = prefix_targets[owner.candidate.target_type][
                    owner.candidate.prefix_index
                ]
                distance = int(np.count_nonzero(signature != target))
                key = (owner.candidate.prefix_index, owner.candidate.suffix_index)
                if distance < owner.candidate.sketch_distance and (
                    key not in best_by_pair or distance < best_by_pair[key][0]
                ):
                    best_by_pair[key] = distance, trial
            refined_suffixes = [item[1] for item in best_by_pair.values()]

        refined_prefixes = []
        if prefix_trials:
            trial_targets = self.sketch.prefix_target_signatures(prefix_trials)
            best_by_pair = {}
            for trial_index, (trial, owner) in enumerate(zip(prefix_trials, prefix_owners)):
                target = trial_targets[owner.candidate.target_type][trial_index]
                suffix_signature = suffix_signatures[owner.candidate.suffix_index]
                distance = int(np.count_nonzero(target != suffix_signature))
                key = (owner.candidate.prefix_index, owner.candidate.suffix_index)
                if distance < owner.candidate.sketch_distance and (
                    key not in best_by_pair or distance < best_by_pair[key][0]
                ):
                    best_by_pair[key] = distance, trial
            refined_prefixes = [item[1] for item in best_by_pair.values()]
        return refined_prefixes, refined_suffixes

    def _niche_select(
        self,
        evaluations: Sequence[JoinEvaluation],
        count: int,
        algebraic: bool,
    ) -> list[JoinEvaluation]:
        groups: dict[int, list[JoinEvaluation]] = {}
        for evaluation in evaluations:
            niche = len(evaluation.factor_ids) // self.config.length_niche_width
            groups.setdefault(niche, []).append(evaluation)
        ranker = (
            JoinEvaluation.algebra_rank
            if algebraic
            else JoinEvaluation.rank
        )
        for group in groups.values():
            group.sort(key=ranker, reverse=True)
        selected = []
        seen = set()
        depth = 0
        ordered_niches = sorted(groups)
        while len(selected) < count:
            progress = False
            for niche in ordered_niches:
                group = groups[niche]
                if depth >= len(group):
                    continue
                candidate = group[depth]
                key = (
                    candidate.prefix.factor_ids,
                    candidate.suffix.factor_ids,
                )
                if key in seen:
                    continue
                selected.append(candidate)
                seen.add(key)
                progress = True
                if len(selected) >= count:
                    break
            if not progress and depth >= max(
                (len(group) for group in groups.values()),
                default=0,
            ):
                break
            depth += 1
        return selected

    def _select_elites(
        self,
        evaluations: Sequence[JoinEvaluation],
    ) -> tuple[list[JoinEvaluation], list[JoinEvaluation]]:
        algebra_count = round(
            self.config.elite_pairs * self.config.algebra_elite_fraction
        )
        endpoint_count = self.config.elite_pairs - algebra_count
        endpoint = self._niche_select(evaluations, endpoint_count, algebraic=False)
        algebra = self._niche_select(evaluations, algebra_count, algebraic=True)
        combined = []
        seen = set()
        for evaluation in endpoint + algebra:
            key = (evaluation.prefix.factor_ids, evaluation.suffix.factor_ids)
            if key in seen:
                continue
            seen.add(key)
            combined.append(evaluation)
        if len(combined) < self.config.elite_pairs:
            for evaluation in evaluations:
                key = (evaluation.prefix.factor_ids, evaluation.suffix.factor_ids)
                if key in seen:
                    continue
                seen.add(key)
                combined.append(evaluation)
                if len(combined) >= self.config.elite_pairs:
                    break
        return combined, algebra

    def _next_population(
        self,
        role: str,
        target_size: int,
        elites: Sequence[Segment],
        refined: Sequence[Segment],
        generation: int,
    ) -> list[Segment]:
        population = []
        seen = set()

        def add(segment: Segment) -> None:
            if len(population) < target_size and segment.factor_ids not in seen:
                seen.add(segment.factor_ids)
                population.append(segment)

        carry_count = max(1, round(target_size * self.config.carry_fraction))
        for segment in elites[:carry_count]:
            add(segment)
        for segment in refined:
            add(segment)

        random_count = round(target_size * self.config.random_fraction)
        mutation_target = target_size - random_count
        parents = list(elites) or [
            self._random_segment(role, generation)
        ]
        failed = 0
        while len(population) < mutation_target and failed < target_size * 20:
            parent = self.rng.choice(parents)
            child = self.mutator.mutate(parent, self._next_id(role, generation))
            before = len(population)
            add(child)
            failed = failed + 1 if len(population) == before else 0
        while len(population) < target_size:
            add(self._random_segment(role, generation))
        return population

    @staticmethod
    def _unique_elite_segments(
        evaluations: Sequence[JoinEvaluation],
        role: str,
        count: int,
    ) -> list[Segment]:
        selected = []
        seen = set()
        for evaluation in evaluations:
            segment = evaluation.prefix if role == "prefix" else evaluation.suffix
            if segment.factor_ids in seen:
                continue
            seen.add(segment.factor_ids)
            selected.append(segment)
            if len(selected) >= count:
                break
        return selected

    def _save_json(self, name: str, payload) -> None:
        (self.run_dir / name).write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def _save_checkpoint(
        self,
        generation: int,
        prefixes: list[Segment],
        suffixes: list[Segment],
    ) -> None:
        checkpoint = self.run_dir / "checkpoint.pkl.gz"
        payload = {
            "generation": generation,
            "prefixes": prefixes,
            "suffixes": suffixes,
            "rng_state": self.rng.getstate(),
            "segment_counter": self.segment_counter,
            "best": self.best,
            "best_algebraic": self.best_algebraic,
            "kernel_hits": self.kernel_hits,
            "run_dir": str(self.run_dir),
        }
        temporary = checkpoint.with_suffix(".tmp.gz")
        with gzip.open(temporary, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(checkpoint)
        self.pointer_path.write_text(str(checkpoint.resolve()) + "\n", encoding="utf-8")

    def _load_checkpoint(self, checkpoint: Path) -> None:
        with gzip.open(checkpoint, "rb") as handle:
            payload = pickle.load(handle)
        self.start_generation = payload["generation"]
        self.prefixes = payload["prefixes"]
        self.suffixes = payload["suffixes"]
        self.rng.setstate(payload["rng_state"])
        self.segment_counter = payload["segment_counter"]
        self.best = payload["best"]
        self.best_algebraic = payload.get("best_algebraic")
        self.kernel_hits = payload["kernel_hits"]
        self.run_dir = Path(payload["run_dir"])
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        started = time.perf_counter()
        completed_generation = self.start_generation
        stop_reason = "max_generations"
        for generation in range(self.start_generation, self.config.generations):
            generation_started = time.perf_counter()
            prefix_targets = self.sketch.prefix_target_signatures(self.prefixes)
            suffix_signatures = self.sketch.suffix_signatures(self.suffixes)
            suffix_index = SuffixLSHIndex(
                self.config,
                self.suffixes,
                suffix_signatures,
                self.rng,
            )
            candidates = self._retrieve_candidates(
                prefix_targets,
                suffix_signatures,
                suffix_index,
            )
            evaluations = self._evaluate_joins(candidates)
            if not evaluations:
                raise RuntimeError("bidirectional retrieval produced no legal joins")

            generation_best = evaluations[0]
            generation_algebraic = max(
                evaluations,
                key=JoinEvaluation.algebra_rank,
            )
            if self.best is None or generation_best.rank() > self.best.rank():
                self.best = generation_best
                self._save_json("best_candidate.json", self.best.summary())
            if (
                self.best_algebraic is None
                or generation_algebraic.algebra_rank()
                > self.best_algebraic.algebra_rank()
            ):
                self.best_algebraic = generation_algebraic
                self._save_json(
                    "best_algebraic_candidate.json",
                    self.best_algebraic.summary(),
                )
            new_hits = [item for item in evaluations if item.word.has_kernel]
            for hit in new_hits:
                summary = hit.summary()
                if tuple(summary["factor_ids"]) not in {
                    tuple(existing["factor_ids"]) for existing in self.kernel_hits
                }:
                    self.kernel_hits.append(summary)
            self.kernel_hits = self.kernel_hits[: self.config.max_kernel_hits]
            self._save_json("kernel_hits.json", self.kernel_hits)

            elite_evaluations, algebra_elites = self._select_elites(evaluations)
            refined_prefixes, refined_suffixes = self._targeted_refinement(
                algebra_elites,
                prefix_targets,
                suffix_signatures,
                generation + 1,
            )
            prefix_elites = self._unique_elite_segments(
                elite_evaluations,
                "prefix",
                self.config.elite_pairs,
            )
            suffix_elites = self._unique_elite_segments(
                elite_evaluations,
                "suffix",
                self.config.elite_pairs,
            )
            next_prefixes = self._next_population(
                "prefix",
                self.config.prefix_count,
                prefix_elites,
                refined_prefixes,
                generation + 1,
            )
            next_suffixes = self._next_population(
                "suffix",
                self.config.suffix_count,
                suffix_elites,
                refined_suffixes,
                generation + 1,
            )

            summary = {
                "generation": generation,
                "prefixes": len(self.prefixes),
                "suffixes": len(self.suffixes),
                "retrieved_joins": len(candidates),
                "evaluated_joins": len(evaluations),
                "best_final_projlen": generation_best.word.final_projlen,
                "best_final_horizon": len(generation_best.factor_ids),
                "best_sketch_distance": generation_algebraic.candidate.sketch_distance,
                "best_algebraic_final_projlen": generation_algebraic.word.final_projlen,
                "best_algebraic_horizon": len(generation_algebraic.factor_ids),
                "best_largest_drop": max(item.word.largest_drop for item in evaluations),
                "kernel_hits_this_generation": len(new_hits),
                "kernel_hits_total": len(self.kernel_hits),
                "best_target_type": generation_best.candidate.target_type,
                "best_horizon": len(generation_best.factor_ids),
                "refined_prefixes": len(refined_prefixes),
                "refined_suffixes": len(refined_suffixes),
                "suffix_index": suffix_index.stats(),
                "elapsed_sec": round(time.perf_counter() - generation_started, 4),
            }
            with self.generation_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary) + "\n")
            print(json.dumps(summary), flush=True)

            completed_generation = generation + 1
            self.prefixes = next_prefixes
            self.suffixes = next_suffixes
            self._save_checkpoint(
                completed_generation,
                self.prefixes,
                self.suffixes,
            )
            if new_hits and self.config.stop_at_kernel:
                stop_reason = "kernel_found"
                break
            if self.stop_requested:
                stop_reason = "signal_checkpoint"
                break

        final_summary = {
            "run_dir": str(self.run_dir),
            "algorithm": "bidirectional_matrix_search_v5",
            "p": self.config.p,
            "n": self.config.n,
            "completed_generations": completed_generation,
            "stop_reason": stop_reason,
            "prefix_count": self.config.prefix_count,
            "suffix_count": self.config.suffix_count,
            "field": f"GF({self.config.p}^2)",
            "field_points": self.config.field_points,
            "num_kernel_hits": len(self.kernel_hits),
            "best": self.best.summary() if self.best else None,
            "best_algebraic": (
                self.best_algebraic.summary() if self.best_algebraic else None
            ),
            "elapsed_sec": round(time.perf_counter() - started, 4),
        }
        self._save_json("summary.json", final_summary)
        return final_summary
