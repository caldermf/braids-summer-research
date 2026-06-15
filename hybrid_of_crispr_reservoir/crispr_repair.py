from __future__ import annotations

import math
import random
from dataclasses import asdict, replace
from pathlib import Path

try:
    from crispr_algorithms.crispr_trajectory_search_v4.config import (
        ISLAND_NAMES,
        SearchConfig,
    )
    from crispr_algorithms.crispr_trajectory_search_v4.evaluators import make_evaluator
    from crispr_algorithms.crispr_trajectory_search_v4.gnf import GNFAutomaton
    from crispr_algorithms.crispr_trajectory_search_v4.models import (
        MutationRecord,
        Trajectory,
    )
    from crispr_algorithms.crispr_trajectory_search_v4.mutation import (
        StructuralMutationPlanner,
    )
    from crispr_algorithms.crispr_trajectory_search_v4.transition_model import (
        TransitionModel,
    )
except ModuleNotFoundError:
    from crispr_trajectory_search_v4.config import ISLAND_NAMES, SearchConfig
    from crispr_trajectory_search_v4.evaluators import make_evaluator
    from crispr_trajectory_search_v4.gnf import GNFAutomaton
    from crispr_trajectory_search_v4.models import MutationRecord, Trajectory
    from crispr_trajectory_search_v4.mutation import StructuralMutationPlanner
    from crispr_trajectory_search_v4.transition_model import TransitionModel

from .checkpoint import Candidate, select_diverse_candidates
from .config import CrisprConfig
from .io_utils import append_jsonl, write_json


def repair_tail_metrics(history: tuple[int, ...], base_depth: int) -> dict:
    if not history:
        raise ValueError("cannot score an empty projective-length history")
    start = min(len(history) - 1, max(0, base_depth - 1))
    tail = tuple(history[start:])
    running_peak = tail[0]
    max_drawdown = 0
    for value in tail:
        running_peak = max(running_peak, value)
        max_drawdown = max(max_drawdown, running_peak - value)

    terminal = tail[-min(12, len(tail)) :]
    descent_steps = sum(
        right < left for left, right in zip(terminal, terminal[1:])
    )
    rise_steps = sum(right > left for left, right in zip(terminal, terminal[1:]))
    terminal_slope = (
        (terminal[0] - terminal[-1]) / (len(terminal) - 1)
        if len(terminal) > 1
        else 0.0
    )
    weights = tuple(range(1, len(terminal) + 1))
    weighted_area = sum(weight * value for weight, value in zip(weights, terminal))
    weighted_area /= max(1, sum(weights))
    return {
        "tail": tail,
        "final": tail[-1],
        "peak": max(tail),
        "mean": sum(tail) / len(tail),
        "max_drawdown": max_drawdown,
        "terminal_slope": terminal_slope,
        "terminal_descent_fraction": descent_steps / max(1, len(terminal) - 1),
        "terminal_rise_fraction": rise_steps / max(1, len(terminal) - 1),
        "terminal_weighted_area": weighted_area,
    }


def _apply_repair_scores(evaluation, base_depth: int, max_depth: int):
    metrics = repair_tail_metrics(evaluation.projlen_history, base_depth)
    span = max(1, max_depth - base_depth)
    depth_bonus = (evaluation.trajectory.horizon - base_depth) / span
    kernel_bonus = 1_000_000.0 if evaluation.has_kernel else 0.0
    final = metrics["final"]
    peak = metrics["peak"]
    mean = metrics["mean"]
    drop = metrics["max_drawdown"]
    slope = metrics["terminal_slope"]
    descent = metrics["terminal_descent_fraction"]
    rise = metrics["terminal_rise_fraction"]
    area = metrics["terminal_weighted_area"]

    scores = {
        "endpoint": kernel_bonus - final + 0.30 * drop + 0.75 * depth_bonus,
        "envelope": (
            kernel_bonus
            - 0.50 * peak
            - 0.30 * mean
            - 0.20 * final
            + 0.50 * depth_bonus
        ),
        "collapse": (
            kernel_bonus
            + 1.40 * drop
            + 2.00 * slope
            + 2.00 * descent
            - 0.45 * final
            - rise
            + depth_bonus
        ),
        "suffix": (
            kernel_bonus
            - area
            - 1.50 * rise
            + 0.50 * drop
            + slope
            + depth_bonus
        ),
    }
    evaluation.island_scores = scores
    evaluation.score = scores[evaluation.trajectory.island]
    return evaluation


def _repair_rank(evaluation, island: str) -> tuple:
    return (
        1 if evaluation.has_kernel else 0,
        evaluation.score_for(island),
        -evaluation.final_projlen,
        evaluation.trajectory.horizon,
        evaluation.novelty,
    )


def _select_ranked_diverse(evaluations, island: str, count: int, seen_words=None):
    if count <= 0:
        return []
    selected = []
    seen_words = set() if seen_words is None else seen_words
    seen_states = set()
    ordered = sorted(
        evaluations,
        key=lambda item: _repair_rank(item, island),
        reverse=True,
    )
    for candidate in ordered:
        word = candidate.trajectory.factor_ids
        if word in seen_words:
            continue
        if candidate.matrix_fingerprint in seen_states and len(selected) < count // 2:
            continue
        selected.append(candidate)
        seen_words.add(word)
        seen_states.add(candidate.matrix_fingerprint)
        if len(selected) >= count:
            return selected
    for candidate in ordered:
        word = candidate.trajectory.factor_ids
        if word in seen_words:
            continue
        selected.append(candidate)
        seen_words.add(word)
        if len(selected) >= count:
            break
    return selected


def select_repair_elites(
    evaluations,
    island: str,
    count: int,
    active_max: int,
    boundary_fraction: float,
    boundary_margin: int,
):
    items = list(evaluations)
    boundary_target = min(count, math.ceil(count * boundary_fraction))
    boundary = [
        item
        for item in items
        if item.trajectory.horizon >= active_max - boundary_margin
    ]
    seen_words = set()
    selected = _select_ranked_diverse(
        boundary,
        island,
        boundary_target,
        seen_words,
    )
    selected.extend(
        _select_ranked_diverse(
            items,
            island,
            count - len(selected),
            seen_words,
        )
    )
    return selected


def _trajectory_id(island: str, generation: int, index: int) -> str:
    return f"reservoir-crispr-{island}-g{generation:03d}-{index:08d}"


def _v4_config(
    branch: CrisprConfig,
    p: int,
    n: int,
    base_depth: int,
    max_depth: int,
) -> SearchConfig:
    return SearchConfig(
        p=p,
        n=n,
        min_horizon=base_depth,
        initial_max_horizon=max_depth,
        hard_max_horizon=max_depth,
        horizon_boundary_margin=branch.boundary_margin,
        horizon_boundary_elite_fraction=branch.boundary_fraction,
        population_size=branch.population_per_island * len(ISLAND_NAMES),
        min_generations=1,
        max_generations=max(1, branch.generations),
        backend=branch.backend,
        device=branch.device,
        seed=branch.seed,
        mcts_enabled=False,
        stop_at_kernel=False,
    )


def _as_island(evaluation, island: str, trajectory_id: str):
    trajectory = replace(
        evaluation.trajectory,
        island=island,
        trajectory_id=trajectory_id,
    )
    return replace(
        evaluation,
        trajectory=trajectory,
        score=evaluation.score_for(island),
        island_scores=dict(evaluation.island_scores),
    )


def _evaluation_summary(evaluation, base_depth: int) -> dict:
    metrics = repair_tail_metrics(
        evaluation.projlen_history,
        base_depth,
    )
    return {
        **evaluation.summary(),
        "projlen_history": list(evaluation.projlen_history),
        "kernel_matches": list(evaluation.kernel_matches),
        "terminal_metrics": {
            key: value for key, value in metrics.items() if key != "tail"
        },
    }


def _population_summary(
    populations: dict[str, list],
    active_max: int,
    boundary_margin: int,
) -> dict:
    champions = {}
    for island, population in populations.items():
        champion = max(population, key=lambda item: _repair_rank(item, island))
        champions[island] = {
            **champion.summary(),
            "projlen_history": list(champion.projlen_history),
        }
    all_items = [item for population in populations.values() for item in population]
    boundary_by_island = {
        island: sum(
            item.trajectory.horizon >= active_max - boundary_margin
            for item in population
        )
        for island, population in populations.items()
    }
    return {
        "active_max_depth": active_max,
        "evaluations": len(all_items),
        "unique_words": len({item.trajectory.factor_ids for item in all_items}),
        "unique_matrix_states": len({item.matrix_fingerprint for item in all_items}),
        "depth_min": min(item.trajectory.horizon for item in all_items),
        "depth_max": max(item.trajectory.horizon for item in all_items),
        "boundary_count": sum(boundary_by_island.values()),
        "boundary_by_island": boundary_by_island,
        "best_final_projlen": min(item.final_projlen for item in all_items),
        "kernel_hits": sum(item.has_kernel for item in all_items),
        "champions": champions,
    }


def run_crispr_repair(
    candidates: list[Candidate],
    branch: CrisprConfig,
    *,
    p: int,
    n: int,
    base_depth: int,
    max_depth: int,
    output_dir: str | Path,
    stop_at_kernel: bool = True,
) -> dict:
    if max_depth <= base_depth:
        raise ValueError("CRISPR maximum depth must exceed reservoir depth")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events_path = output / "generations.jsonl"
    if events_path.exists():
        events_path.unlink()

    rng = random.Random(branch.seed)
    selected = select_diverse_candidates(candidates, branch.pool_size, branch.seed)
    if not selected:
        raise ValueError("CRISPR repair received an empty reservoir pool")

    config = _v4_config(branch, p, n, base_depth, max_depth)
    config.validate()
    automaton = GNFAutomaton(n)
    evaluator = make_evaluator(config)
    transition_models = {
        island: TransitionModel(config, automaton) for island in ISLAND_NAMES
    }
    planners = {
        island: StructuralMutationPlanner(
            config,
            island,
            automaton,
            transition_models[island],
            random.Random(branch.seed + 1009 * (index + 1)),
        )
        for index, island in enumerate(ISLAND_NAMES)
    }

    seed_words = []
    seen = set()
    for candidate in selected:
        if candidate.factor_ids not in seen:
            seen.add(candidate.factor_ids)
            seed_words.append(candidate.factor_ids)
    seed_trajectories = [
        Trajectory(
            factor_ids=word,
            island="endpoint",
            origin=f"paper_reservoir_depth{base_depth}",
            trajectory_id=_trajectory_id("seed", 0, index),
        )
        for index, word in enumerate(seed_words)
    ]
    seed_evaluations = evaluator.evaluate(seed_trajectories)
    for item in seed_evaluations:
        _apply_repair_scores(item, base_depth, max_depth)

    populations: dict[str, list] = {}
    for island_index, island in enumerate(ISLAND_NAMES):
        offset = island_index * max(1, len(seed_evaluations) // len(ISLAND_NAMES))
        rotated = seed_evaluations[offset:] + seed_evaluations[:offset]
        chosen = rotated[: branch.population_per_island]
        populations[island] = [
            _as_island(item, island, _trajectory_id(island, 0, index))
            for index, item in enumerate(chosen)
        ]
        transition_models[island].update(populations[island])

    kernel_hits = {}
    for population in populations.values():
        for item in population:
            if item.has_kernel:
                kernel_hits.setdefault(item.trajectory.factor_ids, item)
    append_jsonl(
        events_path,
        {
            "generation": 0,
            **_population_summary(
                populations,
                base_depth,
                branch.boundary_margin,
            ),
        },
    )

    generations_to_run = 0 if kernel_hits and stop_at_kernel else branch.generations
    for generation in range(1, generations_to_run + 1):
        span = max_depth - base_depth
        active_max = min(
            max_depth,
            base_depth + max(1, math.ceil(span * generation / branch.generations)),
        )
        next_populations = {}

        for island in ISLAND_NAMES:
            current = populations[island]
            elite_count = max(
                1,
                round(branch.population_per_island * branch.elite_fraction),
            )
            retained = select_repair_elites(
                current,
                island,
                elite_count,
                active_max,
                branch.boundary_fraction,
                branch.boundary_margin,
            )
            parent_count = max(1, round(len(current) * branch.parent_fraction))
            parents = select_repair_elites(
                current,
                island,
                parent_count,
                active_max,
                branch.boundary_fraction,
                branch.boundary_margin,
            )

            target_offspring = max(
                branch.population_per_island - len(retained),
                branch.population_per_island * branch.offspring_multiplier,
            )
            boundary_target = math.ceil(
                branch.population_per_island * branch.boundary_fraction
            )
            boundary_parents = [
                item
                for item in parents
                if item.trajectory.horizon < active_max
            ]
            children = []
            seen_words = {item.trajectory.factor_ids for item in retained}
            attempts = 0
            while len(children) < target_offspring and attempts < target_offspring * 30:
                attempts += 1
                current_boundary = sum(
                    item.trajectory.horizon >= active_max - branch.boundary_margin
                    for item in retained
                ) + sum(
                    child.horizon >= active_max - branch.boundary_margin
                    for child in children
                )
                force_boundary = (
                    current_boundary < boundary_target and bool(boundary_parents)
                )
                parent = rng.choice(boundary_parents if force_boundary else parents)
                factors = parent.trajectory.factor_ids
                should_append = len(factors) < active_max and (
                    force_boundary or rng.random() < branch.append_fraction
                )
                if should_append:
                    append_length = (
                        active_max - len(factors)
                        if force_boundary
                        else min(active_max - len(factors), rng.choice((1, 1, 2, 3)))
                    )
                    addition = automaton.sample_bridge(
                        left=factors[-1],
                        right=None,
                        length=append_length,
                        rng=rng,
                        chooser=transition_models[island].choose,
                        absolute_start=len(factors),
                        horizon=len(factors) + append_length,
                    )
                    child = Trajectory(
                        factor_ids=factors + addition,
                        island=island,
                        origin="reservoir_boundary_append",
                        parent_id=parent.trajectory.trajectory_id,
                        parent_score=parent.score_for(island),
                        mutation_records=(
                            MutationRecord(
                                island=island,
                                action="append",
                                mode="boundary" if force_boundary else "terminal",
                                start=len(factors),
                                removed_length=0,
                                inserted_length=append_length,
                                location_bin=7,
                            ),
                        ),
                    )
                else:
                    child = planners[island].make_child(
                        parent,
                        active_max_horizon=active_max,
                        stagnant=False,
                        force_large=False,
                        use_learned=True,
                        two_mutations=rng.random() < 0.20,
                    )
                if child.factor_ids in seen_words:
                    continue
                if not automaton.is_legal(child.factor_ids):
                    raise AssertionError("CRISPR planner emitted an illegal GNF word")
                seen_words.add(child.factor_ids)
                child.trajectory_id = _trajectory_id(island, generation, len(children))
                children.append(child)

            child_evaluations = evaluator.evaluate(children)
            for item in child_evaluations:
                _apply_repair_scores(item, base_depth, max_depth)
                planners[island].observe(item)
                if item.has_kernel:
                    kernel_hits.setdefault(item.trajectory.factor_ids, item)

            next_populations[island] = select_repair_elites(
                retained + child_evaluations,
                island,
                branch.population_per_island,
                active_max,
                branch.boundary_fraction,
                branch.boundary_margin,
            )
            transition_models[island].update(next_populations[island])
            planners[island].decay_statistics()

        populations = next_populations
        event = {
            "generation": generation,
            **_population_summary(
                populations,
                active_max,
                branch.boundary_margin,
            ),
        }
        append_jsonl(events_path, event)
        print(
            f"[crispr] generation={generation} active_max={active_max} "
            f"depth_max={event['depth_max']} best_final={event['best_final_projlen']} "
            f"hits={len(kernel_hits)}",
            flush=True,
        )
        if kernel_hits and stop_at_kernel:
            break

    all_items = [item for population in populations.values() for item in population]
    best = sorted(
        all_items,
        key=lambda item: (
            0 if item.has_kernel else 1,
            item.final_projlen,
            -item.trajectory.horizon,
            item.normalized_terminal_area,
        ),
    )[:100]
    result = {
        "branch": "crispr_repair_after_paper_reservoir",
        "config": asdict(branch),
        "reservoir_depth": base_depth,
        "reservoir_seeds": len(selected),
        "kernel_hits": [
            _evaluation_summary(item, base_depth) for item in kernel_hits.values()
        ],
        "best": [_evaluation_summary(item, base_depth) for item in best],
        "mutation_stats": {
            island: planners[island].stats_json() for island in ISLAND_NAMES
        },
        "transition_models": {
            island: transition_models[island].top_transitions()
            for island in ISLAND_NAMES
        },
    }
    write_json(output / "result.json", result)
    return result
