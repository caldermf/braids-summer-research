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
    from crispr_algorithms.crispr_trajectory_search_v4.islands import (
        island_rank,
        select_island_elites,
    )
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
    from crispr_trajectory_search_v4.islands import island_rank, select_island_elites
    from crispr_trajectory_search_v4.models import MutationRecord, Trajectory
    from crispr_trajectory_search_v4.mutation import StructuralMutationPlanner
    from crispr_trajectory_search_v4.transition_model import TransitionModel

from .candidates import Candidate, select_diverse_candidates
from .config import CrisprConfig
from .io_utils import append_jsonl, write_json


def _trajectory_id(island: str, generation: int, index: int) -> str:
    return f"hybrid-crispr-{island}-g{generation:03d}-{index:08d}"


def _v4_config(
    branch: CrisprConfig,
    p: int,
    n: int,
    min_depth: int,
    max_depth: int,
) -> SearchConfig:
    return SearchConfig(
        p=p,
        n=n,
        min_horizon=min_depth,
        initial_max_horizon=max_depth,
        hard_max_horizon=max_depth,
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


def _evaluation_summary(evaluation) -> dict:
    return {
        **evaluation.summary(),
        "projlen_history": list(evaluation.projlen_history),
        "kernel_matches": list(evaluation.kernel_matches),
    }


def _population_summary(populations: dict[str, list], active_max: int) -> dict:
    champions = {}
    for island, population in populations.items():
        champion = max(population, key=lambda item: island_rank(item, island))
        champions[island] = _evaluation_summary(champion)
    all_items = [item for population in populations.values() for item in population]
    return {
        "active_max_depth": active_max,
        "evaluations": len(all_items),
        "unique_words": len(
            {item.trajectory.factor_ids for item in all_items}
        ),
        "unique_matrix_states": len(
            {item.matrix_fingerprint for item in all_items}
        ),
        "depth_min": min(item.trajectory.horizon for item in all_items),
        "depth_max": max(item.trajectory.horizon for item in all_items),
        "best_final_projlen": min(item.final_projlen for item in all_items),
        "best_min_projlen": min(item.min_projlen for item in all_items),
        "kernel_hits": sum(item.has_kernel for item in all_items),
        "champions": champions,
    }


def run_crispr_branch(
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
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events_path = output / "generations.jsonl"
    if events_path.exists():
        events_path.unlink()

    rng = random.Random(branch.seed)
    selected = select_diverse_candidates(candidates, branch.pool_size, branch.seed)
    if not selected:
        raise ValueError("CRISPR branch received an empty candidate pool")

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

    # Evaluate each unique reservoir seed once, then reinterpret the same exact
    # trajectory under each island objective.
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
            origin="paper_reservoir_depth35",
            trajectory_id=_trajectory_id("seed", 0, index),
        )
        for index, word in enumerate(seed_words)
    ]
    seed_evaluations = evaluator.evaluate(seed_trajectories)

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
    initial = {"generation": 0, **_population_summary(populations, base_depth)}
    append_jsonl(events_path, initial)

    for generation in range(1, branch.generations + 1):
        span = max_depth - base_depth
        active_max = min(
            max_depth,
            base_depth + max(1, math.ceil(span * generation / branch.generations)),
        )
        next_populations = {}

        for island in ISLAND_NAMES:
            current = populations[island]
            elite_count = max(1, round(branch.population_per_island * branch.elite_fraction))
            retained = select_island_elites(
                current,
                island,
                elite_count,
                config.length_niche_width,
            )
            parent_count = max(
                1,
                round(len(current) * branch.parent_fraction),
            )
            parents = select_island_elites(
                current,
                island,
                parent_count,
                config.length_niche_width,
            )

            target_offspring = max(
                branch.population_per_island - len(retained),
                branch.population_per_island * branch.offspring_multiplier,
            )
            children = []
            seen_words = {item.trajectory.factor_ids for item in retained}
            attempts = 0
            while len(children) < target_offspring and attempts < target_offspring * 20:
                attempts += 1
                parent = rng.choice(parents)
                factors = parent.trajectory.factor_ids
                if len(factors) < active_max and rng.random() < branch.append_fraction:
                    addition = automaton.sample_bridge(
                        left=factors[-1],
                        right=None,
                        length=1,
                        rng=rng,
                        chooser=transition_models[island].choose,
                        absolute_start=len(factors),
                        horizon=len(factors) + 1,
                    )
                    child = Trajectory(
                        factor_ids=factors + addition,
                        island=island,
                        origin="reservoir_prefix_append",
                        parent_id=parent.trajectory.trajectory_id,
                        parent_score=parent.score_for(island),
                        mutation_records=(
                            MutationRecord(
                                island=island,
                                action="append",
                                mode="terminal",
                                start=len(factors),
                                removed_length=0,
                                inserted_length=1,
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
                child.trajectory_id = _trajectory_id(
                    island,
                    generation,
                    len(children),
                )
                children.append(child)

            child_evaluations = evaluator.evaluate(children)
            for item in child_evaluations:
                planners[island].observe(item)
                if item.has_kernel:
                    kernel_hits.setdefault(item.trajectory.factor_ids, item)

            combined = retained + child_evaluations
            next_populations[island] = select_island_elites(
                combined,
                island,
                branch.population_per_island,
                config.length_niche_width,
            )
            transition_models[island].update(next_populations[island])
            planners[island].decay_statistics()

        populations = next_populations
        event = {
            "generation": generation,
            **_population_summary(populations, active_max),
        }
        append_jsonl(events_path, event)
        print(
            f"[crispr] generation={generation} active_max={active_max} "
            f"best_final={event['best_final_projlen']} hits={len(kernel_hits)}",
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
            item.normalized_peak,
            item.normalized_terminal_area,
        ),
    )[:100]
    result = {
        "branch": "crispr_repair",
        "config": asdict(branch),
        "reservoir_seeds": len(selected),
        "kernel_hits": [_evaluation_summary(item) for item in kernel_hits.values()],
        "best": [_evaluation_summary(item) for item in best],
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
