from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .checkpoint import load_checkpoints
from .edits import EditGeometry, enumerate_and_apply, sample_balanced_geometries
from .exact import Evaluation, require_compatible_cuda
from .gnf import GNFAutomaton
from .io_utils import append_jsonl, read_json, write_json
from .model import load_model
from .percentiles import LengthPercentiles
from structural_experiments.objectives import make_objective_evaluator


def _initial_words(candidates, limit: int, seed: int) -> list[tuple[int, ...]]:
    rng = random.Random(seed)
    ordered = sorted(candidates, key=lambda item: item.author_projlen)
    groups = defaultdict(list)
    for candidate in ordered:
        groups[
            (candidate.checkpoint, candidate.author_projlen, candidate.factor_ids[-1])
        ].append(candidate)
    for values in groups.values():
        rng.shuffle(values)
    words = []
    seen = set()
    keys = sorted(groups)
    while keys and len(words) < limit:
        remaining = []
        for key in keys:
            if not groups[key]:
                continue
            candidate = groups[key].pop()
            if candidate.factor_ids not in seen:
                seen.add(candidate.factor_ids)
                words.append(candidate.factor_ids)
            if groups[key]:
                remaining.append(key)
            if len(words) >= limit:
                break
        keys = remaining
    return words


@torch.no_grad()
def _choose_guided_geometries(
    model,
    population: list[Evaluation],
    *,
    min_length: int,
    max_length: int,
    actions_per_parent: int,
    geometry_candidates_per_parent: int,
    exploration_fraction: float,
    rng: random.Random,
    device: str,
) -> tuple[dict[tuple[int, ...], list[tuple[EditGeometry, str]]], Counter]:
    """Rank a balanced, multi-scale candidate set for every parent."""
    target = torch.device(device)
    choices = {}
    origins = Counter()
    for batch_start in range(0, len(population), 32):
        batch = population[batch_start : batch_start + 32]
        width = max(item.length for item in batch)
        tokens = torch.zeros(len(batch), width, dtype=torch.long, device=target)
        histories = torch.zeros(
            len(batch), width, dtype=torch.float32, device=target
        )
        lengths = torch.tensor(
            [item.length for item in batch], dtype=torch.long, device=target
        )
        geometries_by_parent = []
        for parent_index, parent in enumerate(batch):
            tokens[parent_index, : parent.length] = torch.tensor(
                [value + 1 for value in parent.factor_ids],
                dtype=torch.long,
                device=target,
            )
            histories[parent_index, : parent.length] = torch.tensor(
                parent.projlen_history, dtype=torch.float32, device=target
            )
            geometries_by_parent.append(
                sample_balanced_geometries(
                    parent.length,
                    geometry_candidates_per_parent,
                    rng,
                    min_length=min_length,
                    max_length=max_length,
                    max_delete=model.config.max_delete,
                    max_insert=model.config.max_insert,
                    max_net_delta=model.config.max_net_delta,
                )
            )
        hidden, pooled = model.encode(tokens, histories, lengths)
        flat = [
            (parent_index, geometry)
            for parent_index, geometries in enumerate(geometries_by_parent)
            for geometry in geometries
        ]
        scored_by_parent = [[] for _ in batch]
        for action_start in range(0, len(flat), 16_384):
            chunk = flat[action_start : action_start + 16_384]
            action_parents = torch.tensor(
                [parent_index for parent_index, _ in chunk],
                dtype=torch.long,
                device=target,
            )
            actions = torch.tensor(
                [
                    (geometry.start, geometry.delete_length, geometry.insert_length)
                    for _, geometry in chunk
                ],
                dtype=torch.long,
                device=target,
            )
            scores = model.score_encoded(
                hidden, pooled, lengths, action_parents, actions
            ).cpu().tolist()
            for score, (parent_index, geometry) in zip(scores, chunk):
                scored_by_parent[parent_index].append((score, geometry))
        for parent, scored in zip(batch, scored_by_parent):
            scored.sort(key=lambda pair: pair[0], reverse=True)
            exploit_count = min(
                actions_per_parent,
                max(1, round(actions_per_parent * (1.0 - exploration_fraction))),
            )
            model_chosen = [geometry for _, geometry in scored[:exploit_count]]
            remaining = [geometry for _, geometry in scored[exploit_count:]]
            random_count = min(actions_per_parent - len(model_chosen), len(remaining))
            explored = rng.sample(remaining, random_count) if random_count else []
            chosen = [
                (geometry, "model_geometry") for geometry in model_chosen
            ] + [
                (geometry, "exploration_geometry") for geometry in explored
            ]
            choices[parent.factor_ids] = chosen
            origins["model_geometry"] += len(model_chosen)
            origins["exploration_geometry"] += random_count
    return choices, origins


def _select_population(
    evaluations: list[Evaluation],
    baseline: LengthPercentiles,
    population_size: int,
) -> list[Evaluation]:
    unique = {}
    for evaluation in evaluations:
        current = unique.get(evaluation.factor_ids)
        if current is None or evaluation.final_projlen < current.final_projlen:
            unique[evaluation.factor_ids] = evaluation
    groups = defaultdict(list)
    for evaluation in unique.values():
        groups[evaluation.length].append(evaluation)
    for group in groups.values():
        group.sort(key=lambda item: item.final_projlen)

    lengths = sorted(groups)
    niche_quota = max(2, population_size // max(1, 2 * len(lengths)))
    selected = []
    seen = set()
    for length in lengths:
        for evaluation in groups[length][:niche_quota]:
            selected.append(evaluation)
            seen.add(evaluation.factor_ids)
            if len(selected) >= population_size:
                return selected
    remainder = [
        item
        for item in unique.values()
        if item.factor_ids not in seen
    ]
    remainder.sort(
        key=lambda item: (
            baseline.quality(item.length, item.final_projlen),
            item.final_projlen,
        )
    )
    selected.extend(remainder[: population_size - len(selected)])
    return selected


def _best_summary(evaluation: Evaluation, baseline: LengthPercentiles) -> dict:
    return {
        **evaluation.summary(),
        "length_conditioned_percentile": baseline.quality(
            evaluation.length, evaluation.final_projlen
        ),
    }


def run_guided_repair(
    *,
    checkpoints: list[str | Path],
    baseline_path: str | Path,
    output_dir: str | Path,
    model_path: str | Path | None = None,
    mode: str = "guided",
    population_size: int = 512,
    generations: int = 40,
    actions_per_parent: int = 4,
    replacements_per_action: int = 4,
    exploration_fraction: float = 0.15,
    geometry_candidates_per_parent: int = 1_024,
    stagnation_generations: int = 15,
    restart_fraction: float = 0.25,
    backend: str = "torch",
    device: str = "cuda",
    eval_batch_size: int = 10_000,
    stop_at_kernel: bool = True,
    seed: int = 1,
) -> dict:
    if mode not in {"guided", "random"}:
        raise ValueError("mode must be 'guided' or 'random'")
    if mode == "guided" and model_path is None:
        raise ValueError("guided repair requires a trained model")
    if geometry_candidates_per_parent < actions_per_parent:
        raise ValueError("geometry candidate pool must cover selected actions")
    if stagnation_generations < 1 or not 0.0 <= restart_fraction < 1.0:
        raise ValueError("invalid stagnation restart parameters")
    if device.startswith("cuda"):
        require_compatible_cuda(torch)
    started = time.perf_counter()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    events_path = output / "generations.jsonl"
    if events_path.exists():
        events_path.unlink()

    metadata, candidates = load_checkpoints(checkpoints)
    p, n = int(metadata["p"]), int(metadata["n"])
    baseline = LengthPercentiles.load(baseline_path)
    if (baseline.p, baseline.n) != (p, n):
        raise ValueError("baseline p/n does not match the reservoir checkpoint")
    baseline_summary_path = Path(baseline_path).resolve().with_name(
        "dataset_summary.json"
    )
    if baseline_summary_path.is_file():
        baseline_summary = read_json(baseline_summary_path)
        if baseline_summary.get("objective", "ordinary_projlen") != metadata.get(
            "objective", "ordinary_projlen"
        ):
            raise ValueError("baseline objective does not match the reservoir checkpoint")
        if baseline_summary.get("generator_index") != metadata.get("generator_index"):
            raise ValueError("baseline generator does not match the reservoir checkpoint")
    model = None
    model_payload = None
    if model_path is not None:
        model, model_payload = load_model(str(model_path), device=device)
        if (model.config.p, model.config.n) != (p, n):
            raise ValueError("model p/n does not match the reservoir checkpoint")
        expected_objective = metadata.get("objective", "ordinary_projlen")
        if model_payload.get("objective", "ordinary_projlen") != expected_objective:
            raise ValueError("model objective does not match the reservoir checkpoint")
        if model_payload.get("generator_index") != metadata.get("generator_index"):
            raise ValueError("model generator does not match the reservoir checkpoint")

    min_length = min(baseline.values)
    max_length = min(max(baseline.values), model.config.max_length if model else max(baseline.values))
    max_delete = model.config.max_delete if model else 16
    max_insert = model.config.max_insert if model else 16
    max_net_delta = model.config.max_net_delta if model else 3
    rng = random.Random(seed)
    automaton = GNFAutomaton(n=n)
    evaluator = make_objective_evaluator(
        metadata=metadata,
        backend=backend,
        device=device,
        batch_size=eval_batch_size,
        max_length=max_length,
    )
    initial = _initial_words(candidates, population_size, seed)
    population = evaluator.evaluate(initial)
    cache = {item.factor_ids: item for item in population}
    discovery = {
        item.factor_ids: {"origin": "reservoir", "generation": 0}
        for item in population
    }
    all_kernel_hits = [item for item in population if item.has_kernel]
    best_seen = min(
        population,
        key=lambda item: baseline.quality(item.length, item.final_projlen),
    )
    restart_words = [
        candidate.factor_ids
        for candidate in candidates
        if candidate.factor_ids not in cache
    ]
    rng.shuffle(restart_words)
    restart_cursor = 0
    stagnant_for = 0
    restart_events = 0

    for generation in range(generations + 1):
        if generation == 0:
            children = []
            proposal_origins = Counter()
            duplicate_rejections = 0
            restart_triggered = False
        else:
            proposals = []
            proposal_origins = Counter()
            evaluated_origins = Counter()
            duplicate_rejections = 0
            duplicate_origins = Counter()
            guided_choices = {}
            if mode == "guided":
                guided_choices, proposal_origins = _choose_guided_geometries(
                    model,
                    population,
                    min_length=min_length,
                    max_length=max_length,
                    actions_per_parent=actions_per_parent,
                    geometry_candidates_per_parent=geometry_candidates_per_parent,
                    exploration_fraction=exploration_fraction,
                    rng=rng,
                    device=device,
                )
            for parent in population:
                if mode == "guided":
                    chosen = guided_choices.get(parent.factor_ids, [])
                else:
                    geometries = sample_balanced_geometries(
                        parent.length,
                        geometry_candidates_per_parent,
                        rng,
                        min_length=min_length,
                        max_length=max_length,
                        max_delete=max_delete,
                        max_insert=max_insert,
                        max_net_delta=max_net_delta,
                    )
                    random_geometries = rng.sample(
                        geometries,
                        min(actions_per_parent, len(geometries)),
                    )
                    chosen = [
                        (geometry, "random_geometry")
                        for geometry in random_geometries
                    ]
                    proposal_origins["random_geometry"] += len(chosen)
                for geometry, origin in chosen:
                    generated = enumerate_and_apply(
                        parent.factor_ids,
                        [geometry],
                        replacements_per_action,
                        automaton,
                        rng,
                    )
                    proposals.extend(
                        (generated_geometry, child, origin, parent.factor_ids)
                        for generated_geometry, child in generated
                    )
            unseen_words = []
            unseen_metadata = []
            seen_generation = set()
            for geometry, child, origin, parent_word in proposals:
                if child in cache or child in seen_generation:
                    duplicate_rejections += 1
                    duplicate_origins[origin] += 1
                    continue
                seen_generation.add(child)
                unseen_words.append(child)
                unseen_metadata.append(
                    {
                        "origin": origin,
                        "generation": generation,
                        "geometry": geometry.to_dict(),
                        "parent_factor_ids": list(parent_word),
                    }
                )
            children = evaluator.evaluate(unseen_words)
            for child, metadata_row in zip(children, unseen_metadata):
                discovery[child.factor_ids] = metadata_row
                evaluated_origins[metadata_row["origin"]] += 1
            cache.update((item.factor_ids, item) for item in children)
            all_kernel_hits.extend(item for item in children if item.has_kernel)
            population = _select_population(
                population + children,
                baseline,
                population_size,
            )
            candidate_best = min(
                population,
                key=lambda item: baseline.quality(item.length, item.final_projlen),
            )
            improved = baseline.quality(
                candidate_best.length, candidate_best.final_projlen
            ) < baseline.quality(
                best_seen.length, best_seen.final_projlen
            )
            if improved:
                best_seen = candidate_best
                stagnant_for = 0
            else:
                stagnant_for += 1

            restart_triggered = (
                restart_fraction > 0.0 and stagnant_for >= stagnation_generations
            )
            if restart_triggered:
                requested = max(1, round(population_size * restart_fraction))
                words = []
                while restart_cursor < len(restart_words) and len(words) < requested:
                    word = restart_words[restart_cursor]
                    restart_cursor += 1
                    if word not in cache:
                        words.append(word)
                while len(words) < requested:
                    word = automaton.sample_uniform(
                        rng.randint(min_length, max_length),
                        rng,
                    )
                    if word not in cache and word not in words:
                        words.append(word)
                restarted = evaluator.evaluate(words)
                for item in restarted:
                    discovery[item.factor_ids] = {
                        "origin": "stagnation_restart",
                        "generation": generation,
                    }
                cache.update((item.factor_ids, item) for item in restarted)
                children.extend(restarted)
                all_kernel_hits.extend(item for item in restarted if item.has_kernel)
                evaluated_origins["stagnation_restart"] += len(restarted)
                population = _select_population(
                    population + restarted,
                    baseline,
                    population_size,
                )
                candidate_best = min(
                    population,
                    key=lambda item: baseline.quality(item.length, item.final_projlen),
                )
                if baseline.quality(
                    candidate_best.length, candidate_best.final_projlen
                ) < baseline.quality(best_seen.length, best_seen.final_projlen):
                    best_seen = candidate_best
                stagnant_for = 0
                restart_events += 1

        row = {
            "generation": generation,
            "mode": mode,
            "population": len(population),
            "children_evaluated": len(children),
            "cache_entries": len(cache),
            "duplicate_rejections": duplicate_rejections,
            "stagnant_for": stagnant_for,
            "restart_triggered": restart_triggered,
            "restart_events_total": restart_events,
            "proposal_origins": dict(proposal_origins),
            "evaluated_children_by_origin": dict(evaluated_origins) if generation else {},
            "duplicate_rejections_by_origin": dict(duplicate_origins) if generation else {},
            "kernel_hits_total": len(all_kernel_hits),
            "lengths": dict(sorted(Counter(item.length for item in population).items())),
            "best_projlen_by_length": {
                str(length): min(
                    item.final_projlen for item in population if item.length == length
                )
                for length in sorted({item.length for item in population})
            },
            "best": _best_summary(best_seen, baseline),
            "best_discovery": discovery[best_seen.factor_ids],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        append_jsonl(events_path, row)
        print(row, flush=True)
        if all_kernel_hits and stop_at_kernel:
            break

    unique_hits = {}
    for hit in all_kernel_hits:
        unique_hits.setdefault(hit.factor_ids, hit)
    hits = [
        {**item.summary(), "discovery": discovery[item.factor_ids]}
        for item in unique_hits.values()
    ]
    result = {
        "format": "crispr-transformer-wide-edit-repair-run-v3",
        "mode": mode,
        "p": p,
        "n": n,
        "objective": metadata.get("objective", "ordinary_projlen"),
        "generator_index": metadata.get("generator_index"),
        "model": str(Path(model_path).resolve()) if model_path else None,
        "model_best_epoch": model_payload.get("best_epoch") if model_payload else None,
        "baseline": str(Path(baseline_path).resolve()),
        "checkpoints": [str(Path(path).resolve()) for path in checkpoints],
        "completed_generations": generation,
        "population_size": population_size,
        "geometry_candidates_per_parent": geometry_candidates_per_parent,
        "stagnation_generations": stagnation_generations,
        "restart_fraction": restart_fraction,
        "restart_events": restart_events,
        "unique_evaluations": len(cache),
        "num_kernel_hits": len(hits),
        "kernel_hits": hits,
        "best": _best_summary(best_seen, baseline),
        "best_discovery": discovery[best_seen.factor_ids],
        "backend": backend,
        "device": device,
        "seed": seed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output / "summary.json", result)
    write_json(output / "best_candidate.json", result["best"])
    write_json(output / "kernel_hits.json", hits)
    return result
