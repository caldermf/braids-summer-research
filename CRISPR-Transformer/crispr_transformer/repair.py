from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .checkpoint import load_checkpoints
from .edits import EditGeometry, enumerate_and_apply, valid_geometries
from .exact import Evaluation, make_evaluator, require_compatible_cuda
from .gnf import GNFAutomaton
from .io_utils import append_jsonl, write_json
from .model import load_model
from .percentiles import LengthPercentiles


def _initial_words(candidates, limit: int, seed: int) -> list[tuple[int, ...]]:
    rng = random.Random(seed)
    ordered = sorted(candidates, key=lambda item: item.author_projlen)
    groups = defaultdict(list)
    for candidate in ordered:
        groups[(candidate.author_projlen, candidate.factor_ids[-1])].append(candidate)
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
    exploration_fraction: float,
    rng: random.Random,
    device: str,
) -> tuple[dict[tuple[int, ...], list[EditGeometry]], Counter]:
    """Encode parents in batches, then score every permitted integer geometry."""
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
                valid_geometries(
                    parent.length,
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
            chosen = [geometry for _, geometry in scored[:exploit_count]]
            remaining = [geometry for _, geometry in scored[exploit_count:]]
            random_count = min(actions_per_parent - len(chosen), len(remaining))
            if random_count:
                chosen.extend(rng.sample(remaining, random_count))
            choices[parent.factor_ids] = chosen
            origins["model_geometry"] += len(chosen) - random_count
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
    model = None
    model_payload = None
    if model_path is not None:
        model, model_payload = load_model(str(model_path), device=device)
        if (model.config.p, model.config.n) != (p, n):
            raise ValueError("model p/n does not match the reservoir checkpoint")

    min_length = min(baseline.values)
    max_length = min(max(baseline.values), model.config.max_length if model else max(baseline.values))
    max_delete = model.config.max_delete if model else 16
    max_insert = model.config.max_insert if model else 16
    max_net_delta = model.config.max_net_delta if model else 3
    rng = random.Random(seed)
    automaton = GNFAutomaton(n=n)
    evaluator = make_evaluator(
        p=p,
        n=n,
        backend=backend,
        device=device,
        batch_size=eval_batch_size,
    )
    initial = _initial_words(candidates, population_size, seed)
    population = evaluator.evaluate(initial)
    cache = {item.factor_ids: item for item in population}
    all_kernel_hits = [item for item in population if item.has_kernel]
    best_seen = min(
        population,
        key=lambda item: baseline.quality(item.length, item.final_projlen),
    )

    for generation in range(generations + 1):
        if generation == 0:
            children = []
            proposal_origins = Counter()
            duplicate_rejections = 0
        else:
            proposals = []
            proposal_origins = Counter()
            duplicate_rejections = 0
            guided_choices = {}
            if mode == "guided":
                guided_choices, proposal_origins = _choose_guided_geometries(
                    model,
                    population,
                    min_length=min_length,
                    max_length=max_length,
                    actions_per_parent=actions_per_parent,
                    exploration_fraction=exploration_fraction,
                    rng=rng,
                    device=device,
                )
            for parent in population:
                geometries = valid_geometries(
                    parent.length,
                    min_length=min_length,
                    max_length=max_length,
                    max_delete=max_delete,
                    max_insert=max_insert,
                    max_net_delta=max_net_delta,
                )
                if not geometries:
                    continue
                if mode == "guided":
                    chosen = guided_choices.get(parent.factor_ids, [])
                else:
                    chosen = rng.sample(geometries, min(actions_per_parent, len(geometries)))
                    proposal_origins["random_geometry"] += len(chosen)
                proposals.extend(
                    enumerate_and_apply(
                        parent.factor_ids,
                        chosen,
                        replacements_per_action,
                        automaton,
                        rng,
                    )
                )
            unseen_words = []
            seen_generation = set()
            for _, child in proposals:
                if child in cache or child in seen_generation:
                    duplicate_rejections += 1
                    continue
                seen_generation.add(child)
                unseen_words.append(child)
            children = evaluator.evaluate(unseen_words)
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
            if baseline.quality(candidate_best.length, candidate_best.final_projlen) < baseline.quality(
                best_seen.length, best_seen.final_projlen
            ):
                best_seen = candidate_best

        row = {
            "generation": generation,
            "mode": mode,
            "population": len(population),
            "children_evaluated": len(children),
            "cache_entries": len(cache),
            "duplicate_rejections": duplicate_rejections,
            "proposal_origins": dict(proposal_origins),
            "kernel_hits_total": len(all_kernel_hits),
            "lengths": dict(sorted(Counter(item.length for item in population).items())),
            "best_projlen_by_length": {
                str(length): min(
                    item.final_projlen for item in population if item.length == length
                )
                for length in sorted({item.length for item in population})
            },
            "best": _best_summary(best_seen, baseline),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        append_jsonl(events_path, row)
        print(row, flush=True)
        if all_kernel_hits and stop_at_kernel:
            break

    unique_hits = {}
    for hit in all_kernel_hits:
        unique_hits.setdefault(hit.factor_ids, hit)
    hits = [item.summary() for item in unique_hits.values()]
    result = {
        "format": "crispr-transformer-repair-run-v1",
        "mode": mode,
        "p": p,
        "n": n,
        "model": str(Path(model_path).resolve()) if model_path else None,
        "model_best_epoch": model_payload.get("best_epoch") if model_payload else None,
        "baseline": str(Path(baseline_path).resolve()),
        "checkpoints": [str(Path(path).resolve()) for path in checkpoints],
        "completed_generations": generation,
        "population_size": population_size,
        "unique_evaluations": len(cache),
        "num_kernel_hits": len(hits),
        "kernel_hits": hits,
        "best": _best_summary(best_seen, baseline),
        "backend": backend,
        "device": device,
        "seed": seed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output / "summary.json", result)
    write_json(output / "best_candidate.json", result["best"])
    write_json(output / "kernel_hits.json", hits)
    return result
