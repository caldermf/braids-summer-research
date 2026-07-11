from __future__ import annotations

import random
import time
from collections import defaultdict
from pathlib import Path

from .checkpoint import Candidate, load_checkpoints
from .edits import (
    apply_geometry,
    enumerate_and_apply,
    sample_balanced_geometries,
)
from .exact import Evaluation
from .gnf import GNFAutomaton
from .io_utils import write_json, write_jsonl
from .percentiles import LengthPercentiles
from structural_experiments.objectives import make_objective_evaluator


def _choose_parents(
    candidates: list[Candidate],
    limit: int,
    seed: int,
) -> list[Candidate]:
    if limit <= 0 or limit >= len(candidates):
        return list(candidates)
    rng = random.Random(seed)
    by_checkpoint: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_checkpoint[candidate.checkpoint].append(candidate)
    selected = []
    selected_words = set()
    checkpoint_groups = list(by_checkpoint.values())
    quota = max(1, limit // len(checkpoint_groups))
    for group in checkpoint_groups:
        ordered = sorted(group, key=lambda item: item.author_projlen)
        group_limit = min(quota, len(ordered))
        elite_count = min(group_limit, max(1, round(0.70 * group_limit)))
        choices = ordered[:elite_count]
        remainder = ordered[elite_count:]
        choices.extend(
            rng.sample(remainder, min(group_limit - len(choices), len(remainder)))
        )
        for candidate in choices:
            if candidate.factor_ids not in selected_words:
                selected_words.add(candidate.factor_ids)
                selected.append(candidate)
    if len(selected) < limit:
        remainder = [
            item for item in candidates if item.factor_ids not in selected_words
        ]
        selected.extend(rng.sample(remainder, min(limit - len(selected), len(remainder))))
    rng.shuffle(selected)
    return selected


def _build_random_baseline(
    automaton: GNFAutomaton,
    evaluator,
    min_length: int,
    max_length: int,
    samples_per_length: int,
    rng: random.Random,
) -> dict[int, list[int]]:
    samples: dict[int, list[int]] = defaultdict(list)
    for length in range(min_length, max_length + 1):
        words = [
            automaton.sample_uniform(length, rng)
            for _ in range(samples_per_length)
        ]
        for evaluation in evaluator.evaluate(words):
            samples[length].append(evaluation.final_projlen)
    return samples


def _augment_parent_lengths(
    parents: list[Candidate],
    fraction: float,
    *,
    automaton: GNFAutomaton,
    min_length: int,
    max_length: int,
    max_delete: int,
    max_insert: int,
    max_net_delta: int,
    rng: random.Random,
) -> list[Candidate]:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("augmented_parent_fraction must lie in [0, 1)")
    target = round(len(parents) * fraction)
    if target == 0:
        return parents
    retained = parents[: len(parents) - target]
    augmented = []
    seen = {item.factor_ids for item in parents}
    attempts = 0
    while len(augmented) < target and attempts < target * 50:
        attempts += 1
        source = rng.choice(parents)
        word = source.factor_ids
        for _ in range(rng.randint(1, 6)):
            actions = sample_balanced_geometries(
                len(word),
                64,
                rng,
                min_length=min_length,
                max_length=max_length,
                max_delete=max_delete,
                max_insert=max_insert,
                max_net_delta=max_net_delta,
            )
            if not actions:
                break
            try:
                word = apply_geometry(word, rng.choice(actions), automaton, rng)
            except (ValueError, RuntimeError):
                continue
        if word in seen:
            continue
        seen.add(word)
        augmented.append(
            Candidate(
                factor_ids=word,
                author_projlen=-1,
                matrix_fingerprint="",
                source_index=-1 - len(augmented),
                checkpoint=f"{source.checkpoint}#random-length-augmentation",
            )
        )
    if len(augmented) != target:
        raise RuntimeError("could not construct the requested augmented parent set")
    return retained + augmented


def generate_mutation_dataset(
    *,
    checkpoints: list[str | Path],
    output_dir: str | Path,
    parents_limit: int = 5_000,
    actions_per_parent: int = 16,
    replacements_per_action: int = 4,
    max_delete: int = 16,
    max_insert: int = 16,
    max_net_delta: int = 3,
    min_length: int | None = None,
    max_length: int | None = None,
    baseline_samples_per_length: int = 2_048,
    augmented_parent_fraction: float = 0.25,
    target_top_k: int = 2,
    allow_unconfirmed_handoff: bool = False,
    backend: str = "torch",
    device: str = "cuda",
    eval_batch_size: int = 10_000,
    seed: int = 1,
) -> dict:
    started = time.perf_counter()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata, candidates = load_checkpoints(checkpoints)
    p, n = int(metadata["p"]), int(metadata["n"])
    unconfirmed = [
        item
        for item in metadata.get("checkpoint_metadata", [])
        if item.get("adaptive_downturn")
        and item.get("halt_reason") not in {
            "sustained_downturn_handoff",
            "author_projlen_one",
        }
    ]
    if unconfirmed and not allow_unconfirmed_handoff:
        reasons = sorted({str(item.get("halt_reason")) for item in unconfirmed})
        raise ValueError(
            "adaptive reservoir did not confirm a downturn "
            f"(halt reasons: {reasons}); increase its max depth or explicitly "
            "pass --allow-unconfirmed-handoff"
        )
    parents = _choose_parents(candidates, parents_limit, seed)
    if not parents:
        raise ValueError("reservoir checkpoint has no frontier candidates")

    observed_lengths = [candidate.length for candidate in parents]
    # Leave enough room for repeated small edits to move well beyond the
    # original frontier length (for example, depth 60 down to the known 54).
    default_margin = max(16, max_delete, max_insert)
    min_length = int(min_length or max(2, min(observed_lengths) - default_margin))
    max_length = int(max_length or max(observed_lengths) + default_margin)
    if min_length > min(observed_lengths) or max_length < max(observed_lengths):
        raise ValueError("length bounds must include every selected reservoir parent")

    rng = random.Random(seed)
    automaton = GNFAutomaton(n=n)
    parents = _augment_parent_lengths(
        parents,
        augmented_parent_fraction,
        automaton=automaton,
        min_length=min_length,
        max_length=max_length,
        max_delete=max_delete,
        max_insert=max_insert,
        max_net_delta=max_net_delta,
        rng=rng,
    )
    automaton.assert_legal(candidate.factor_ids for candidate in parents)
    evaluator = make_objective_evaluator(
        metadata=metadata,
        backend=backend,
        device=device,
        batch_size=eval_batch_size,
        max_length=max_length,
    )
    parent_evaluations = evaluator.evaluate(candidate.factor_ids for candidate in parents)
    baseline_samples = _build_random_baseline(
        automaton,
        evaluator,
        min_length,
        max_length,
        baseline_samples_per_length,
        rng,
    )

    raw_groups = []
    kernel_hits = []
    action_counter = 0
    child_counter = 0
    parent_batch_size = max(1, min(128, eval_batch_size // max(1, actions_per_parent)))
    for batch_start in range(0, len(parents), parent_batch_size):
        batch_end = min(len(parents), batch_start + parent_batch_size)
        indexed_proposals = []
        for parent_index in range(batch_start, batch_end):
            parent_eval = parent_evaluations[parent_index]
            geometries = sample_balanced_geometries(
                parent_eval.length,
                actions_per_parent,
                rng,
                min_length=min_length,
                max_length=max_length,
                max_delete=max_delete,
                max_insert=max_insert,
                max_net_delta=max_net_delta,
            )
            proposals = enumerate_and_apply(
                parent_eval.factor_ids,
                geometries,
                replacements_per_action,
                automaton,
                rng,
            )
            indexed_proposals.extend(
                (parent_index, geometry, child)
                for geometry, child in proposals
            )
        child_evaluations = evaluator.evaluate(
            child for _, _, child in indexed_proposals
        )
        grouped: dict[tuple[int, int, int, int], list[Evaluation]] = defaultdict(list)
        for (parent_index, geometry, _), child_eval in zip(
            indexed_proposals, child_evaluations
        ):
            key = (
                parent_index,
                geometry.start,
                geometry.delete_length,
                geometry.insert_length,
            )
            grouped[key].append(child_eval)
            child_counter += 1
            if child_eval.has_kernel:
                kernel_hits.append(child_eval.summary())
        for parent_index in range(batch_start, batch_end):
            candidate = parents[parent_index]
            parent_eval = parent_evaluations[parent_index]
            actions = []
            for (
                grouped_parent,
                start,
                delete_length,
                insert_length,
            ), outcomes in grouped.items():
                if grouped_parent != parent_index:
                    continue
                actions.append(
                    {
                        "start": start,
                        "delete_length": delete_length,
                        "insert_length": insert_length,
                        "child_projlens": [item.final_projlen for item in outcomes],
                        "child_lengths": [item.length for item in outcomes],
                    }
                )
            action_counter += len(actions)
            raw_groups.append(
                {
                    "parent_id": parent_index,
                    "checkpoint": candidate.checkpoint,
                    "source_index": candidate.source_index,
                    "factor_ids": list(parent_eval.factor_ids),
                    "projlen_history": list(parent_eval.projlen_history),
                    "parent_projlen": parent_eval.final_projlen,
                    "actions": actions,
                }
            )
        if batch_end % 100 == 0 or batch_end == len(parents):
            print(
                {
                    "parents_labeled": batch_end,
                    "actions": action_counter,
                    "children": child_counter,
                    "kernel_hits": len(kernel_hits),
                },
                flush=True,
            )

    baseline = LengthPercentiles.from_samples(
        p,
        n,
        baseline_samples,
        effective_sample_size=baseline_samples_per_length,
    )
    baseline_path = baseline.save(output / "length_percentiles.json")
    rows = []
    beneficial = 0
    for group in raw_groups:
        parent_length = len(group["factor_ids"])
        parent_projlen = int(group["parent_projlen"])
        parent_quality = baseline.quality(parent_length, parent_projlen)
        labeled_actions = []
        for action in group["actions"]:
            rewards = [
                baseline.reward(
                    parent_length,
                    parent_projlen,
                    child_length,
                    child_projlen,
                )
                for child_length, child_projlen in zip(
                    action["child_lengths"], action["child_projlens"]
                )
            ]
            if not rewards:
                continue
            top = sorted(rewards, reverse=True)[: max(1, min(target_top_k, len(rewards)))]
            target = sum(top) / len(top)
            beneficial += target > 0
            labeled_actions.append(
                {
                    "start": action["start"],
                    "delete_length": action["delete_length"],
                    "insert_length": action["insert_length"],
                    "target_reward": target,
                    "mean_reward": sum(rewards) / len(rewards),
                    "best_reward": max(rewards),
                    "beneficial_rate": sum(value > 0 for value in rewards) / len(rewards),
                    "samples": len(rewards),
                }
            )
        if len(labeled_actions) >= 2:
            rows.append(
                {
                    **{key: value for key, value in group.items() if key != "actions"},
                    "parent_quality": parent_quality,
                    "actions": labeled_actions,
                }
            )

    dataset_path = write_jsonl(output / "mutation_groups.jsonl.gz", rows)
    summary = {
        "format": "crispr-transformer-wide-edit-mutation-dataset-v3",
        "p": p,
        "n": n,
        "objective": metadata.get("objective", "ordinary_projlen"),
        "generator_index": metadata.get("generator_index"),
        "checkpoints": [str(Path(path).resolve()) for path in checkpoints],
        "dataset": str(dataset_path),
        "length_percentiles": str(baseline_path),
        "parents_available": len(candidates),
        "parents_selected": len(parents),
        "augmented_parent_fraction": augmented_parent_fraction,
        "allow_unconfirmed_handoff": allow_unconfirmed_handoff,
        "parent_groups_written": len(rows),
        "actions_evaluated": action_counter,
        "children_evaluated": child_counter,
        "beneficial_action_labels": beneficial,
        "kernel_hits": kernel_hits,
        "bounds": {
            "min_length": min_length,
            "max_length": max_length,
            "max_delete": max_delete,
            "max_insert": max_insert,
            "max_net_delta": max_net_delta,
        },
        "label": {
            "definition": "q(parent)-q(child)",
            "q": (
                "calibrated lower-tail objective-projlen percentile conditioned on braid length; "
                "every length uses an independent equal-sized uniform-GNF baseline"
            ),
            "effective_sample_size": baseline.effective_sample_size,
            "target": f"mean of best {target_top_k} sampled legal replacements",
        },
        "action_sampling": {
            "strategy": "balanced direct sampling",
            "scale_bands": [[1, 8], [9, 24], [25, max_delete]],
            "location_bands": ["prefix", "interior", "suffix"],
        },
        "baseline_counts": {
            str(length): len(values) for length, values in sorted(baseline.values.items())
        },
        "backend": backend,
        "device": device,
        "seed": seed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output / "dataset_summary.json", summary)
    write_json(output / "kernel_hits.json", kernel_hits)
    return summary
