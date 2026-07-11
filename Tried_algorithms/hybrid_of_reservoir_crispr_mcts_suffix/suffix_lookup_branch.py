from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

try:
    from crispr_algorithms.bidirectional_matrix_search_v5.config import SearchConfig
    from crispr_algorithms.bidirectional_matrix_search_v5.exact_evaluator import (
        make_exact_evaluator,
    )
    from crispr_algorithms.bidirectional_matrix_search_v5.field_sketch import (
        ExtensionFieldSketch,
    )
    from crispr_algorithms.bidirectional_matrix_search_v5.gnf import GNFAutomaton
    from crispr_algorithms.bidirectional_matrix_search_v5.models import (
        JoinCandidate,
        Segment,
    )
    from crispr_algorithms.bidirectional_matrix_search_v5.suffix_index import (
        SuffixLSHIndex,
    )
except ModuleNotFoundError:
    from bidirectional_matrix_search_v5.config import SearchConfig
    from bidirectional_matrix_search_v5.exact_evaluator import make_exact_evaluator
    from bidirectional_matrix_search_v5.field_sketch import ExtensionFieldSketch
    from bidirectional_matrix_search_v5.gnf import GNFAutomaton
    from bidirectional_matrix_search_v5.models import JoinCandidate, Segment
    from bidirectional_matrix_search_v5.suffix_index import SuffixLSHIndex

from .candidates import Candidate, select_diverse_candidates
from .config import SuffixLookupConfig
from .io_utils import append_jsonl, write_json


def _generate_suffixes(
    automaton: GNFAutomaton,
    length: int,
    count: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    starts = list(automaton.suffix_start_ids)
    rng.shuffle(starts)
    output = []

    def visit(prefix: tuple[int, ...]) -> bool:
        if len(prefix) == length:
            output.append(prefix)
            return len(output) >= count
        successors = list(automaton.successors[prefix[-1]])
        rng.shuffle(successors)
        for factor_id in successors:
            if visit(prefix + (factor_id,)):
                return True
        return False

    for factor_id in starts:
        if visit((factor_id,)):
            break
    return output


def _search_config(
    branch: SuffixLookupConfig,
    p: int,
    n: int,
    base_depth: int,
    suffix_length: int,
) -> SearchConfig:
    return SearchConfig(
        p=p,
        n=n,
        prefix_count=branch.prefix_pool_size,
        suffix_count=branch.suffixes_per_length,
        generations=1,
        prefix_length_min=base_depth,
        prefix_length_max=base_depth,
        suffix_length_min=suffix_length,
        suffix_length_max=suffix_length,
        field_points=branch.field_points,
        lsh_tables=branch.lsh_tables,
        lsh_key_components=branch.lsh_key_components,
        max_lsh_candidates=branch.max_lsh_candidates,
        join_candidates_per_prefix=branch.joins_per_prefix,
        elite_pairs=max(1, branch.exact_candidates_per_depth),
        signature_batch_size=20_000,
        exact_batch_size=10_000,
        backend=branch.backend,
        device=branch.device,
        seed=branch.seed + suffix_length,
        stop_at_kernel=False,
        resume_latest=False,
    )


def run_suffix_lookup_branch(
    candidates: list[Candidate],
    branch: SuffixLookupConfig,
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
    events_path = output / "depths.jsonl"
    if events_path.exists():
        events_path.unlink()

    rng = random.Random(branch.seed)
    automaton = GNFAutomaton(n)
    selected = select_diverse_candidates(
        candidates,
        branch.prefix_pool_size,
        branch.seed,
    )
    prefixes = [
        Segment(
            factor_ids=candidate.factor_ids,
            role="prefix",
            segment_id=f"reservoir-prefix-{index}",
            origin="paper_reservoir_depth35",
        )
        for index, candidate in enumerate(selected)
    ]
    kernel_hits = {}
    best_results = []

    for suffix_length in range(1, max_depth - base_depth + 1):
        config = _search_config(branch, p, n, base_depth, suffix_length)
        config.validate()
        suffix_words = _generate_suffixes(
            automaton,
            suffix_length,
            branch.suffixes_per_length,
            rng,
        )
        suffixes = [
            Segment(
                factor_ids=word,
                role="suffix",
                segment_id=f"suffix-L{suffix_length}-{index}",
                origin="fixed_length_library",
            )
            for index, word in enumerate(suffix_words)
        ]

        sketch = ExtensionFieldSketch(config)
        suffix_signatures = sketch.suffix_signatures(suffixes)
        index = SuffixLSHIndex(config, suffixes, suffix_signatures, rng)
        targets = sketch.prefix_target_signatures(prefixes)

        proposals: dict[tuple[int, int], JoinCandidate] = {}
        for prefix_index, prefix in enumerate(prefixes):
            allowed = automaton.successors[prefix.factor_ids[-1]]
            for target_type, signatures in targets.items():
                for suffix_index, distance in index.query(
                    signatures[prefix_index],
                    allowed,
                    branch.joins_per_prefix,
                ):
                    key = (prefix_index, suffix_index)
                    proposal = JoinCandidate(
                        prefix_index=prefix_index,
                        suffix_index=suffix_index,
                        target_type=target_type,
                        sketch_distance=distance,
                    )
                    current = proposals.get(key)
                    if current is None or distance < current.sketch_distance:
                        proposals[key] = proposal

        ordered = sorted(
            proposals.values(),
            key=lambda item: (item.sketch_distance, item.prefix_index, item.suffix_index),
        )[: branch.exact_candidates_per_depth]
        words = [
            prefixes[item.prefix_index].factor_ids
            + suffixes[item.suffix_index].factor_ids
            for item in ordered
        ]
        evaluator = make_exact_evaluator(config)
        evaluations = evaluator.evaluate(words)
        rows = []
        for proposal, evaluation in zip(ordered, evaluations):
            row = {
                "prefix_id": prefixes[proposal.prefix_index].segment_id,
                "suffix_id": suffixes[proposal.suffix_index].segment_id,
                "target_type": proposal.target_type,
                "sketch_distance": proposal.sketch_distance,
                "factor_ids": list(evaluation.factor_ids),
                "depth": len(evaluation.factor_ids),
                "projlen_history": list(evaluation.projlen_history),
                "final_projlen": evaluation.final_projlen,
                "min_projlen": evaluation.min_projlen,
                "peak_projlen": evaluation.peak_projlen,
                "largest_drop": evaluation.largest_drop,
                "kernel_matches": list(evaluation.kernel_matches),
            }
            rows.append(row)
            if evaluation.has_kernel:
                kernel_hits.setdefault(evaluation.factor_ids, row)
        rows.sort(
            key=lambda row: (
                0 if row["kernel_matches"] else 1,
                row["sketch_distance"],
                row["final_projlen"],
                -row["largest_drop"],
            )
        )
        best_results.extend(rows[:100])
        event = {
            "depth": base_depth + suffix_length,
            "suffix_length": suffix_length,
            "suffix_library_size": len(suffixes),
            "proposals": len(proposals),
            "exact_evaluations": len(evaluations),
            "best_sketch_distance": (
                min(row["sketch_distance"] for row in rows) if rows else None
            ),
            "best_final_projlen": (
                min(row["final_projlen"] for row in rows) if rows else None
            ),
            "kernel_hits": len(kernel_hits),
            "index": index.stats(),
        }
        append_jsonl(events_path, event)
        write_json(output / f"depth_{base_depth + suffix_length:03d}.json", rows[:1000])
        print(
            f"[suffix-lookup] depth={event['depth']} suffixes={len(suffixes)} "
            f"proposals={len(proposals)} best={event['best_final_projlen']} "
            f"hits={len(kernel_hits)}",
            flush=True,
        )
        if kernel_hits and stop_at_kernel:
            break

    best_results.sort(
        key=lambda row: (
            0 if row["kernel_matches"] else 1,
            row["sketch_distance"],
            row["final_projlen"],
            -row["largest_drop"],
        )
    )
    result = {
        "branch": "suffix_lookup",
        "config": asdict(branch),
        "reservoir_prefixes": len(prefixes),
        "kernel_hits": list(kernel_hits.values()),
        "best": best_results[:200],
    }
    write_json(output / "result.json", result)
    return result
