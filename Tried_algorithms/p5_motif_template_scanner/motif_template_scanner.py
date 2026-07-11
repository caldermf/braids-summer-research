#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
STRUCTURAL_ROOT = REPO_ROOT / "structural-kernel-experiments"
DEFAULT_AUTHOR_REPO = STRUCTURAL_ROOT / "third_party" / "braids_project"
if str(STRUCTURAL_ROOT) not in sys.path:
    sys.path.insert(0, str(STRUCTURAL_ROOT))

from crispr_transformer.gnf import GNFAutomaton  # noqa: E402


@dataclass(frozen=True)
class SeedWord:
    source: str
    power: int
    factor_ids: tuple[int, ...]
    source_record_index: int | None = None
    scalar: int | None = None


@dataclass(frozen=True)
class MotifTemplate:
    template_id: int
    seed_index: int
    power: int
    start: int
    block: tuple[int, ...]
    repeats: int
    prefix: tuple[int, ...]
    suffix: tuple[int, ...]

    @property
    def end(self) -> int:
        return self.start + len(self.block) * self.repeats

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "seed_index": self.seed_index,
            "power": self.power,
            "start": self.start,
            "end": self.end,
            "block": list(self.block),
            "block_length": len(self.block),
            "repeats": self.repeats,
            "prefix_length": len(self.prefix),
            "suffix_length": len(self.suffix),
        }


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    power: int
    factor_ids: tuple[int, ...]
    source: str
    template_id: int | None = None
    metadata: dict | None = None

    def key(self) -> tuple[int, tuple[int, ...]]:
        return self.power, self.factor_ids

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "power": self.power,
            "factor_ids": list(self.factor_ids),
            "length": len(self.factor_ids),
            "source": self.source,
            "template_id": self.template_id,
            "metadata": self.metadata or {},
        }


def _read_json(path: str | Path) -> dict:
    input_path = Path(path)
    if input_path.suffix == ".gz":
        with gzip.open(input_path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(input_path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _parse_int_list(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_seed_word(value: str) -> SeedWord:
    if ":" not in value:
        raise ValueError("--seed-word must have form POWER:f1,f2,...")
    power_text, factors_text = value.split(":", 1)
    return SeedWord(
        source="cli_seed_word",
        power=int(power_text),
        factor_ids=_parse_int_list(factors_text),
    )


def load_seed_words(checkpoint: Path | None, cli_seed_words: Sequence[str]) -> list[SeedWord]:
    seeds: list[SeedWord] = []
    if checkpoint is not None:
        payload = _read_json(checkpoint)
        for index, record in enumerate(payload.get("collision_records", [])):
            quotient = record.get("quotient", {})
            factor_ids = tuple(int(value) for value in quotient.get("factor_ids", []))
            if not factor_ids:
                continue
            seeds.append(
                SeedWord(
                    source="collision_quotient",
                    power=int(quotient.get("power", 0)),
                    factor_ids=factor_ids,
                    source_record_index=index,
                    scalar=record.get("match", {}).get("scalar"),
                )
            )
        for index, record in enumerate(payload.get("kernel_candidates", [])):
            factor_ids = tuple(int(value) for value in record.get("factor_ids", []))
            if not factor_ids:
                continue
            seeds.append(
                SeedWord(
                    source="kernel_candidate",
                    power=int(record.get("power", 0)),
                    factor_ids=factor_ids,
                    source_record_index=index,
                )
            )
    seeds.extend(_parse_seed_word(value) for value in cli_seed_words)

    unique: dict[tuple[int, tuple[int, ...]], SeedWord] = {}
    for seed in seeds:
        unique.setdefault((seed.power, seed.factor_ids), seed)
    return list(unique.values())


def discover_templates(
    seeds: Sequence[SeedWord],
    *,
    min_block_len: int,
    max_block_len: int,
    min_repeats: int,
    top_templates: int,
) -> list[MotifTemplate]:
    templates: list[MotifTemplate] = []
    seen = set()
    for seed_index, seed in enumerate(seeds):
        word = seed.factor_ids
        for block_len in range(min_block_len, max_block_len + 1):
            if block_len * min_repeats > len(word):
                continue
            cursor = 0
            while cursor + block_len * min_repeats <= len(word):
                block = word[cursor : cursor + block_len]
                repeats = 1
                probe = cursor + block_len
                while probe + block_len <= len(word) and word[probe : probe + block_len] == block:
                    repeats += 1
                    probe += block_len
                if repeats >= min_repeats:
                    key = (seed_index, cursor, block, repeats)
                    if key not in seen:
                        seen.add(key)
                        templates.append(
                            MotifTemplate(
                                template_id=-1,
                                seed_index=seed_index,
                                power=seed.power,
                                start=cursor,
                                block=block,
                                repeats=repeats,
                                prefix=word[:cursor],
                                suffix=word[probe:],
                            )
                        )
                    cursor = max(cursor + 1, probe - block_len + 1)
                else:
                    cursor += 1

    templates.sort(
        key=lambda item: (
            -(len(item.block) * item.repeats),
            -item.repeats,
            item.seed_index,
            item.start,
            len(item.block),
        )
    )
    output = []
    for template_id, template in enumerate(templates[:top_templates]):
        output.append(
            MotifTemplate(
                template_id=template_id,
                seed_index=template.seed_index,
                power=template.power,
                start=template.start,
                block=template.block,
                repeats=template.repeats,
                prefix=template.prefix,
                suffix=template.suffix,
            )
        )
    return output


def balanced_power(factor_count: int) -> int:
    return -(factor_count // 2)


def candidate_powers(
    *,
    observed_power: int,
    factor_count: int,
    power_offsets: Sequence[int],
    include_observed_power: bool,
    include_balanced_power: bool,
) -> tuple[int, ...]:
    bases = []
    if include_observed_power:
        bases.append(observed_power)
    if include_balanced_power:
        bases.append(balanced_power(factor_count))
    if not bases:
        bases.append(observed_power)
    powers = sorted({base + offset for base in bases for offset in power_offsets})
    return tuple(powers)


def is_legal_factors(automaton: GNFAutomaton, factors: Sequence[int]) -> bool:
    return bool(factors) and automaton.is_legal(tuple(factors))


def legal_single_replacements(
    automaton: GNFAutomaton,
    word: tuple[int, ...],
    position: int,
) -> tuple[int, ...]:
    left = word[position - 1] if position > 0 else None
    right = word[position + 1] if position + 1 < len(word) else None
    choices = automaton.first_ids if left is None else automaton.successors[left]
    output = []
    for choice in choices:
        if choice == word[position]:
            continue
        if right is None or right in automaton.successors[choice]:
            output.append(choice)
    return tuple(output)


def add_candidate(
    candidates: dict[tuple[int, tuple[int, ...]], Candidate],
    *,
    power: int,
    factors: tuple[int, ...],
    source: str,
    template_id: int | None,
    metadata: dict,
    automaton: GNFAutomaton,
    max_candidates: int,
) -> None:
    if len(candidates) >= max_candidates:
        return
    if not is_legal_factors(automaton, factors):
        return
    key = (power, factors)
    if key in candidates:
        return
    candidates[key] = Candidate(
        candidate_id=len(candidates),
        power=power,
        factor_ids=factors,
        source=source,
        template_id=template_id,
        metadata=metadata,
    )


def boundary_positions(length: int, boundaries: Sequence[int], radius: int) -> tuple[int, ...]:
    positions = set()
    for boundary in boundaries:
        for position in range(boundary - radius, boundary + radius + 1):
            if 0 <= position < length:
                positions.add(position)
    return tuple(sorted(positions))


def bridge_replace(
    automaton: GNFAutomaton,
    word: tuple[int, ...],
    start: int,
    size: int,
    rng: random.Random,
) -> tuple[int, ...] | None:
    end = start + size
    if start < 0 or end > len(word) or size <= 0:
        return None
    left = word[start - 1] if start > 0 else None
    right = word[end] if end < len(word) else None
    try:
        replacement = automaton.sample_bridge(left, right, size, rng)
    except ValueError:
        return None
    if replacement == word[start:end]:
        return None
    return word[:start] + replacement + word[end:]


def generate_candidates(
    seeds: Sequence[SeedWord],
    templates: Sequence[MotifTemplate],
    *,
    automaton: GNFAutomaton,
    repeat_counts: Sequence[int],
    power_offsets: Sequence[int],
    include_observed_power: bool,
    include_balanced_power: bool,
    single_mutation_radius: int,
    max_single_mutations_per_template: int,
    bridge_sizes: Sequence[int],
    bridge_samples_per_template: int,
    max_candidates: int,
    rng: random.Random,
) -> list[Candidate]:
    candidates: dict[tuple[int, tuple[int, ...]], Candidate] = {}

    for seed_index, seed in enumerate(seeds):
        for power in candidate_powers(
            observed_power=seed.power,
            factor_count=len(seed.factor_ids),
            power_offsets=power_offsets,
            include_observed_power=include_observed_power,
            include_balanced_power=include_balanced_power,
        ):
            add_candidate(
                candidates,
                power=power,
                factors=seed.factor_ids,
                source="seed",
                template_id=None,
                metadata={"seed_index": seed_index, "seed_source": seed.source},
                automaton=automaton,
                max_candidates=max_candidates,
            )

    block_library = sorted({template.block for template in templates}, key=lambda b: (len(b), b))

    for template in templates:
        if len(candidates) >= max_candidates:
            break
        base_repeat_counts = sorted(set(repeat_counts) | {template.repeats})
        repeat_variants: list[tuple[int, tuple[int, ...]]] = []
        for repeat_count in base_repeat_counts:
            if repeat_count < 0:
                continue
            factors = template.prefix + template.block * repeat_count + template.suffix
            repeat_variants.append((repeat_count, factors))
            for power in candidate_powers(
                observed_power=template.power,
                factor_count=len(factors),
                power_offsets=power_offsets,
                include_observed_power=include_observed_power,
                include_balanced_power=include_balanced_power,
            ):
                add_candidate(
                    candidates,
                    power=power,
                    factors=factors,
                    source="repeat_count",
                    template_id=template.template_id,
                    metadata={
                        "repeat_count": repeat_count,
                        "block_length": len(template.block),
                    },
                    automaton=automaton,
                    max_candidates=max_candidates,
                )

        for repeat_count, factors in repeat_variants:
            if len(candidates) >= max_candidates:
                break
            boundaries = (
                len(template.prefix),
                len(template.prefix) + len(template.block) * repeat_count,
            )
            positions = boundary_positions(len(factors), boundaries, single_mutation_radius)
            mutation_rows = []
            for position in positions:
                for replacement in legal_single_replacements(automaton, factors, position):
                    mutation_rows.append((position, replacement))
            rng.shuffle(mutation_rows)
            for position, replacement in mutation_rows[:max_single_mutations_per_template]:
                mutated = factors[:position] + (replacement,) + factors[position + 1 :]
                for power in candidate_powers(
                    observed_power=template.power,
                    factor_count=len(mutated),
                    power_offsets=power_offsets,
                    include_observed_power=include_observed_power,
                    include_balanced_power=include_balanced_power,
                ):
                    add_candidate(
                        candidates,
                        power=power,
                        factors=mutated,
                        source="boundary_single_mutation",
                        template_id=template.template_id,
                        metadata={
                            "repeat_count": repeat_count,
                            "position": position,
                            "old": factors[position],
                            "new": replacement,
                        },
                        automaton=automaton,
                        max_candidates=max_candidates,
                    )
                if len(candidates) >= max_candidates:
                    break

            bridge_anchors = set()
            for boundary in boundaries:
                for size in bridge_sizes:
                    bridge_anchors.add((boundary - size, size))
                    bridge_anchors.add((boundary, size))
            bridge_anchors = list(bridge_anchors)
            rng.shuffle(bridge_anchors)
            samples_used = 0
            for start, size in bridge_anchors:
                if samples_used >= bridge_samples_per_template:
                    break
                mutated = bridge_replace(automaton, factors, start, size, rng)
                if mutated is None:
                    continue
                samples_used += 1
                for power in candidate_powers(
                    observed_power=template.power,
                    factor_count=len(mutated),
                    power_offsets=power_offsets,
                    include_observed_power=include_observed_power,
                    include_balanced_power=include_balanced_power,
                ):
                    add_candidate(
                        candidates,
                        power=power,
                        factors=mutated,
                        source="boundary_bridge_mutation",
                        template_id=template.template_id,
                        metadata={
                            "repeat_count": repeat_count,
                            "start": start,
                            "size": size,
                        },
                        automaton=automaton,
                        max_candidates=max_candidates,
                    )
                if len(candidates) >= max_candidates:
                    break

        for donor_block in block_library:
            if len(donor_block) != len(template.block) or donor_block == template.block:
                continue
            for repeat_count in base_repeat_counts:
                factors = template.prefix + donor_block * repeat_count + template.suffix
                for power in candidate_powers(
                    observed_power=template.power,
                    factor_count=len(factors),
                    power_offsets=power_offsets,
                    include_observed_power=include_observed_power,
                    include_balanced_power=include_balanced_power,
                ):
                    add_candidate(
                        candidates,
                        power=power,
                        factors=factors,
                        source="block_swap",
                        template_id=template.template_id,
                        metadata={
                            "repeat_count": repeat_count,
                            "old_block": list(template.block),
                            "new_block": list(donor_block),
                        },
                        automaton=automaton,
                        max_candidates=max_candidates,
                    )

    return list(candidates.values())


def setup_author_imports(author_repo: Path):
    if not (author_repo / "peyl" / "braid.py").exists():
        raise FileNotFoundError(f"vendored peyl package is missing at {author_repo}")
    if str(author_repo) not in sys.path:
        sys.path.insert(0, str(author_repo))
    import peyl  # type: ignore
    from peyl import polymat  # type: ignore
    from peyl.braidsearch import evaluate_braids  # type: ignore

    return peyl, polymat, evaluate_braids


def scalar_identity_metrics(polymat_module, image: np.ndarray) -> dict:
    projected = polymat_module.projectivise(image)
    width = int(projected.shape[-1])
    matrix_count = int(np.count_nonzero(projected))

    if projected.shape[0] != projected.shape[1]:
        return {
            "projective_width": width,
            "scalar_identity": False,
            "identity_defect": matrix_count,
            "nonzero_terms": matrix_count,
            "reason": "non_square_matrix",
        }

    dim = projected.shape[0]
    diagonal = np.stack([projected[i, i, :] for i in range(dim)])
    scalar_poly = diagonal[0]
    diagonal_mismatch_terms = int(np.count_nonzero(diagonal - scalar_poly[None, :]))

    off_diagonal_terms = 0
    for row in range(dim):
        for column in range(dim):
            if row != column:
                off_diagonal_terms += int(np.count_nonzero(projected[row, column, :]))

    scalar_nonzero_degrees = int(np.count_nonzero(scalar_poly))
    scalar_extra_degrees = max(0, scalar_nonzero_degrees - 1)
    scalar_zero_penalty = 1 if scalar_nonzero_degrees == 0 else 0
    identity_defect = (
        off_diagonal_terms
        + diagonal_mismatch_terms
        + scalar_extra_degrees
        + scalar_zero_penalty
    )
    scalar_identity = identity_defect == 0
    scalar = None
    degree = None
    if scalar_identity:
        nonzero = np.flatnonzero(scalar_poly)
        degree = int(nonzero[0])
        scalar = int(scalar_poly[degree])
    return {
        "projective_width": width,
        "scalar_identity": bool(scalar_identity),
        "identity_defect": int(identity_defect),
        "off_diagonal_terms": off_diagonal_terms,
        "diagonal_mismatch_terms": diagonal_mismatch_terms,
        "scalar_nonzero_degrees": scalar_nonzero_degrees,
        "scalar_extra_degrees": scalar_extra_degrees,
        "nonzero_terms": matrix_count,
        "scalar": scalar,
        "scalar_degree": degree,
    }


def evaluate_candidates(
    candidates: Sequence[Candidate],
    *,
    author_repo: Path,
    p: int,
    n: int,
    r: int,
    batch_size: int,
    output_path: Path,
) -> list[dict]:
    peyl, polymat_module, evaluate_braids = setup_author_imports(author_repo)
    rep = peyl.JonesSummand(n=n, r=r, p=p)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    rows: list[dict] = []
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start : start + batch_size]
        braids = [
            peyl.GNF(n=n, power=candidate.power, factors=candidate.factor_ids)
            for candidate in chunk
        ]
        images = evaluate_braids(rep, braids)
        chunk_rows = []
        for candidate, image in zip(chunk, images):
            metrics = scalar_identity_metrics(polymat_module, image)
            row = {
                **candidate.to_dict(),
                "target_p": p,
                "metrics": metrics,
            }
            rows.append(row)
            chunk_rows.append(row)
        _append_jsonl(output_path, chunk_rows)
        print(
            json.dumps(
                {
                    "evaluated": min(start + len(chunk), len(candidates)),
                    "total": len(candidates),
                    "best_width_so_far": min(
                        row["metrics"]["projective_width"] for row in rows
                    ),
                    "best_defect_so_far": min(
                        row["metrics"]["identity_defect"] for row in rows
                    ),
                    "kernel_hits_so_far": sum(
                        1 for row in rows if row["metrics"]["scalar_identity"]
                    ),
                }
            ),
            flush=True,
        )
    return rows


def summarise_evaluations(rows: Sequence[dict], top_n: int) -> dict:
    by_width = Counter(row["metrics"]["projective_width"] for row in rows)
    by_defect = Counter(row["metrics"]["identity_defect"] for row in rows)
    ranked = sorted(
        rows,
        key=lambda row: (
            row["metrics"]["identity_defect"],
            row["metrics"]["projective_width"],
            row["length"],
            row["candidate_id"],
        ),
    )
    width_ranked = sorted(
        rows,
        key=lambda row: (
            row["metrics"]["projective_width"],
            row["metrics"]["identity_defect"],
            row["length"],
            row["candidate_id"],
        ),
    )
    kernels = [row for row in rows if row["metrics"]["scalar_identity"]]
    return {
        "evaluated_candidates": len(rows),
        "kernel_hits": len(kernels),
        "projective_width_histogram": dict(sorted(by_width.items())),
        "identity_defect_histogram": dict(sorted(by_defect.items())[:50]),
        "best_by_identity_defect": ranked[:top_n],
        "best_by_projective_width": width_ranked[:top_n],
        "kernel_candidates": kernels[:top_n],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate p=7 motif/template variants from p=5 collision quotient "
            "kernels and score them exactly."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed-word", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path, default=DEFAULT_AUTHOR_REPO)
    parser.add_argument("--target-p", type=int, default=7)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-block-len", type=int, default=6)
    parser.add_argument("--max-block-len", type=int, default=18)
    parser.add_argument("--min-repeats", type=int, default=2)
    parser.add_argument("--top-templates", type=int, default=24)
    parser.add_argument(
        "--repeat-counts",
        default="0,1,2,3,4,5,6,7,8",
        help="Comma-separated repeat counts to try for every discovered block.",
    )
    parser.add_argument(
        "--power-offsets",
        default="0",
        help="Comma-separated offsets from observed and/or balanced Delta power.",
    )
    parser.add_argument("--no-observed-power", action="store_true")
    parser.add_argument("--no-balanced-power", action="store_true")
    parser.add_argument("--single-mutation-radius", type=int, default=3)
    parser.add_argument("--max-single-mutations-per-template", type=int, default=200)
    parser.add_argument("--bridge-sizes", default="1,2,3")
    parser.add_argument("--bridge-samples-per-template", type=int, default=80)
    parser.add_argument("--max-candidates", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--top-output", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    rng = random.Random(args.seed)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    seeds = load_seed_words(args.checkpoint, args.seed_word)
    if not seeds:
        raise ValueError("no seed words found; pass --checkpoint or --seed-word")

    automaton = GNFAutomaton(args.n)
    legal_seeds = []
    skipped_seeds = []
    for index, seed in enumerate(seeds):
        if is_legal_factors(automaton, seed.factor_ids):
            legal_seeds.append(seed)
        else:
            skipped_seeds.append(index)
    if not legal_seeds:
        raise ValueError("none of the seed words are legal positive GNF factor sequences")

    templates = discover_templates(
        legal_seeds,
        min_block_len=args.min_block_len,
        max_block_len=args.max_block_len,
        min_repeats=args.min_repeats,
        top_templates=args.top_templates,
    )
    if not templates:
        raise ValueError("no repeated motifs found; lower --min-repeats or block lengths")

    repeat_counts = _parse_int_list(args.repeat_counts)
    power_offsets = _parse_int_list(args.power_offsets)
    bridge_sizes = _parse_int_list(args.bridge_sizes)

    candidates = generate_candidates(
        legal_seeds,
        templates,
        automaton=automaton,
        repeat_counts=repeat_counts,
        power_offsets=power_offsets,
        include_observed_power=not args.no_observed_power,
        include_balanced_power=not args.no_balanced_power,
        single_mutation_radius=args.single_mutation_radius,
        max_single_mutations_per_template=args.max_single_mutations_per_template,
        bridge_sizes=bridge_sizes,
        bridge_samples_per_template=args.bridge_samples_per_template,
        max_candidates=args.max_candidates,
        rng=rng,
    )

    generation_summary = {
        "seed_count": len(seeds),
        "legal_seed_count": len(legal_seeds),
        "skipped_seed_indices": skipped_seeds,
        "template_count": len(templates),
        "candidate_count": len(candidates),
        "candidate_sources": dict(Counter(candidate.source for candidate in candidates)),
        "candidate_lengths": dict(
            sorted(Counter(len(candidate.factor_ids) for candidate in candidates).items())
        ),
    }
    _write_json(
        output / "motifs.json",
        {
            "format": "p5-motif-template-scanner-motifs-v1",
            "source_checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "seeds": [
                {
                    "source": seed.source,
                    "power": seed.power,
                    "length": len(seed.factor_ids),
                    "factor_ids": list(seed.factor_ids),
                    "source_record_index": seed.source_record_index,
                    "scalar": seed.scalar,
                }
                for seed in legal_seeds
            ],
            "templates": [template.to_dict() for template in templates],
            "generation_summary": generation_summary,
        },
    )

    rows = evaluate_candidates(
        candidates,
        author_repo=args.author_repo,
        p=args.target_p,
        n=args.n,
        r=args.r,
        batch_size=args.batch_size,
        output_path=output / "evaluations.jsonl",
    )

    evaluation_summary = summarise_evaluations(rows, top_n=args.top_output)
    summary = {
        "format": "p5-motif-template-scanner-summary-v1",
        "metadata": {
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "author_repo": str(args.author_repo),
            "target_p": args.target_p,
            "n": args.n,
            "r": args.r,
            "seed": args.seed,
            "min_block_len": args.min_block_len,
            "max_block_len": args.max_block_len,
            "min_repeats": args.min_repeats,
            "top_templates": args.top_templates,
            "repeat_counts": list(repeat_counts),
            "power_offsets": list(power_offsets),
            "include_observed_power": not args.no_observed_power,
            "include_balanced_power": not args.no_balanced_power,
            "single_mutation_radius": args.single_mutation_radius,
            "max_single_mutations_per_template": args.max_single_mutations_per_template,
            "bridge_sizes": list(bridge_sizes),
            "bridge_samples_per_template": args.bridge_samples_per_template,
            "max_candidates": args.max_candidates,
            "batch_size": args.batch_size,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        "generation_summary": generation_summary,
        "evaluation_summary": evaluation_summary,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps({"summary": str(output / "summary.json"), **generation_summary}), flush=True)


if __name__ == "__main__":
    main()
