from __future__ import annotations

import random
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from peyl.braid_data import simple_factor_id_maps

from .exact import ExactEngine
from .io_utils import read_json


@dataclass(frozen=True)
class Candidate:
    factor_ids: tuple[int, ...]
    author_projlen: int
    matrix_fingerprint: str
    source_index: int
    source: str = "frontier"

    @property
    def depth(self) -> int:
        return len(self.factor_ids)

    @property
    def last_factor_id(self) -> int:
        return self.factor_ids[-1]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["factor_ids"] = list(self.factor_ids)
        return payload


def _map_records(records: list[dict], metadata: dict, source: str) -> list[Candidate]:
    perm_to_id, _ = simple_factor_id_maps(metadata["n"])
    candidates = []
    for index, item in enumerate(records):
        if item["power"] != 0:
            raise ValueError("the hybrid currently requires Delta power zero")
        try:
            factor_ids = tuple(
                perm_to_id[tuple(permutation)]
                for permutation in item["factor_permutations"]
            )
        except KeyError as exc:
            raise ValueError(f"unknown simple-factor permutation: {exc}") from exc
        expected_depth = int(item.get("depth", metadata["actual_depth"]))
        if len(factor_ids) != expected_depth:
            raise ValueError("checkpoint candidate has the wrong Garside depth")
        candidates.append(
            Candidate(
                factor_ids=factor_ids,
                author_projlen=int(item["author_projlen"]),
                matrix_fingerprint=item["matrix_fingerprint"],
                source_index=index,
                source=source,
            )
        )
    return candidates


def load_checkpoint(path: str | Path) -> tuple[dict, list[Candidate], list[Candidate]]:
    payload = read_json(path)
    if payload.get("format") != "paper-tracker-reservoir-run-v1":
        raise ValueError("not a reservoir-first paper Tracker checkpoint")
    metadata = payload["metadata"]
    frontier = _map_records(payload["candidates"], metadata, "frontier")
    suspected = _map_records(
        payload.get("kernel_candidates", []),
        metadata,
        "author_projlen_one",
    )
    return metadata, frontier, suspected


def verify_author_kernel_candidates(
    metadata: dict,
    frontier: Iterable[Candidate],
    suspected: Iterable[Candidate],
) -> dict:
    possible = list(suspected)
    possible.extend(item for item in frontier if item.author_projlen == 1)
    unique = {}
    for candidate in possible:
        unique.setdefault(candidate.factor_ids, candidate)

    engine = ExactEngine(p=int(metadata["p"]), n=int(metadata["n"]))
    evaluated = [engine.evaluate(word) for word in unique]
    hits = [state.summary() for state in evaluated if state.kernel_matches]
    false_positives = [
        state.summary() for state in evaluated if not state.kernel_matches
    ]
    return {
        "author_projlen_one_candidates": len(possible),
        "unique_candidates_verified": len(evaluated),
        "kernel_hits": hits,
        "false_positives": false_positives,
    }


def _round_robin(groups: dict[tuple, deque[Candidate]]) -> Iterable[Candidate]:
    keys = sorted(groups)
    while keys:
        next_keys = []
        for key in keys:
            group = groups[key]
            if group:
                yield group.popleft()
            if group:
                next_keys.append(key)
        keys = next_keys


def select_diverse_candidates(
    candidates: Iterable[Candidate],
    limit: int,
    seed: int,
) -> list[Candidate]:
    candidate_list = list(candidates)
    if limit <= 0 or limit >= len(candidate_list):
        limit = len(candidate_list)
    rng = random.Random(seed)
    rng.shuffle(candidate_list)

    unique_groups: dict[tuple, deque[Candidate]] = defaultdict(deque)
    duplicate_groups: dict[tuple, deque[Candidate]] = defaultdict(deque)
    seen_states = set()
    for candidate in sorted(candidate_list, key=lambda item: item.author_projlen):
        key = (candidate.author_projlen, candidate.last_factor_id)
        if candidate.matrix_fingerprint in seen_states:
            duplicate_groups[key].append(candidate)
        else:
            unique_groups[key].append(candidate)
            seen_states.add(candidate.matrix_fingerprint)

    selected = []
    seen_words = set()
    projlens = sorted({item.author_projlen for item in candidate_list})
    for projlen in projlens:
        for all_groups in (unique_groups, duplicate_groups):
            groups = {
                key: group for key, group in all_groups.items() if key[0] == projlen
            }
            for candidate in _round_robin(groups):
                if candidate.factor_ids in seen_words:
                    continue
                selected.append(candidate)
                seen_words.add(candidate.factor_ids)
                if len(selected) >= limit:
                    return selected
    return selected


def candidate_summary(candidates: Iterable[Candidate]) -> dict:
    items = list(candidates)
    return {
        "count": len(items),
        "depths": dict(sorted(Counter(item.depth for item in items).items())),
        "author_projlen": dict(
            sorted(Counter(item.author_projlen for item in items).items())
        ),
        "last_factor_classes": len({item.last_factor_id for item in items}),
        "unique_matrix_states": len({item.matrix_fingerprint for item in items}),
        "unique_words": len({item.factor_ids for item in items}),
    }
