from __future__ import annotations

import random
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from peyl.braid_data import simple_factor_id_maps

from .io_utils import read_json, write_json


@dataclass(frozen=True)
class Candidate:
    factor_ids: tuple[int, ...]
    author_projlen: int
    matrix_fingerprint: str
    source_index: int

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


def load_author_checkpoint(path: str | Path) -> tuple[dict, list[Candidate]]:
    payload = read_json(path)
    if payload.get("format") != "paper-tracker-frontier-v1":
        raise ValueError("not a paper Tracker frontier checkpoint")
    metadata = payload["metadata"]
    perm_to_id, _ = simple_factor_id_maps(metadata["n"])
    candidates = []
    for index, item in enumerate(payload["candidates"]):
        if item["power"] != 0:
            raise ValueError("the hybrid currently requires Delta power zero")
        try:
            factor_ids = tuple(
                perm_to_id[tuple(permutation)]
                for permutation in item["factor_permutations"]
            )
        except KeyError as exc:
            raise ValueError(f"unknown simple-factor permutation: {exc}") from exc
        if len(factor_ids) != metadata["target_depth"]:
            raise ValueError("checkpoint candidate has the wrong Garside depth")
        candidates.append(
            Candidate(
                factor_ids=factor_ids,
                author_projlen=int(item["author_projlen"]),
                matrix_fingerprint=item["matrix_fingerprint"],
                source_index=index,
            )
        )
    return metadata, candidates


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
    """
    Preserve the paper's low-projlen frontier while balancing terminal factors
    and exact projective matrix states. Exact state duplicates are filled last.
    """
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
                key: group
                for key, group in all_groups.items()
                if key[0] == projlen
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


def write_branch_pool(
    output_path: str | Path,
    checkpoint_path: str | Path,
    metadata: dict,
    candidates: Iterable[Candidate],
) -> Path:
    items = list(candidates)
    return write_json(
        output_path,
        {
            "format": "hybrid-branch-pool-v1",
            "source_checkpoint": str(Path(checkpoint_path).resolve()),
            "metadata": metadata,
            "summary": candidate_summary(items),
            "candidates": [item.to_dict() for item in items],
        },
    )
