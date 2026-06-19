from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .braid_data import simple_factor_id_maps
from .io_utils import read_json


@dataclass(frozen=True)
class Candidate:
    factor_ids: tuple[int, ...]
    author_projlen: int
    matrix_fingerprint: str
    source_index: int
    checkpoint: str

    @property
    def length(self) -> int:
        return len(self.factor_ids)


def load_checkpoint(path: str | Path) -> tuple[dict, list[Candidate]]:
    source = Path(path).resolve()
    payload = read_json(source)
    if payload.get("format") != "paper-tracker-reservoir-run-v1":
        raise ValueError(f"{source} is not a paper Tracker reservoir checkpoint")
    metadata = payload["metadata"]
    perm_to_id, _ = simple_factor_id_maps(int(metadata["n"]))
    candidates = []
    for index, item in enumerate(payload["candidates"]):
        if int(item["power"]) != 0:
            raise ValueError("only Delta-power-zero reservoir candidates are supported")
        factor_ids = tuple(
            perm_to_id[tuple(permutation)]
            for permutation in item["factor_permutations"]
        )
        if len(factor_ids) != int(item.get("depth", metadata["actual_depth"])):
            raise ValueError("checkpoint candidate has the wrong Garside length")
        candidates.append(
            Candidate(
                factor_ids=factor_ids,
                author_projlen=int(item["author_projlen"]),
                matrix_fingerprint=item["matrix_fingerprint"],
                source_index=index,
                checkpoint=str(source),
            )
        )
    return metadata, candidates


def load_checkpoints(paths: list[str | Path]) -> tuple[dict, list[Candidate]]:
    if not paths:
        raise ValueError("at least one reservoir checkpoint is required")
    common = None
    candidates = []
    seen = set()
    for path in paths:
        metadata, items = load_checkpoint(path)
        identity = (int(metadata["p"]), int(metadata["n"]), int(metadata.get("r", 1)))
        if common is None:
            common = identity
        elif identity != common:
            raise ValueError("all checkpoints must have the same p, n, and r")
        for candidate in items:
            if candidate.factor_ids not in seen:
                seen.add(candidate.factor_ids)
                candidates.append(candidate)
    return {"p": common[0], "n": common[1], "r": common[2]}, candidates
