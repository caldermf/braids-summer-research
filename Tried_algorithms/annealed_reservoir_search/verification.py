from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from peyl.braid_data import (
    append_factor_to_burau_matrix,
    identity_burau_matrix,
    polynomial_matrix_projlen,
    projective_kernel_match,
    simple_factor_burau_table,
    simple_factor_id_maps,
)


def _fingerprint(matrix: Any) -> str:
    array = np.asarray(matrix)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _factor_ids(record: dict, n: int) -> tuple[int, ...]:
    permutation_to_id, _ = simple_factor_id_maps(n)
    if int(record.get("power", 0)) != 0:
        raise ValueError("exact verification currently requires Delta power zero")
    return tuple(
        permutation_to_id[tuple(permutation)]
        for permutation in record["factor_permutations"]
    )


def _evaluate_word(factor_ids: tuple[int, ...], p: int, n: int) -> dict:
    matrix = identity_burau_matrix(p=p, n=n)
    simple_table = simple_factor_burau_table(p=p, n=n)
    history = []
    matches = []
    for depth, factor_id in enumerate(factor_ids, start=1):
        matrix = append_factor_to_burau_matrix(
            current_matrix=matrix,
            factor_id=factor_id,
            simple_table=simple_table,
            p=p,
        )
        projlen = polynomial_matrix_projlen(matrix)
        history.append(projlen)
        if projlen == 0:
            match = projective_kernel_match(matrix, p=p, n=n)
            if match.get("matches"):
                matches.append({"depth": depth, **match})
    return {
        "factor_ids": list(factor_ids),
        "depth": len(factor_ids),
        "projlen_history": history,
        "final_projlen": history[-1] if history else 0,
        "matrix_fingerprint": _fingerprint(matrix),
        "kernel_matches": matches,
    }


def verify_author_candidates(metadata: dict, records: list[dict]) -> dict:
    """Independently verify every unique author projlen-one candidate."""
    p = int(metadata["p"])
    n = int(metadata["n"])
    words: dict[tuple[int, ...], dict] = {}
    for record in records:
        words.setdefault(_factor_ids(record, n), record)

    evaluated = [_evaluate_word(word, p=p, n=n) for word in words]
    return {
        "author_projlen_one_candidates": len(records),
        "unique_candidates_verified": len(evaluated),
        "kernel_hits": [item for item in evaluated if item["kernel_matches"]],
        "false_positives": [
            item for item in evaluated if not item["kernel_matches"]
        ],
    }
