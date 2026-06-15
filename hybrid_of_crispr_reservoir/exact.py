from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from crispr_algorithms.crispr_trajectory_search_v4.evaluators import (
        canonical_matrix_fingerprint,
    )
except ModuleNotFoundError:
    from crispr_trajectory_search_v4.evaluators import canonical_matrix_fingerprint
from peyl.braid_data import (
    append_factor_to_burau_matrix,
    identity_burau_matrix,
    polynomial_matrix_projlen,
    projective_kernel_match,
    simple_factor_burau_table,
)


@dataclass
class ExactState:
    factor_ids: tuple[int, ...]
    matrix: Any
    projlen_history: tuple[int, ...]
    kernel_matches: tuple[dict, ...]

    @property
    def final_projlen(self) -> int:
        return self.projlen_history[-1]

    @property
    def matrix_fingerprint(self) -> str:
        return canonical_matrix_fingerprint(self.matrix)

    def summary(self) -> dict:
        return {
            "factor_ids": list(self.factor_ids),
            "depth": len(self.factor_ids),
            "projlen_history": list(self.projlen_history),
            "final_projlen": self.final_projlen,
            "matrix_fingerprint": self.matrix_fingerprint,
            "kernel_matches": list(self.kernel_matches),
        }


class ExactEngine:
    def __init__(self, p: int, n: int = 4):
        self.p = p
        self.n = n
        self.simple_table = simple_factor_burau_table(p=p, n=n)

    def evaluate(self, factor_ids: tuple[int, ...]) -> ExactState:
        matrix = identity_burau_matrix(p=self.p, n=self.n)
        history = []
        matches = []
        for depth, factor_id in enumerate(factor_ids, start=1):
            matrix = append_factor_to_burau_matrix(
                current_matrix=matrix,
                factor_id=factor_id,
                simple_table=self.simple_table,
                p=self.p,
            )
            projlen = polynomial_matrix_projlen(matrix)
            history.append(projlen)
            if projlen == 0:
                match = projective_kernel_match(matrix, p=self.p, n=self.n)
                if match.get("matches"):
                    matches.append({"depth": depth, **match})
        return ExactState(
            factor_ids=factor_ids,
            matrix=matrix,
            projlen_history=tuple(history),
            kernel_matches=tuple(matches),
        )
