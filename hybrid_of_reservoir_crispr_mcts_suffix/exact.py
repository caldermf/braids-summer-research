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
    def min_projlen(self) -> int:
        return min(self.projlen_history)

    @property
    def has_kernel(self) -> bool:
        return bool(self.kernel_matches)

    @property
    def matrix_fingerprint(self) -> str:
        return canonical_matrix_fingerprint(self.matrix)

    def summary(self) -> dict:
        return {
            "factor_ids": list(self.factor_ids),
            "depth": len(self.factor_ids),
            "projlen_history": list(self.projlen_history),
            "final_projlen": self.final_projlen,
            "min_projlen": self.min_projlen,
            "matrix_fingerprint": self.matrix_fingerprint,
            "kernel_matches": list(self.kernel_matches),
        }


class ExactEngine:
    def __init__(self, p: int, n: int = 4):
        self.p = p
        self.n = n
        self.simple_table = simple_factor_burau_table(p=p, n=n)

    def identity(self) -> ExactState:
        return ExactState(
            factor_ids=(),
            matrix=identity_burau_matrix(p=self.p, n=self.n),
            projlen_history=(),
            kernel_matches=(),
        )

    def extend(self, state: ExactState, factor_id: int) -> ExactState:
        matrix = append_factor_to_burau_matrix(
            current_matrix=state.matrix,
            factor_id=factor_id,
            simple_table=self.simple_table,
            p=self.p,
        )
        projlen = polynomial_matrix_projlen(matrix)
        matches = list(state.kernel_matches)
        if projlen == 0:
            match = projective_kernel_match(matrix, p=self.p, n=self.n)
            if match.get("matches"):
                matches.append({"depth": len(state.factor_ids) + 1, **match})
        return ExactState(
            factor_ids=state.factor_ids + (factor_id,),
            matrix=matrix,
            projlen_history=state.projlen_history + (projlen,),
            kernel_matches=tuple(matches),
        )

    def evaluate(self, factor_ids: tuple[int, ...]) -> ExactState:
        state = self.identity()
        for factor_id in factor_ids:
            state = self.extend(state, factor_id)
        return state
