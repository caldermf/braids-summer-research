from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Verification:
    target_match: bool
    quotient_nontrivial: bool
    quotient_kernel: bool
    quotient_power: int
    quotient_factors: tuple[int, ...]
    metrics: dict


class ReverseAlgebra:
    """Exact projective factor removal and Garside verification."""

    def __init__(self, env, proper_factor_ids: Sequence[int], target_power: int):
        self.env = env
        self.target_power = int(target_power)
        self.target_label = "identity" if self.target_power % 2 == 0 else "delta"
        self.target = env.target_exact(self.target_label)
        self.identity_digest = env.exact_digest(env.identity_exact)
        self.inverse_factor_polymats = self._inverse_factor_table(proper_factor_ids)
        self._verify_inverse_table(proper_factor_ids)

    def _inverse_factor_table(self, proper_factor_ids: Sequence[int]) -> dict[int, np.ndarray]:
        _, artin_inverses = self.env.rep.artin_gens_invs()
        out: dict[int, np.ndarray] = {}
        for factor_id in proper_factor_ids:
            permutation = self.env.nf_table.divs[int(factor_id)]
            inverse_letters = [artin_inverses[index] for index in reversed(permutation.shortlex())]
            out[int(factor_id)] = functools.reduce(
                self.env.rep.mul, inverse_letters, self.env.rep.id()
            )
        return out

    def _verify_inverse_table(self, proper_factor_ids: Sequence[int]) -> None:
        expected = self.identity_digest
        for factor_id in proper_factor_ids:
            product = self.env.rep.mul(
                self.env.factor_polymats[int(factor_id)],
                self.inverse_factor_polymats[int(factor_id)],
            )
            if self.env.exact_digest(product) != expected:
                raise RuntimeError(f"failed to construct inverse for factor {factor_id}")

    def legal_predecessors(self, right_factor: int | None) -> tuple[int, ...]:
        if right_factor is None:
            return tuple(sorted(self.inverse_factor_polymats))
        return tuple(
            factor_id
            for factor_id in sorted(self.inverse_factor_polymats)
            if bool(self.env.nf_table.is_normalised[factor_id][int(right_factor)])
        )

    def remove(self, residual: np.ndarray, factor_id: int) -> np.ndarray:
        return self.env.rep.mul(residual, self.inverse_factor_polymats[int(factor_id)])

    def projlen(self, residual: np.ndarray) -> int:
        normalized = self.env.polymat.projectivise(np.asarray(residual))
        return int(normalized.shape[-1] - 1)

    def is_identity(self, residual: np.ndarray) -> bool:
        return self.env.exact_target_metrics(residual, "identity")["target_match"]

    def same_projective_matrix(self, left: np.ndarray, right: np.ndarray) -> bool:
        a = self.env.polymat.projectivise(np.asarray(left)) % self.env.p
        b = self.env.polymat.projectivise(np.asarray(right)) % self.env.p
        return a.shape == b.shape and bool(np.array_equal(a, b))

    def verify_target_preimage(self, factors: Sequence[int]) -> Verification:
        factors = tuple(int(x) for x in factors)
        image = self.env.exact_evaluate(factors)
        metrics = self.env.exact_target_metrics(image, self.target_label)

        target_braid = self.env.GNF(self.env.n, self.target_power, ())
        braid = self.env.GNF(self.env.n, 0, factors)
        quotient = braid * target_braid.inv()
        nontrivial = quotient.power != 0 or bool(quotient.factors)

        from peyl.braidsearch import evaluate_braid_factors

        quotient_image = evaluate_braid_factors(self.env.rep, quotient)
        quotient_kernel = bool(
            self.env.exact_target_metrics(quotient_image, "identity")["target_match"]
        )
        return Verification(
            target_match=bool(metrics["target_match"]),
            quotient_nontrivial=bool(nontrivial),
            quotient_kernel=quotient_kernel,
            quotient_power=int(quotient.power),
            quotient_factors=tuple(int(x) for x in quotient.factors),
            metrics=metrics,
        )

    def verify_collision(self, left: Sequence[int], right: Sequence[int]) -> Verification:
        left_braid = self.env.GNF(self.env.n, 0, tuple(int(x) for x in left))
        right_braid = self.env.GNF(self.env.n, 0, tuple(int(x) for x in right))
        quotient = left_braid * right_braid.inv()
        nontrivial = quotient.power != 0 or bool(quotient.factors)

        from peyl.braidsearch import evaluate_braid_factors

        quotient_image = evaluate_braid_factors(self.env.rep, quotient)
        metrics = self.env.exact_target_metrics(quotient_image, "identity")
        return Verification(
            target_match=bool(metrics["target_match"]),
            quotient_nontrivial=bool(nontrivial),
            quotient_kernel=bool(metrics["target_match"]),
            quotient_power=int(quotient.power),
            quotient_factors=tuple(int(x) for x in quotient.factors),
            metrics=metrics,
        )

