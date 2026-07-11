from __future__ import annotations

from typing import Dict, Optional, Tuple

from peyl.braid_data import delta_burau_matrix


def normalize_entry(entry: dict, p: int) -> dict:
    return {
        int(exp): int(coeff) % p
        for exp, coeff in entry.items()
        if int(coeff) % p != 0
    }


def entry_width(entry: dict) -> int:
    if not entry:
        return 0
    exponents = tuple(entry)
    return max(exponents) - min(exponents) + 1


def monomial_mul_entry(scalar: Tuple[int, int], entry: dict, p: int) -> dict:
    scalar_exp, scalar_coeff = scalar
    return {
        exp + scalar_exp: value
        for exp, coeff in entry.items()
        if (value := scalar_coeff * coeff % p)
    }


def entry_difference(actual: dict, expected: dict, p: int) -> float:
    actual = normalize_entry(actual, p)
    expected = normalize_entry(expected, p)
    if actual == expected:
        return 0.0
    support_penalty = len(set(actual.items()).symmetric_difference(expected.items()))
    width_penalty = abs(entry_width(actual) - entry_width(expected))
    return float(support_penalty + 0.25 * width_penalty)


def projective_identity_distance(poly_mat, p: int, n: int) -> float:
    size = n - 1
    penalty = 0.0
    diagonal_scalars: list[Tuple[int, int]] = []

    for i in range(size):
        for j in range(size):
            entry = normalize_entry(poly_mat[i][j], p)
            if i != j:
                penalty += 2.0 * len(entry) + 0.25 * entry_width(entry)
            elif len(entry) == 1:
                diagonal_scalars.append(next(iter(entry.items())))
            else:
                penalty += 2.0 + 1.5 * len(entry) + 0.25 * entry_width(entry)

    if not diagonal_scalars:
        return penalty + 3.0 * size

    counts: Dict[Tuple[int, int], int] = {}
    for scalar in diagonal_scalars:
        counts[scalar] = counts.get(scalar, 0) + 1
    reference_exp, reference_coeff = max(counts, key=counts.get)

    for exp, coeff in diagonal_scalars:
        if (exp, coeff) == (reference_exp, reference_coeff):
            continue
        penalty += 1.0 + 0.1 * min(20, abs(exp - reference_exp))
        if coeff != reference_coeff:
            penalty += 0.5

    return penalty


def projective_target_distance(poly_mat, target_mat, p: int, n: int) -> float:
    size = n - 1
    scalar: Optional[Tuple[int, int]] = None
    penalty = 0.0

    for i in range(size):
        for j in range(size):
            target = normalize_entry(target_mat[i][j], p)
            actual = normalize_entry(poly_mat[i][j], p)
            if not target or not actual:
                continue
            if len(target) == 1 and len(actual) == 1:
                target_exp, target_coeff = next(iter(target.items()))
                actual_exp, actual_coeff = next(iter(actual.items()))
                scalar = (
                    actual_exp - target_exp,
                    actual_coeff * pow(target_coeff, -1, p) % p,
                )
                break
        if scalar is not None:
            break

    if scalar is None:
        penalty += 5.0

    for i in range(size):
        for j in range(size):
            target = normalize_entry(target_mat[i][j], p)
            actual = normalize_entry(poly_mat[i][j], p)
            if not target:
                penalty += 2.0 * len(actual) + 0.25 * entry_width(actual)
            elif scalar is None:
                penalty += (
                    3.0
                    if not actual
                    else 1.0 + abs(len(actual) - len(target)) + 0.25 * entry_width(actual)
                )
            else:
                penalty += entry_difference(actual, monomial_mul_entry(scalar, target, p), p)

    return penalty


class PeriodicDistance:
    def __init__(self, p: int, n: int):
        self.p = p
        self.n = n
        self.delta_target = delta_burau_matrix(p=p, n=n)

    def __call__(self, poly_mat) -> float:
        return min(
            projective_identity_distance(poly_mat, p=self.p, n=self.n),
            projective_target_distance(
                poly_mat,
                self.delta_target,
                p=self.p,
                n=self.n,
            ),
        )
