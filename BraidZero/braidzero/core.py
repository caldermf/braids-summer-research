from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


FlatMatrix = tuple[int, ...]
MatrixTuple = tuple[FlatMatrix, ...]
Fingerprint = tuple[int, ...]


def setup_author_imports(author_repo: Path):
    author_repo = Path(author_repo).resolve()
    if not (author_repo / "peyl" / "braid.py").exists():
        raise FileNotFoundError(f"peyl package not found under author repo: {author_repo}")
    if str(author_repo) not in sys.path:
        sys.path.insert(0, str(author_repo))

    import peyl  # type: ignore
    from peyl import polymat  # type: ignore
    from peyl.braid import GNF  # type: ignore
    from peyl.braidsearch import symmetric_table  # type: ignore

    return peyl, polymat, GNF, symmetric_table


def parse_int_list(value: str | Sequence[int] | None, *, default: Sequence[int]) -> tuple[int, ...]:
    if value is None or value == "":
        return tuple(int(x) for x in default)
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
        if not items:
            return tuple(int(x) for x in default)
        return tuple(int(part) for part in items)
    return tuple(int(x) for x in value)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def word_digest(power: int, factors: Sequence[int]) -> str:
    encoded = json.dumps(
        [int(power), [int(x) for x in factors]],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def identity_flat(dim: int) -> FlatMatrix:
    return tuple(1 if row == col else 0 for row in range(dim) for col in range(dim))


def mat_mul_flat(left: FlatMatrix, right: FlatMatrix, dim: int, p: int) -> FlatMatrix:
    out: list[int] = []
    for row in range(dim):
        row_offset = row * dim
        for col in range(dim):
            value = 0
            for mid in range(dim):
                value += left[row_offset + mid] * right[mid * dim + col]
            out.append(value % p)
    return tuple(out)


def mat_inv_flat(matrix: FlatMatrix, dim: int, p: int) -> FlatMatrix:
    rows = [
        [matrix[row * dim + col] % p for col in range(dim)]
        + [1 if row == col else 0 for col in range(dim)]
        for row in range(dim)
    ]
    for col in range(dim):
        pivot = None
        for row in range(col, dim):
            if rows[row][col] % p:
                pivot = row
                break
        if pivot is None:
            raise ValueError("singular matrix")
        if pivot != col:
            rows[col], rows[pivot] = rows[pivot], rows[col]
        inv = pow(rows[col][col] % p, -1, p)
        rows[col] = [(value * inv) % p for value in rows[col]]
        for row in range(dim):
            if row == col:
                continue
            factor = rows[row][col] % p
            if factor:
                rows[row] = [
                    (rows[row][idx] - factor * rows[col][idx]) % p
                    for idx in range(2 * dim)
                ]
    return tuple(rows[row][dim + col] % p for row in range(dim) for col in range(dim))


def normalize_flat(matrix: FlatMatrix, p: int) -> FlatMatrix:
    for value in matrix:
        value %= p
        if value:
            inv = pow(value, -1, p)
            return tuple((entry * inv) % p for entry in matrix)
    raise ValueError("zero matrix cannot be projectively normalized")


def is_scalar_flat(matrix: FlatMatrix, dim: int, p: int) -> bool:
    scalar = matrix[0] % p
    if scalar == 0:
        return False
    for row in range(dim):
        for col in range(dim):
            value = matrix[row * dim + col] % p
            if row == col and value != scalar:
                return False
            if row != col and value != 0:
                return False
    return True


def specialize_polymat(poly_matrix: np.ndarray, t_value: int, p: int) -> FlatMatrix:
    dim = int(poly_matrix.shape[0])
    powers = np.array(
        [pow(t_value % p, degree, p) for degree in range(poly_matrix.shape[-1])],
        dtype=np.int64,
    )
    specialized = np.tensordot(poly_matrix.astype(np.int64) % p, powers, axes=([-1], [0])) % p
    return tuple(int(value) % p for value in specialized.reshape(dim * dim))


def scalar_identity_metrics(polymat_module, image: np.ndarray) -> dict:
    projected = polymat_module.projectivise(image)
    projlen = int(projected.shape[-1])
    dim = int(projected.shape[0])
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
    return {
        "projlen": projlen,
        "scalar_identity": identity_defect == 0,
        "identity_defect": int(identity_defect),
        "off_diagonal_terms": int(off_diagonal_terms),
        "diagonal_mismatch_terms": int(diagonal_mismatch_terms),
        "scalar_nonzero_degrees": int(scalar_nonzero_degrees),
        "scalar_extra_degrees": int(scalar_extra_degrees),
        "nonzero_terms": int(np.count_nonzero(projected)),
    }


def exact_matrix_digest(polymat_module, image: np.ndarray) -> str:
    projected = polymat_module.projectivise(image)
    digest = hashlib.sha1()
    digest.update(json.dumps(list(projected.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(projected.astype(np.int32, copy=False).tobytes(order="C"))
    return digest.hexdigest()


class BraidEnvironment:
    """Exact and finite-shadow arithmetic for the B4 two-rowed search."""

    verifier_version = "braidzero-exact-peyl-v1"

    def __init__(
        self,
        *,
        author_repo: Path,
        n: int = 4,
        r: int = 1,
        p: int = 7,
        t_values: Sequence[int] | None = None,
    ):
        if p <= 1:
            raise ValueError("BraidZero finite shadows require p > 1")
        default_t_values = tuple(range(1, p))
        self.author_repo = Path(author_repo).resolve()
        self.n = int(n)
        self.r = int(r)
        self.p = int(p)
        self.t_values = tuple(int(x) % p for x in (t_values or default_t_values))
        if not self.t_values or any(x == 0 for x in self.t_values):
            raise ValueError("t-values must be nonzero modulo p")

        self.peyl, self.polymat, self.GNF, symmetric_table = setup_author_imports(self.author_repo)
        self.rep = self.peyl.JonesSummand(n=self.n, r=self.r, p=self.p)
        self.dim = int(self.rep.dimension())
        self.nf_table = self.GNF._nf_table(self.n)
        self.identity_exact = self.rep.id()
        self.identity_finite = tuple(identity_flat(self.dim) for _ in self.t_values)

        sym_table = symmetric_table(self.rep)
        self.factor_polymats: dict[int, np.ndarray] = {}
        self.factor_finite: dict[int, MatrixTuple] = {}
        for factor_id, perm in enumerate(self.nf_table.divs):
            poly = self.polymat.projectivise(sym_table[perm])
            self.factor_polymats[factor_id] = poly
            self.factor_finite[factor_id] = tuple(
                specialize_polymat(poly, t_value, self.p) for t_value in self.t_values
            )

        self.first_ids = tuple(sorted(self.nf_table.follows[self.nf_table.D]))
        self.successors = {
            factor_id: tuple(sorted(self.nf_table.follows[factor_id]))
            for factor_id in range(self.nf_table.order)
        }

    @property
    def representation_label(self) -> str:
        return f"two-rowed ({self.n - self.r},{self.r}) of B_{self.n} over F_{self.p}"

    def legal_next(self, factors: Sequence[int]) -> tuple[int, ...]:
        if not factors:
            return self.first_ids
        return self.successors[int(factors[-1])]

    def is_legal(self, factors: Sequence[int]) -> bool:
        if not factors:
            return True
        return self.nf_table.is_factors_normalised(tuple(int(x) for x in factors))

    def count_normal_forms(self, length: int, following: int | None = None) -> int:
        return int(self.nf_table.count_normal_forms(length, following=following))

    def normal_forms(self, length: int, following: int | None = None) -> Iterable[tuple[int, ...]]:
        yield from self.nf_table.normal_forms(length, following=following)

    def sample_normal_form(self, length: int, rng: random.Random) -> tuple[int, ...]:
        return self.nf_table.sample(length, rng)

    def finite_mul(self, left: MatrixTuple, right: MatrixTuple) -> MatrixTuple:
        return tuple(mat_mul_flat(a, b, self.dim, self.p) for a, b in zip(left, right))

    def finite_inverse(self, matrices: MatrixTuple) -> MatrixTuple:
        return tuple(mat_inv_flat(matrix, self.dim, self.p) for matrix in matrices)

    def finite_key(self, matrices: MatrixTuple) -> Fingerprint:
        chunks = [normalize_flat(matrix, self.p) for matrix in matrices]
        return tuple(entry for chunk in chunks for entry in chunk)

    def finite_scalar_flags(self, matrices: MatrixTuple) -> list[bool]:
        return [is_scalar_flat(matrix, self.dim, self.p) for matrix in matrices]

    def finite_append(self, matrices: MatrixTuple, factor_id: int) -> MatrixTuple:
        return self.finite_mul(matrices, self.factor_finite[int(factor_id)])

    def finite_evaluate(self, factors: Sequence[int]) -> MatrixTuple:
        matrices = self.identity_finite
        for factor_id in factors:
            matrices = self.finite_append(matrices, int(factor_id))
        return matrices

    def exact_append(self, image: np.ndarray, factor_id: int) -> np.ndarray:
        return self.rep.mul(image, self.factor_polymats[int(factor_id)])

    def exact_evaluate(self, factors: Sequence[int]) -> np.ndarray:
        image = self.identity_exact
        for factor_id in factors:
            image = self.exact_append(image, int(factor_id))
        return image

    def exact_append_sequence(self, image: np.ndarray, factors: Sequence[int]) -> np.ndarray:
        out = image
        for factor_id in factors:
            out = self.exact_append(out, int(factor_id))
        return out

    def exact_metrics(self, image: np.ndarray) -> dict:
        return scalar_identity_metrics(self.polymat, image)

    def exact_digest(self, image: np.ndarray) -> str:
        return exact_matrix_digest(self.polymat, image)

    def braid_is_nontrivial_positive_gnf(self, factors: Sequence[int]) -> bool:
        return len(tuple(factors)) > 0 and self.is_legal(factors)
