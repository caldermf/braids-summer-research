from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from peyl.braid_data import delta_burau_matrix, simple_factor_burau_table

from .config import SearchConfig
from .models import Segment


def first_quadratic_nonresidue(p: int) -> int:
    for value in range(2, p):
        if pow(value, (p - 1) // 2, p) == p - 1:
            return value
    raise ValueError(f"could not find a quadratic nonresidue modulo {p}")


class ExtensionFieldSketch:
    """
    Evaluate Burau matrices at several points of GF(p^2).

    Matrices are canonicalized up to multiplication by a nonzero field scalar.
    A suffix of a projective kernel therefore matches either P^-1 or
    P^-1*Delta exactly at every selected field point.
    """

    def __init__(self, config: SearchConfig):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for V5 matrix sketches") from exc

        self.torch = torch
        self.config = config
        self.p = config.p
        self.nonresidue = first_quadratic_nonresidue(config.p)
        self.device = torch.device(config.device if config.backend == "torch" else "cpu")
        self.points = self._field_points(config.field_points)
        self.inverse_table = torch.tensor(
            [0] + [pow(value, -1, self.p) for value in range(1, self.p)],
            dtype=torch.int64,
            device=self.device,
        )

        simple_table = simple_factor_burau_table(p=config.p, n=config.n)
        max_factor_id = max(simple_table)
        factor_values = torch.zeros(
            max_factor_id + 1,
            len(self.points),
            3,
            3,
            2,
            dtype=torch.int64,
        )
        for factor_id, matrix in simple_table.items():
            factor_values[factor_id] = torch.tensor(
                self._evaluate_polynomial_matrix(matrix),
                dtype=torch.int64,
            )
        self.factor_values = factor_values.to(self.device)
        self.delta_values = torch.tensor(
            self._evaluate_polynomial_matrix(delta_burau_matrix(config.p, config.n)),
            dtype=torch.int64,
            device=self.device,
        )

    @property
    def signature_width(self) -> int:
        return len(self.points) * 3 * 3 * 2

    def _field_points(self, count: int) -> tuple[tuple[int, int], ...]:
        points = [
            (real, imaginary)
            for imaginary in range(1, self.p)
            for real in range(self.p)
        ]
        return tuple(points[:count])

    def _pair_add(self, left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
        return ((left[0] + right[0]) % self.p, (left[1] + right[1]) % self.p)

    def _pair_mul(self, left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
        return (
            (left[0] * right[0] + self.nonresidue * left[1] * right[1]) % self.p,
            (left[0] * right[1] + left[1] * right[0]) % self.p,
        )

    def _pair_inverse(self, value: tuple[int, int]) -> tuple[int, int]:
        denominator = (
            value[0] * value[0] - self.nonresidue * value[1] * value[1]
        ) % self.p
        scale = pow(denominator, -1, self.p)
        return value[0] * scale % self.p, -value[1] * scale % self.p

    def _pair_power(self, value: tuple[int, int], exponent: int) -> tuple[int, int]:
        if exponent < 0:
            return self._pair_power(self._pair_inverse(value), -exponent)
        result = (1, 0)
        base = value
        while exponent:
            if exponent & 1:
                result = self._pair_mul(result, base)
            base = self._pair_mul(base, base)
            exponent //= 2
        return result

    def _evaluate_polynomial_matrix(self, matrix) -> list:
        evaluated = []
        for point in self.points:
            point_matrix = []
            for row in matrix:
                point_row = []
                for entry in row:
                    value = (0, 0)
                    for exponent, coefficient in entry.items():
                        term = self._pair_mul(
                            (int(coefficient) % self.p, 0),
                            self._pair_power(point, int(exponent)),
                        )
                        value = self._pair_add(value, term)
                    point_row.append(value)
                point_matrix.append(point_row)
            evaluated.append(point_matrix)
        return evaluated

    def _mul(self, left, right):
        torch = self.torch
        real = left[..., 0] * right[..., 0]
        real += self.nonresidue * left[..., 1] * right[..., 1]
        imaginary = left[..., 0] * right[..., 1] + left[..., 1] * right[..., 0]
        return torch.stack(
            (torch.remainder(real, self.p), torch.remainder(imaginary, self.p)),
            dim=-1,
        )

    def _sub(self, left, right):
        return self.torch.remainder(left - right, self.p)

    def _inverse(self, value):
        denominator = self.torch.remainder(
            value[..., 0] * value[..., 0]
            - self.nonresidue * value[..., 1] * value[..., 1],
            self.p,
        )
        if self.torch.any(denominator == 0):
            raise ArithmeticError("attempted to invert zero in GF(p^2)")
        scale = self.inverse_table[denominator]
        return self.torch.stack(
            (
                self.torch.remainder(value[..., 0] * scale, self.p),
                self.torch.remainder(-value[..., 1] * scale, self.p),
            ),
            dim=-1,
        )

    def _matmul(self, left, right):
        products = self._mul(
            left[:, :, :, :, None, :],
            right[:, :, None, :, :, :],
        )
        return self.torch.remainder(products.sum(dim=3), self.p)

    def _identity(self, batch_size: int):
        state = self.torch.zeros(
            batch_size,
            len(self.points),
            3,
            3,
            2,
            dtype=self.torch.int64,
            device=self.device,
        )
        diagonal = self.torch.arange(3, device=self.device)
        state[:, :, diagonal, diagonal, 0] = 1
        return state

    def _states(self, segments: Sequence[Segment]):
        if not segments:
            return self._identity(0)
        max_length = max(segment.length for segment in segments)
        padded_words = [
            list(segment.factor_ids) + [0] * (max_length - segment.length)
            for segment in segments
        ]
        words = self.torch.tensor(
            padded_words,
            dtype=self.torch.long,
            device=self.device,
        )
        lengths = self.torch.tensor(
            [segment.length for segment in segments],
            dtype=self.torch.long,
            device=self.device,
        )
        for index, segment in enumerate(segments):
            words[index, : segment.length] = self.torch.tensor(
                segment.factor_ids,
                dtype=self.torch.long,
                device=self.device,
            )

        state = self._identity(len(segments))
        for depth in range(max_length):
            active = depth < lengths
            if not self.torch.any(active):
                break
            product = self._matmul(state, self.factor_values[words[:, depth]])
            state = self.torch.where(
                active[:, None, None, None, None],
                product,
                state,
            )
        return state

    def _matrix_inverse(self, matrix):
        a, b, c = matrix[..., 0, 0, :], matrix[..., 0, 1, :], matrix[..., 0, 2, :]
        d, e, f = matrix[..., 1, 0, :], matrix[..., 1, 1, :], matrix[..., 1, 2, :]
        g, h, i = matrix[..., 2, 0, :], matrix[..., 2, 1, :], matrix[..., 2, 2, :]

        c00 = self._sub(self._mul(e, i), self._mul(f, h))
        c01 = self._sub(self._mul(f, g), self._mul(d, i))
        c02 = self._sub(self._mul(d, h), self._mul(e, g))
        c10 = self._sub(self._mul(c, h), self._mul(b, i))
        c11 = self._sub(self._mul(a, i), self._mul(c, g))
        c12 = self._sub(self._mul(b, g), self._mul(a, h))
        c20 = self._sub(self._mul(b, f), self._mul(c, e))
        c21 = self._sub(self._mul(c, d), self._mul(a, f))
        c22 = self._sub(self._mul(a, e), self._mul(b, d))

        determinant = self._mul(a, c00)
        determinant = self.torch.remainder(
            determinant + self._mul(b, c01) + self._mul(c, c02),
            self.p,
        )
        scale = self._inverse(determinant)
        adjugate = self.torch.stack(
            (
                self.torch.stack((c00, c10, c20), dim=-2),
                self.torch.stack((c01, c11, c21), dim=-2),
                self.torch.stack((c02, c12, c22), dim=-2),
            ),
            dim=-3,
        )
        return self._mul(adjugate, scale[..., None, None, :])

    def _canonical(self, matrix) -> np.ndarray:
        batch_size = matrix.shape[0]
        flat = matrix.reshape(batch_size, len(self.points), 9, 2)
        nonzero = self.torch.any(flat != 0, dim=-1)
        if self.torch.any(~self.torch.any(nonzero, dim=-1)):
            raise ArithmeticError("zero matrix cannot be projectively normalized")
        first = self.torch.argmax(nonzero.to(self.torch.int64), dim=-1)
        gather_index = first[..., None, None].expand(batch_size, len(self.points), 1, 2)
        pivot = self.torch.gather(flat, 2, gather_index).squeeze(2)
        normalized = self._mul(flat, self._inverse(pivot)[..., None, :])
        return normalized.reshape(batch_size, -1).to(self.torch.uint8).cpu().numpy()

    def suffix_signatures(self, segments: Iterable[Segment]) -> np.ndarray:
        segment_list = list(segments)
        output = []
        for start in range(0, len(segment_list), self.config.signature_batch_size):
            states = self._states(
                segment_list[start : start + self.config.signature_batch_size]
            )
            output.append(self._canonical(states))
        if not output:
            return np.empty((0, self.signature_width), dtype=np.uint8)
        return np.concatenate(output, axis=0)

    def prefix_target_signatures(
        self,
        segments: Iterable[Segment],
    ) -> dict[str, np.ndarray]:
        segment_list = list(segments)
        identity_output = []
        delta_output = []
        for start in range(0, len(segment_list), self.config.signature_batch_size):
            states = self._states(
                segment_list[start : start + self.config.signature_batch_size]
            )
            inverse = self._matrix_inverse(states)
            identity_output.append(self._canonical(inverse))
            delta = self.delta_values[None, ...].expand(states.shape[0], -1, -1, -1, -1)
            delta_output.append(self._canonical(self._matmul(inverse, delta)))
        empty = np.empty((0, self.signature_width), dtype=np.uint8)
        return {
            "identity": np.concatenate(identity_output, axis=0) if identity_output else empty,
            "delta": np.concatenate(delta_output, axis=0) if delta_output else empty,
        }

    @staticmethod
    def distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Component Hamming distance between canonical GF(p^2) signatures."""
        return np.count_nonzero(left != right, axis=-1)
