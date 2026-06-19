from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .braid_data import (
    append_factor_to_burau_matrix,
    identity_burau_matrix,
    polynomial_matrix_projlen,
    projective_kernel_match,
    simple_factor_burau_table,
)


def require_compatible_cuda(torch, required_partition: str = "scavenge_gpu") -> None:
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if partition != required_partition:
        raise RuntimeError(
            f"CUDA is restricted to partition {required_partition!r}; "
            f"active partition is {partition!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    major, minor = torch.cuda.get_device_capability(0)
    required_arch = f"sm_{major}{minor}"
    supported_arches = set(torch.cuda.get_arch_list())
    if required_arch not in supported_arches:
        raise RuntimeError(
            f"GPU {torch.cuda.get_device_name(0)!r} requires {required_arch}, but "
            f"PyTorch {torch.__version__} only contains {sorted(supported_arches)}. "
            "Use a PyTorch CUDA 13.0+ build; the project setup script installs "
            "a separate compatible environment."
        )


@dataclass(frozen=True)
class Evaluation:
    factor_ids: tuple[int, ...]
    projlen_history: tuple[int, ...]
    kernel_matches: tuple[dict, ...] = ()

    @property
    def length(self) -> int:
        return len(self.factor_ids)

    @property
    def final_projlen(self) -> int:
        return self.projlen_history[-1]

    @property
    def has_kernel(self) -> bool:
        return bool(self.kernel_matches)

    def summary(self) -> dict:
        return {
            "factor_ids": list(self.factor_ids),
            "length": self.length,
            "projlen_history": list(self.projlen_history),
            "final_projlen": self.final_projlen,
            "kernel_matches": list(self.kernel_matches),
        }


class CPUExactEvaluator:
    def __init__(self, p: int, n: int = 4, **_: Any):
        self.p = int(p)
        self.n = int(n)
        self.simple_table = simple_factor_burau_table(p=self.p, n=self.n)

    def evaluate_one(self, factors: Sequence[int]) -> Evaluation:
        word = tuple(int(value) for value in factors)
        matrix = identity_burau_matrix(p=self.p, n=self.n)
        history = []
        matches = []
        for depth, factor_id in enumerate(word, start=1):
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
        return Evaluation(word, tuple(history), tuple(matches))

    def evaluate(self, words: Iterable[Sequence[int]]) -> list[Evaluation]:
        return [self.evaluate_one(word) for word in words]


class TorchExactEvaluator:
    """Batched exact-mod-p coefficient propagation for B4."""

    def __init__(
        self,
        p: int,
        n: int = 4,
        device: str = "cuda",
        batch_size: int = 10_000,
        required_partition: str = "scavenge_gpu",
    ):
        if n != 4:
            raise ValueError("the torch evaluator currently supports n=4")
        import torch

        self.torch = torch
        self.p = int(p)
        self.n = int(n)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        if self.device.type == "cuda":
            require_compatible_cuda(torch, required_partition)
        self.simple_table = simple_factor_burau_table(p=self.p, n=self.n)
        self.max_factor_id = max(self.simple_table)
        self.max_shift = 4
        self.coefficients = self._coefficient_table().to(self.device)
        self.cpu_verifier = CPUExactEvaluator(p=self.p, n=self.n)

    def _coefficient_table(self):
        torch = self.torch
        table = torch.zeros(
            self.max_factor_id + 1,
            self.max_shift + 1,
            3,
            3,
            dtype=torch.float32,
        )
        for factor_id, matrix in self.simple_table.items():
            for row in range(3):
                for column in range(3):
                    entry = matrix[row][column]
                    if len(entry) > 1:
                        raise ValueError("simple factor entry is not monomial")
                    if entry:
                        exponent, coefficient = next(iter(entry.items()))
                        table[factor_id, exponent, row, column] = coefficient % self.p
        return table

    def _same_length(self, words: Sequence[tuple[int, ...]]) -> list[Evaluation]:
        torch = self.torch
        length = len(words[0])
        tokens = torch.tensor(words, dtype=torch.long, device=self.device)
        width = self.max_shift * length + 1
        state = torch.zeros(len(words), 3, 3, width, device=self.device)
        diagonal = torch.arange(3, device=self.device)
        state[:, diagonal, diagonal, 0] = 1.0
        histories = torch.empty(len(words), length, dtype=torch.int16, device=self.device)
        for depth in range(length):
            factor_ids = tokens[:, depth]
            next_state = torch.zeros_like(state)
            for shift in range(self.max_shift + 1):
                right = self.coefficients[factor_ids, shift]
                source = state[..., : width - shift].permute(0, 3, 1, 2)
                product = torch.matmul(source, right[:, None, :, :])
                next_state[..., shift:] += product.permute(0, 2, 3, 1)
            state = torch.remainder(next_state, self.p)
            support = torch.any(state != 0, dim=(1, 2))
            first = torch.argmax(support.to(torch.int16), dim=1)
            last = width - 1 - torch.argmax(
                torch.flip(support, dims=(1,)).to(torch.int16), dim=1
            )
            histories[:, depth] = (last - first).to(torch.int16)
        output = [
            Evaluation(word, tuple(history))
            for word, history in zip(words, histories.cpu().tolist())
        ]
        for index, evaluation in enumerate(output):
            if 0 in evaluation.projlen_history:
                output[index] = self.cpu_verifier.evaluate_one(evaluation.factor_ids)
        return output

    def evaluate(self, words: Iterable[Sequence[int]]) -> list[Evaluation]:
        word_list = [tuple(int(value) for value in word) for word in words]
        grouped: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
        for index, word in enumerate(word_list):
            grouped[len(word)].append((index, word))
        output: list[Evaluation | None] = [None] * len(word_list)
        for group in grouped.values():
            for start in range(0, len(group), self.batch_size):
                chunk = group[start : start + self.batch_size]
                evaluated = self._same_length([word for _, word in chunk])
                for (index, _), result in zip(chunk, evaluated):
                    output[index] = result
        return [item for item in output if item is not None]


def make_evaluator(
    p: int,
    n: int,
    backend: str,
    device: str,
    batch_size: int,
):
    if backend == "cpu":
        return CPUExactEvaluator(p=p, n=n)
    if backend == "torch":
        return TorchExactEvaluator(
            p=p,
            n=n,
            device=device,
            batch_size=batch_size,
        )
    raise ValueError("backend must be 'cpu' or 'torch'")
