from __future__ import annotations

import os
from collections import defaultdict
from typing import Iterable, Sequence

from peyl.braid_data import (
    append_factor_to_burau_matrix,
    identity_burau_matrix,
    polynomial_matrix_projlen,
    projective_kernel_match,
    simple_factor_burau_table,
)

from .config import SearchConfig
from .models import WordEvaluation


def maximum_drawdown(values: Sequence[int]) -> int:
    running_max = values[0]
    best = 0
    for value in values[1:]:
        running_max = max(running_max, value)
        best = max(best, running_max - value)
    return best


class CPUExactEvaluator:
    def __init__(self, config: SearchConfig):
        self.config = config
        self.simple_table = simple_factor_burau_table(config.p, config.n)

    def evaluate_one(self, factor_ids: tuple[int, ...]) -> WordEvaluation:
        matrix = identity_burau_matrix(self.config.p, self.config.n)
        history = []
        matches = []
        for depth, factor_id in enumerate(factor_ids, start=1):
            matrix = append_factor_to_burau_matrix(
                matrix,
                factor_id,
                self.simple_table,
                self.config.p,
            )
            projlen = polynomial_matrix_projlen(matrix)
            history.append(projlen)
            if projlen == 0:
                match = projective_kernel_match(matrix, self.config.p, self.config.n)
                if match.get("matches"):
                    matches.append({"depth": depth, **match})
        return WordEvaluation(
            factor_ids=factor_ids,
            projlen_history=tuple(history),
            final_projlen=history[-1],
            min_projlen=min(history),
            peak_projlen=max(history),
            largest_drop=maximum_drawdown(history),
            kernel_matches=tuple(matches),
        )

    def evaluate(self, words: Iterable[tuple[int, ...]]) -> list[WordEvaluation]:
        return [self.evaluate_one(tuple(word)) for word in words]


class TorchExactEvaluator:
    """Batched exact-mod-p coefficient evolution, with CPU certification."""

    def __init__(self, config: SearchConfig):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for backend='torch'") from exc
        if config.n != 4:
            raise ValueError("the torch exact evaluator currently supports n=4")
        self.torch = torch
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda":
            partition = os.environ.get("SLURM_JOB_PARTITION")
            if partition != config.required_cuda_partition:
                raise RuntimeError(
                    f"CUDA is restricted to {config.required_cuda_partition!r}; "
                    f"active partition is {partition!r}"
                )
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
        self.simple_table = simple_factor_burau_table(config.p, config.n)
        self.max_factor_id = max(self.simple_table)
        self.max_shift = 4
        coefficients = torch.zeros(
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
                    if not entry:
                        continue
                    if len(entry) != 1:
                        raise ValueError("simple factor entry is not monomial")
                    exponent, coefficient = next(iter(entry.items()))
                    coefficients[factor_id, exponent, row, column] = coefficient % config.p
        self.coefficients = coefficients.to(self.device)
        self.cpu_verifier = CPUExactEvaluator(config)

    def _same_horizon(self, words: Sequence[tuple[int, ...]]) -> list[WordEvaluation]:
        torch = self.torch
        horizon = len(words[0])
        word_tensor = torch.tensor(words, dtype=torch.long, device=self.device)
        width = self.max_shift * horizon + 1
        state = torch.zeros(
            len(words),
            3,
            3,
            width,
            dtype=torch.float32,
            device=self.device,
        )
        diagonal = torch.arange(3, device=self.device)
        state[:, diagonal, diagonal, 0] = 1
        histories = torch.empty(
            len(words),
            horizon,
            dtype=torch.int16,
            device=self.device,
        )
        for depth in range(horizon):
            right_ids = word_tensor[:, depth]
            next_state = torch.zeros_like(state)
            for shift in range(self.max_shift + 1):
                right = self.coefficients[right_ids, shift]
                source = state[..., : width - shift].permute(0, 3, 1, 2)
                product = torch.matmul(source, right[:, None, :, :])
                next_state[..., shift:] += product.permute(0, 2, 3, 1)
            state = torch.remainder(next_state, self.config.p)
            support = torch.any(state != 0, dim=(1, 2))
            first = torch.argmax(support.to(torch.int16), dim=1)
            last = width - 1 - torch.argmax(
                torch.flip(support, dims=(1,)).to(torch.int16),
                dim=1,
            )
            histories[:, depth] = (last - first).to(torch.int16)

        output = []
        for word, history in zip(words, histories.cpu().tolist()):
            result = WordEvaluation(
                factor_ids=word,
                projlen_history=tuple(history),
                final_projlen=history[-1],
                min_projlen=min(history),
                peak_projlen=max(history),
                largest_drop=maximum_drawdown(history),
            )
            if 0 in history:
                result = self.cpu_verifier.evaluate_one(word)
            output.append(result)
        return output

    def evaluate(self, words: Iterable[tuple[int, ...]]) -> list[WordEvaluation]:
        word_list = [tuple(word) for word in words]
        grouped: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
        for index, word in enumerate(word_list):
            grouped[len(word)].append((index, word))
        output: list[WordEvaluation | None] = [None] * len(word_list)
        for group in grouped.values():
            for start in range(0, len(group), self.config.exact_batch_size):
                chunk = group[start : start + self.config.exact_batch_size]
                evaluated = self._same_horizon([word for _, word in chunk])
                for (index, _), result in zip(chunk, evaluated):
                    output[index] = result
        return [item for item in output if item is not None]


def make_exact_evaluator(config: SearchConfig):
    if config.backend == "cpu":
        return CPUExactEvaluator(config)
    return TorchExactEvaluator(config)
