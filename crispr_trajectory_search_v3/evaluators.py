from __future__ import annotations

import hashlib
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
from .fitness import build_evaluation
from .models import Trajectory, TrajectoryEvaluation


def canonical_matrix_fingerprint(matrix) -> str:
    """Collision-resistant digest of an exact polynomial matrix."""
    frozen = tuple(
        tuple(
            tuple(sorted((int(degree), int(coefficient)) for degree, coefficient in entry.items()))
            for entry in row
        )
        for row in matrix
    )
    return hashlib.blake2b(repr(frozen).encode("ascii"), digest_size=16).hexdigest()


class CPUTrajectoryEvaluator:
    """Exact reference evaluator using peyl's polynomial dictionaries."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.simple_table = simple_factor_burau_table(p=config.p, n=config.n)

    def evaluate_one(self, trajectory: Trajectory) -> TrajectoryEvaluation:
        matrix = identity_burau_matrix(p=self.config.p, n=self.config.n)
        projlen_history = []
        kernel_depths = []
        kernel_matches = []

        for depth, factor_id in enumerate(trajectory.factor_ids, start=1):
            matrix = append_factor_to_burau_matrix(
                current_matrix=matrix,
                factor_id=factor_id,
                simple_table=self.simple_table,
                p=self.config.p,
            )
            projlen_history.append(polynomial_matrix_projlen(matrix))
            match = projective_kernel_match(
                matrix,
                p=self.config.p,
                n=self.config.n,
            )
            if match.get("matches"):
                kernel_depths.append(depth)
                kernel_matches.append(match)

        return build_evaluation(
            trajectory=trajectory,
            projlen_history=projlen_history,
            config=self.config,
            kernel_depths=kernel_depths,
            kernel_matches=kernel_matches,
            matrix_fingerprint=canonical_matrix_fingerprint(matrix),
        )

    def evaluate(self, trajectories: Iterable[Trajectory]) -> list[TrajectoryEvaluation]:
        return [self.evaluate_one(trajectory) for trajectory in trajectories]


class TorchTrajectoryEvaluator:
    """
    Batched dense evaluator for CUDA or CPU PyTorch devices.

    The simple positive Garside-factor matrices have monomial entries with
    shifts in 0..4. At each depth, five batched 3x3 matrix products update the
    coefficient tensor exactly modulo p.
    """

    def __init__(self, config: SearchConfig):
        if config.n != 4:
            raise ValueError("the torch evaluator currently supports n=4")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for backend='torch'") from exc

        self.torch = torch
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda":
            active_partition = os.environ.get("SLURM_JOB_PARTITION")
            if active_partition != config.required_cuda_partition:
                raise RuntimeError(
                    "CUDA execution is restricted to the Slurm partition "
                    f"{config.required_cuda_partition!r}; active partition is "
                    f"{active_partition!r}. Submit the project scavenge_gpu job script."
                )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this PyTorch environment")

        self.simple_table = simple_factor_burau_table(p=config.p, n=config.n)
        self.max_factor_id = max(self.simple_table)
        self.max_shift = 4
        self.coefficients = self._build_coefficient_table().to(self.device)

        verifier_config = SearchConfig(**{**config.__dict__, "backend": "cpu"})
        self.cpu_verifier = CPUTrajectoryEvaluator(verifier_config)

    def _build_coefficient_table(self):
        torch = self.torch
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
                    if len(entry) > 1:
                        raise ValueError("simple-factor entry is not monomial")
                    if not entry:
                        continue
                    exponent, coefficient = next(iter(entry.items()))
                    if exponent < 0 or exponent > self.max_shift:
                        raise ValueError(f"unsupported simple-factor shift {exponent}")
                    coefficients[factor_id, exponent, row, column] = (
                        coefficient % self.config.p
                    )
        return coefficients

    def _matrix_fingerprints(self, state) -> list[str]:
        """Two independent exact-in-float64 linear hashes of final matrix states."""
        torch = self.torch
        flattened = state.to(torch.float64).flatten(start_dim=1)
        positions = torch.arange(
            1,
            flattened.shape[1] + 1,
            dtype=torch.float64,
            device=self.device,
        )
        modulus = 2_147_483_647
        weights_a = torch.remainder(positions * 1_000_003 + 97, modulus)
        weights_b = torch.remainder(
            positions * positions * 1_000_033 + positions * 193 + 17,
            modulus,
        )
        hash_a = torch.remainder((flattened * weights_a).sum(dim=1), modulus)
        hash_b = torch.remainder((flattened * weights_b).sum(dim=1), modulus)
        return [
            f"{int(left):08x}{int(right):08x}"
            for left, right in zip(hash_a.cpu().tolist(), hash_b.cpu().tolist())
        ]

    def _evaluate_same_horizon(
        self,
        trajectories: Sequence[Trajectory],
    ) -> list[TrajectoryEvaluation]:
        torch = self.torch
        horizon = trajectories[0].horizon
        if any(trajectory.horizon != horizon for trajectory in trajectories):
            raise ValueError("torch evaluation batch must have one horizon")

        words = torch.tensor(
            [trajectory.factor_ids for trajectory in trajectories],
            dtype=torch.long,
            device=self.device,
        )
        batch_size = len(trajectories)
        width = self.max_shift * horizon + 1
        state = torch.zeros(
            batch_size,
            3,
            3,
            width,
            dtype=torch.float32,
            device=self.device,
        )
        diagonal = torch.arange(3, device=self.device)
        state[:, diagonal, diagonal, 0] = 1.0
        histories = torch.empty(
            batch_size,
            horizon,
            dtype=torch.int16,
            device=self.device,
        )

        for depth in range(horizon):
            factor_ids = words[:, depth]
            next_state = torch.zeros_like(state)
            for shift in range(self.max_shift + 1):
                right = self.coefficients[factor_ids, shift]
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

        history_rows = histories.cpu().tolist()
        matrix_fingerprints = self._matrix_fingerprints(state)
        evaluations = [
            build_evaluation(
                trajectory=trajectory,
                projlen_history=history,
                config=self.config,
                matrix_fingerprint=matrix_fingerprint,
            )
            for trajectory, history, matrix_fingerprint in zip(
                trajectories,
                history_rows,
                matrix_fingerprints,
            )
        ]

        # Projlen zero is necessary for a projective identity/Delta match.
        # Verify those rare candidates with the exact dictionary arithmetic.
        for index, evaluation in enumerate(evaluations):
            if 0 not in evaluation.projlen_history:
                continue
            evaluations[index] = self.cpu_verifier.evaluate_one(evaluation.trajectory)

        return evaluations

    def evaluate(self, trajectories: Iterable[Trajectory]) -> list[TrajectoryEvaluation]:
        by_horizon: dict[int, list[tuple[int, Trajectory]]] = defaultdict(list)
        trajectory_list = list(trajectories)
        for index, trajectory in enumerate(trajectory_list):
            by_horizon[trajectory.horizon].append((index, trajectory))

        output: list[TrajectoryEvaluation | None] = [None] * len(trajectory_list)
        for indexed_group in by_horizon.values():
            for start in range(0, len(indexed_group), self.config.eval_batch_size):
                chunk = indexed_group[start : start + self.config.eval_batch_size]
                chunk_evaluations = self._evaluate_same_horizon(
                    [trajectory for _, trajectory in chunk]
                )
                for (original_index, _), evaluation in zip(chunk, chunk_evaluations):
                    output[original_index] = evaluation

        if any(evaluation is None for evaluation in output):
            raise AssertionError("torch evaluator failed to fill every output slot")
        return [evaluation for evaluation in output if evaluation is not None]


def make_evaluator(config: SearchConfig):
    if config.backend == "cpu":
        return CPUTrajectoryEvaluator(config)
    if config.backend == "torch":
        return TorchTrajectoryEvaluator(config)
    raise ValueError(f"unknown backend: {config.backend}")
