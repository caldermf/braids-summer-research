from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from crispr_transformer.braid_data import (
    burau_mod_p_polynomial_matrix,
    factor_ids_to_artin_word,
    polynomial_matrix_projlen,
    projective_kernel_match,
)
from crispr_transformer.exact import Evaluation, require_compatible_cuda

from .bootstrap import ensure_author_peyl


SIGMA_FACTOR_IDS = {1: 6, 2: 2, 3: 1}


def commutator_artin_word(
    factor_ids: Sequence[int], generator_index: int, n: int = 4
) -> tuple[int, ...]:
    """Return the Artin word for [sigma_i, g^-1]."""
    if generator_index not in SIGMA_FACTOR_IDS:
        raise ValueError("generator_index must be 1, 2, or 3")
    positive = tuple(factor_ids_to_artin_word(factor_ids, d=0, n=n))
    inverse = tuple(-value for value in reversed(positive))
    return (generator_index, *inverse, -generator_index, *positive)


def commutator_is_nontrivial(
    factor_ids: Sequence[int], generator_index: int, n: int = 4
) -> bool:
    """Decide braid-group nontriviality exactly using Garside normal form."""
    ensure_author_peyl()
    from peyl.braid import GNF  # type: ignore

    g = GNF(n=n, power=0, factors=tuple(int(value) for value in factor_ids))
    sigma = GNF(n=n, power=0, factors=(SIGMA_FACTOR_IDS[generator_index],))
    commutator = sigma * g.inv() * sigma.inv() * g
    return commutator != GNF.identity(n)


class CPUCommutatorEvaluator:
    """Exact reference evaluator for C_i(g) = [sigma_i, g^-1]."""

    def __init__(self, p: int, generator_index: int, n: int = 4, **_):
        if n != 4:
            raise ValueError("the commutator experiment currently supports B4")
        self.p = int(p)
        self.n = int(n)
        self.generator_index = int(generator_index)

    def evaluate_one(self, factors: Sequence[int]) -> Evaluation:
        word = tuple(int(value) for value in factors)
        history = []
        matches = []
        for depth in range(1, len(word) + 1):
            prefix = word[:depth]
            matrix = burau_mod_p_polynomial_matrix(
                commutator_artin_word(prefix, self.generator_index, self.n),
                p=self.p,
                n=self.n,
            )
            projlen = polynomial_matrix_projlen(matrix)
            nontrivial = None
            if projlen == 0:
                nontrivial = commutator_is_nontrivial(
                    prefix, self.generator_index, self.n
                )
                if not nontrivial:
                    # Prevent the learner from obtaining a perfect score by
                    # moving g into the centralizer of sigma_i. This remains
                    # length-scaled and safely below int16 limits.
                    projlen = 8 * depth
                match = projective_kernel_match(matrix, p=self.p, n=self.n)
                if match.get("matches") and nontrivial:
                    matches.append(
                        {
                            "depth": depth,
                            "objective": "commutator_projlen",
                            "generator_index": self.generator_index,
                            "commutator_nontrivial": True,
                            **match,
                        }
                    )
            history.append(projlen)
        return Evaluation(word, tuple(history), tuple(matches))

    def evaluate(self, words: Iterable[Sequence[int]]) -> list[Evaluation]:
        return [self.evaluate_one(word) for word in words]


class TorchCommutatorEvaluator:
    """Batched GPU evaluator using C_{gb}=T_b C_g M_b."""

    def __init__(
        self,
        p: int,
        generator_index: int,
        n: int = 4,
        device: str = "cuda",
        batch_size: int = 2_000,
        max_length: int = 220,
        degree_multiplier: int = 4,
        required_partition: str = "scavenge_gpu",
        table_path: str | Path | None = None,
        **_,
    ):
        if n != 4:
            raise ValueError("the commutator experiment currently supports B4")
        import torch

        from third_party.commutator_search.braid_search import compute_projlen_batch
        from third_party.commutator_search.commutator_braid_search import (
            CommutatorConfig,
            CommutatorFastPolyMatmul,
            load_tables_for_commutator,
        )

        self.torch = torch
        self.compute_projlen_batch = compute_projlen_batch
        self.p = int(p)
        self.n = int(n)
        self.generator_index = int(generator_index)
        self.device = torch.device(device)
        if self.device.type == "cuda":
            require_compatible_cuda(torch, required_partition)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.cpu_verifier = CPUCommutatorEvaluator(
            p=self.p, n=self.n, generator_index=self.generator_index
        )
        config = CommutatorConfig(
            prime=self.p,
            generator_index=self.generator_index,
            max_length=self.max_length,
            degree_multiplier=int(degree_multiplier),
            device=str(self.device),
            matmul_chunk_size=self.batch_size,
        )
        default_table = (
            Path(__file__).resolve().parents[1]
            / "third_party"
            / "commutator_search"
            / "precomputed_tables"
            / f"tables_B4_r1_p{self.p}.pt"
        )
        table = Path(table_path or default_table)
        simple, twisted, _, _, _ = load_tables_for_commutator(config, str(table))
        self.D = config.degree_window
        self.center = config.center
        self.fast = CommutatorFastPolyMatmul(
            simple, twisted, self.D, self.device
        )

    def _same_length(self, words: list[tuple[int, ...]]) -> list[Evaluation]:
        torch = self.torch
        length = len(words[0])
        if length > self.max_length:
            raise ValueError(
                f"word length {length} exceeds commutator window {self.max_length}"
            )
        tokens = torch.tensor(words, dtype=torch.long, device=self.device)
        state = torch.zeros(
            len(words), 3, 3, self.D, dtype=torch.long, device=self.device
        )
        diagonal = torch.arange(3, device=self.device)
        state[:, diagonal, diagonal, self.center] = 1
        histories = torch.empty(
            len(words), length, dtype=torch.int16, device=self.device
        )
        for depth in range(length):
            expanded = self.fast.commutator_expand_batch(
                state,
                tokens[:, depth],
                self.p,
                chunk_size=self.batch_size,
            )
            state = self.fast.recenter_after_convolution(expanded, self.p)
            # The professor's engine calls support width "projlen", so a
            # monomial has value 1. The transformer code consistently uses
            # max_degree-min_degree, where a monomial has value 0.
            histories[:, depth] = (
                self.compute_projlen_batch(state) - 1
            ).to(torch.int16)

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


def make_commutator_evaluator(
    *,
    p: int,
    n: int,
    generator_index: int,
    backend: str,
    device: str,
    batch_size: int,
    max_length: int,
):
    if backend == "cpu":
        return CPUCommutatorEvaluator(
            p=p, n=n, generator_index=generator_index
        )
    if backend == "torch":
        return TorchCommutatorEvaluator(
            p=p,
            n=n,
            generator_index=generator_index,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
    raise ValueError("backend must be 'cpu' or 'torch'")
