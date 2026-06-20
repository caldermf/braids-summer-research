from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

from .bootstrap import ensure_author_peyl

ensure_author_peyl()

from peyl.braid import GNF  # type: ignore  # noqa: E402


def gnf_from_positive_artin_word(word: Iterable[int]) -> GNF:
    letters = tuple(int(letter) for letter in word)
    if any(letter not in (1, 2, 3) for letter in letters):
        raise ValueError("Datta minimal form requires a positive B4 Artin word")
    return GNF.from_artin_word(4, ((letter - 1, 1) for letter in letters))


def gnf_from_factor_ids(factor_ids: Sequence[int]) -> GNF:
    if not factor_ids:
        return GNF.identity(4)
    return GNF(n=4, power=0, factors=tuple(int(value) for value in factor_ids))


def positive_artin_length(braid: GNF) -> int:
    word = braid.artin_word()
    if any(exponent != 1 for _, exponent in word):
        raise ValueError("braid is not positive")
    return len(word)


@lru_cache(maxsize=250_000)
def minimal_word_from_gnf(power: int, factors: tuple[int, ...]) -> tuple[int, ...]:
    """Return Datta's lexicographically minimal positive Artin word.

    At each step we greedily remove the smallest Artin generator that
    left-divides the remaining positive braid. This is equivalent to choosing
    the lexicographically smallest representative in the positive braid monoid.
    """
    remainder = GNF(n=4, power=int(power), factors=tuple(factors))
    total_length = positive_artin_length(remainder)
    generators = GNF.artin_gens(4)
    output: list[int] = []

    for _ in range(total_length):
        for index, generator in enumerate(generators, start=1):
            quotient = generator.inv() * remainder
            if quotient.inf() >= 0:
                output.append(index)
                remainder = quotient
                break
        else:
            raise RuntimeError("positive braid had no positive Artin left divisor")

    if remainder != GNF.identity(4):
        raise RuntimeError("minimal-word extraction did not consume the braid")
    return tuple(output)


def factor_ids_to_minimal_word(factor_ids: Sequence[int]) -> tuple[int, ...]:
    braid = gnf_from_factor_ids(factor_ids)
    return minimal_word_from_gnf(braid.power, tuple(braid.factors))


def minimal_word_to_blocks(word: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    """Parse prod_i sigma_1^a_i sigma_3^b_i sigma_2^c_i."""
    letters = tuple(int(letter) for letter in word)
    if any(letter not in (1, 2, 3) for letter in letters):
        raise ValueError("minimal word must use the positive B4 generators")

    blocks: list[tuple[int, int, int]] = []
    offset = 0
    while offset < len(letters):
        counts = [0, 0, 0]
        while offset < len(letters) and letters[offset] == 1:
            counts[0] += 1
            offset += 1
        while offset < len(letters) and letters[offset] == 3:
            counts[1] += 1
            offset += 1
        while offset < len(letters) and letters[offset] == 2:
            counts[2] += 1
            offset += 1
        if counts == [0, 0, 0]:
            raise ValueError(f"word is not in Datta minimal block form at offset {offset}")
        blocks.append(tuple(counts))
    return tuple(blocks)
