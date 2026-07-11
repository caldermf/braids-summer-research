from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Sequence

from crispr_transformer.gnf import GNFAutomaton

from .minimal_form import factor_ids_to_minimal_word, minimal_word_to_blocks


@dataclass(frozen=True)
class DattaDefect:
    block_index: int
    condition: str
    values: tuple[int, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["values"] = list(self.values)
        return payload


@dataclass(frozen=True)
class DattaAnalysis:
    minimal_word: tuple[int, ...]
    blocks: tuple[tuple[int, int, int], ...]
    defects: tuple[DattaDefect, ...]

    @property
    def is_normal(self) -> bool:
        return not self.defects

    @property
    def is_exceptional(self) -> bool:
        return bool(self.defects)

    def to_dict(self, include_word: bool = True) -> dict:
        result = {
            "is_normal": self.is_normal,
            "is_exceptional": self.is_exceptional,
            "defect_count": len(self.defects),
            "defect_conditions": [defect.condition for defect in self.defects],
            "blocks": [list(block) for block in self.blocks],
        }
        if include_word:
            result["minimal_word"] = list(self.minimal_word)
        return result


def normality_defects(
    blocks: Sequence[tuple[int, int, int]],
) -> tuple[DattaDefect, ...]:
    """Evaluate Definition 1.3 in Datta, arXiv:2209.10826v1.

    Indices in the paper are one-based. Conditions only involve p < n, so the
    final block is never used as the left-hand block of a condition.
    """
    defects: list[DattaDefect] = []
    for index in range(max(0, len(blocks) - 1)):
        a, b, c = blocks[index]
        next_a, _, next_c = blocks[index + 1]

        if a == 0 and b == 1 and c in (1, 2):
            defects.append(DattaDefect(index, "1.3(i)-local-c", (a, b, c)))
        if a == 0 and b == 1 and next_a + 1 >= c and next_c == 1:
            defects.append(
                DattaDefect(
                    index,
                    "1.3(i)-next-c",
                    (a, b, c, next_a, next_c),
                )
            )
        if b > 0 and a >= b + 1 and c == 1 and next_a == 2:
            defects.append(
                DattaDefect(
                    index,
                    "1.3(ii)-unbalanced",
                    (a, b, c, next_a),
                )
            )
        if a == 1 and b == 1 and c == 1 and next_a == 2:
            defects.append(
                DattaDefect(
                    index,
                    "1.3(ii)-111",
                    (a, b, c, next_a),
                )
            )
    return tuple(defects)


@lru_cache(maxsize=250_000)
def _analyze_cached(factor_ids: tuple[int, ...]) -> DattaAnalysis:
    word = factor_ids_to_minimal_word(factor_ids)
    blocks = minimal_word_to_blocks(word)
    return DattaAnalysis(word, blocks, normality_defects(blocks))


def analyze_factor_ids(factor_ids: Sequence[int]) -> DattaAnalysis:
    return _analyze_cached(tuple(int(value) for value in factor_ids))


def exceptionality_persistence(
    factor_ids: Sequence[int], automaton: GNFAutomaton | None = None
) -> dict:
    factors = tuple(int(value) for value in factor_ids)
    if not factors:
        raise ValueError("persistence requires a nonempty GNF word")
    graph = automaton or GNFAutomaton(n=4)
    successors = graph.successors[factors[-1]]
    exceptional = [
        successor
        for successor in successors
        if analyze_factor_ids((*factors, successor)).is_exceptional
    ]
    return {
        "legal_successors": len(successors),
        "exceptional_successors": len(exceptional),
        "exceptionality_persistence": len(exceptional) / len(successors),
        "exceptional_factor_ids": exceptional,
    }
