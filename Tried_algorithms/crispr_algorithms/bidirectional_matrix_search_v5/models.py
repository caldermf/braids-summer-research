from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Segment:
    factor_ids: tuple[int, ...]
    role: str
    segment_id: str
    origin: str = "random"
    parent_id: str | None = None

    @property
    def length(self) -> int:
        return len(self.factor_ids)


@dataclass(frozen=True)
class JoinCandidate:
    prefix_index: int
    suffix_index: int
    target_type: str
    sketch_distance: int


@dataclass
class WordEvaluation:
    factor_ids: tuple[int, ...]
    projlen_history: tuple[int, ...]
    final_projlen: int
    min_projlen: int
    peak_projlen: int
    largest_drop: int
    kernel_matches: tuple[dict[str, Any], ...] = ()

    @property
    def has_kernel(self) -> bool:
        return bool(self.kernel_matches)


@dataclass
class JoinEvaluation:
    candidate: JoinCandidate
    prefix: Segment
    suffix: Segment
    word: WordEvaluation

    @property
    def factor_ids(self) -> tuple[int, ...]:
        return self.prefix.factor_ids + self.suffix.factor_ids

    def rank(self) -> tuple:
        return (
            1 if self.word.has_kernel else 0,
            -self.word.final_projlen,
            -self.candidate.sketch_distance,
            self.word.largest_drop,
            -len(self.factor_ids),
        )

    def algebra_rank(self) -> tuple:
        return (
            1 if self.word.has_kernel else 0,
            -self.candidate.sketch_distance,
            -self.word.final_projlen,
            self.word.largest_drop,
            -len(self.factor_ids),
        )

    def summary(self) -> dict:
        return {
            "prefix_id": self.prefix.segment_id,
            "suffix_id": self.suffix.segment_id,
            "prefix_length": self.prefix.length,
            "suffix_length": self.suffix.length,
            "horizon": len(self.factor_ids),
            "target_type": self.candidate.target_type,
            "sketch_distance": self.candidate.sketch_distance,
            "final_projlen": self.word.final_projlen,
            "min_projlen": self.word.min_projlen,
            "peak_projlen": self.word.peak_projlen,
            "largest_drop": self.word.largest_drop,
            "kernel_matches": list(self.word.kernel_matches),
            "factor_ids": list(self.factor_ids),
            "projlen_history": list(self.word.projlen_history),
        }
