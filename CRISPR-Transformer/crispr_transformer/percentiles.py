from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path

from .io_utils import read_json, write_json


@dataclass(frozen=True)
class LengthPercentiles:
    """Empirical projective-length CDFs indexed by exact braid length."""

    p: int
    n: int
    values: dict[int, tuple[int, ...]]

    def quality(self, length: int, projlen: int) -> float:
        """Return a lower-tail percentile; smaller values are better."""
        sample = self.values.get(int(length))
        if not sample:
            raise KeyError(f"no projective-length baseline for length {length}")
        rank = bisect.bisect_left(sample, int(projlen))
        if rank == 0 and projlen < sample[0]:
            # Preserve resolution below the observed minimum instead of
            # assigning every exceptionally good candidate the same rank.
            floor_rank = 0.5 / (len(sample) + 1.0)
            return floor_rank * (max(0, projlen) + 1.0) / (sample[0] + 1.0)
        return (rank + 0.5) / (len(sample) + 1.0)

    def reward(
        self,
        parent_length: int,
        parent_projlen: int,
        child_length: int,
        child_projlen: int,
    ) -> float:
        return self.quality(parent_length, parent_projlen) - self.quality(
            child_length,
            child_projlen,
        )

    def to_dict(self) -> dict:
        return {
            "format": "crispr-transformer-length-percentiles-v1",
            "p": self.p,
            "n": self.n,
            "values": {str(length): list(values) for length, values in self.values.items()},
        }

    def save(self, path: str | Path) -> Path:
        return write_json(path, self.to_dict())

    @classmethod
    def from_samples(cls, p: int, n: int, samples: dict[int, list[int]]):
        values = {
            int(length): tuple(sorted(int(value) for value in projlens))
            for length, projlens in samples.items()
            if projlens
        }
        if not values:
            raise ValueError("percentile baseline cannot be empty")
        return cls(p=int(p), n=int(n), values=values)

    @classmethod
    def load(cls, path: str | Path):
        payload = read_json(path)
        if payload.get("format") != "crispr-transformer-length-percentiles-v1":
            raise ValueError("not a CRISPR-Transformer percentile baseline")
        return cls.from_samples(
            p=int(payload["p"]),
            n=int(payload["n"]),
            samples={int(key): values for key, values in payload["values"].items()},
        )
