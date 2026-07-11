from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def install_peyl(author_repo: Path) -> None:
    author_repo = author_repo.resolve()
    if not (author_repo / "peyl" / "braid.py").is_file():
        raise FileNotFoundError(f"No peyl package under {author_repo}")
    if str(author_repo) not in sys.path:
        sys.path.insert(0, str(author_repo))


@dataclass(frozen=True)
class JonesSpec:
    n: int = 4
    r: int = 1
    p: int = 5

    @property
    def name(self) -> str:
        return f"jones_two_row_({self.n-self.r},{self.r})_B{self.n}_F{self.p}"


class JonesAdapter:
    """Thin adapter around the established peyl evaluator; no algebra is reimplemented."""

    def __init__(self, author_repo: Path, spec: JonesSpec):
        install_peyl(author_repo)
        from peyl.braidsearch import JonesSummand, evaluate_prefixes_of_same_length
        from peyl import polymat

        self.spec = spec
        self.rep = JonesSummand(n=spec.n, r=spec.r, p=spec.p)
        self._evaluate_prefixes = evaluate_prefixes_of_same_length
        self.polymat = polymat

    @property
    def dimension(self) -> int:
        return self.rep.dimension()

    def evaluate_prefixes(self, braids):
        return self._evaluate_prefixes(self.rep, braids)

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        image = self.polymat.projectivise(np.asarray(image)) % self.spec.p
        if image.ndim != 3 or image.shape[:2] != (self.dimension, self.dimension):
            raise ValueError(f"Unexpected Jones image shape {image.shape}")
        if not np.any(image):
            raise ValueError("Representation image is zero")
        return image.astype(np.int16, copy=False)

    def projlen(self, image: np.ndarray) -> int:
        # peyl's historical projlen is the occupied slot count. Research terminology
        # uses deg-val, hence subtract one after projectivisation.
        normalized = self.normalize_image(image)
        return int(normalized.shape[-1] - 1)

    def degree_major(self, image: np.ndarray) -> list:
        return np.moveaxis(self.normalize_image(image), -1, 0).tolist()

