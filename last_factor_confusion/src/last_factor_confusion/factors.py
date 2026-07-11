from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class FactorTable:
    permutations: tuple[tuple[int, ...], ...]

    @classmethod
    def from_peyl(cls, n: int = 4) -> "FactorTable":
        from peyl.permutations import SymmetricGroup

        group = SymmetricGroup(n)
        identity = tuple(group.id().word)
        delta = tuple(group.longest_element().word)
        proper = sorted(
            tuple(perm.word)
            for perm in group.elements()
            if tuple(perm.word) not in (identity, delta)
        )
        return cls(tuple(proper))

    def class_id(self, permutation) -> int:
        return self.permutations.index(tuple(permutation.word if hasattr(permutation, "word") else permutation))

    def checksum(self) -> str:
        payload = json.dumps(self.permutations, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def as_dict(self) -> dict:
        return {
            "convention": "lexicographic proper nontrivial simple permutations; identity and Delta excluded",
            "classes": [list(x) for x in self.permutations],
            "sha256": self.checksum(),
        }
