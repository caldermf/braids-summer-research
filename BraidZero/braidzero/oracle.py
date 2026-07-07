from __future__ import annotations

import random
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from .core import BraidEnvironment, Fingerprint, MatrixTuple


@dataclass(frozen=True)
class ShadowRecord:
    record_id: int
    factors: tuple[int, ...]
    first: int
    last: int
    length: int
    source: str


class ShadowOracle:
    """
    A finite-shadow bank indexed by projective specialized matrix tuples.

    The same bank is used in two ways:
    - collision query: find partner words v with S(v) = S(u);
    - scalar-completion query: find suffixes s with S(s) = S(u)^(-1).
    """

    def __init__(
        self,
        *,
        env: BraidEnvironment,
        bank_length: int,
        records: Sequence[ShadowRecord],
        index: dict[Fingerprint, list[ShadowRecord]],
        metadata: dict,
    ):
        self.env = env
        self.bank_length = int(bank_length)
        self.records = tuple(records)
        self.index = index
        self.metadata = dict(metadata)

    @classmethod
    def build(
        cls,
        *,
        env: BraidEnvironment,
        bank_length: int,
        mode: str,
        samples: int,
        seed: int,
        max_exhaustive: int,
        max_records_per_key: int,
        progress_interval_seconds: float = 30.0,
    ) -> "ShadowOracle":
        if bank_length <= 0:
            raise ValueError("bank_length must be positive")
        if mode not in {"auto", "exhaustive", "random"}:
            raise ValueError("mode must be auto, exhaustive, or random")

        start_time = time.time()
        rng = random.Random(seed)
        total_normal_forms = env.count_normal_forms(bank_length)
        use_exhaustive = mode == "exhaustive" or (
            mode == "auto" and total_normal_forms <= max_exhaustive
        )
        if mode == "exhaustive" and total_normal_forms > max_exhaustive:
            raise ValueError(
                f"refusing exhaustive bank of {total_normal_forms:,} forms; "
                f"increase --max-exhaustive-bank or use --bank-mode random"
            )

        def iter_random_unique():
            seen: set[tuple[int, ...]] = set()
            attempts = 0
            max_attempts = max(samples * 100, 1000)
            while len(seen) < samples and attempts < max_attempts:
                attempts += 1
                factors = env.sample_normal_form(bank_length, rng)
                if factors in seen:
                    continue
                seen.add(factors)
                yield factors

        if use_exhaustive:
            source_iter = env.normal_forms(bank_length)
            generation_mode = "exhaustive"
            requested = total_normal_forms
        else:
            source_iter = iter_random_unique()
            generation_mode = "random"
            requested = samples

        index: dict[Fingerprint, list[ShadowRecord]] = defaultdict(list)
        records: list[ShadowRecord] = []
        capped = 0
        finite_singular = 0
        last_progress = start_time

        for factors in source_iter:
            try:
                key = env.finite_key(env.finite_evaluate(factors))
            except ValueError:
                finite_singular += 1
                continue

            bucket = index[key]
            if max_records_per_key > 0 and len(bucket) >= max_records_per_key:
                capped += 1
                continue
            record = ShadowRecord(
                record_id=len(records),
                factors=tuple(int(x) for x in factors),
                first=int(factors[0]),
                last=int(factors[-1]),
                length=len(factors),
                source=generation_mode,
            )
            records.append(record)
            bucket.append(record)

            now = time.time()
            if now - last_progress >= progress_interval_seconds:
                print(
                    json.dumps(
                        {
                        "phase": "oracle_build",
                        "records": len(records),
                        "keys": len(index),
                        "capped": capped,
                        "elapsed_seconds": round(now - start_time, 2),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                last_progress = now

        bucket_sizes = [len(bucket) for bucket in index.values()]
        first_factor_counts = Counter(record.first for record in records)
        metadata = {
            "bank_length": bank_length,
            "bank_mode_requested": mode,
            "bank_mode_actual": generation_mode,
            "requested_records": requested,
            "total_normal_forms": total_normal_forms,
            "records": len(records),
            "keys": len(index),
            "finite_singular": finite_singular,
            "capped_records": capped,
            "max_records_per_key": max_records_per_key,
            "max_bucket_size": max(bucket_sizes) if bucket_sizes else 0,
            "mean_bucket_size": (sum(bucket_sizes) / len(bucket_sizes)) if bucket_sizes else 0.0,
            "first_factor_counts": dict(sorted(first_factor_counts.items())),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        return cls(env=env, bank_length=bank_length, records=records, index=dict(index), metadata=metadata)

    def count_key(self, key: Fingerprint) -> int:
        return len(self.index.get(key, ()))

    def collision_partners(self, key: Fingerprint, *, limit: int) -> tuple[int, tuple[ShadowRecord, ...]]:
        records = tuple(self.index.get(key, ()))
        total = len(records)
        if limit > 0:
            records = records[:limit]
        return total, records

    def scalar_suffixes(
        self,
        prefix_matrices: MatrixTuple,
        *,
        legal_first: Sequence[int],
        limit: int,
    ) -> tuple[int, tuple[ShadowRecord, ...]]:
        needed = self.env.finite_key(self.env.finite_inverse(prefix_matrices))
        legal = set(int(x) for x in legal_first)
        records = tuple(record for record in self.index.get(needed, ()) if record.first in legal)
        total = len(records)
        if limit > 0:
            records = records[:limit]
        return total, records
