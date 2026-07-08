from __future__ import annotations

import gzip
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
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

    @staticmethod
    def _cache_signature(env: BraidEnvironment, bank_length: int) -> dict:
        return {
            "n": env.n,
            "r": env.r,
            "p": env.p,
            "t_values": list(env.t_values),
            "bank_length": int(bank_length),
            "dim": env.dim,
            "representation": env.representation_label,
            "verifier_version": env.verifier_version,
        }

    @staticmethod
    def _open_cache(path: Path, mode: str):
        if path.suffix == ".gz":
            return gzip.open(path, mode, encoding="utf-8")
        return path.open(mode, encoding="utf-8")

    @staticmethod
    def _stable_key_shard(key: Fingerprint, shard_count: int) -> int:
        encoded = json.dumps(list(key), separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha1(encoded).digest()
        return int.from_bytes(digest[:8], "big") % shard_count

    def save_cache(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record_keys: dict[int, Fingerprint] = {}
        for key, bucket in self.index.items():
            for record in bucket:
                record_keys[record.record_id] = key

        header = {
            "format": "braidzero-shadow-bank-cache-v1",
            "kind": "header",
            "signature": self._cache_signature(self.env, self.bank_length),
            "metadata": self.metadata,
            "records": len(self.records),
        }
        with self._open_cache(path, "wt") as handle:
            handle.write(json.dumps(header, sort_keys=True) + "\n")
            for record in self.records:
                key = record_keys.get(record.record_id)
                if key is None:
                    continue
                handle.write(
                    json.dumps(
                        {
                            "kind": "record",
                            "record_id": record.record_id,
                            "factors": list(record.factors),
                            "key": list(key),
                            "source": record.source,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    @classmethod
    def load_cache(
        cls,
        *,
        env: BraidEnvironment,
        path: Path,
        max_records_per_key: int,
        shard_count: int = 1,
        shard_index: int = 0,
        shard_by: str = "none",
    ) -> "ShadowOracle":
        path = Path(path)
        if shard_by not in {"none", "record", "key"}:
            raise ValueError("bank shard mode must be none, record, or key")
        shard_count = int(shard_count)
        shard_index = int(shard_index)
        if shard_count < 1:
            raise ValueError("--bank-shard-count must be at least 1")
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError("--bank-shard-index must be in [0, bank_shard_count)")

        start_time = time.time()
        index: dict[Fingerprint, list[ShadowRecord]] = defaultdict(list)
        records: list[ShadowRecord] = []
        capped = 0
        source_records = 0
        selected_source_records = 0
        first_factor_counts: Counter[int] = Counter()

        with cls._open_cache(path, "rt") as handle:
            first_line = handle.readline()
            if not first_line:
                raise ValueError(f"empty shadow bank cache: {path}")
            header = json.loads(first_line)
            if header.get("format") != "braidzero-shadow-bank-cache-v1":
                raise ValueError(f"unsupported shadow bank cache format in {path}")
            expected = cls._cache_signature(env, int(header["signature"]["bank_length"]))
            if header.get("signature") != expected:
                raise ValueError(
                    "shadow bank cache signature does not match this run: "
                    f"cache={header.get('signature')} expected={expected}"
                )
            bank_length = int(header["signature"]["bank_length"])

            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("kind") != "record":
                    continue
                source_record_id = int(payload["record_id"])
                source_records += 1
                key = tuple(int(x) for x in payload["key"])
                if shard_by == "record" and source_record_id % shard_count != shard_index:
                    continue
                if shard_by == "key" and cls._stable_key_shard(key, shard_count) != shard_index:
                    continue

                selected_source_records += 1
                bucket = index[key]
                if max_records_per_key > 0 and len(bucket) >= max_records_per_key:
                    capped += 1
                    continue
                factors = tuple(int(x) for x in payload["factors"])
                record = ShadowRecord(
                    record_id=len(records),
                    factors=factors,
                    first=int(factors[0]),
                    last=int(factors[-1]),
                    length=len(factors),
                    source=f"cache:{payload.get('source', 'unknown')}",
                )
                records.append(record)
                bucket.append(record)
                first_factor_counts[record.first] += 1

        bucket_sizes = [len(bucket) for bucket in index.values()]
        source_metadata = header.get("metadata", {})
        metadata = {
            **source_metadata,
            "bank_mode_actual": "cache",
            "bank_cache_path": str(path),
            "bank_cache_source_records": source_records,
            "bank_cache_selected_source_records": selected_source_records,
            "bank_shard_by": shard_by,
            "bank_shard_count": shard_count,
            "bank_shard_index": shard_index,
            "records": len(records),
            "keys": len(index),
            "capped_records": capped,
            "max_records_per_key": max_records_per_key,
            "max_bucket_size": max(bucket_sizes) if bucket_sizes else 0,
            "mean_bucket_size": (sum(bucket_sizes) / len(bucket_sizes)) if bucket_sizes else 0.0,
            "first_factor_counts": dict(sorted(first_factor_counts.items())),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        return cls(env=env, bank_length=bank_length, records=records, index=dict(index), metadata=metadata)

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
        target_matrices: MatrixTuple | None = None,
        limit: int,
    ) -> tuple[int, tuple[ShadowRecord, ...]]:
        target_matrices = target_matrices if target_matrices is not None else self.env.identity_finite
        needed = self.env.finite_key(
            self.env.finite_mul(self.env.finite_inverse(prefix_matrices), target_matrices)
        )
        legal = set(int(x) for x in legal_first)
        records = tuple(record for record in self.index.get(needed, ()) if record.first in legal)
        total = len(records)
        if limit > 0:
            records = records[:limit]
        return total, records
