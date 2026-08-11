from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")


def parse_csv_ints(text: str | None, default: Sequence[int]) -> tuple[int, ...]:
    if not text:
        return tuple(int(x) for x in default)
    out = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    return out or tuple(int(x) for x in default)


def parse_csv(text: str | None, default: Sequence[str]) -> tuple[str, ...]:
    if not text:
        return tuple(default)
    out = tuple(part.strip() for part in text.split(",") if part.strip())
    return out or tuple(default)


def canonical_key(braid) -> tuple[int, tuple[int, ...]]:
    return int(braid.inf()), tuple(int(x) for x in braid.factors)


def key_digest(key: tuple[int, tuple[int, ...]], n: int) -> str:
    payload = {"n": int(n), "infimum": int(key[0]), "factor_ids": list(key[1])}
    return hashlib.sha1(
        ("mod-synthesis-gnf-v1:" + json.dumps(payload, sort_keys=True, separators=(",", ":"))).encode("utf-8")
    ).hexdigest()


def artifact_checksum(output_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "summary.json":
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class Parent:
    braid: object
    source: str
    label: str
    seed_projlen: int | None = None
    seed_identity_defect: int | None = None


@dataclass(frozen=True)
class EvalRecord:
    braid: object
    source: str
    operation: str
    parents: tuple[str, ...]
    image: np.ndarray
    projlen: int
    identity_defect: int
    scalar_identity: bool
    delta_match: bool
    delta_defect: int
    nonzero_terms: int
    image_digest: str
    residual_norm: int
    residual_sig: tuple[tuple[int, int, int, int], ...]

    @property
    def key(self) -> tuple[int, tuple[int, ...]]:
        return canonical_key(self.braid)


class ModSynthesis:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = self.output_dir / "candidates.jsonl"
        self.progress_path = self.output_dir / "progress.jsonl"
        self.best_path = self.output_dir / "best.jsonl"
        if args.overwrite:
            for path in (self.candidates_path, self.progress_path, self.best_path):
                if path.exists():
                    path.unlink()

        root = Path(__file__).resolve().parents[2]
        bz = root / "BraidZero"
        if str(bz) not in sys.path:
            sys.path.insert(0, str(bz))
        from braidzero.core import BraidEnvironment  # type: ignore

        self.env = BraidEnvironment(
            author_repo=Path(args.author_repo),
            n=args.n,
            r=args.r,
            p=args.p,
            t_values=tuple(range(1, args.p)),
        )
        from peyl.braidsearch import evaluate_braid_factors  # type: ignore

        self.evaluate_braid_factors = evaluate_braid_factors
        self.GNF = self.env.GNF
        self.rng = random.Random(args.seed)
        self.seen_eval_keys: set[tuple[int, tuple[int, ...], str]] = set()
        self.seen_candidate_keys: set[tuple[int, tuple[int, ...]]] = set()
        self.exact_evaluations = 0
        self.synthesized = 0
        self.candidates = 0

    def braid_from_factors(self, factors: Sequence[int], infimum: int = 0, *, normalize: bool = True):
        factors = tuple(int(x) for x in factors)
        if not normalize and self.env.nf_table.is_factors_normalised(factors):
            return self.GNF(self.env.n, int(infimum), factors)
        braid = self.GNF(self.env.n, int(infimum), ())
        table = self.env.nf_table
        for factor in factors:
            if factor == table.id:
                simple = self.GNF(self.env.n, 0, ())
            elif factor == table.D:
                simple = self.GNF(self.env.n, 1, ())
            else:
                simple = self.GNF(self.env.n, 0, (int(factor),))
            braid = braid * simple
        return braid

    def add_parent(self, parents: dict[tuple[int, tuple[int, ...]], Parent], braid, *, source: str, label: str, projlen=None, identity_defect=None) -> None:
        key = canonical_key(braid)
        if key in parents:
            return
        parents[key] = Parent(
            braid=braid,
            source=source,
            label=label,
            seed_projlen=None if projlen is None else int(projlen),
            seed_identity_defect=None if identity_defect is None else int(identity_defect),
        )

    def load_jsonl_parents(self) -> dict[tuple[int, tuple[int, ...]], Parent]:
        parents: dict[tuple[int, tuple[int, ...]], Parent] = {}
        paths: list[Path] = []
        for pattern in self.args.parent_jsonl or []:
            matches = [Path(p) for p in glob.glob(pattern)]
            paths.extend(matches if matches else [Path(pattern)])
        keys = ("factor_ids", "precursor_factor_ids", "power_factor_ids_raw")
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for index, line in enumerate(handle):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    for key in keys:
                        if key not in row:
                            continue
                        try:
                            factors = [int(x) for x in row[key]]
                            if not factors:
                                continue
                            braid = self.braid_from_factors(
                                factors,
                                int(row.get("infimum", row.get("garside_power", 0)) or 0),
                                normalize=True,
                            )
                        except Exception:
                            continue
                        label = f"{path}:{index}:{key}"
                        self.add_parent(
                            parents,
                            braid,
                            source=str(path),
                            label=label,
                            projlen=row.get("projlen", row.get("precursor_projlen", row.get("power_projlen"))),
                            identity_defect=row.get("identity_defect", row.get("precursor_identity_defect", row.get("power_identity_defect"))),
                        )
                        if len(parents) >= self.args.max_parents:
                            return parents
        return parents

    def load_sqlite_parents(self) -> dict[tuple[int, tuple[int, ...]], Parent]:
        parents: dict[tuple[int, tuple[int, ...]], Parent] = {}
        if not self.args.sqlite_db:
            return parents
        db = Path(self.args.sqlite_db)
        if not db.exists():
            return parents
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=120)
        conn.execute("PRAGMA busy_timeout=120000")
        clauses = ["i.p=?", "i.n=?", "i.r=?", "b.factor_ids_json IS NOT NULL"]
        params: list[object] = [self.args.p, self.args.n, self.args.r]
        if self.args.min_length is not None:
            clauses.append("b.length >= ?")
            params.append(int(self.args.min_length))
        if self.args.max_length is not None:
            clauses.append("b.length <= ?")
            params.append(int(self.args.max_length))
        if self.args.min_projlen is not None:
            clauses.append("i.projlen >= ?")
            params.append(int(self.args.min_projlen))
        if self.args.max_projlen is not None:
            clauses.append("i.projlen <= ?")
            params.append(int(self.args.max_projlen))
        order = "i.projlen ASC, i.identity_defect ASC, b.length ASC"
        if self.args.parent_order == "random":
            order = "abs(random())"
        elif self.args.parent_order == "long":
            order = "b.length DESC, i.projlen ASC"
        query = f"""
            SELECT b.factor_ids_json, b.infimum, i.projlen, i.identity_defect, b.braid_digest
            FROM projlen_images i
            JOIN braids b ON b.braid_digest = i.braid_digest
            WHERE {' AND '.join(clauses)}
            ORDER BY {order}
            LIMIT {int(self.args.max_parents)}
        """
        for factors_json, infimum, projlen, identity_defect, digest in conn.execute(query, params):
            try:
                braid = self.braid_from_factors(json.loads(factors_json), int(infimum or 0), normalize=True)
            except Exception:
                continue
            self.add_parent(
                parents,
                braid,
                source=str(db),
                label=str(digest),
                projlen=projlen,
                identity_defect=identity_defect,
            )
        conn.close()
        return parents

    def load_lake_parents(self) -> dict[tuple[int, tuple[int, ...]], Parent]:
        parents: dict[tuple[int, tuple[int, ...]], Parent] = {}
        if not self.args.lake_root:
            return parents
        try:
            import duckdb  # type: ignore
        except Exception:
            return parents
        lake_root = Path(self.args.lake_root)
        pattern = lake_root / f"p={self.args.p}" / f"n={self.args.n}_r={self.args.r}" / "**" / "*.parquet"
        matches = glob.glob(str(pattern), recursive=True)
        if not matches:
            return parents
        conn = duckdb.connect(database=":memory:")
        conn.execute("SET threads=1")
        conn.execute("SET memory_limit='3GB'")
        expr = "read_parquet(?, union_by_name=true)"
        clauses = ["p = ?", "n = ?", "r = ?", "factor_ids_json IS NOT NULL"]
        params: list[object] = [str(pattern), self.args.p, self.args.n, self.args.r]
        if self.args.min_length is not None:
            clauses.append("length >= ?")
            params.append(int(self.args.min_length))
        if self.args.max_length is not None:
            clauses.append("length <= ?")
            params.append(int(self.args.max_length))
        if self.args.min_projlen is not None:
            clauses.append("projlen >= ?")
            params.append(int(self.args.min_projlen))
        if self.args.max_projlen is not None:
            clauses.append("projlen <= ?")
            params.append(int(self.args.max_projlen))
        seed_text = str(int(self.args.seed))
        order = f"projlen ASC NULLS LAST, identity_defect ASC NULLS LAST, hash(braid_digest || '{seed_text}')"
        if self.args.parent_order == "random":
            order = f"hash(braid_digest || '{seed_text}')"
        elif self.args.parent_order == "long":
            order = f"length DESC, projlen ASC NULLS LAST, hash(braid_digest || '{seed_text}')"
        query = f"""
            SELECT
              braid_digest,
              any_value(factor_ids_json) AS factor_ids_json,
              min(length) AS length,
              min(projlen) AS projlen,
              min(identity_defect) AS identity_defect
            FROM {expr}
            WHERE {' AND '.join(clauses)}
            GROUP BY braid_digest
            ORDER BY {order}
            LIMIT {int(self.args.max_parents)}
        """
        for digest, factors_json, _length, projlen, identity_defect in conn.execute(query, params).fetchall():
            try:
                braid = self.braid_from_factors(json.loads(factors_json), 0, normalize=True)
            except Exception:
                continue
            self.add_parent(
                parents,
                braid,
                source=str(lake_root),
                label=str(digest),
                projlen=projlen,
                identity_defect=identity_defect,
            )
        conn.close()
        return parents

    def load_bootstrap_parents(self) -> dict[tuple[int, tuple[int, ...]], Parent]:
        parents: dict[tuple[int, tuple[int, ...]], Parent] = {}
        if self.args.bootstrap_length <= 0 and self.args.random_parents <= 0:
            return parents
        B = self.env.peyl.BraidGroup(self.env.n)
        for length in range(1, self.args.bootstrap_length + 1):
            for braid in B.all_of_garside_length(length):
                self.add_parent(parents, braid, source="bootstrap", label=f"bootstrap:L{length}")
                if len(parents) >= self.args.max_parents:
                    return parents
        for _ in range(self.args.random_parents):
            length = self.rng.randint(max(1, self.args.random_min_length), max(1, self.args.random_max_length))
            braid = B.sample_braid_perm(length, rand=self.rng)
            self.add_parent(parents, braid, source="random", label=f"random:L{length}")
            if len(parents) >= self.args.max_parents:
                return parents
        return parents

    def load_parents(self) -> list[Parent]:
        merged: dict[tuple[int, tuple[int, ...]], Parent] = {}
        for loader in (
            self.load_jsonl_parents,
            self.load_sqlite_parents,
            self.load_lake_parents,
            self.load_bootstrap_parents,
        ):
            for key, parent in loader().items():
                merged.setdefault(key, parent)
                if len(merged) >= self.args.max_parents:
                    break
            if len(merged) >= self.args.max_parents:
                break
        parents = list(merged.values())
        if self.args.parent_order == "random":
            self.rng.shuffle(parents)
        else:
            parents.sort(
                key=lambda parent: (
                    10**9 if parent.seed_projlen is None else parent.seed_projlen,
                    10**9 if parent.seed_identity_defect is None else parent.seed_identity_defect,
                    parent.braid.garside_length(),
                    key_digest(canonical_key(parent.braid), self.env.n),
                )
            )
        return parents[: self.args.max_parents]

    def residual_signature(self, image: np.ndarray) -> tuple[tuple[int, int, int, int], ...]:
        projected = self.env.polymat.projectivise(image)
        dim = int(projected.shape[0])
        entries: list[tuple[int, int, int, int]] = []
        for row in range(dim):
            for col in range(dim):
                if row == col:
                    continue
                for degree, coeff in enumerate(projected[row, col, :]):
                    value = int(coeff) % self.env.p
                    if value:
                        entries.append((row, col, degree, value))
        base = projected[0, 0, :]
        for row in range(1, dim):
            width = max(base.shape[0], projected[row, row, :].shape[0])
            for degree in range(width):
                left = int(projected[row, row, degree]) if degree < projected.shape[-1] else 0
                right = int(base[degree]) if degree < base.shape[0] else 0
                value = (left - right) % self.env.p
                if value:
                    entries.append((row, row, degree, value))
        return tuple(sorted(entries))

    def negative_signature(self, sig: tuple[tuple[int, int, int, int], ...]) -> tuple[tuple[int, int, int, int], ...]:
        p = self.env.p
        return tuple((row, col, deg, (-coeff) % p) for row, col, deg, coeff in sig)

    def evaluate(self, braid, *, operation: str, source: str, parents: Sequence[str]) -> EvalRecord | None:
        key = (int(braid.inf()), tuple(int(x) for x in braid.factors), operation)
        if key in self.seen_eval_keys:
            return None
        self.seen_eval_keys.add(key)
        try:
            image = self.evaluate_braid_factors(self.env.rep, braid)
        except Exception as exc:
            append_jsonl(
                self.progress_path,
                {"phase": "eval_error", "operation": operation, "error": repr(exc), "length": int(braid.garside_length())},
            )
            return None
        metrics = self.env.exact_metrics(image)
        delta_metrics = self.env.exact_target_metrics(image, "delta")
        sig = self.residual_signature(image)
        record = EvalRecord(
            braid=braid,
            source=source,
            operation=operation,
            parents=tuple(parents),
            image=image,
            projlen=int(metrics["projlen"]),
            identity_defect=int(metrics["identity_defect"]),
            scalar_identity=bool(metrics["scalar_identity"]),
            delta_match=bool(delta_metrics["target_match"]),
            delta_defect=int(delta_metrics["target_defect"]),
            nonzero_terms=int(metrics["nonzero_terms"]),
            image_digest=self.env.exact_digest(image),
            residual_norm=len(sig),
            residual_sig=sig,
        )
        self.exact_evaluations += 1
        self.check_candidate(record)
        return record

    def candidate_row(self, record: EvalRecord) -> dict:
        key = canonical_key(record.braid)
        return {
            "kind": "mod_synthesis_verified_candidate",
            "n": int(self.env.n),
            "r": int(self.env.r),
            "p": int(self.env.p),
            "representation": self.env.representation_label,
            "operation": record.operation,
            "source": record.source,
            "parent_labels": list(record.parents),
            "infimum": int(key[0]),
            "factor_ids": list(key[1]),
            "garside_length": int(record.braid.garside_length()),
            "artin_word": [int(x) for x in record.braid.magma_artin_word()],
            "word_digest": key_digest(key, self.env.n),
            "image_digest": record.image_digest,
            "projlen": int(record.projlen),
            "identity_defect": int(record.identity_defect),
            "scalar_identity": bool(record.scalar_identity),
            "delta_match": bool(record.delta_match),
            "delta_defect": int(record.delta_defect),
            "nonzero_terms": int(record.nonzero_terms),
            "verifier_version": self.env.verifier_version,
        }

    def check_candidate(self, record: EvalRecord) -> None:
        if not (record.scalar_identity or record.delta_match):
            return
        key = canonical_key(record.braid)
        if key in self.seen_candidate_keys:
            return
        self.seen_candidate_keys.add(key)
        self.candidates += 1
        row = self.candidate_row(record)
        append_jsonl(self.candidates_path, row)
        print(
            "FOUND mod-synthesis candidate "
            f"p={self.env.p} op={record.operation} length={row['garside_length']} "
            f"projlen={record.projlen} identity_defect={record.identity_defect} "
            f"delta_match={record.delta_match}",
            flush=True,
        )

    def best_rows(self, records: Sequence[EvalRecord], limit: int = 30) -> None:
        ordered = sorted(records, key=lambda r: (r.identity_defect, r.delta_defect, r.projlen, r.braid.garside_length()))
        for record in ordered[:limit]:
            append_jsonl(
                self.best_path,
                {
                    "operation": record.operation,
                    "length": int(record.braid.garside_length()),
                    "infimum": int(record.braid.inf()),
                    "factor_ids": list(record.braid.factors),
                    "projlen": int(record.projlen),
                    "identity_defect": int(record.identity_defect),
                    "delta_defect": int(record.delta_defect),
                    "residual_norm": int(record.residual_norm),
                    "image_digest": record.image_digest,
                    "parents": list(record.parents),
                },
            )

    def synthesize_powers(self, parents: Sequence[Parent], exponents: Sequence[int]) -> list[tuple[object, str, tuple[str, ...]]]:
        out = []
        for parent in parents:
            for exp in exponents:
                out.append((parent.braid ** int(exp), f"power_{exp}", (parent.label,)))
        return out

    def synthesize_short_commutators(self, parents: Sequence[Parent]) -> list[tuple[object, str, tuple[str, ...]]]:
        out = []
        table = self.env.nf_table
        simple_braids = []
        for factor in self.env.first_ids:
            if int(factor) not in (table.id, table.D):
                simple_braids.append(self.GNF(self.env.n, 0, (int(factor),)))
        simple_braids = simple_braids[: self.args.short_conjugators]
        for parent in parents[: self.args.unary_pool_size]:
            a = parent.braid
            ainv = a.inv()
            for idx, g in enumerate(simple_braids):
                gin = g.inv()
                comm = a * g * ainv * gin
                out.append((comm, "short_commutator", (parent.label, f"simple_{idx}")))
                if self.args.power_commutators:
                    out.append((comm ** self.args.p, "short_commutator_power_p", (parent.label, f"simple_{idx}")))
        return out

    def parent_pairs(self, records: Sequence[EvalRecord]) -> list[tuple[EvalRecord, EvalRecord]]:
        pool = sorted(records, key=lambda r: (r.identity_defect, r.residual_norm, r.projlen))[: self.args.pair_pool_size]
        pairs = [(pool[i], pool[j]) for i in range(len(pool)) for j in range(i + 1, len(pool))]
        self.rng.shuffle(pairs)
        return pairs[: self.args.max_pairs]

    def synthesize_pairs(self, records: Sequence[EvalRecord]) -> list[tuple[object, str, tuple[str, ...]]]:
        out = []
        ops = set(parse_csv(self.args.operations, ("powers", "collisions", "quotients", "commutators", "residual")))
        for a, b in self.parent_pairs(records):
            A, B = a.braid, b.braid
            labels = (key_digest(a.key, self.env.n), key_digest(b.key, self.env.n))
            if "products" in ops:
                out.append((A * B, "product_ab", labels))
                out.append((B * A, "product_ba", labels))
            if "quotients" in ops:
                out.append((A * B.inv(), "quotient_abinv", labels))
                out.append((B * A.inv(), "quotient_bainv", labels))
            if "commutators" in ops:
                comm = A * B * A.inv() * B.inv()
                out.append((comm, "commutator", labels))
                if self.args.power_commutators:
                    out.append((comm ** self.args.p, "commutator_power_p", labels))
        return out

    def synthesize_collisions(self, records: Sequence[EvalRecord]) -> list[tuple[object, str, tuple[str, ...]]]:
        out = []
        by_digest: dict[str, list[EvalRecord]] = defaultdict(list)
        for record in records:
            by_digest[record.image_digest].append(record)
        for digest, group in by_digest.items():
            if len(group) < 2:
                continue
            for i in range(min(len(group), self.args.collision_group_limit)):
                for j in range(i + 1, min(len(group), self.args.collision_group_limit)):
                    a, b = group[i], group[j]
                    if a.key == b.key:
                        continue
                    labels = (key_digest(a.key, self.env.n), key_digest(b.key, self.env.n))
                    out.append((a.braid * b.braid.inv(), "image_collision_quotient", labels))
                    out.append((b.braid * a.braid.inv(), "image_collision_quotient_reverse", labels))
        return out

    def synthesize_residual_cancellations(self, records: Sequence[EvalRecord]) -> list[tuple[object, str, tuple[str, ...]]]:
        out = []
        by_sig: dict[tuple[tuple[int, int, int, int], ...], list[EvalRecord]] = defaultdict(list)
        for record in records:
            if 0 < record.residual_norm <= self.args.max_residual_norm:
                by_sig[record.residual_sig].append(record)
        used = 0
        for record in sorted(records, key=lambda r: (r.residual_norm, r.identity_defect, r.projlen)):
            if record.residual_norm <= 0 or record.residual_norm > self.args.max_residual_norm:
                continue
            targets = by_sig.get(self.negative_signature(record.residual_sig), [])
            for other in targets[: self.args.residual_pair_limit]:
                if record.key == other.key:
                    continue
                labels = (key_digest(record.key, self.env.n), key_digest(other.key, self.env.n))
                out.append((record.braid * other.braid, "residual_cancel_product", labels))
                out.append((record.braid * other.braid.inv(), "residual_cancel_quotient", labels))
                used += 1
                if used >= self.args.max_residual_pairs:
                    return out
        return out

    def run(self) -> dict:
        started_at = time.time()
        parents = self.load_parents()
        append_jsonl(
            self.progress_path,
            {
                "phase": "parents_loaded",
                "parents": len(parents),
                "p": self.args.p,
                "n": self.args.n,
                "r": self.args.r,
            },
        )
        print(f"Loaded {len(parents)} parents for p={self.args.p}", flush=True)

        records: list[EvalRecord] = []
        for parent in parents:
            record = self.evaluate(parent.braid, operation="parent", source=parent.source, parents=(parent.label,))
            if record is not None:
                records.append(record)
        self.best_rows(records)

        operations = set(parse_csv(self.args.operations, ("powers", "collisions", "quotients", "commutators", "residual")))
        batches: list[tuple[str, list[tuple[object, str, tuple[str, ...]]]]] = []
        if "powers" in operations:
            batches.append(("powers", self.synthesize_powers(parents[: self.args.unary_pool_size], parse_csv_ints(self.args.power_exponents, (self.args.p,)))))
        if "short_commutators" in operations:
            batches.append(("short_commutators", self.synthesize_short_commutators(parents)))
        if "collisions" in operations:
            batches.append(("collisions", self.synthesize_collisions(records)))
        pair_ops = operations.intersection({"products", "quotients", "commutators"})
        if pair_ops:
            batches.append(("pair_ops", self.synthesize_pairs(records)))
        if "residual" in operations:
            batches.append(("residual", self.synthesize_residual_cancellations(records)))

        for phase, jobs in batches:
            if self.args.max_synthesized_per_phase > 0:
                self.rng.shuffle(jobs)
                jobs = jobs[: self.args.max_synthesized_per_phase]
            phase_records = []
            for braid, operation, labels in jobs:
                self.synthesized += 1
                record = self.evaluate(braid, operation=operation, source=phase, parents=labels)
                if record is not None:
                    phase_records.append(record)
            records.extend(phase_records)
            self.best_rows(phase_records)
            append_jsonl(
                self.progress_path,
                {
                    "phase": "synthesis_phase_done",
                    "name": phase,
                    "jobs": len(jobs),
                    "phase_evaluated": len(phase_records),
                    "exact_evaluations": self.exact_evaluations,
                    "candidates": self.candidates,
                    "best_identity_defect": min((r.identity_defect for r in records), default=None),
                    "best_projlen": min((r.projlen for r in records), default=None),
                    "elapsed_seconds": round(time.time() - started_at, 2),
                },
            )
            print(f"Finished {phase}: jobs={len(jobs)} candidates={self.candidates}", flush=True)

        summary = {
            "format": "mod-synthesis-summary-v1",
            "status": "clean",
            "method": "modular_congruence_synthesis",
            "prime": int(self.args.p),
            "n": int(self.args.n),
            "r": int(self.args.r),
            "representation": self.env.representation_label,
            "seed": int(self.args.seed),
            "parents": len(parents),
            "synthesized": int(self.synthesized),
            "exact_evaluations": int(self.exact_evaluations),
            "verified_candidates": int(self.candidates),
            "best_projlen": min((r.projlen for r in records), default=None),
            "best_identity_defect": min((r.identity_defect for r in records), default=None),
            "best_delta_defect": min((r.delta_defect for r in records), default=None),
            "elapsed_seconds": round(time.time() - started_at, 2),
            "candidate_path": str(self.candidates_path),
            "best_path": str(self.best_path),
            "verifier_version": self.env.verifier_version,
        }
        write_json(self.output_dir / "summary.json", {**summary, "artifact_checksum": artifact_checksum(self.output_dir)})
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modular congruence synthesis search for braid kernel candidates.")
    parser.add_argument("--author-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--parent-jsonl", action="append", help="JSONL/glob containing factor_ids or precursor_factor_ids.")
    parser.add_argument("--sqlite-db", help="cross_prime_projlen.sqlite path.")
    parser.add_argument("--lake-root", help="BraidLake root.")
    parser.add_argument("--max-parents", type=int, default=5000)
    parser.add_argument("--parent-order", choices=["projlen", "random", "long"], default="projlen")
    parser.add_argument("--min-length", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--min-projlen", type=int)
    parser.add_argument("--max-projlen", type=int)
    parser.add_argument("--bootstrap-length", type=int, default=0)
    parser.add_argument("--random-parents", type=int, default=0)
    parser.add_argument("--random-min-length", type=int, default=8)
    parser.add_argument("--random-max-length", type=int, default=40)

    parser.add_argument("--operations", default="powers,collisions,quotients,commutators,residual,short_commutators")
    parser.add_argument("--power-exponents", help="Comma list; default is p.")
    parser.add_argument("--unary-pool-size", type=int, default=5000)
    parser.add_argument("--pair-pool-size", type=int, default=300)
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--max-synthesized-per-phase", type=int, default=50000)
    parser.add_argument("--power-commutators", action="store_true")
    parser.add_argument("--short-conjugators", type=int, default=22)
    parser.add_argument("--collision-group-limit", type=int, default=12)
    parser.add_argument("--max-residual-norm", type=int, default=80)
    parser.add_argument("--max-residual-pairs", type=int, default=20000)
    parser.add_argument("--residual-pair-limit", type=int, default=6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = ModSynthesis(args).run()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
