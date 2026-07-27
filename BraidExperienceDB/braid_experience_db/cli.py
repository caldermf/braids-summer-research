from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
SUPPORTED_SUFFIXES = {".json", ".jsonl", ".csv"}
FACTOR_KEYS = {
    "factor_ids",
    "child_factors",
    "parent_factors",
    "precursor_factor_ids",
    "power_factor_ids_raw",
    "suffix",
    "quotient_factors",
    "left_suffix",
    "right_suffix",
}
ARTIN_ONLY_KEYS = {"artin_word", "word"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def parse_int_list(value: Any) -> Optional[List[int]]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(value, list):
        return None
    if any(isinstance(item, list) for item in value):
        return None
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return None


def is_factor_list(value: Any) -> bool:
    factors = parse_int_list(value)
    return factors is not None and all(0 <= item <= 10_000 for item in factors)


def infer_family(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        index = parts.index("results")
        if index + 1 < len(parts):
            return parts[index + 1]
    return path.parent.name


def infer_prime_from_text(text: str) -> Optional[int]:
    patterns = [
        r"(?:^|[_/\-])p(\d+)(?:[_/\-]|$)",
        r"(?:^|[_/\-])mod(\d+)(?:[_/\-]|$)",
        r"F_(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def infer_int_from_path(path: Path, prefix: str) -> Optional[int]:
    match = re.search(rf"{re.escape(prefix)}(\d+)", str(path))
    return int(match.group(1)) if match else None


def first_int(*values: Any) -> Optional[int]:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def first_str(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return None


def truthy(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(bool(value))
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "yes", "clean", "1"}:
            return 1
        if lowered in {"false", "no", "malformed", "0"}:
            return 0
    return None


@dataclass
class RunContext:
    run_id: str
    prime: Optional[int]
    n: Optional[int]
    r: Optional[int]
    representation: Optional[str]
    method: str
    run_group: str
    seed: Optional[int]
    task: Optional[str]
    source_dir: str
    config_json: Optional[str]
    summary_json: Optional[str]
    status: Optional[str]


@dataclass
class ExtractedBraid:
    label: str
    factors: List[int]
    source_record_kind: str
    length: int
    infimum: int
    garside_power: int
    n: Optional[int]
    r: Optional[int]
    p: Optional[int]
    projlen: Optional[int]
    identity_defect: Optional[int]
    delta_defect: Optional[int]
    scalar_identity: Optional[int]
    score: Optional[float]
    matrix_digest: Optional[str]
    finite_shadow_digest: Optional[str]
    selected_by_json: Optional[str]
    was_expanded: Optional[int]
    was_exactly_checked: Optional[int]
    verified_kernel: Optional[int]
    status: str
    metrics_json: Optional[str]


class ExperienceDB:
    def __init__(self, path: Path, prime: int):
        self.path = path
        self.prime = prime
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.create_schema()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY,
              prime INTEGER,
              n INTEGER,
              r INTEGER,
              representation TEXT,
              method TEXT,
              run_group TEXT,
              seed INTEGER,
              task TEXT,
              source_dir TEXT,
              config_json TEXT,
              summary_json TEXT,
              status TEXT,
              imported_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS braids (
              braid_digest TEXT PRIMARY KEY,
              n INTEGER,
              infimum INTEGER NOT NULL,
              garside_power INTEGER NOT NULL,
              length INTEGER NOT NULL,
              factor_ids_json TEXT NOT NULL,
              factor_ids_text TEXT NOT NULL,
              first_seen_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
              observation_id TEXT PRIMARY KEY,
              braid_digest TEXT NOT NULL,
              run_id TEXT NOT NULL,
              prime INTEGER NOT NULL,
              n INTEGER,
              r INTEGER,
              source_file TEXT NOT NULL,
              line_number INTEGER,
              record_label TEXT,
              record_kind TEXT,
              method TEXT,
              length INTEGER,
              projlen INTEGER,
              identity_defect INTEGER,
              delta_defect INTEGER,
              scalar_identity INTEGER,
              score REAL,
              matrix_digest TEXT,
              finite_shadow_digest TEXT,
              selected_by_json TEXT,
              was_expanded INTEGER,
              was_exactly_checked INTEGER,
              verified_kernel INTEGER,
              status TEXT,
              metrics_json TEXT,
              observed_at TEXT NOT NULL,
              FOREIGN KEY(braid_digest) REFERENCES braids(braid_digest),
              FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS malformed_records (
              malformed_id TEXT PRIMARY KEY,
              source_file TEXT NOT NULL,
              line_number INTEGER,
              reason TEXT NOT NULL,
              raw_text TEXT,
              imported_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS imports (
              source_file TEXT PRIMARY KEY,
              source_checksum TEXT,
              imported_at TEXT NOT NULL,
              records_seen INTEGER NOT NULL,
              aggregate_records INTEGER NOT NULL,
              observations_inserted INTEGER NOT NULL,
              malformed_records INTEGER NOT NULL,
              status TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_braids_length ON braids(length);
            CREATE INDEX IF NOT EXISTS idx_obs_braid ON observations(braid_digest);
            CREATE INDEX IF NOT EXISTS idx_obs_run ON observations(run_id);
            CREATE INDEX IF NOT EXISTS idx_obs_length_projlen ON observations(length, projlen);
            CREATE INDEX IF NOT EXISTS idx_obs_matrix ON observations(matrix_digest);
            CREATE INDEX IF NOT EXISTS idx_obs_method ON observations(method);
            CREATE INDEX IF NOT EXISTS idx_obs_verified ON observations(verified_kernel);
            """
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
            ("prime", str(self.prime)),
        )
        self.conn.commit()

    def has_clean_import(self, source_file: str, checksum: str) -> bool:
        row = self.conn.execute(
            "SELECT status FROM imports WHERE source_file=? AND source_checksum=?",
            (source_file, checksum),
        ).fetchone()
        return bool(row and row[0] == "clean")

    def upsert_run(self, ctx: RunContext) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO runs
            (run_id, prime, n, r, representation, method, run_group, seed, task,
             source_dir, config_json, summary_json, status, imported_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ctx.run_id,
                self.prime,
                ctx.n,
                ctx.r,
                ctx.representation,
                ctx.method,
                ctx.run_group,
                ctx.seed,
                ctx.task,
                ctx.source_dir,
                ctx.config_json,
                ctx.summary_json,
                ctx.status,
                utc_now(),
            ),
        )

    def braid_digest(self, n: Optional[int], infimum: int, factors: Sequence[int]) -> str:
        payload = {"n": n, "infimum": infimum, "factor_ids": list(factors)}
        return sha1_text("gnf-factor-digest-v1:" + compact_json(payload))

    def insert_observation(
        self,
        *,
        ctx: RunContext,
        braid: ExtractedBraid,
        source_file: str,
        line_number: Optional[int],
    ) -> bool:
        n = braid.n if braid.n is not None else ctx.n
        r = braid.r if braid.r is not None else ctx.r
        digest = self.braid_digest(n, braid.infimum, braid.factors)
        factor_ids_json = compact_json(braid.factors)
        factor_ids_text = ",".join(str(item) for item in braid.factors)
        self.conn.execute(
            """
            INSERT OR IGNORE INTO braids
            (braid_digest, n, infimum, garside_power, length, factor_ids_json,
             factor_ids_text, first_seen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                digest,
                n,
                braid.infimum,
                braid.garside_power,
                braid.length,
                factor_ids_json,
                factor_ids_text,
                utc_now(),
            ),
        )
        obs_payload = {
            "source_file": source_file,
            "line_number": line_number,
            "run_id": ctx.run_id,
            "label": braid.label,
            "digest": digest,
            "kind": braid.source_record_kind,
        }
        observation_id = sha1_text("observation-v1:" + compact_json(obs_payload))
        cur = self.conn.execute(
            """
            INSERT OR IGNORE INTO observations
            (observation_id, braid_digest, run_id, prime, n, r, source_file,
             line_number, record_label, record_kind, method, length, projlen,
             identity_defect, delta_defect, scalar_identity, score, matrix_digest,
             finite_shadow_digest, selected_by_json, was_expanded,
             was_exactly_checked, verified_kernel, status, metrics_json, observed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation_id,
                digest,
                ctx.run_id,
                self.prime,
                n,
                r,
                source_file,
                line_number,
                braid.label,
                braid.source_record_kind,
                ctx.method,
                braid.length,
                braid.projlen,
                braid.identity_defect,
                braid.delta_defect,
                braid.scalar_identity,
                braid.score,
                braid.matrix_digest,
                braid.finite_shadow_digest,
                braid.selected_by_json,
                braid.was_expanded,
                braid.was_exactly_checked,
                braid.verified_kernel,
                braid.status,
                braid.metrics_json,
                utc_now(),
            ),
        )
        return cur.rowcount > 0

    def insert_malformed(
        self, source_file: str, line_number: Optional[int], reason: str, raw_text: Optional[str]
    ) -> bool:
        malformed_id = sha1_text(
            "malformed-v1:" + compact_json(
                {
                    "source_file": source_file,
                    "line_number": line_number,
                    "reason": reason,
                    "raw_text": raw_text,
                }
            )
        )
        cur = self.conn.execute(
            """
            INSERT OR IGNORE INTO malformed_records
            (malformed_id, source_file, line_number, reason, raw_text, imported_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (malformed_id, source_file, line_number, reason, raw_text, utc_now()),
        )
        return cur.rowcount > 0

    def finish_import(
        self,
        source_file: str,
        checksum: str,
        records_seen: int,
        aggregate_records: int,
        observations_inserted: int,
        malformed_records: int,
        status: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO imports
            (source_file, source_checksum, imported_at, records_seen, aggregate_records,
             observations_inserted, malformed_records, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_file,
                checksum,
                utc_now(),
                records_seen,
                aggregate_records,
                observations_inserted,
                malformed_records,
                status,
            ),
        )
        self.conn.commit()


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def find_run_dir(path: Path) -> Path:
    current = path.parent
    for candidate in [current] + list(current.parents):
        if (candidate / "config.json").is_file() or (candidate / "summary.json").is_file():
            return candidate
        if candidate.name == "results":
            break
    return path.parent


def make_context(path: Path, row: Optional[Dict[str, Any]] = None, forced_prime: Optional[int] = None) -> RunContext:
    row = row or {}
    run_dir = find_run_dir(path)
    config = load_json_if_exists(run_dir / "config.json") or {}
    summary = load_json_if_exists(run_dir / "summary.json") or {}
    summary_config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    prime = first_int(
        forced_prime,
        row.get("p"),
        row.get("prime"),
        config.get("p"),
        config.get("prime"),
        summary.get("p"),
        summary.get("prime"),
        summary_config.get("p"),
        infer_prime_from_text(str(path)),
    )
    n = first_int(row.get("n"), config.get("n"), summary.get("n"), summary_config.get("n"), 4)
    r = first_int(row.get("r"), config.get("r"), summary.get("r"), summary_config.get("r"), 1)
    method = first_str(
        row.get("method"),
        row.get("kind"),
        summary.get("method"),
        config.get("method"),
        infer_family(path),
    ) or "unknown"
    if "results" in path.parts:
        index = path.parts.index("results")
        run_group = "/".join(path.parts[index + 1 : min(len(path.parts), index + 3)])
    else:
        run_group = run_dir.name
    seed = first_int(row.get("seed"), config.get("seed"), summary.get("seed"), infer_int_from_path(path, "seed"))
    task = first_str(row.get("task"), row.get("task_id"), run_dir.name if run_dir != path.parent else None)
    representation = first_str(
        row.get("representation"),
        summary.get("representation"),
        config.get("representation"),
        f"JonesSummand(n={n},r={r})" if n is not None and r is not None else None,
    )
    status = first_str(row.get("status"), summary.get("status"), config.get("status"))
    run_id = sha1_text("run-v1:" + str(run_dir.resolve()))
    return RunContext(
        run_id=run_id,
        prime=prime,
        n=n,
        r=r,
        representation=representation,
        method=method,
        run_group=run_group,
        seed=seed,
        task=task,
        source_dir=str(run_dir),
        config_json=compact_json(config) if config else None,
        summary_json=compact_json(summary) if summary else None,
        status=status,
    )


def prefix_for_label(label: str) -> Optional[str]:
    if "precursor" in label:
        return "precursor"
    if "power" in label:
        return "power"
    if "child" in label:
        return "child"
    if "parent" in label:
        return "parent"
    return None


def metric_value(row: Dict[str, Any], local: Dict[str, Any], label: str, base: str) -> Any:
    prefix = prefix_for_label(label)
    candidates: List[str] = []
    if prefix:
        candidates.append(f"{prefix}_{base}")
    candidates.append(base)
    if base == "projlen":
        candidates.append("projective_width")
    for source in (local, row):
        metrics = source.get("metrics") if isinstance(source.get("metrics"), dict) else {}
        for key in candidates:
            if key in source:
                return source[key]
            if key in metrics:
                return metrics[key]
    return None


def int_metric(row: Dict[str, Any], local: Dict[str, Any], label: str, base: str) -> Optional[int]:
    value = metric_value(row, local, label, base)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def float_metric(row: Dict[str, Any], local: Dict[str, Any], label: str, base: str) -> Optional[float]:
    value = metric_value(row, local, label, base)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_selected_by(row: Dict[str, Any], local: Dict[str, Any]) -> Optional[str]:
    selected: List[Any] = []
    for source in (local, row):
        for key in ("parent_selected_by", "reservoir_memberships", "selected_by", "heuristic"):
            value = source.get(key)
            if isinstance(value, list):
                selected.extend(value)
            elif value is not None:
                selected.append(value)
    if not selected:
        return None
    deduped = []
    seen = set()
    for item in selected:
        text = str(item)
        if text not in seen:
            seen.add(text)
            deduped.append(text)
    return compact_json(deduped)


def matrix_digest_for(row: Dict[str, Any], local: Dict[str, Any], label: str) -> Optional[str]:
    prefix = prefix_for_label(label)
    keys: List[str] = []
    if prefix == "precursor":
        keys += ["base_matrix_digest", "precursor_matrix_digest"]
    elif prefix == "power":
        keys += ["power_matrix_digest"]
    keys += ["matrix_digest", "image_digest", "base_matrix_digest", "power_matrix_digest"]
    for source in (local, row):
        for key in keys:
            value = source.get(key)
            if value is not None:
                return str(value)
    return None


def verified_kernel_for(row: Dict[str, Any], local: Dict[str, Any], label: str) -> Optional[int]:
    verification = row.get("verification") if isinstance(row.get("verification"), dict) else {}
    kernel_match = row.get("kernel_match") if isinstance(row.get("kernel_match"), dict) else {}
    if verification.get("quotient_kernel") is not None:
        return truthy(verification.get("quotient_kernel"))
    if row.get("verified_kernel") is not None:
        return truthy(row.get("verified_kernel"))
    if kernel_match.get("matches") is not None:
        return truthy(kernel_match.get("matches"))
    kind = str(row.get("kind", ""))
    if "verified_power_scalar" in kind and "power" in label:
        return truthy(row.get("power_scalar_identity", True))
    if "paper_kernels" in str(row.get("source", "")):
        return 1
    return None


def metrics_payload(row: Dict[str, Any], local: Dict[str, Any]) -> Optional[str]:
    payload: Dict[str, Any] = {}
    if isinstance(row.get("metrics"), dict):
        payload.update(row["metrics"])
    if isinstance(local.get("metrics"), dict):
        payload.update(local["metrics"])
    for key in (
        "projlen",
        "projective_width",
        "identity_defect",
        "delta_defect",
        "score",
        "mcts_cost",
        "visits",
        "value",
        "average_value",
        "power_projlen",
        "precursor_projlen",
        "collapse_excess",
        "collapse_ratio_milli",
    ):
        if key in local:
            payload[key] = local[key]
        elif key in row:
            payload[key] = row[key]
    return compact_json(payload) if payload else None


def make_extracted(
    row: Dict[str, Any],
    local: Dict[str, Any],
    label: str,
    factors: Sequence[int],
) -> ExtractedBraid:
    factors = list(factors)
    record_kind = str(row.get("kind") or local.get("kind") or label)
    length = first_int(
        metric_value(row, local, label, "length"),
        len(factors),
    ) or len(factors)
    infimum = first_int(row.get("infimum"), local.get("infimum"), row.get("garside_power"), 0) or 0
    garside_power = first_int(row.get("garside_power"), local.get("garside_power"), row.get("power"), 0) or 0
    scalar_identity = truthy(metric_value(row, local, label, "scalar_identity"))
    if scalar_identity is None and "power" in label:
        scalar_identity = truthy(row.get("power_scalar_identity"))
    was_exactly_checked = 1 if metrics_payload(row, local) is not None else None
    status = str(row.get("status") or local.get("status") or "clean")
    return ExtractedBraid(
        label=label,
        factors=factors,
        source_record_kind=record_kind,
        length=length,
        infimum=infimum,
        garside_power=garside_power,
        n=first_int(row.get("n"), local.get("n")),
        r=first_int(row.get("r"), local.get("r")),
        p=first_int(row.get("p"), row.get("prime"), local.get("p"), local.get("prime")),
        projlen=int_metric(row, local, label, "projlen"),
        identity_defect=int_metric(row, local, label, "identity_defect"),
        delta_defect=int_metric(row, local, label, "delta_defect"),
        scalar_identity=scalar_identity,
        score=float_metric(row, local, label, "score"),
        matrix_digest=matrix_digest_for(row, local, label),
        finite_shadow_digest=first_str(row.get("finite_shadow_digest"), local.get("finite_shadow_digest")),
        selected_by_json=collect_selected_by(row, local),
        was_expanded=truthy(row.get("was_expanded") if row.get("was_expanded") is not None else local.get("was_expanded")),
        was_exactly_checked=was_exactly_checked,
        verified_kernel=verified_kernel_for(row, local, label),
        status=status,
        metrics_json=metrics_payload(row, local),
    )


def walk_factor_records(
    row: Dict[str, Any],
    current: Any,
    path: Tuple[str, ...] = (),
) -> Iterator[Tuple[str, Dict[str, Any], List[int]]]:
    if isinstance(current, dict):
        for key, value in current.items():
            if key in ARTIN_ONLY_KEYS:
                continue
            if key in FACTOR_KEYS and is_factor_list(value):
                factors = parse_int_list(value)
                if factors is not None:
                    yield ".".join(path + (key,)), current, factors
            if isinstance(value, (dict, list)):
                yield from walk_factor_records(row, value, path + (key,))
    elif isinstance(current, list):
        for index, value in enumerate(current):
            if isinstance(value, (dict, list)):
                yield from walk_factor_records(row, value, path + (str(index),))


def extract_from_json_row(row: Dict[str, Any]) -> List[ExtractedBraid]:
    extracted: List[ExtractedBraid] = []
    seen_labels: set = set()
    for label, local, factors in walk_factor_records(row, row):
        key = (label, tuple(factors))
        if key in seen_labels:
            continue
        seen_labels.add(key)
        extracted.append(make_extracted(row, local, label, factors))

    # MCTS policy-target records store child actions separately from the parent factor list.
    parent = parse_int_list(row.get("factor_ids"))
    actions = row.get("actions")
    if parent is not None and isinstance(actions, list):
        for index, action_row in enumerate(actions):
            if not isinstance(action_row, dict) or action_row.get("action") is None:
                continue
            try:
                child = parent + [int(action_row["action"])]
            except (TypeError, ValueError):
                continue
            local = dict(action_row)
            local["metrics"] = {
                "projlen": action_row.get("child_projlen"),
                "identity_defect": action_row.get("child_identity_defect"),
                "score": action_row.get("child_mcts_cost"),
            }
            label = f"policy_child.{index}"
            extracted.append(make_extracted(row, local, label, child))
    return extracted


def iter_json_records(obj: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(obj, dict):
        if obj.keys() & FACTOR_KEYS or "rollout_state" in obj or ("actions" in obj and "factor_ids" in obj):
            yield obj
        for value in obj.values():
            if isinstance(value, (dict, list)):
                yield from iter_json_records(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_json_records(value)


def iter_kernel_db_records(obj: Dict[str, Any]) -> Iterator[Tuple[int, Dict[str, Any]]]:
    primes = obj.get("primes")
    if not isinstance(primes, dict):
        return
    for prime_text, prime_block in primes.items():
        try:
            prime = int(prime_text)
        except ValueError:
            continue
        elements = prime_block.get("elements") if isinstance(prime_block, dict) else None
        if not isinstance(elements, dict):
            continue
        for key, value in elements.items():
            if not isinstance(value, dict):
                continue
            word = parse_int_list(value.get("word"))
            if word is None:
                continue
            row = {
                "factor_ids": word,
                "length": value.get("length", len(word)),
                "p": prime,
                "n": value.get("n", 4),
                "r": value.get("r", 1),
                "kind": "known_kernel_db",
                "verified_kernel": True,
                "status": "clean",
                "source": "kernel_db.json",
                "kernel_db_key": key,
            }
            yield prime, row


def read_jsonl(path: Path) -> Iterator[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                yield line_number, None, stripped
                continue
            if isinstance(value, dict):
                yield line_number, value, None
            else:
                yield line_number, {"value": value}, None


def read_json(path: Path) -> Iterator[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        yield exc.lineno, None, exc.msg
        return
    if isinstance(value, dict) and "primes" in value:
        for _prime, row in iter_kernel_db_records(value):
            yield 0, row, None
        return
    for row in iter_json_records(value):
        yield 0, row, None


def read_csv_rows(path: Path) -> Iterator[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            if "factor_ids" not in row:
                yield line_number, {}, None
                continue
            yield line_number, row, None


def read_records(path: Path) -> Iterator[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    if path.suffix == ".jsonl":
        yield from read_jsonl(path)
    elif path.suffix == ".json":
        yield from read_json(path)
    elif path.suffix == ".csv":
        yield from read_csv_rows(path)


class ImportManager:
    def __init__(
        self,
        out_dir: Path,
        primes: Sequence[int],
        force_prime: Optional[int] = None,
        store_unknown: bool = False,
    ):
        self.out_dir = out_dir
        self.allowed = set(int(p) for p in primes)
        self.force_prime = force_prime
        self.store_unknown = store_unknown
        self.dbs: Dict[int, ExperienceDB] = {}

    def close(self) -> None:
        for db in self.dbs.values():
            db.close()

    def db(self, prime: int) -> ExperienceDB:
        if prime not in self.dbs:
            self.dbs[prime] = ExperienceDB(self.out_dir / f"p{prime}.sqlite", prime)
        return self.dbs[prime]

    def import_file(self, path: Path, force: bool = False) -> Dict[str, int]:
        path = path.resolve()
        source_file = str(path)
        checksum = sha256_file(path)
        counts = {
            "files": 1,
            "records_seen": 0,
            "aggregate_records": 0,
            "observations_inserted": 0,
            "malformed_records": 0,
            "skipped_existing": 0,
            "skipped_prime": 0,
        }
        if not force:
            reusable = True
            for prime in self.allowed:
                db = self.db(prime)
                if db.has_clean_import(source_file, checksum):
                    continue
                reusable = False
                break
            if reusable:
                counts["skipped_existing"] = 1
                return counts

        per_prime = {prime: {"seen": 0, "aggregate": 0, "inserted": 0, "malformed": 0} for prime in self.allowed}
        for line_number, row, malformed in read_records(path):
            counts["records_seen"] += 1
            if malformed is not None:
                prime = self.force_prime or infer_prime_from_text(str(path))
                if prime in self.allowed:
                    inserted = self.db(prime).insert_malformed(source_file, line_number, malformed, malformed)
                    counts["malformed_records"] += int(inserted)
                    per_prime[prime]["malformed"] += int(inserted)
                continue
            if row is None:
                continue
            ctx0 = make_context(path, row, self.force_prime)
            prime = ctx0.prime
            if prime is None:
                counts["aggregate_records"] += 1
                continue
            if prime not in self.allowed:
                counts["skipped_prime"] += 1
                continue
            db = self.db(prime)
            ctx = make_context(path, row, prime)
            db.upsert_run(ctx)
            extracted = extract_from_json_row(row)
            per_prime[prime]["seen"] += 1
            if not extracted:
                counts["aggregate_records"] += 1
                per_prime[prime]["aggregate"] += 1
                continue
            for braid in extracted:
                if braid.p is not None and braid.p != prime:
                    continue
                try:
                    inserted = db.insert_observation(
                        ctx=ctx,
                        braid=braid,
                        source_file=source_file,
                        line_number=line_number,
                    )
                    counts["observations_inserted"] += int(inserted)
                    per_prime[prime]["inserted"] += int(inserted)
                except Exception as exc:
                    raw = compact_json(row)[:4000]
                    malformed_inserted = db.insert_malformed(
                        source_file, line_number, f"observation insert failed: {exc}", raw
                    )
                    counts["malformed_records"] += int(malformed_inserted)
                    per_prime[prime]["malformed"] += int(malformed_inserted)

        for prime, local in per_prime.items():
            self.db(prime).finish_import(
                source_file=source_file,
                checksum=checksum,
                records_seen=local["seen"],
                aggregate_records=local["aggregate"],
                observations_inserted=local["inserted"],
                malformed_records=local["malformed"],
                status="clean",
            )
        return counts


def find_supported_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root] if root.suffix in SUPPORTED_SUFFIXES else []
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in SUPPORTED_SUFFIXES:
            if ".git" in path.parts:
                continue
            if path.name.startswith("."):
                continue
            files.append(path)
    return sorted(files)


def parse_primes(text: str) -> List[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def summarize_db(path: Path) -> Dict[str, Any]:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    out: Dict[str, Any] = {"db": str(path)}
    for table in ("runs", "braids", "observations", "malformed_records", "imports"):
        out[table] = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
    out["by_method"] = [
        dict(row)
        for row in conn.execute(
            """
            SELECT method, COUNT(*) AS observations, COUNT(DISTINCT braid_digest) AS unique_braids,
                   MIN(projlen) AS min_projlen, MAX(length) AS max_length
            FROM observations
            GROUP BY method
            ORDER BY observations DESC
            LIMIT 40
            """
        )
    ]
    out["by_length"] = [
        dict(row)
        for row in conn.execute(
            """
            SELECT length, COUNT(DISTINCT braid_digest) AS unique_braids,
                   MIN(projlen) AS min_projlen
            FROM observations
            GROUP BY length
            ORDER BY length
            """
        )
    ]
    out["verified"] = [
        dict(row)
        for row in conn.execute(
            """
            SELECT braid_digest, length, MIN(projlen) AS min_projlen, COUNT(*) AS observations
            FROM observations
            WHERE verified_kernel=1
            GROUP BY braid_digest, length
            ORDER BY length, braid_digest
            LIMIT 50
            """
        )
    ]
    conn.close()
    return out


def export_seen(db_path: Path, out_path: Path, kind: str, min_length: Optional[int], max_length: Optional[int]) -> int:
    conn = sqlite3.connect(str(db_path))
    clauses: List[str] = []
    params: List[Any] = []
    if min_length is not None:
        clauses.append("length >= ?")
        params.append(min_length)
    if max_length is not None:
        clauses.append("length <= ?")
        params.append(max_length)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "braid":
        query = f"SELECT braid_digest AS digest FROM braids{where} ORDER BY braid_digest"
    elif kind == "matrix":
        query = (
            "SELECT DISTINCT matrix_digest AS digest FROM observations "
            "WHERE matrix_digest IS NOT NULL"
        )
        if clauses:
            query += " AND " + " AND ".join(clauses)
        query += " ORDER BY matrix_digest"
    else:
        raise ValueError(f"Unknown export kind {kind}")
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for (digest,) in conn.execute(query, params):
            if digest:
                handle.write(str(digest) + "\n")
                count += 1
    conn.close()
    return count


def cmd_import_root(args: argparse.Namespace) -> None:
    roots = [Path(item) for item in args.results_root]
    manager = ImportManager(
        out_dir=Path(args.out_dir),
        primes=parse_primes(args.primes),
        force_prime=args.prime,
    )
    totals = {
        "files": 0,
        "records_seen": 0,
        "aggregate_records": 0,
        "observations_inserted": 0,
        "malformed_records": 0,
        "skipped_existing": 0,
        "skipped_prime": 0,
    }
    try:
        files: List[Path] = []
        for root in roots:
            files.extend(find_supported_files(root))
        if args.limit_files is not None:
            files = files[: args.limit_files]
        for index, path in enumerate(files, start=1):
            result = manager.import_file(path, force=args.force)
            for key, value in result.items():
                totals[key] = totals.get(key, 0) + value
            if args.progress_every and (index % args.progress_every == 0 or index == len(files)):
                print(json.dumps({"processed_files": index, "total_files": len(files), **totals}), flush=True)
    finally:
        manager.close()
    print(json.dumps({"status": "clean", **totals}, indent=2, sort_keys=True))


def cmd_summarize(args: argparse.Namespace) -> None:
    print(json.dumps(summarize_db(Path(args.db)), indent=2, sort_keys=True))


def cmd_export_seen(args: argparse.Namespace) -> None:
    count = export_seen(Path(args.db), Path(args.out), args.kind, args.min_length, args.max_length)
    print(json.dumps({"status": "clean", "kind": args.kind, "out": args.out, "digests": count}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and query per-prime braid experience databases")
    sub = parser.add_subparsers(dest="command", required=True)

    import_root = sub.add_parser("import-root", help="Import supported files under one or more roots")
    import_root.add_argument("--results-root", action="append", required=True,
                             help="File or directory to scan. Pass more than once for multiple roots.")
    import_root.add_argument("--out-dir", required=True)
    import_root.add_argument("--primes", default="2,3,5,7")
    import_root.add_argument("--prime", type=int, help="Force missing-prime records into this prime")
    import_root.add_argument("--force", action="store_true", help="Re-import even if checksum is already clean")
    import_root.add_argument("--limit-files", type=int)
    import_root.add_argument("--progress-every", type=int, default=100)
    import_root.set_defaults(func=cmd_import_root)

    summarize = sub.add_parser("summarize", help="Print coverage summary for one prime DB")
    summarize.add_argument("--db", required=True)
    summarize.set_defaults(func=cmd_summarize)

    export = sub.add_parser("export-seen", help="Export seen braid or matrix digests")
    export.add_argument("--db", required=True)
    export.add_argument("--kind", choices=["braid", "matrix"], required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--min-length", type=int)
    export.add_argument("--max-length", type=int)
    export.set_defaults(func=cmd_export_seen)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
