from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Sequence

from braidzero.core import BraidEnvironment

from .common import braid_digest, compact_json, factor_ids_json, parse_factors, utc_now, write_json


LOCAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluated_braids (
  braid_digest TEXT PRIMARY KEY,
  n INTEGER NOT NULL,
  r INTEGER NOT NULL,
  p INTEGER NOT NULL,
  length INTEGER NOT NULL,
  factor_ids_json TEXT NOT NULL,
  projlen INTEGER NOT NULL,
  identity_defect INTEGER NOT NULL,
  scalar_identity INTEGER NOT NULL,
  source TEXT NOT NULL,
  verifier_version TEXT NOT NULL,
  evaluated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidates (
  braid_digest TEXT PRIMARY KEY,
  n INTEGER NOT NULL,
  r INTEGER NOT NULL,
  p INTEGER NOT NULL,
  length INTEGER NOT NULL,
  factor_ids_json TEXT NOT NULL,
  projlen INTEGER NOT NULL,
  identity_defect INTEGER NOT NULL,
  scalar_identity INTEGER NOT NULL,
  source TEXT NOT NULL,
  recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoints (
  checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
  current_length INTEGER NOT NULL,
  buckets_json TEXT NOT NULL,
  rng_state_b64 TEXT NOT NULL,
  inserted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bfs_queue (
  braid_digest TEXT PRIMARY KEY,
  factor_ids_json TEXT NOT NULL,
  depth_from_parent INTEGER NOT NULL,
  parent_digest TEXT,
  status INTEGER NOT NULL DEFAULT 0,
  inserted_at TEXT NOT NULL,
  finished_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_eval_length_projlen ON evaluated_braids(length, projlen);
CREATE INDEX IF NOT EXISTS idx_eval_pnr ON evaluated_braids(p, n, r);
CREATE INDEX IF NOT EXISTS idx_queue_status_depth ON bfs_queue(status, depth_from_parent);
"""


def open_local_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=120000")
    conn.executescript(LOCAL_SCHEMA)
    return conn


def insert_eval(
    conn: sqlite3.Connection,
    *,
    env: BraidEnvironment,
    factors: Sequence[int],
    metrics: dict[str, Any],
    source: str,
) -> int:
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO evaluated_braids
        (braid_digest, n, r, p, length, factor_ids_json, projlen,
         identity_defect, scalar_identity, source, verifier_version, evaluated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            braid_digest(env.n, factors),
            env.n,
            env.r,
            env.p,
            len(factors),
            factor_ids_json(factors),
            int(metrics["projlen"]),
            int(metrics["identity_defect"]),
            int(bool(metrics["scalar_identity"])),
            source,
            env.verifier_version,
            utc_now(),
        ),
    )
    return int(cur.rowcount)


def insert_candidate(
    conn: sqlite3.Connection,
    *,
    env: BraidEnvironment,
    factors: Sequence[int],
    metrics: dict[str, Any],
    source: str,
) -> int:
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO candidates
        (braid_digest, n, r, p, length, factor_ids_json, projlen,
         identity_defect, scalar_identity, source, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            braid_digest(env.n, factors),
            env.n,
            env.r,
            env.p,
            len(factors),
            factor_ids_json(factors),
            int(metrics["projlen"]),
            int(metrics["identity_defect"]),
            int(bool(metrics["scalar_identity"])),
            source,
            utc_now(),
        ),
    )
    return int(cur.rowcount)


def enqueue(
    conn: sqlite3.Connection,
    *,
    n: int,
    factors: Sequence[int],
    depth: int,
    parent_digest: str | None,
) -> int:
    digest = braid_digest(n, factors)
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO bfs_queue
        (braid_digest, factor_ids_json, depth_from_parent, parent_digest, status, inserted_at)
        VALUES (?, ?, ?, ?, 0, ?)
        """,
        (digest, factor_ids_json(factors), int(depth), parent_digest, utc_now()),
    )
    return int(cur.rowcount)


def load_parent_rows(path: Path, limit: int) -> list[tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit and len(rows) >= limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            factors = parse_factors(record.get("factor_ids") or record.get("factor_ids_json"))
            if factors:
                rows.append(factors)
    return rows


def queue_stats(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT status, COUNT(*) FROM bfs_queue GROUP BY status").fetchall()
    out = {f"status_{int(status)}": int(count) for status, count in rows}
    out["queued_total"] = sum(out.values())
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    candidates_path = output_dir / "candidates.jsonl"
    conn = open_local_db(output_dir / "local_run.sqlite")
    env = BraidEnvironment(author_repo=Path(args.author_repo), n=args.n, r=args.r, p=args.p)

    config = vars(args).copy()
    config["verifier_version"] = env.verifier_version
    config["representation"] = env.representation_label
    write_json(output_dir / "config.json", config)

    if not bool(conn.execute("SELECT COUNT(*) FROM bfs_queue").fetchone()[0]):
        parents = load_parent_rows(Path(args.parent_jsonl), args.max_parents)
        inserted = 0
        for factors in parents:
            if not env.is_legal(factors):
                continue
            inserted += enqueue(
                conn,
                n=env.n,
                factors=factors,
                depth=0,
                parent_digest=braid_digest(env.n, factors),
            )
        conn.commit()
        print(json.dumps({"phase": "parents_loaded", "parents": len(parents), "queued": inserted}, sort_keys=True), flush=True)

    evaluated = int(conn.execute("SELECT COUNT(*) FROM evaluated_braids").fetchone()[0])
    candidates = int(conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
    loops = 0
    status = "clean"
    reason = ""

    while True:
        if args.max_evals and evaluated >= int(args.max_evals):
            status = "budget_exhausted"
            reason = f"max_evals={args.max_evals}"
            break
        row = conn.execute(
            """
            SELECT braid_digest, factor_ids_json, depth_from_parent
            FROM bfs_queue
            WHERE status=0
            ORDER BY depth_from_parent, braid_digest
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            break

        digest, factors_text, depth = row
        factors = parse_factors(factors_text)
        image = env.exact_evaluate(factors)
        metrics = env.exact_metrics(image)
        inserted_eval = insert_eval(
            conn,
            env=env,
            factors=factors,
            metrics=metrics,
            source=f"local_bfs_depth_{int(depth)}",
        )
        evaluated += inserted_eval
        if bool(metrics["scalar_identity"]):
            candidates += insert_candidate(
                conn,
                env=env,
                factors=factors,
                metrics=metrics,
                source=f"local_bfs_depth_{int(depth)}",
            )
            with candidates_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "braid_digest": braid_digest(env.n, factors),
                            "factor_ids": list(factors),
                            "length": len(factors),
                            "projlen": int(metrics["projlen"]),
                            "identity_defect": int(metrics["identity_defect"]),
                            "scalar_identity": bool(metrics["scalar_identity"]),
                            "source": "local_bfs",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

        if int(depth) < int(args.depth):
            for factor_id in env.legal_next(factors):
                child = tuple([*factors, int(factor_id)])
                enqueue(
                    conn,
                    n=env.n,
                    factors=child,
                    depth=int(depth) + 1,
                    parent_digest=str(digest),
                )

        conn.execute("UPDATE bfs_queue SET status=1, finished_at=? WHERE braid_digest=?", (utc_now(), digest))
        loops += 1

        if loops % int(args.commit_every) == 0:
            conn.commit()
            stats = queue_stats(conn)
            progress = {
                "phase": "progress",
                "processed_queue_items": loops,
                "evaluated_braids": evaluated,
                "candidates": candidates,
                "elapsed_seconds": round(time.time() - started, 2),
                **stats,
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(progress, sort_keys=True) + "\n")
            print(json.dumps(progress, sort_keys=True), flush=True)

    conn.commit()
    stats = queue_stats(conn)
    summary = {
        "status": status,
        "reason": reason,
        "method": "braid_lake_local_bfs",
        "prime": env.p,
        "n": env.n,
        "r": env.r,
        "depth": int(args.depth),
        "max_parents": int(args.max_parents),
        "max_evals": int(args.max_evals),
        "evaluated_braids": int(conn.execute("SELECT COUNT(*) FROM evaluated_braids").fetchone()[0]),
        "total_candidates": int(conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]),
        "local_db": str(output_dir / "local_run.sqlite"),
        "elapsed_seconds": round(time.time() - started, 2),
        "verifier_version": env.verifier_version,
        **stats,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    conn.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local BFS expansion from BraidLake parent lists")
    parser.add_argument("--author-repo", required=True)
    parser.add_argument("--parent-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--p", type=int, default=7)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-parents", type=int, default=1000)
    parser.add_argument("--max-evals", type=int, default=500000)
    parser.add_argument("--commit-every", type=int, default=1000)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
