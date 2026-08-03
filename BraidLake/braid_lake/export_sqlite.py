from __future__ import annotations

import argparse
import glob
import json
import sqlite3
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

from .common import (
    append_jsonl,
    lake_partition_dir,
    last_jsonl,
    read_json_if_exists,
    source_from_run_dir,
    utc_now,
    write_json,
)


LOCAL_COLUMNS = [
    "braid_digest",
    "n",
    "r",
    "p",
    "length",
    "factor_ids_json",
    "projlen",
    "identity_defect",
    "scalar_identity",
    "source",
    "verifier_version",
    "evaluated_at",
]


def import_arrow():
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    return pa, pq


def parquet_write(rows: list[dict[str, Any]], path: Path, compression: str) -> None:
    if not rows:
        return
    pa, pq = import_arrow()
    table = pa.Table.from_pylist(rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        pq.write_table(table, tmp, compression=compression)
    except Exception:
        if compression != "snappy":
            pq.write_table(table, tmp, compression="snappy")
        else:
            raise
    tmp.replace(path)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def infer_local_metadata(local_db: Path) -> dict[str, Any]:
    run_dir = local_db.parent
    summary = read_json_if_exists(run_dir / "summary.json")
    progress = last_jsonl(run_dir / "progress.jsonl") or {}
    source = source_from_run_dir(run_dir)
    mode = summary.get("mode") or progress.get("mode") or "unknown"
    status = summary.get("status") or ("incomplete" if progress else "unknown")
    return {
        "run_dir": str(run_dir),
        "source": source,
        "mode": mode,
        "run_status": status,
        "summary_total_evaluated": summary.get("total_evaluated"),
        "progress_length": progress.get("length"),
        "progress_total_evaluated": progress.get("total_evaluated"),
    }


def first_local_pnr(conn: sqlite3.Connection) -> tuple[int, int, int] | None:
    row = conn.execute(
        "SELECT p, n, r FROM evaluated_braids WHERE p IS NOT NULL AND n IS NOT NULL AND r IS NOT NULL LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return int(row[0]), int(row[1]), int(row[2])


def export_local_run(
    *,
    local_db: Path,
    lake_root: Path,
    chunk_size: int,
    force: bool,
    compression: str,
    manifest: Path,
) -> dict[str, Any]:
    local_db = Path(local_db)
    meta = infer_local_metadata(local_db)
    conn = sqlite3.connect(f"file:{local_db}?mode=ro", uri=True, timeout=120)
    if not table_exists(conn, "evaluated_braids"):
        conn.close()
        raise RuntimeError(f"evaluated_braids table not found in {local_db}")

    pnr = first_local_pnr(conn)
    if pnr is None:
        conn.close()
        return {
            "status": "empty",
            "local_db": str(local_db),
            "row_count": 0,
            **meta,
        }
    p, n, r = pnr
    out_dir = lake_partition_dir(lake_root, p=p, n=n, r=r, kind="local_run", source=meta["source"])
    success_path = out_dir / "_SUCCESS.json"
    if success_path.exists() and not force:
        payload = read_json_if_exists(success_path)
        payload["status"] = "skipped_existing"
        conn.close()
        return payload

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part-*.parquet"):
        if force:
            old.unlink()

    lo, hi, total = conn.execute("SELECT MIN(rowid), MAX(rowid), COUNT(*) FROM evaluated_braids").fetchone()
    if lo is None:
        conn.close()
        return {"status": "empty", "local_db": str(local_db), "row_count": 0, **meta}

    start = int(lo)
    hi = int(hi)
    part = 0
    exported = 0
    started_at = time.time()
    while start <= hi:
        end = start + int(chunk_size) - 1
        rows_raw = conn.execute(
            f"""
            SELECT {", ".join(LOCAL_COLUMNS)}
            FROM evaluated_braids
            WHERE rowid BETWEEN ? AND ?
            ORDER BY rowid
            """,
            (start, end),
        ).fetchall()
        rows = []
        for raw in rows_raw:
            row = dict(zip(LOCAL_COLUMNS, raw))
            row.update(
                {
                    "global_source": meta["source"],
                    "mode": meta["mode"],
                    "run_status": meta["run_status"],
                    "local_db": str(local_db),
                    "exported_at": utc_now(),
                }
            )
            rows.append(row)
        if rows:
            parquet_write(rows, out_dir / f"part-{part:06d}.parquet", compression)
            exported += len(rows)
            part += 1
            print(
                json.dumps(
                    {
                        "phase": "local_chunk_exported",
                        "local_db": str(local_db),
                        "rows": len(rows),
                        "rowid_start": start,
                        "rowid_end": end,
                        "exported": exported,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        start = end + 1

    conn.close()
    payload = {
        "status": "clean",
        "kind": "local_run",
        "local_db": str(local_db),
        "lake_dir": str(out_dir),
        "row_count": int(total),
        "exported_rows": exported,
        "parts": part,
        "p": p,
        "n": n,
        "r": r,
        "elapsed_seconds": round(time.time() - started_at, 2),
        **meta,
    }
    write_json(success_path, payload)
    append_jsonl(manifest, payload)
    return payload


def export_global_db(
    *,
    db: Path,
    lake_root: Path,
    p: int,
    n: int,
    r: int,
    chunk_size: int,
    force: bool,
    compression: str,
    manifest: Path,
) -> dict[str, Any]:
    source = f"global_db:{Path(db).name}:p{p}:n{n}:r{r}"
    out_dir = lake_partition_dir(lake_root, p=p, n=n, r=r, kind="global_db", source=source)
    success_path = out_dir / "_SUCCESS.json"
    if success_path.exists() and not force:
        payload = read_json_if_exists(success_path)
        payload["status"] = "skipped_existing"
        return payload

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part-*.parquet"):
        if force:
            old.unlink()

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=300)
    conn.execute("PRAGMA busy_timeout=300000")
    lo, hi, total = conn.execute(
        """
        SELECT MIN(rowid), MAX(rowid), COUNT(*)
        FROM projlen_images
        WHERE p=? AND n=? AND r=?
        """,
        (p, n, r),
    ).fetchone()
    if lo is None:
        conn.close()
        return {"status": "empty", "db": str(db), "row_count": 0}

    start = int(lo)
    hi = int(hi)
    part = 0
    exported = 0
    started_at = time.time()
    while start <= hi:
        end = start + int(chunk_size) - 1
        rows_raw = conn.execute(
            """
            SELECT
              b.braid_digest,
              b.n,
              i.r,
              i.p,
              b.length,
              b.factor_ids_json,
              i.projlen,
              i.identity_defect,
              i.scalar_identity,
              i.source,
              i.verifier_version,
              i.observed_at
            FROM projlen_images i
            JOIN braids b ON b.braid_digest=i.braid_digest
            WHERE i.rowid BETWEEN ? AND ?
              AND i.p=? AND i.n=? AND i.r=?
            ORDER BY i.rowid
            """,
            (start, end, p, n, r),
        ).fetchall()
        rows = []
        for raw in rows_raw:
            row = dict(zip(LOCAL_COLUMNS, raw))
            row.update(
                {
                    "global_source": source,
                    "mode": "global_db",
                    "run_status": "global_db",
                    "local_db": "",
                    "exported_at": utc_now(),
                }
            )
            rows.append(row)
        if rows:
            parquet_write(rows, out_dir / f"part-{part:06d}.parquet", compression)
            exported += len(rows)
            part += 1
            print(
                json.dumps(
                    {
                        "phase": "global_chunk_exported",
                        "db": str(db),
                        "rows": len(rows),
                        "rowid_start": start,
                        "rowid_end": end,
                        "exported": exported,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        start = end + 1

    conn.close()
    payload = {
        "status": "clean",
        "kind": "global_db",
        "db": str(db),
        "lake_dir": str(out_dir),
        "row_count": int(total),
        "exported_rows": exported,
        "parts": part,
        "p": p,
        "n": n,
        "r": r,
        "source": source,
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    write_json(success_path, payload)
    append_jsonl(manifest, payload)
    return payload


def cmd_local_runs(args: argparse.Namespace) -> None:
    paths = [Path(p) for p in sorted(glob.glob(args.input_glob))]
    print("local DBs found:", len(paths), flush=True)
    ok = 0
    for idx, path in enumerate(paths, 1):
        try:
            print(f"[{idx}/{len(paths)}] exporting {path}", flush=True)
            payload = export_local_run(
                local_db=path,
                lake_root=Path(args.lake_root),
                chunk_size=args.chunk_size,
                force=args.force,
                compression=args.compression,
                manifest=Path(args.manifest),
            )
            print(json.dumps(payload, sort_keys=True), flush=True)
            if payload.get("status") in {"clean", "skipped_existing"}:
                ok += 1
        except Exception:
            traceback.print_exc()
            if not args.keep_going:
                raise
    print("exports ok:", ok, "of", len(paths), flush=True)


def cmd_global_db(args: argparse.Namespace) -> None:
    payload = export_global_db(
        db=Path(args.db),
        lake_root=Path(args.lake_root),
        p=args.p,
        n=args.n,
        r=args.r,
        chunk_size=args.chunk_size,
        force=args.force,
        compression=args.compression,
        manifest=Path(args.manifest),
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export braid SQLite DBs into BraidLake Parquet shards")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lake-root", default="results/BraidLake")
    common.add_argument("--manifest", default="results/BraidLake/manifest.jsonl")
    common.add_argument("--chunk-size", type=int, default=250_000)
    common.add_argument("--compression", default="zstd")
    common.add_argument("--force", action="store_true")

    local = sub.add_parser("local-runs", parents=[common])
    local.add_argument("--input-glob", required=True)
    local.add_argument("--keep-going", action="store_true")
    local.set_defaults(func=cmd_local_runs)

    global_db = sub.add_parser("global-db", parents=[common])
    global_db.add_argument("--db", required=True)
    global_db.add_argument("--p", type=int, required=True)
    global_db.add_argument("--n", type=int, required=True)
    global_db.add_argument("--r", type=int, required=True)
    global_db.set_defaults(func=cmd_global_db)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

