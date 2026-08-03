from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def import_duckdb():
    import duckdb  # type: ignore

    return duckdb


def sql_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def parquet_glob(lake_root: Path, p: int | None, n: int | None, r: int | None) -> str:
    root = Path(lake_root)
    if p is None:
        return str(root / "**" / "*.parquet")
    if n is None or r is None:
        return str(root / f"p={int(p)}" / "**" / "*.parquet")
    return str(root / f"p={int(p)}" / f"n={int(n)}_r={int(r)}" / "**" / "*.parquet")


def read_parquet_expr(args: argparse.Namespace) -> str:
    return f"read_parquet({sql_quote(parquet_glob(Path(args.lake_root), args.p, args.n, args.r))}, union_by_name=true)"


def where_clause(args: argparse.Namespace) -> str:
    clauses = []
    if args.p is not None:
        clauses.append(f"p = {int(args.p)}")
    if args.n is not None:
        clauses.append(f"n = {int(args.n)}")
    if args.r is not None:
        clauses.append(f"r = {int(args.r)}")
    if args.min_length is not None:
        clauses.append(f"length >= {int(args.min_length)}")
    if args.max_length is not None:
        clauses.append(f"length <= {int(args.max_length)}")
    if args.min_projlen is not None:
        clauses.append(f"projlen >= {int(args.min_projlen)}")
    if args.max_projlen is not None:
        clauses.append(f"projlen <= {int(args.max_projlen)}")
    if args.mode:
        clauses.append(f"mode = {sql_quote(args.mode)}")
    if args.source_like:
        clauses.append(f"global_source LIKE {sql_quote(args.source_like)}")
    return " AND ".join(clauses) if clauses else "TRUE"


def print_rows(rows: list[dict[str, Any]], fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    for row in rows:
        print("\t".join(str(row.get(key, "")) for key in row))


def cmd_coverage(args: argparse.Namespace) -> None:
    duckdb = import_duckdb()
    conn = duckdb.connect(database=":memory:")
    expr = read_parquet_expr(args)
    where = where_clause(args)
    count_expr = "COUNT(*)" if args.observations else "COUNT(DISTINCT braid_digest)"
    query = f"""
        SELECT
          length,
          {count_expr} AS braids,
          COUNT(*) AS rows,
          MIN(projlen) AS min_projlen,
          MAX(projlen) AS max_projlen
        FROM {expr}
        WHERE {where}
        GROUP BY length
        ORDER BY length
    """
    rows = [dict(zip([d[0] for d in conn.description], row)) for row in conn.execute(query).fetchall()]
    print_rows(rows, args.format)


def cmd_best(args: argparse.Namespace) -> None:
    duckdb = import_duckdb()
    conn = duckdb.connect(database=":memory:")
    expr = read_parquet_expr(args)
    where = where_clause(args)
    query = f"""
        SELECT
          braid_digest,
          length,
          projlen,
          identity_defect,
          factor_ids_json,
          global_source,
          mode
        FROM {expr}
        WHERE {where}
        QUALIFY row_number() OVER (
          PARTITION BY length
          ORDER BY projlen ASC NULLS LAST, identity_defect ASC NULLS LAST, hash(braid_digest)
        ) <= {int(args.per_length)}
        ORDER BY length, projlen ASC NULLS LAST, identity_defect ASC NULLS LAST
    """
    rows = [dict(zip([d[0] for d in conn.description], row)) for row in conn.execute(query).fetchall()]
    print_rows(rows, args.format)


def cmd_summary(args: argparse.Namespace) -> None:
    duckdb = import_duckdb()
    conn = duckdb.connect(database=":memory:")
    expr = read_parquet_expr(args)
    where = where_clause(args)
    query = f"""
        SELECT
          COUNT(*) AS rows,
          COUNT(DISTINCT braid_digest) AS distinct_braids,
          MIN(length) AS min_length,
          MAX(length) AS max_length,
          MIN(projlen) AS min_projlen,
          MAX(projlen) AS max_projlen
        FROM {expr}
        WHERE {where}
    """
    row = dict(zip([d[0] for d in conn.description], conn.execute(query).fetchone()))
    print(json.dumps(row, indent=2, sort_keys=True))


def cmd_candidates(args: argparse.Namespace) -> None:
    duckdb = import_duckdb()
    conn = duckdb.connect(database=":memory:")
    expr = read_parquet_expr(args)
    where = where_clause(args)
    query = f"""
        SELECT
          braid_digest,
          length,
          projlen,
          identity_defect,
          scalar_identity,
          factor_ids_json,
          global_source,
          mode
        FROM {expr}
        WHERE {where}
          AND scalar_identity = 1
        ORDER BY length, projlen, braid_digest
        LIMIT {int(args.limit)}
    """
    rows = [dict(zip([d[0] for d in conn.description], row)) for row in conn.execute(query).fetchall()]
    print_rows(rows, args.format)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query BraidLake Parquet data with DuckDB")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lake-root", default="results/BraidLake")
    common.add_argument("--p", type=int)
    common.add_argument("--n", type=int)
    common.add_argument("--r", type=int)
    common.add_argument("--min-length", type=int)
    common.add_argument("--max-length", type=int)
    common.add_argument("--min-projlen", type=int)
    common.add_argument("--max-projlen", type=int)
    common.add_argument("--mode")
    common.add_argument("--source-like")
    common.add_argument("--format", choices=["json", "tsv"], default="tsv")

    coverage = sub.add_parser("coverage", parents=[common])
    coverage.add_argument("--observations", action="store_true", help="Count rows instead of distinct braid digests")
    coverage.set_defaults(func=cmd_coverage)

    best = sub.add_parser("best", parents=[common])
    best.add_argument("--per-length", type=int, default=10)
    best.set_defaults(func=cmd_best)

    summary = sub.add_parser("summary", parents=[common])
    summary.set_defaults(func=cmd_summary)

    candidates = sub.add_parser("candidates", parents=[common])
    candidates.add_argument("--limit", type=int, default=50)
    candidates.set_defaults(func=cmd_candidates)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
