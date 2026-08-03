from __future__ import annotations

import argparse
import json
from pathlib import Path

from .query_lake import parquet_glob, sql_quote


def import_duckdb():
    import duckdb  # type: ignore

    return duckdb


def build_where(args: argparse.Namespace) -> str:
    clauses = [
        f"p = {int(args.p)}",
        f"n = {int(args.n)}",
        f"r = {int(args.r)}",
        "factor_ids_json IS NOT NULL",
    ]
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
    if args.exclude_source_like:
        clauses.append(f"(global_source IS NULL OR global_source NOT LIKE {sql_quote(args.exclude_source_like)})")
    return " AND ".join(clauses)


def order_expr(args: argparse.Namespace) -> str:
    seed_text = str(int(args.seed))
    if args.order == "random":
        return f"hash(braid_digest || {sql_quote(seed_text)})"
    if args.order == "projlen":
        return f"projlen ASC NULLS LAST, hash(braid_digest || {sql_quote(seed_text)})"
    if args.order == "long":
        return f"length DESC, projlen ASC NULLS LAST, hash(braid_digest || {sql_quote(seed_text)})"
    if args.order == "short":
        return f"length ASC, projlen ASC NULLS LAST, hash(braid_digest || {sql_quote(seed_text)})"
    if args.order == "mixed":
        return f"(projlen * 1000000 + (hash(braid_digest || {sql_quote(seed_text)}) % 1000000)) ASC"
    raise ValueError(f"unknown order: {args.order}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample parent braids from BraidLake")
    parser.add_argument("--lake-root", default="results/BraidLake")
    parser.add_argument("--output", required=True)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--min-length", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--min-projlen", type=int)
    parser.add_argument("--max-projlen", type=int)
    parser.add_argument("--mode")
    parser.add_argument("--source-like")
    parser.add_argument("--exclude-source-like")
    parser.add_argument("--order", choices=["random", "projlen", "long", "short", "mixed"], default="random")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--oversample-factor", type=int, default=3)
    args = parser.parse_args()

    duckdb = import_duckdb()
    conn = duckdb.connect(database=":memory:")
    expr = f"read_parquet({sql_quote(parquet_glob(Path(args.lake_root), args.p, args.n, args.r))}, union_by_name=true)"
    where = build_where(args)
    # Deduplicate first, then rank. any_value is OK because braid_digest fixes the word.
    inner_limit = max(int(args.limit) * max(1, int(args.oversample_factor)), int(args.limit))
    query = f"""
        WITH candidates AS (
          SELECT
            braid_digest,
            any_value(factor_ids_json) AS factor_ids_json,
            any_value(global_source) AS global_source,
            any_value(mode) AS mode,
            min(length) AS length,
            min(projlen) AS projlen,
            min(identity_defect) AS identity_defect
          FROM {expr}
          WHERE {where}
          GROUP BY braid_digest
          ORDER BY {order_expr(args)}
          LIMIT {inner_limit}
        )
        SELECT *
        FROM candidates
        ORDER BY {order_expr(args)}
        LIMIT {int(args.limit)}
    """
    rows = conn.execute(query).fetchall()
    names = [d[0] for d in conn.description]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for raw in rows:
            row = dict(zip(names, raw))
            factors = json.loads(row["factor_ids_json"])
            handle.write(
                json.dumps(
                    {
                        "braid_digest": row["braid_digest"],
                        "factor_ids": [int(x) for x in factors],
                        "factor_ids_json": row["factor_ids_json"],
                        "length": int(row["length"]),
                        "projlen": None if row["projlen"] is None else int(row["projlen"]),
                        "identity_defect": None
                        if row["identity_defect"] is None
                        else int(row["identity_defect"]),
                        "source": row["global_source"],
                        "mode": row["mode"],
                        "sample_order": args.order,
                        "sample_seed": int(args.seed),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    print(json.dumps({"output": str(output), "rows": len(rows), "order": args.order}, sort_keys=True))


if __name__ == "__main__":
    main()
