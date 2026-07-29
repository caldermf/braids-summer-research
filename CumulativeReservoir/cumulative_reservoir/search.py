from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pickle
import random
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from braidzero.core import BraidEnvironment, write_json


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

CREATE INDEX IF NOT EXISTS idx_eval_length_projlen ON evaluated_braids(length, projlen);
CREATE INDEX IF NOT EXISTS idx_eval_pnr ON evaluated_braids(p, n, r);
"""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def braid_digest(n: int, factors: Sequence[int], infimum: int = 0) -> str:
    payload = {"n": int(n), "infimum": int(infimum), "factor_ids": [int(x) for x in factors]}
    return sha1_text("gnf-factor-digest-v1:" + compact_json(payload))


def encode_rng_state(rng: random.Random) -> str:
    return base64.b64encode(pickle.dumps(rng.getstate())).decode("ascii")


def decode_rng_state(text: str) -> object:
    return pickle.loads(base64.b64decode(text.encode("ascii")))


@dataclass
class State:
    factors: tuple[int, ...]
    image: np.ndarray
    projlen: int
    identity_defect: int
    scalar_identity: bool
    score: int

    @property
    def length(self) -> int:
        return len(self.factors)


class Reservoir:
    def __init__(self, bucket_size: int, rng: random.Random):
        self.bucket_size = int(bucket_size)
        self.rng = rng
        self.buckets: dict[tuple[Any, ...], list[State]] = defaultdict(list)
        self.seen_by_bucket: dict[tuple[Any, ...], int] = defaultdict(int)

    def add(self, key: tuple[Any, ...], state: State) -> None:
        bucket = self.buckets[key]
        self.seen_by_bucket[key] += 1
        seen = self.seen_by_bucket[key]
        if len(bucket) < self.bucket_size:
            bucket.append(state)
            return
        index = self.rng.randrange(seen)
        if index < self.bucket_size:
            bucket[index] = state

    def keys_for_length(self, length: int) -> list[tuple[Any, ...]]:
        return sorted(key for key in self.buckets if int(key[0]) == int(length))

    def select(self, length: int, use_best: int) -> list[State]:
        selected: list[State] = []
        for key in self.keys_for_length(length):
            bucket = self.buckets[key]
            remaining = int(use_best) - len(selected)
            if remaining <= 0:
                break
            if len(bucket) <= remaining:
                selected.extend(bucket)
            else:
                selected.extend(self.rng.sample(bucket, remaining))
                break
        return selected

    def discard_length(self, length: int) -> None:
        for key in list(self.buckets):
            if int(key[0]) <= int(length):
                del self.buckets[key]
                self.seen_by_bucket.pop(key, None)

    def stats(self) -> dict[str, Any]:
        best: dict[int, int] = {}
        counts: dict[str, int] = {}
        for key, bucket in self.buckets.items():
            length = int(key[0])
            score = int(key[1])
            best[length] = score if length not in best else min(best[length], score)
            counts[json.dumps(key)] = len(bucket)
        return {
            "bucket_count": len(self.buckets),
            "live_states": sum(len(v) for v in self.buckets.values()),
            "best_score_by_length": dict(sorted(best.items())),
            "bucket_counts": counts,
        }


def local_conn(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")
    conn.executescript(LOCAL_SCHEMA)
    return conn


def metric_from_image(env: BraidEnvironment, image: np.ndarray) -> dict[str, Any]:
    return env.exact_metrics(image)


def power_image(env: BraidEnvironment, image: np.ndarray, power: int) -> np.ndarray:
    out = image
    for _ in range(int(power) - 1):
        out = env.rep.mul(out, image)
    return out


def score_state(env: BraidEnvironment, image: np.ndarray, mode: str, power: int) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
    base = metric_from_image(env, image)
    if mode == "paper":
        return int(base["projlen"]), base, None
    if mode == "power_v2":
        pimg = power_image(env, image, power)
        pmet = metric_from_image(env, pimg)
        # Two-level key is handled by bucket_key; scalar score keeps stdout readable.
        return int(pmet["projlen"]), base, pmet
    raise ValueError(f"unknown mode {mode}")


def bucket_key(state: State, mode: str, power_metrics: dict[str, Any] | None = None) -> tuple[Any, ...]:
    if mode == "paper":
        return (state.length, state.score)
    if mode == "power_v2":
        base_bin = (state.projlen // 4) * 4
        return (state.length, state.score, base_bin)
    raise ValueError(f"unknown mode {mode}")


def insert_eval(
    conn: sqlite3.Connection,
    *,
    env: BraidEnvironment,
    factors: Sequence[int],
    metrics: dict[str, Any],
    source: str,
) -> None:
    conn.execute(
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
            compact_json([int(x) for x in factors]),
            int(metrics["projlen"]),
            int(metrics["identity_defect"]),
            int(bool(metrics["scalar_identity"])),
            source,
            env.verifier_version,
            now(),
        ),
    )


def insert_candidate(
    conn: sqlite3.Connection,
    *,
    env: BraidEnvironment,
    factors: Sequence[int],
    metrics: dict[str, Any],
    source: str,
) -> None:
    conn.execute(
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
            compact_json([int(x) for x in factors]),
            int(metrics["projlen"]),
            int(metrics["identity_defect"]),
            int(bool(metrics["scalar_identity"])),
            source,
            now(),
        ),
    )


def load_seed_rows(
    global_db: Path,
    *,
    p: int,
    n: int,
    r: int,
    seed_length: int,
    min_projlen: int,
    max_projlen: int,
    limit: int,
    order: str,
    rng: random.Random,
) -> list[tuple[str, tuple[int, ...], int]]:
    conn = sqlite3.connect(str(global_db))
    rows = conn.execute(
        """
        SELECT b.braid_digest, b.factor_ids_json, i.projlen
        FROM projlen_images i
        JOIN braids b ON b.braid_digest = i.braid_digest
        WHERE i.p=? AND i.n=? AND i.r=?
          AND i.verifier_version='braidzero-exact-peyl-v1'
          AND i.source='computed-exact'
          AND b.length=?
          AND i.projlen >= ?
          AND i.projlen <= ?
        ORDER BY i.projlen, b.braid_digest
        """,
        (int(p), int(n), int(r), int(seed_length), int(min_projlen), int(max_projlen)),
    ).fetchall()
    conn.close()
    out = [(row[0], tuple(int(x) for x in json.loads(row[1])), int(row[2])) for row in rows]
    if order == "random":
        rng.shuffle(out)
    elif order == "hash":
        out.sort(key=lambda row: row[0])
    elif order == "projlen":
        pass
    else:
        raise ValueError(f"unknown seed order {order}")
    return out[: int(limit)] if limit else out


def save_checkpoint(conn: sqlite3.Connection, reservoir: Reservoir, rng: random.Random, current_length: int) -> None:
    buckets: dict[str, list[list[int]]] = {}
    for key, states in reservoir.buckets.items():
        buckets[json.dumps(key)] = [[int(x) for x in state.factors] for state in states]
    conn.execute(
        "INSERT INTO checkpoints(current_length, buckets_json, rng_state_b64, inserted_at) VALUES (?, ?, ?, ?)",
        (int(current_length), json.dumps(buckets, sort_keys=True), encode_rng_state(rng), now()),
    )
    conn.commit()


def load_checkpoint(conn: sqlite3.Connection, env: BraidEnvironment, mode: str, power: int, bucket_size: int, rng: random.Random) -> tuple[int, Reservoir] | None:
    row = conn.execute(
        "SELECT current_length, buckets_json, rng_state_b64 FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    rng.setstate(decode_rng_state(row[2]))
    reservoir = Reservoir(bucket_size=bucket_size, rng=rng)
    payload = json.loads(row[1])
    for key_text, factors_list in payload.items():
        key = tuple(json.loads(key_text))
        for factors in factors_list:
            image = env.exact_evaluate(factors)
            score, base, pmet = score_state(env, image, mode, power)
            state = State(tuple(factors), image, int(base["projlen"]), int(base["identity_defect"]), bool(base["scalar_identity"]), score)
            reservoir.add(key, state)
    return int(row[0]), reservoir


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    env = BraidEnvironment(author_repo=Path(args.author_repo), n=args.n, r=args.r, p=args.p)
    conn = local_conn(output_dir / "local_run.sqlite")

    config = vars(args).copy()
    config["representation"] = env.representation_label
    config["verifier_version"] = env.verifier_version
    write_json(output_dir / "config.json", config)

    loaded = load_checkpoint(conn, env, args.mode, args.power or args.p, args.bucket_size, rng)
    if loaded is None:
        current_length = int(args.seed_length)
        reservoir = Reservoir(bucket_size=args.bucket_size, rng=rng)
        seeds = load_seed_rows(
            Path(args.global_db),
            p=args.p,
            n=args.n,
            r=args.r,
            seed_length=args.seed_length,
            min_projlen=args.seed_min_projlen,
            max_projlen=args.seed_max_projlen,
            limit=args.seed_limit,
            order=args.seed_order,
            rng=rng,
        )
        assigned = [
            item for idx, item in enumerate(seeds)
            if idx % int(args.seed_shard_count) == int(args.seed_shard_index)
        ]
        for _, factors, _seed_projlen in assigned:
            image = env.exact_evaluate(factors)
            score, base, pmet = score_state(env, image, args.mode, args.power or args.p)
            state = State(tuple(factors), image, int(base["projlen"]), int(base["identity_defect"]), bool(base["scalar_identity"]), score)
            reservoir.add(bucket_key(state, args.mode, pmet), state)
            insert_eval(conn, env=env, factors=factors, metrics=base, source="seed")
        conn.commit()
        save_checkpoint(conn, reservoir, rng, current_length)
    else:
        current_length, reservoir = loaded
        assigned = []

    progress_path = output_dir / "progress.jsonl"
    candidates_path = output_dir / "candidates.jsonl"
    total_evaluated = 0
    total_candidates = 0

    while current_length < int(args.target_length):
        parents = reservoir.select(current_length, args.use_best)
        next_length = current_length + 1
        next_reservoir = Reservoir(bucket_size=args.bucket_size, rng=rng)
        length_evaluated = 0
        for parent in parents:
            for factor_id in env.legal_next(parent.factors):
                child_factors = tuple([*parent.factors, int(factor_id)])
                child_image = env.exact_append(parent.image, int(factor_id))
                score, base, pmet = score_state(env, child_image, args.mode, args.power or args.p)
                state = State(child_factors, child_image, int(base["projlen"]), int(base["identity_defect"]), bool(base["scalar_identity"]), score)
                next_reservoir.add(bucket_key(state, args.mode, pmet), state)
                insert_eval(conn, env=env, factors=child_factors, metrics=base, source=f"{args.mode}_length_{next_length}")
                length_evaluated += 1
                total_evaluated += 1
                if bool(base["scalar_identity"]):
                    insert_candidate(conn, env=env, factors=child_factors, metrics=base, source=f"{args.mode}_length_{next_length}")
                    with candidates_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "factor_ids": list(child_factors),
                            "length": len(child_factors),
                            "projlen": int(base["projlen"]),
                            "identity_defect": int(base["identity_defect"]),
                            "scalar_identity": bool(base["scalar_identity"]),
                            "mode": args.mode,
                        }, sort_keys=True) + "\n")
                    total_candidates += 1
        conn.commit()
        reservoir = next_reservoir
        current_length = next_length
        save_checkpoint(conn, reservoir, rng, current_length)
        stats = reservoir.stats()
        row = {
            "phase": "length_done",
            "mode": args.mode,
            "length": current_length,
            "seed_length": args.seed_length,
            "seed_min_projlen": args.seed_min_projlen,
            "seed_max_projlen": args.seed_max_projlen,
            "seed_order": args.seed_order,
            "parents": len(parents),
            "length_evaluated": length_evaluated,
            "total_evaluated": total_evaluated,
            "total_candidates": total_candidates,
            "elapsed_seconds": round(time.time() - start, 2),
            **stats,
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "status": "clean",
        "method": "cumulative_local_db_reservoir",
        "mode": args.mode,
        "prime": args.p,
        "n": args.n,
        "r": args.r,
        "seed": args.seed,
        "seed_length": args.seed_length,
        "seed_min_projlen": args.seed_min_projlen,
        "seed_max_projlen": args.seed_max_projlen,
        "seed_order": args.seed_order,
        "seed_shard_index": args.seed_shard_index,
        "seed_shard_count": args.seed_shard_count,
        "target_length": args.target_length,
        "total_evaluated": total_evaluated,
        "total_candidates": total_candidates,
        "elapsed_seconds": round(time.time() - start, 2),
        "local_db": str(output_dir / "local_run.sqlite"),
        "verifier_version": env.verifier_version,
    }
    write_json(output_dir / "summary.json", summary)
    conn.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cumulative local-DB reservoir search")
    parser.add_argument("--author-repo", required=True)
    parser.add_argument("--global-db", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["paper", "power_v2"], required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--p", type=int, default=7)
    parser.add_argument("--power", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seed-length", type=int, default=8)
    parser.add_argument("--seed-min-projlen", type=int, default=0)
    parser.add_argument("--seed-max-projlen", type=int, default=16)
    parser.add_argument("--seed-limit", type=int, default=0)
    parser.add_argument("--seed-order", choices=["projlen", "hash", "random"], default="projlen")
    parser.add_argument("--seed-shard-count", type=int, default=1)
    parser.add_argument("--seed-shard-index", type=int, default=0)
    parser.add_argument("--target-length", type=int, default=40)
    parser.add_argument("--bucket-size", type=int, default=3000)
    parser.add_argument("--use-best", type=int, default=50000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
