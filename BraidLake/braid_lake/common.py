from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Optional, Sequence


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def parse_factors(value: Any) -> tuple[int, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        value = json.loads(value)
    return tuple(int(x) for x in value)


def factor_ids_json(factors: Sequence[int]) -> str:
    return compact_json([int(x) for x in factors])


def braid_digest(n: int, factors: Sequence[int], infimum: int = 0) -> str:
    payload = {"n": int(n), "infimum": int(infimum), "factor_ids": [int(x) for x in factors]}
    return sha1_text("gnf-factor-digest-v1:" + compact_json(payload))


def safe_slug(text: str, *, max_prefix: int = 120) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "__", text.strip())
    cleaned = cleaned.strip("._-")
    digest = sha1_text(text)[:12]
    if len(cleaned) > max_prefix:
        cleaned = cleaned[:max_prefix].rstrip("._-")
    return f"{cleaned}__{digest}" if cleaned else digest


def lake_partition_dir(lake_root: Path, *, p: int, n: int, r: int, kind: str, source: str) -> Path:
    return (
        Path(lake_root)
        / f"p={int(p)}"
        / f"n={int(n)}_r={int(r)}"
        / f"kind={safe_slug(kind, max_prefix=40)}"
        / f"source={safe_slug(source)}"
    )


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def last_jsonl(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 1024 * 1024))
            lines = handle.read().decode("utf-8", "ignore").splitlines()
        for line in reversed(lines):
            if line.strip():
                return json.loads(line)
    except Exception:
        return None
    return None


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")


def source_from_run_dir(run_dir: Path) -> str:
    parts = Path(run_dir).parts
    if "results" in parts:
        idx = parts.index("results")
        return "/".join(parts[idx:])
    return str(run_dir)
