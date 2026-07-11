from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable


def read_json(path: str | Path) -> Any:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if destination.suffix == ".gz" else open
    with opener(destination, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=None if destination.suffix == ".gz" else 2)
    return destination


def read_jsonl(path: str | Path) -> Iterable[dict]:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if destination.suffix == ".gz" else open
    with opener(destination, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return destination


def append_jsonl(path: str | Path, row: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")

