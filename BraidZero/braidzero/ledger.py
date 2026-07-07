from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from .core import append_jsonl, sha256_file, write_json


class RunLedger:
    def __init__(self, *, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = self.output_dir / "candidates.jsonl"
        self.collisions_path = self.output_dir / "collisions.jsonl"
        self.training_path = self.output_dir / "training_examples.jsonl"
        self.progress_path = self.output_dir / "progress.jsonl"
        self.ledger_path = self.output_dir / "run_ledger.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        for path in (
            self.candidates_path,
            self.collisions_path,
            self.training_path,
            self.progress_path,
            self.ledger_path,
        ):
            path.write_text("", encoding="utf-8")

    def progress(self, row: dict) -> None:
        append_jsonl(self.progress_path, {"time": time.time(), **row})

    def candidate(self, row: dict) -> None:
        append_jsonl(self.candidates_path, row)

    def collision(self, row: dict) -> None:
        append_jsonl(self.collisions_path, row)

    def training_example(self, row: dict) -> None:
        append_jsonl(self.training_path, row)

    def artifact_records(self) -> list[dict]:
        paths: Iterable[Path] = (
            self.output_dir / "config.json",
            self.output_dir / "oracle_summary.json",
            self.candidates_path,
            self.collisions_path,
            self.training_path,
            self.progress_path,
            self.summary_path,
        )
        records: list[dict] = []
        for path in paths:
            records.append(
                {
                    "artifact_path": str(path),
                    "artifact_checksum": sha256_file(path),
                    "artifact_bytes": path.stat().st_size if path.exists() else None,
                }
            )
        return records

    def finalize(self, *, summary: dict, ledger_row: dict) -> None:
        write_json(self.summary_path, summary)
        ledger_row = {
            **ledger_row,
            "artifacts": self.artifact_records(),
        }
        append_jsonl(self.ledger_path, ledger_row)
