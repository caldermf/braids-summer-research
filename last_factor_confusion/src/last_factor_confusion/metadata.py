from __future__ import annotations

import hashlib
from pathlib import Path


STATUSES = {"clean", "truncated", "cancelled", "malformed"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest: dict) -> None:
    required = {
        "prime", "representation", "seed", "method", "length_range", "split",
        "model_config", "exact_evaluations", "best_projlen", "confusion_summary",
        "artifact_path", "artifact_checksum", "verifier_version", "status",
    }
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"Manifest missing fields: {sorted(missing)}")
    if manifest["status"] not in STATUSES:
        raise ValueError(f"Invalid status {manifest['status']!r}")

