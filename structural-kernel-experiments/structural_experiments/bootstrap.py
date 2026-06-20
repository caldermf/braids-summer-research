from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTHOR_REPO = PROJECT_ROOT / "third_party" / "braids_project"


def ensure_author_peyl() -> Path:
    """Make the vendored paper implementation importable."""
    if not (AUTHOR_REPO / "peyl" / "braid.py").exists():
        raise FileNotFoundError(f"vendored peyl package is missing at {AUTHOR_REPO}")
    path = str(AUTHOR_REPO)
    if path not in sys.path:
        sys.path.insert(0, path)
    return AUTHOR_REPO
