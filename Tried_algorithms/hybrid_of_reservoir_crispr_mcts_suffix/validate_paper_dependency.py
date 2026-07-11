from __future__ import annotations

import json
import sys

from .config import HybridConfig


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("the vendored paper peyl package requires Python >=3.10")

    source_root = HybridConfig().author_repo
    tracker_path = source_root / "peyl" / "braidsearch.py"
    if not tracker_path.is_file():
        raise SystemExit(f"vendored paper tracker is missing: {tracker_path}")

    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "the paper backbone requires NumPy and pandas; install "
            "requirements-cluster.txt into PYTHON_PATH"
        ) from exc

    sys.path.insert(0, str(source_root))
    import peyl  # type: ignore

    rep = peyl.JonesSummand(n=4, r=1, p=5)
    tracker = peyl.Tracker(
        rep=rep,
        bucket_size=2,
        bucket_keys=("length", "projlen"),
        criterion=lambda frame: frame["length"] >= 1,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "paper_source": str(source_root),
                "tracker": f"{type(tracker).__module__}.{type(tracker).__name__}",
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
