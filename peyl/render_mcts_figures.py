#!/usr/bin/env python3
"""
Render figures for an existing MCTS run directory.

If matplotlib is installed this writes PNGs. Otherwise it writes dependency-free
SVGs, which works on Bouchet's default Python.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from peyl.mcts_figure_utils import render_surprise_beam_figures


def parse_args():
    parser = argparse.ArgumentParser(description="Render figures from MCTS iterations.jsonl")
    parser.add_argument("run_dir", help="Run directory containing iterations.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    written = render_surprise_beam_figures(run_dir, plt=plt)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "format": "png" if plt is not None else "svg",
                "figures": [str(path) for path in written],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
