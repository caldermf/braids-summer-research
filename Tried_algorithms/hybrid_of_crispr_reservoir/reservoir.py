from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .config import ReservoirConfig


def run_author_reservoir(
    config: ReservoirConfig,
    author_repo: str | Path,
    output_path: str | Path,
    python_executable: str | Path | None = None,
) -> Path:
    worker = Path(__file__).with_name("author_reservoir_worker.py")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_executable or sys.executable),
        str(worker),
        "--author-repo",
        str(Path(author_repo).resolve()),
        "--output",
        str(output.resolve()),
        "--n",
        str(config.n),
        "--r",
        str(config.r),
        "--p",
        str(config.p),
        "--bootstrap-depth",
        str(config.bootstrap_depth),
        "--target-depth",
        str(config.target_depth),
        "--step-size",
        str(config.step_size),
        "--bucket-size",
        str(config.bucket_size),
        "--use-best",
        str(config.use_best),
        "--seed",
        str(config.seed),
    ]
    if not config.stop_at_author_projlen_one:
        command.append("--continue-after-projlen-one")
    subprocess.run(command, check=True)
    return output
