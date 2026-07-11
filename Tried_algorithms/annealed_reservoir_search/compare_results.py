from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize matched paper and annealed reservoir runs."
    )
    parser.add_argument(
        "results_root",
        nargs="?",
        default="results/annealed_reservoir_search",
    )
    parser.add_argument("--output")
    return parser


def _last_progress(summary_path: Path) -> dict:
    progress_path = summary_path.parent / "progress.jsonl"
    if not progress_path.exists():
        return {}
    rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[-1] if rows else {}


def load_runs(root: Path) -> list[dict]:
    runs = []
    for summary_path in sorted(root.glob("**/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = summary["metadata"]
        final_progress = _last_progress(summary_path)
        runs.append(
            {
                "path": str(summary_path.parent),
                "selection_mode": metadata["selection_mode"],
                "seed": int(metadata["seed"]),
                "status": summary["status"],
                "kernel_hits": len(summary["exact_verification"]["kernel_hits"]),
                "actual_depth": int(metadata["actual_depth"]),
                "elapsed_seconds": float(metadata["elapsed_seconds"]),
                "cumulative_reservoir_offers": int(
                    metadata.get(
                        "cumulative_reservoir_offers",
                        final_progress.get("cumulative_reservoir_offers", 0),
                    )
                ),
                "final_min_author_projlen": final_progress.get(
                    "lowest_author_projlen"
                ),
                "final_selected_parents": final_progress.get("selected_braids"),
                "final_effective_buckets": final_progress.get(
                    "effective_selected_buckets"
                ),
            }
        )
    return runs


def aggregate(runs: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[run["selection_mode"]].append(run)

    output = {}
    for mode, items in sorted(grouped.items()):
        successes = sum(item["status"] == "kernel_found" for item in items)
        output[mode] = {
            "runs": len(items),
            "successes": successes,
            "success_rate": successes / len(items),
            "total_kernel_hits": sum(item["kernel_hits"] for item in items),
            "mean_elapsed_seconds": statistics.mean(
                item["elapsed_seconds"] for item in items
            ),
            "mean_reservoir_offers": statistics.mean(
                item["cumulative_reservoir_offers"] for item in items
            ),
            "median_final_min_author_projlen": statistics.median(
                item["final_min_author_projlen"] for item in items
            ),
            "seeds": sorted(item["seed"] for item in items),
        }
    return output


def main() -> None:
    args = _parser().parse_args()
    root = Path(args.results_root).resolve()
    runs = load_runs(root)
    if not runs:
        raise FileNotFoundError(f"no summary.json files found below {root}")
    payload = {"results_root": str(root), "runs": runs, "aggregate": aggregate(runs)}
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
