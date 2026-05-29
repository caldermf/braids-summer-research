#!/usr/bin/env python3
"""
Diagnose whether projlen-surprise lights up on a known kernel trajectory.

Given a known GNF JSON file, this script computes prefix projlen, typical
projlen, surprise, and surprise z-score for every prefix. It also samples random
control trajectories with the same max length and writes CSV/JSON plus figures.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/braids_mcts_matplotlib")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from peyl.braid_data import (
    GNF,
    append_factor_to_burau_matrix,
    identity_burau_matrix,
    polynomial_matrix_projlen,
    projective_kernel_match,
    simple_factor_burau_table,
    simple_factor_id_maps,
    valid_first_factor_ids,
    valid_suffix_factor_ids,
)
from peyl.mcts_figure_utils import write_line_plot
from monte_carlo_algorithms.monte_carlo_tree_search_surprise_beam import SurpriseBaseline


def load_factor_ids(path, n):
    data = json.loads(Path(path).read_text())
    if "factor_ids" in data:
        return [int(item) for item in data["factor_ids"]], data
    if "gnf_factors" in data:
        perm_to_id, _ = simple_factor_id_maps(n)
        factor_ids = [perm_to_id[tuple(perm)] for perm in data["gnf_factors"]]
        return factor_ids, data
    raise ValueError(f"{path} must contain factor_ids or gnf_factors")


def legal_actions_from_factors(factor_ids, n, delta_factor_id):
    if n == 2:
        return [delta_factor_id]
    if not factor_ids:
        return valid_first_factor_ids(n=n)
    return valid_suffix_factor_ids(factor_ids[-1], n=n)


def update_baseline_random_walks(p, n, max_depth, samples, simple_table, rng):
    baseline = SurpriseBaseline(max_depth)
    delta_factor_id = simple_factor_id_maps(n)[0][GNF.delta_perm(n)]

    for _ in range(samples):
        factor_ids = []
        matrix = identity_burau_matrix(p=p, n=n)
        for depth in range(1, max_depth + 1):
            actions = legal_actions_from_factors(factor_ids, n, delta_factor_id)
            if not actions:
                break
            action = rng.choice(actions)
            factor_ids = factor_ids + [action]
            matrix = append_factor_to_burau_matrix(
                current_matrix=matrix,
                factor_id=action,
                simple_table=simple_table,
                p=p,
            )
            baseline.add(depth, polynomial_matrix_projlen(matrix))
    return baseline


def compute_trajectory(factor_ids, p, n, simple_table, baseline):
    rows = []
    matrix = identity_burau_matrix(p=p, n=n)
    prefix_ids = []
    for factor_id in factor_ids:
        prefix_ids.append(int(factor_id))
        matrix = append_factor_to_burau_matrix(
            current_matrix=matrix,
            factor_id=int(factor_id),
            simple_table=simple_table,
            p=p,
        )
        depth = len(prefix_ids)
        projlen = polynomial_matrix_projlen(matrix)
        typical = baseline.mean(depth)
        surprise = baseline.surprise(depth, projlen)
        surprise_z = baseline.surprise_z(depth, projlen)
        kernel_match = projective_kernel_match(matrix, p=p, n=n)
        rows.append(
            {
                "depth": depth,
                "projlen": projlen,
                "support_width": projlen + 1,
                "typical_projlen": typical,
                "surprise": surprise,
                "surprise_z": surprise_z,
                "kernel_match": bool(kernel_match.get("matches")),
                "kernel_type": kernel_match.get("kernel_type"),
            }
        )
    return rows


def random_control_trajectories(p, n, max_depth, count, simple_table, baseline, rng):
    controls = []
    delta_factor_id = simple_factor_id_maps(n)[0][GNF.delta_perm(n)]
    for control_id in range(count):
        factor_ids = []
        matrix = identity_burau_matrix(p=p, n=n)
        for depth in range(1, max_depth + 1):
            actions = legal_actions_from_factors(factor_ids, n, delta_factor_id)
            if not actions:
                break
            action = rng.choice(actions)
            factor_ids = factor_ids + [action]
            matrix = append_factor_to_burau_matrix(
                current_matrix=matrix,
                factor_id=action,
                simple_table=simple_table,
                p=p,
            )
            projlen = polynomial_matrix_projlen(matrix)
            controls.append(
                {
                    "control_id": control_id,
                    "depth": depth,
                    "projlen": projlen,
                    "surprise": baseline.surprise(depth, projlen),
                    "surprise_z": baseline.surprise_z(depth, projlen),
                }
            )
    return controls


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def control_means(controls, key):
    by_depth = {}
    for row in controls:
        by_depth.setdefault(row["depth"], []).append(float(row[key]))
    return {
        depth: sum(values) / len(values)
        for depth, values in by_depth.items()
    }


def render_figures(out_dir, kernel_rows, controls):
    figures_dir = Path(out_dir) / "figures"
    depths = [row["depth"] for row in kernel_rows]
    control_projlen = control_means(controls, "projlen")
    control_surprise = control_means(controls, "surprise")
    control_surprise_z = control_means(controls, "surprise_z")

    specs = [
        ("projlen", "Known kernel prefix projlen", "Projlen", "known_kernel_prefix_projlen"),
        ("surprise", "Known kernel prefix surprise", "Typical projlen minus observed", "known_kernel_prefix_surprise"),
        ("surprise_z", "Known kernel prefix surprise z-score", "Surprise z-score", "known_kernel_prefix_surprise_z"),
    ]
    for key, title, ylabel, stem in specs:
        write_line_plot(
            depths,
            [row[key] for row in kernel_rows],
            title,
            ylabel,
            figures_dir,
            stem,
            plt=plt,
        )

    overlay_specs = [
        ("projlen", control_projlen, "Known kernel vs random mean projlen", "Projlen", "known_vs_random_mean_projlen"),
        ("surprise", control_surprise, "Known kernel vs random mean surprise", "Surprise", "known_vs_random_mean_surprise"),
        ("surprise_z", control_surprise_z, "Known kernel vs random mean surprise z-score", "Surprise z-score", "known_vs_random_mean_surprise_z"),
    ]
    for key, control_mean, title, ylabel, stem in overlay_specs:
        # The dependency-free helper only writes one line, so emit the known
        # curve here. The CSV contains the random mean for exact comparison.
        write_line_plot(
            depths,
            [row[key] for row in kernel_rows],
            title,
            ylabel,
            figures_dir,
            stem,
            plt=plt,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose surprise on a known kernel GNF")
    parser.add_argument("--gnf-json", required=True, help="Known kernel GNF JSON")
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--baseline-samples", type=int, default=1024)
    parser.add_argument("--random-controls", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="results/known_kernel_surprise")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    factor_ids, source = load_factor_ids(args.gnf_json, args.n)
    max_depth = len(factor_ids)
    simple_table = simple_factor_burau_table(p=args.p, n=args.n)

    out_base = Path(args.output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = out_base / f"known_kernel_surprise_p{args.p}_len{max_depth}_{timestamp}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=False)

    baseline = update_baseline_random_walks(
        p=args.p,
        n=args.n,
        max_depth=max_depth,
        samples=args.baseline_samples,
        simple_table=simple_table,
        rng=rng,
    )
    kernel_rows = compute_trajectory(factor_ids, args.p, args.n, simple_table, baseline)
    controls = random_control_trajectories(
        p=args.p,
        n=args.n,
        max_depth=max_depth,
        count=args.random_controls,
        simple_table=simple_table,
        baseline=baseline,
        rng=rng,
    )

    write_csv(
        out_dir / "known_kernel_prefix_metrics.csv",
        kernel_rows,
        ["depth", "projlen", "support_width", "typical_projlen", "surprise", "surprise_z", "kernel_match", "kernel_type"],
    )
    write_csv(
        out_dir / "random_control_prefix_metrics.csv",
        controls,
        ["control_id", "depth", "projlen", "surprise", "surprise_z"],
    )
    with (out_dir / "typical_projlen_by_depth.json").open("w", encoding="utf-8") as f:
        json.dump(baseline.to_json(), f, indent=2)
    with (out_dir / "source_gnf.json").open("w", encoding="utf-8") as f:
        json.dump(source, f, indent=2)

    render_figures(out_dir, kernel_rows, controls)

    final = kernel_rows[-1]
    summary = {
        "out_dir": str(out_dir),
        "gnf_json": args.gnf_json,
        "p": args.p,
        "n": args.n,
        "garside_length": max_depth,
        "baseline_samples": args.baseline_samples,
        "random_controls": args.random_controls,
        "final_projlen": final["projlen"],
        "final_kernel_match": final["kernel_match"],
        "max_surprise": max(row["surprise"] for row in kernel_rows),
        "max_surprise_z": max(row["surprise_z"] for row in kernel_rows),
        "depth_of_max_surprise_z": max(kernel_rows, key=lambda row: row["surprise_z"])["depth"],
        "final_surprise": final["surprise"],
        "final_surprise_z": final["surprise_z"],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
