#!/usr/bin/env python3
"""
Continue a saved MCTS candidate with surprise-scored beam search.

This is a focused completion search. It starts from a saved candidate prefix,
then searches only legal GNF suffixes up to max_depth. It is meant to answer:
"Can this unusually low-projlen prefix be completed to a kernel element?"
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/braids_mcts_matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from peyl.braid_data import (
    GNF,
    append_factor_to_burau_matrix,
    identity_burau_matrix,
    polynomial_matrix_projlen,
    projective_kernel_match,
    serialize_prefix_state,
    simple_factor_burau_table,
    simple_factor_id_maps,
    valid_first_factor_ids,
    valid_suffix_factor_ids,
)
from peyl.mcts_figure_utils import write_line_plot
from monte_carlo_algorithms.monte_carlo_tree_search_surprise_beam import (
    BucketReservoir,
    SurpriseBaseline,
)


@dataclass
class ContinuationConfig:
    candidate_json: str
    p: int = 5
    n: int = 4
    max_depth: int = 59
    baseline_samples: int = 512
    beam_width: int = 32
    reservoir_size: int = 32
    score_noise: float = 0.0
    seed: int = 1
    output_dir: str = "results/continuations"


class ContinuationSearch:
    def __init__(self, config: ContinuationConfig):
        if config.n < 2:
            raise ValueError("n must be at least 2")
        if config.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if config.baseline_samples <= 0:
            raise ValueError("baseline_samples must be positive")

        self.config = config
        self.rng = random.Random(config.seed)
        self.simple_table = simple_factor_burau_table(p=config.p, n=config.n)
        self.delta_factor_id = simple_factor_id_maps(config.n)[0][GNF.delta_perm(config.n)]
        self.candidate = self.load_candidate(config.candidate_json)
        self.start_factor_ids = [int(item) for item in self.candidate["factor_ids"]]
        if len(self.start_factor_ids) >= config.max_depth:
            raise ValueError("candidate length must be smaller than max_depth")

        self.start_matrix = self.matrix_from_factor_ids(self.start_factor_ids)
        self.baseline = self.estimate_surprise_baseline()
        self.best_score: Optional[dict] = None
        self.kernel_hits: List[dict] = []

        self.run_dir = self.create_run_directory()
        self.figures_dir = self.run_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.iterations_path = self.run_dir / "iterations.jsonl"

        with (self.run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2)
        with (self.run_dir / "start_candidate.json").open("w", encoding="utf-8") as f:
            json.dump(self.candidate, f, indent=2)
        with (self.run_dir / "typical_projlen_by_depth.json").open("w", encoding="utf-8") as f:
            json.dump(self.baseline.to_json(), f, indent=2)

    def load_candidate(self, path):
        path = Path(path)
        candidate = json.loads(path.read_text())
        if "factor_ids" not in candidate:
            raise ValueError(f"{path} does not contain factor_ids")
        return candidate

    def create_run_directory(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = Path(self.config.output_dir)
        start_len = len(self.start_factor_ids)
        run_dir = base_dir / f"continue_len{start_len}_to{self.config.max_depth}_{timestamp}_seed{self.config.seed}"
        suffix = 1
        unique_run_dir = run_dir
        while unique_run_dir.exists():
            suffix += 1
            unique_run_dir = base_dir / f"{run_dir.name}_{suffix}"
        unique_run_dir.mkdir(parents=True, exist_ok=False)
        return unique_run_dir

    def matrix_from_factor_ids(self, factor_ids):
        matrix = identity_burau_matrix(p=self.config.p, n=self.config.n)
        for factor_id in factor_ids:
            matrix = append_factor_to_burau_matrix(
                current_matrix=matrix,
                factor_id=factor_id,
                simple_table=self.simple_table,
                p=self.config.p,
            )
        return matrix

    def first_actions(self) -> List[int]:
        if self.config.n == 2:
            return [self.delta_factor_id]
        return valid_first_factor_ids(n=self.config.n)

    def legal_actions_from_factors(self, factor_ids: List[int]) -> List[int]:
        if self.config.n == 2:
            return [self.delta_factor_id]
        if not factor_ids:
            return self.first_actions()
        return valid_suffix_factor_ids(factor_ids[-1], n=self.config.n)

    def estimate_surprise_baseline(self) -> SurpriseBaseline:
        baseline = SurpriseBaseline(self.config.max_depth)
        for _ in range(self.config.baseline_samples):
            factor_ids = []
            matrix = identity_burau_matrix(p=self.config.p, n=self.config.n)
            for depth in range(1, self.config.max_depth + 1):
                actions = self.legal_actions_from_factors(factor_ids)
                if not actions:
                    break
                action = self.rng.choice(actions)
                factor_ids = factor_ids + [action]
                matrix = append_factor_to_burau_matrix(
                    current_matrix=matrix,
                    factor_id=action,
                    simple_table=self.simple_table,
                    p=self.config.p,
                )
                baseline.add(depth, polynomial_matrix_projlen(matrix))
        return baseline

    def score_prefix(self, factor_ids, matrix):
        projlen = polynomial_matrix_projlen(matrix)
        depth = len(factor_ids)
        kernel_match = projective_kernel_match(matrix, p=self.config.p, n=self.config.n)
        typical_projlen = self.baseline.mean(depth)
        surprise = self.baseline.surprise(depth, projlen)
        surprise_z = self.baseline.surprise_z(depth, projlen)
        value = self.value_from_surprise(surprise, surprise_z, kernel_match, depth)
        state = serialize_prefix_state(
            factor_ids,
            poly_mat=matrix,
            p=self.config.p,
            n=self.config.n,
        )
        state["gnf_d"] = 0
        return {
            "factor_ids": list(factor_ids),
            "depth": depth,
            "projlen": projlen,
            "projlen_per_length": projlen / max(1, depth),
            "typical_projlen": typical_projlen,
            "surprise": surprise,
            "surprise_z": surprise_z,
            "kernel_match": kernel_match,
            "value": value,
            "state": state,
        }

    def value_from_surprise(self, surprise, surprise_z, kernel_match, depth):
        if depth <= 0:
            value = 0.0
        else:
            value = surprise_z
            value += 0.01 * surprise / max(1.0, float(depth))
            value += 0.001 * min(depth, self.config.max_depth) / self.config.max_depth
        if kernel_match.get("matches") and depth > 0:
            value += 1000.0
        return value

    def rank_value(self, score):
        if self.config.score_noise <= 0:
            return score["value"]
        return score["value"] + self.rng.gauss(0.0, self.config.score_noise)

    def update_best(self, score):
        if self.best_score is None or score["value"] > self.best_score["value"]:
            self.best_score = score

    def run(self):
        start_time = time.time()
        reservoir = BucketReservoir(self.config.reservoir_size, self.rng)
        start_score = self.score_prefix(self.start_factor_ids, self.start_matrix)
        reservoir.add(start_score)
        self.update_best(start_score)
        if start_score["kernel_match"].get("matches"):
            self.kernel_hits.append(start_score)

        beam = [(list(self.start_factor_ids), self.start_matrix)]
        step = 0

        while beam and len(beam[0][0]) < self.config.max_depth:
            step += 1
            candidates = []
            for factor_ids, matrix in beam:
                actions = self.legal_actions_from_factors(factor_ids)
                shuffled_actions = list(actions)
                self.rng.shuffle(shuffled_actions)
                for action in shuffled_actions:
                    child_factor_ids = factor_ids + [action]
                    child_matrix = append_factor_to_burau_matrix(
                        current_matrix=matrix,
                        factor_id=action,
                        simple_table=self.simple_table,
                        p=self.config.p,
                    )
                    score = self.score_prefix(child_factor_ids, child_matrix)
                    reservoir.add(score)
                    self.update_best(score)
                    candidates.append((self.rank_value(score), score, child_factor_ids, child_matrix))
                    if score["kernel_match"].get("matches"):
                        self.kernel_hits.append(score)

            if not candidates:
                break

            candidates.sort(key=lambda item: item[0], reverse=True)
            beam = [
                (factor_ids, matrix)
                for _, _, factor_ids, matrix in candidates[: self.config.beam_width]
            ]
            best_this_step = max((item[1] for item in candidates), key=lambda score: score["value"])
            record = {
                "iteration": step,
                "path_depth": len(beam[0][0]) if beam else None,
                "best_prefix_depth": best_this_step["depth"],
                "best_prefix_projlen": best_this_step["projlen"],
                "best_prefix_value": best_this_step["value"],
                "best_prefix_typical_projlen": best_this_step["typical_projlen"],
                "best_prefix_surprise": best_this_step["surprise"],
                "best_prefix_surprise_z": best_this_step["surprise_z"],
                "best_value": self.best_score["value"],
                "best_projlen": self.best_score["projlen"],
                "kernel_hits_this_iteration": sum(
                    1 for _, score, _, _ in candidates if score["kernel_match"].get("matches")
                ),
                "num_candidates": len(candidates),
                "beam_width": len(beam),
                "reservoir_summary": reservoir.summary(),
                "best_prefix_state": best_this_step["state"],
            }
            with self.iterations_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        self.save_outputs(reservoir, elapsed_sec=round(time.time() - start_time, 4))
        self.write_plots()
        return self.summary(elapsed_sec=round(time.time() - start_time, 4))

    def summary(self, elapsed_sec):
        return {
            "run_dir": str(self.run_dir),
            "start_garside_length": len(self.start_factor_ids),
            "max_depth": self.config.max_depth,
            "beam_width": self.config.beam_width,
            "best_value": self.best_score["value"] if self.best_score else None,
            "best_projlen": self.best_score["projlen"] if self.best_score else None,
            "best_garside_length": self.best_score["depth"] if self.best_score else None,
            "num_kernel_hits": len(self.kernel_hits),
            "elapsed_sec": elapsed_sec,
        }

    def save_outputs(self, reservoir, elapsed_sec):
        best_state = dict(self.best_score["state"])
        best_state["value"] = self.best_score["value"]
        best_state["typical_projlen"] = self.best_score["typical_projlen"]
        best_state["surprise"] = self.best_score["surprise"]
        best_state["surprise_z"] = self.best_score["surprise_z"]
        with (self.run_dir / "best_candidate.json").open("w", encoding="utf-8") as f:
            json.dump(best_state, f, indent=2)

        hits = []
        for hit_score in self.kernel_hits:
            hit = dict(hit_score["state"])
            hit["value"] = hit_score["value"]
            hit["typical_projlen"] = hit_score["typical_projlen"]
            hit["surprise"] = hit_score["surprise"]
            hit["surprise_z"] = hit_score["surprise_z"]
            hits.append(hit)
        with (self.run_dir / "kernel_hits.json").open("w", encoding="utf-8") as f:
            json.dump(hits, f, indent=2)

        with (self.run_dir / "reservoir_summary.json").open("w", encoding="utf-8") as f:
            json.dump(reservoir.summary(), f, indent=2)

        with (self.run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(self.summary(elapsed_sec), f, indent=2)

    def write_plots(self):
        if not self.iterations_path.exists():
            return
        records = []
        with self.iterations_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            return
        iterations = [record["iteration"] for record in records]

        specs = [
            ("best_prefix_projlen", "Best child projective length per continuation step", "Best child projlen", "best_prefix_projlen_per_iteration"),
            ("best_projlen", "Best projective length over time", "Best projlen", "best_projlen_over_time"),
            ("best_value", "Best value over time", "Best value", "best_value_over_time"),
            ("best_prefix_surprise", "Best child surprise per continuation step", "Typical projlen minus observed projlen", "best_prefix_surprise_per_iteration"),
            ("best_prefix_surprise_z", "Best child surprise z-score per continuation step", "Surprise z-score", "best_prefix_surprise_z_per_iteration"),
            ("path_depth", "Continuation depth over time", "Depth", "selected_depth_over_time"),
            ("kernel_hits_this_iteration", "Kernel hits per continuation step", "Kernel hits", "kernel_hits_per_iteration"),
        ]
        for key, title, ylabel, stem in specs:
            if key not in records[0]:
                continue
            write_line_plot(
                iterations,
                [record[key] for record in records],
                title,
                ylabel,
                self.figures_dir,
                stem,
                plt=plt if HAS_MATPLOTLIB else None,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Continue a saved MCTS candidate by beam search")
    parser.add_argument("--candidate-json", required=True, help="Path to best_candidate.json")
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=59)
    parser.add_argument("--baseline-samples", type=int, default=512)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--reservoir-size", type=int, default=32)
    parser.add_argument("--score-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="results/continuations")
    args = parser.parse_args()
    return ContinuationConfig(
        candidate_json=args.candidate_json,
        p=args.p,
        n=args.n,
        max_depth=args.max_depth,
        baseline_samples=args.baseline_samples,
        beam_width=args.beam_width,
        reservoir_size=args.reservoir_size,
        score_noise=args.score_noise,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def main():
    search = ContinuationSearch(parse_args())
    print(json.dumps(search.run(), indent=2))


if __name__ == "__main__":
    main()
