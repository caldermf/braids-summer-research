#!/usr/bin/env python3
"""
Monte Carlo tree search with trajectory-aware surprise scoring.

This variant builds on monte_carlo_tree_search_surprise_beam.py. The older
surprise-beam search scores a candidate mostly by the current prefix's surprise.
Here, each beam item also carries the surprise-z history of the prefixes that
led to it, and the beam ranks candidates by a weighted mix of:

- latest prefix surprise-z
- average of the top-k surprise-z values seen along that trajectory

The goal is to prefer branches whose prefixes repeatedly look unusual, not just
branches that have one lucky low-projlen endpoint.
"""

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peyl.braid_data import (
    append_factor_to_burau_matrix,
    identity_burau_matrix,
)
from monte_carlo_algorithms.monte_carlo_tree_search_surprise_beam import (
    BucketReservoir,
    MCTSConfig,
    MCTSNode,
    MonteCarloTreeSearch,
)


@dataclass
class TrajectoryMCTSConfig(MCTSConfig):
    trajectory_top_k: int = 5
    latest_surprise_weight: float = 0.7
    history_surprise_weight: float = 0.3


class TrajectorySurpriseMCTS(MonteCarloTreeSearch):
    def __init__(self, config: TrajectoryMCTSConfig):
        if config.trajectory_top_k <= 0:
            raise ValueError("trajectory_top_k must be positive")
        if config.latest_surprise_weight < 0 or config.history_surprise_weight < 0:
            raise ValueError("trajectory surprise weights must be nonnegative")
        super().__init__(config)

    @property
    def config(self) -> TrajectoryMCTSConfig:
        return self._config

    @config.setter
    def config(self, value: TrajectoryMCTSConfig) -> None:
        self._config = value

    def create_run_directory(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = Path(self.config.output_dir)
        run_dir = base_dir / f"mcts_trajectory_surprise_{timestamp}_seed{self.config.seed}"
        suffix = 1
        unique_run_dir = run_dir
        while unique_run_dir.exists():
            suffix += 1
            unique_run_dir = base_dir / f"{run_dir.name}_{suffix}"
        unique_run_dir.mkdir(parents=True, exist_ok=False)
        return unique_run_dir

    def trajectory_signal(self, history: List[float], latest_surprise_z: float) -> Tuple[float, float]:
        values = history + [latest_surprise_z]
        top_k = sorted(values, reverse=True)[: self.config.trajectory_top_k]
        top_k_mean = sum(top_k) / len(top_k)
        signal = (
            self.config.latest_surprise_weight * latest_surprise_z
            + self.config.history_surprise_weight * top_k_mean
        )
        return signal, top_k_mean

    def score_prefix_with_history(
        self,
        factor_ids: List[int],
        burau_matrix,
        surprise_z_history: List[float],
    ) -> dict:
        score = self.score_prefix(factor_ids, burau_matrix)
        trajectory_signal, top_k_mean = self.trajectory_signal(
            surprise_z_history,
            float(score["surprise_z"]),
        )

        depth = int(score["depth"])
        value = trajectory_signal
        if depth > 0:
            value += 0.01 * float(score["surprise"]) / max(1.0, float(depth))
            value += 0.001 * min(depth, self.config.max_depth) / self.config.max_depth
        if score["kernel_match"].get("matches") and depth > 0:
            value += 1000.0

        full_history = surprise_z_history + [float(score["surprise_z"])]
        score["single_prefix_value"] = score["value"]
        score["value"] = value
        score["trajectory_signal"] = trajectory_signal
        score["trajectory_top_k_mean_surprise_z"] = top_k_mean
        score["trajectory_surprise_z_history"] = full_history
        score["state"]["single_prefix_value"] = score["single_prefix_value"]
        score["state"]["trajectory_signal"] = trajectory_signal
        score["state"]["trajectory_top_k_mean_surprise_z"] = top_k_mean
        score["state"]["trajectory_surprise_z_history"] = full_history
        return score

    def reconstruct_history_for_node(self, node: MCTSNode) -> Tuple[List[float], dict]:
        """
        Recompute prefix surprise history for a tree node's existing factor list.

        Tree nodes store only their endpoint matrix. The trajectory score needs
        prefix history too, so we replay the short GNF word from the identity.
        """
        if not node.factor_ids:
            start_score = self.score_prefix([], identity_burau_matrix(p=self.config.p, n=self.config.n))
            return [], start_score

        factor_ids = []
        matrix = identity_burau_matrix(p=self.config.p, n=self.config.n)
        history = []
        current_score = None
        for action in node.factor_ids:
            factor_ids = factor_ids + [action]
            matrix = append_factor_to_burau_matrix(
                current_matrix=matrix,
                factor_id=action,
                simple_table=self.simple_table,
                p=self.config.p,
            )
            current_score = self.score_prefix_with_history(factor_ids, matrix, history)
            history = list(current_score["trajectory_surprise_z_history"])

        return history, current_score

    def beam_frontier_playout(self, node: MCTSNode) -> dict:
        """
        Evaluate a node by beam/frontier expansion with trajectory-aware ranking.

        The beam is still the same beam as in surprise-beam MCTS. The difference
        is only the score: every candidate carries surprise-z history for the
        full prefix path, and candidates are ranked by trajectory value.
        """
        reservoir = BucketReservoir(self.config.reservoir_size, self.rng)
        kernel_hits = []

        start_history, start_score = self.reconstruct_history_for_node(node)
        if node.factor_ids:
            reservoir.add(start_score)
            if start_score["kernel_match"].get("matches"):
                kernel_hits.append(start_score)

        beam = [(list(node.factor_ids), node.burau_matrix, start_history)]
        while beam and len(beam[0][0]) < self.config.max_depth:
            candidates = []
            for factor_ids, burau_matrix, surprise_z_history in beam:
                actions = self.legal_actions_from_factors(factor_ids)
                if not actions:
                    continue

                shuffled_actions = list(actions)
                self.rng.shuffle(shuffled_actions)
                for action in shuffled_actions:
                    child_factor_ids = factor_ids + [action]
                    child_matrix = append_factor_to_burau_matrix(
                        current_matrix=burau_matrix,
                        factor_id=action,
                        simple_table=self.simple_table,
                        p=self.config.p,
                    )
                    score = self.score_prefix_with_history(
                        child_factor_ids,
                        child_matrix,
                        surprise_z_history,
                    )
                    reservoir.add(score)
                    candidates.append(
                        (
                            score,
                            child_factor_ids,
                            child_matrix,
                            score["trajectory_surprise_z_history"],
                        )
                    )
                    if score["kernel_match"].get("matches"):
                        kernel_hits.append(score)

            if not candidates:
                break

            candidates.sort(key=lambda item: item[0]["value"], reverse=True)
            beam = [
                (factor_ids, burau_matrix, surprise_z_history)
                for _, factor_ids, burau_matrix, surprise_z_history in candidates[
                    : self.config.beam_width
                ]
            ]

        best_prefix = reservoir.best_score() or start_score
        return {
            "best_prefix": best_prefix,
            "value": best_prefix["value"],
            "kernel_hits": kernel_hits,
            "reservoir_summary": reservoir.summary(),
        }

    def update_best(self, playout_result: dict) -> None:
        super().update_best(playout_result)
        if self.best_candidate is not None and "trajectory_signal" not in self.best_candidate:
            best_prefix = playout_result["best_prefix"]
            self.best_candidate["single_prefix_value"] = best_prefix.get("single_prefix_value")
            self.best_candidate["trajectory_signal"] = best_prefix.get("trajectory_signal")
            self.best_candidate["trajectory_top_k_mean_surprise_z"] = best_prefix.get(
                "trajectory_top_k_mean_surprise_z"
            )
            self.best_candidate["trajectory_surprise_z_history"] = best_prefix.get(
                "trajectory_surprise_z_history"
            )

    def log_iteration(self, iteration: int, path: List[int], playout_result: dict) -> None:
        leaf = self.nodes[path[-1]]
        best_prefix = playout_result["best_prefix"]
        record = {
            "iteration": iteration,
            "path": path,
            "path_depth": leaf.depth,
            "expanded_or_selected_node_id": leaf.node_id,
            "best_prefix_depth": best_prefix["depth"],
            "best_prefix_projlen": best_prefix["projlen"],
            "best_prefix_value": best_prefix["value"],
            "best_prefix_single_prefix_value": best_prefix.get("single_prefix_value"),
            "best_prefix_trajectory_signal": best_prefix.get("trajectory_signal"),
            "best_prefix_trajectory_top_k_mean_surprise_z": best_prefix.get(
                "trajectory_top_k_mean_surprise_z"
            ),
            "best_prefix_typical_projlen": best_prefix["typical_projlen"],
            "best_prefix_surprise": best_prefix["surprise"],
            "best_prefix_surprise_z": best_prefix["surprise_z"],
            "best_value": self.best_value,
            "best_projlen": self.best_projlen,
            "kernel_hits_this_iteration": len(playout_result["kernel_hits"]),
            "reservoir_summary": playout_result["reservoir_summary"],
            "best_prefix_state": best_prefix["state"],
        }
        with self.iterations_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def save_summary_json(self) -> None:
        super().save_summary_json()
        summary_path = self.run_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["trajectory_top_k"] = self.config.trajectory_top_k
        summary["latest_surprise_weight"] = self.config.latest_surprise_weight
        summary["history_surprise_weight"] = self.config.history_surprise_weight
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> TrajectoryMCTSConfig:
    parser = argparse.ArgumentParser(
        description="Run MCTS with trajectory-aware surprise beam playouts."
    )
    parser.add_argument("--p", type=int, default=7, help="Modulus for Burau arithmetic")
    parser.add_argument("--n", type=int, default=4, help="Number of braid strands")
    parser.add_argument("--max-depth", type=int, default=40, help="Maximum search depth")
    parser.add_argument("--iterations", type=int, default=1000, help="MCTS iterations")
    parser.add_argument("--exploration-weight", type=float, default=1.4)
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=256,
        help="Number of random walks used to estimate typical projlen by depth",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=8,
        help="Number of frontier states kept at each beam playout depth",
    )
    parser.add_argument(
        "--reservoir-size",
        type=int,
        default=16,
        help="Number of representative prefixes kept in each projlen bucket",
    )
    parser.add_argument(
        "--trajectory-top-k",
        type=int,
        default=5,
        help="How many best prefix surprise-z values to average for trajectory memory",
    )
    parser.add_argument(
        "--latest-surprise-weight",
        type=float,
        default=0.7,
        help="Weight on the current prefix's surprise-z",
    )
    parser.add_argument(
        "--history-surprise-weight",
        type=float,
        default=0.3,
        help="Weight on the average of top-k prior/current surprise-z values",
    )
    parser.add_argument(
        "--progressive-widening-k",
        type=float,
        default=0.5,
        help="Controls how many children a visited node may expand",
    )
    parser.add_argument(
        "--progressive-widening-alpha",
        type=float,
        default=0.5,
        help="Exponent for progressive widening; lower values go deeper sooner",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    return TrajectoryMCTSConfig(
        p=args.p,
        n=args.n,
        max_depth=args.max_depth,
        iterations=args.iterations,
        exploration_weight=args.exploration_weight,
        baseline_samples=args.baseline_samples,
        beam_width=args.beam_width,
        reservoir_size=args.reservoir_size,
        trajectory_top_k=args.trajectory_top_k,
        latest_surprise_weight=args.latest_surprise_weight,
        history_surprise_weight=args.history_surprise_weight,
        progressive_widening_k=args.progressive_widening_k,
        progressive_widening_alpha=args.progressive_widening_alpha,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def main() -> None:
    search = TrajectorySurpriseMCTS(parse_args())
    print(json.dumps(search.run(), indent=2))


if __name__ == "__main__":
    main()
