#!/usr/bin/env python3
"""
Monte Carlo tree search with late-breakout surprise scoring.

The score is designed around the p=5 diagnostics: known kernels became most
surprising at the endpoint. So this variant rewards high surprise late in the
search, plus a bonus when the current prefix breaks above its earlier best:

    score = depth_fraction^depth_power * latest_surprise_z
          + breakout_weight * max(0, latest_surprise_z - max_prior_surprise_z)

Temporary dips are allowed. A path only gets the breakout bonus when it reaches
a new surprise-z high relative to its own previous prefixes.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peyl.braid_data import append_factor_to_burau_matrix, identity_burau_matrix
from monte_carlo_algorithms.monte_carlo_tree_search_surprise_beam import (
    BucketReservoir,
    MCTSConfig,
    MCTSNode,
    MonteCarloTreeSearch,
)


@dataclass
class BreakoutMCTSConfig(MCTSConfig):
    breakout_weight: float = 0.5
    depth_power: float = 1.0


class BreakoutSurpriseMCTS(MonteCarloTreeSearch):
    def __init__(self, config: BreakoutMCTSConfig):
        if config.breakout_weight < 0:
            raise ValueError("breakout_weight must be nonnegative")
        if config.depth_power <= 0:
            raise ValueError("depth_power must be positive")
        super().__init__(config)

    @property
    def config(self) -> BreakoutMCTSConfig:
        return self._config

    @config.setter
    def config(self, value: BreakoutMCTSConfig) -> None:
        self._config = value

    def create_run_directory(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = Path(self.config.output_dir)
        run_dir = base_dir / f"mcts_breakout_surprise_{timestamp}_seed{self.config.seed}"
        suffix = 1
        unique_run_dir = run_dir
        while unique_run_dir.exists():
            suffix += 1
            unique_run_dir = base_dir / f"{run_dir.name}_{suffix}"
        unique_run_dir.mkdir(parents=True, exist_ok=False)
        return unique_run_dir

    def breakout_signal(self, history: List[float], latest_surprise_z: float, depth: int) -> Tuple[float, float, float]:
        prior_best = max(history) if history else latest_surprise_z
        breakout = max(0.0, latest_surprise_z - prior_best)
        depth_fraction = max(0.0, min(1.0, depth / max(1, self.config.max_depth)))
        depth_weight = depth_fraction ** self.config.depth_power
        signal = depth_weight * latest_surprise_z + self.config.breakout_weight * breakout
        return signal, breakout, depth_weight

    def score_prefix_with_history(self, factor_ids: List[int], burau_matrix, surprise_z_history: List[float]) -> dict:
        score = self.score_prefix(factor_ids, burau_matrix)
        depth = int(score["depth"])
        latest_surprise_z = float(score["surprise_z"])
        signal, breakout, depth_weight = self.breakout_signal(
            surprise_z_history,
            latest_surprise_z,
            depth,
        )

        value = signal
        if depth > 0:
            value += 0.01 * float(score["surprise"]) / max(1.0, float(depth))
            value += 0.001 * min(depth, self.config.max_depth) / self.config.max_depth
        if score["kernel_match"].get("matches") and depth > 0:
            value += 1000.0

        full_history = surprise_z_history + [latest_surprise_z]
        score["single_prefix_value"] = score["value"]
        score["value"] = value
        score["breakout_signal"] = signal
        score["breakout_surprise_z"] = breakout
        score["depth_weight"] = depth_weight
        score["surprise_z_history"] = full_history
        score["state"]["single_prefix_value"] = score["single_prefix_value"]
        score["state"]["breakout_signal"] = signal
        score["state"]["breakout_surprise_z"] = breakout
        score["state"]["depth_weight"] = depth_weight
        score["state"]["surprise_z_history"] = full_history
        return score

    def reconstruct_history_for_node(self, node: MCTSNode) -> Tuple[List[float], dict]:
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
            history = list(current_score["surprise_z_history"])
        return history, current_score

    def beam_frontier_playout(self, node: MCTSNode) -> dict:
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
                        (score, child_factor_ids, child_matrix, score["surprise_z_history"])
                    )
                    if score["kernel_match"].get("matches"):
                        kernel_hits.append(score)

            if not candidates:
                break

            candidates.sort(key=lambda item: item[0]["value"], reverse=True)
            beam = [
                (factor_ids, burau_matrix, surprise_z_history)
                for _, factor_ids, burau_matrix, surprise_z_history in candidates[: self.config.beam_width]
            ]

        best_prefix = reservoir.best_score() or start_score
        return {
            "best_prefix": best_prefix,
            "value": best_prefix["value"],
            "kernel_hits": kernel_hits,
            "reservoir_summary": reservoir.summary(),
        }

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
            "best_prefix_breakout_signal": best_prefix.get("breakout_signal"),
            "best_prefix_breakout_surprise_z": best_prefix.get("breakout_surprise_z"),
            "best_prefix_depth_weight": best_prefix.get("depth_weight"),
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
        summary["breakout_weight"] = self.config.breakout_weight
        summary["depth_power"] = self.config.depth_power
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> BreakoutMCTSConfig:
    parser = argparse.ArgumentParser(description="Run MCTS with late-breakout surprise scoring.")
    parser.add_argument("--p", type=int, default=7)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--exploration-weight", type=float, default=1.4)
    parser.add_argument("--baseline-samples", type=int, default=256)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--reservoir-size", type=int, default=16)
    parser.add_argument("--breakout-weight", type=float, default=0.5)
    parser.add_argument("--depth-power", type=float, default=1.0)
    parser.add_argument("--progressive-widening-k", type=float, default=0.5)
    parser.add_argument("--progressive-widening-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    return BreakoutMCTSConfig(
        p=args.p,
        n=args.n,
        max_depth=args.max_depth,
        iterations=args.iterations,
        exploration_weight=args.exploration_weight,
        baseline_samples=args.baseline_samples,
        beam_width=args.beam_width,
        reservoir_size=args.reservoir_size,
        breakout_weight=args.breakout_weight,
        depth_power=args.depth_power,
        progressive_widening_k=args.progressive_widening_k,
        progressive_widening_alpha=args.progressive_widening_alpha,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def main() -> None:
    search = BreakoutSurpriseMCTS(parse_args())
    print(json.dumps(search.run(), indent=2))


if __name__ == "__main__":
    main()
