#!/usr/bin/env python3
"""
Monte Carlo tree search with surprise scoring and beam/frontier playouts.

This version changes the playout evaluator in two ways:
- projlen is scored relative to an estimated typical projlen at the same length
- each playout keeps a beam/frontier of promising descendants instead of a
  single random or greedy path

For n=2, the search permits the Delta/simple generator as the only move. For
n>2, the search uses the usual positive GNF factors and excludes Delta.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/braids_mcts_matplotlib")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from braid_data import (
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


@dataclass
class MCTSConfig:
    p: int = 7
    n: int = 4
    max_depth: int = 40
    iterations: int = 1000
    exploration_weight: float = 1.4
    baseline_samples: int = 256
    beam_width: int = 8
    reservoir_size: int = 16
    progressive_widening_k: float = 0.5
    progressive_widening_alpha: float = 0.5
    seed: int = 1
    output_dir: str = "results"


@dataclass
class MCTSNode:
    node_id: int
    factor_ids: List[int]
    parent_id: Optional[int]
    action_from_parent: Optional[int]
    burau_matrix: object
    depth: int
    visits: int = 0
    total_value: float = 0.0
    children: Dict[int, int] = field(default_factory=dict)
    untried_actions: Optional[List[int]] = None

    def average_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits


class BucketReservoir:
    """
    Bounded uniform reservoirs inside projective-length buckets.

    The buckets make the playout evidence inspectable: a node was promising
    because its descendants reached, say, projlen 0/2/4, not just because one
    scalar reward happened to be large.
    """

    def __init__(self, size_per_bucket: int, rng: random.Random):
        if size_per_bucket <= 0:
            raise ValueError("reservoir_size must be positive")
        self.size_per_bucket = size_per_bucket
        self.rng = rng
        self.buckets: Dict[int, dict] = {}

    def add(self, score: dict) -> None:
        projlen = int(score["projlen"])
        bucket = self.buckets.setdefault(projlen, {"seen": 0, "items": []})
        bucket["seen"] += 1

        items = bucket["items"]
        if len(items) < self.size_per_bucket:
            items.append(score)
            return

        replace_at = self.rng.randrange(bucket["seen"])
        if replace_at < self.size_per_bucket:
            items[replace_at] = score

    def best_score(self) -> Optional[dict]:
        best = None
        for bucket in self.buckets.values():
            for item in bucket["items"]:
                if best is None or item["value"] > best["value"]:
                    best = item
        return best

    def summary(self) -> List[dict]:
        rows = []
        for projlen in sorted(self.buckets):
            bucket = self.buckets[projlen]
            rows.append(
                {
                    "projlen": projlen,
                    "seen": bucket["seen"],
                    "kept": len(bucket["items"]),
                    "best_depth": max(item["depth"] for item in bucket["items"]),
                    "best_value": max(item["value"] for item in bucket["items"]),
                }
            )
        return rows


class SurpriseBaseline:
    """
    Online mean/std table for typical projlen by braid length.
    """

    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.counts = [0 for _ in range(max_depth + 1)]
        self.means = [0.0 for _ in range(max_depth + 1)]
        self.m2 = [0.0 for _ in range(max_depth + 1)]

    def add(self, depth: int, projlen: int) -> None:
        if depth < 0 or depth > self.max_depth:
            return
        self.counts[depth] += 1
        count = self.counts[depth]
        delta = float(projlen) - self.means[depth]
        self.means[depth] += delta / count
        delta2 = float(projlen) - self.means[depth]
        self.m2[depth] += delta * delta2

    def mean(self, depth: int) -> float:
        if depth < 0 or depth > self.max_depth:
            return 2.0 * depth
        if self.counts[depth] == 0:
            return 2.0 * depth
        return self.means[depth]

    def std(self, depth: int) -> float:
        if depth < 0 or depth > self.max_depth or self.counts[depth] < 2:
            return 1.0
        return max(1.0, math.sqrt(self.m2[depth] / (self.counts[depth] - 1)))

    def surprise(self, depth: int, projlen: int) -> float:
        return self.mean(depth) - float(projlen)

    def surprise_z(self, depth: int, projlen: int) -> float:
        return self.surprise(depth, projlen) / self.std(depth)

    def to_json(self) -> List[dict]:
        rows = []
        for depth in range(1, self.max_depth + 1):
            if self.counts[depth] == 0:
                continue
            rows.append(
                {
                    "depth": depth,
                    "count": self.counts[depth],
                    "mean_projlen": self.mean(depth),
                    "std_projlen": self.std(depth),
                }
            )
        return rows


class MonteCarloTreeSearch:
    def __init__(self, config: MCTSConfig):
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
        self.baseline = self.estimate_surprise_baseline()

        self.nodes: Dict[int, MCTSNode] = {}
        root = self.make_root_node()
        self.nodes[root.node_id] = root
        self.root_id = root.node_id
        self.next_node_id = 1

        self.best_candidate: Optional[dict] = None
        self.best_value = float("-inf")
        self.best_projlen: Optional[int] = None
        self.best_candidate_by_depth: Dict[int, dict] = {}
        self.best_projlen_by_depth: Dict[int, int] = {}
        self.kernel_hits: List[dict] = []

        self.run_dir = self.create_run_directory()
        self.figures_dir = self.run_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.iterations_path = self.run_dir / "iterations.jsonl"
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2)
        with (self.run_dir / "typical_projlen_by_depth.json").open("w", encoding="utf-8") as f:
            json.dump(self.baseline.to_json(), f, indent=2)

    def create_run_directory(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = Path(self.config.output_dir)
        run_dir = base_dir / f"mcts_surprise_beam_{timestamp}_seed{self.config.seed}"
        suffix = 1
        unique_run_dir = run_dir
        while unique_run_dir.exists():
            suffix += 1
            unique_run_dir = base_dir / f"{run_dir.name}_{suffix}"
        unique_run_dir.mkdir(parents=True, exist_ok=False)
        return unique_run_dir

    def estimate_surprise_baseline(self) -> SurpriseBaseline:
        """
        Estimate typical projlen by length using random valid GNF walks.

        This gives the search a length-aware target: a length-40 braid with
        projlen 30 may be much more interesting than a length-1 braid with
        projlen 2, even though the raw projlen is larger.
        """
        baseline = SurpriseBaseline(self.config.max_depth)

        for _ in range(self.config.baseline_samples):
            factor_ids = []
            burau_matrix = identity_burau_matrix(p=self.config.p, n=self.config.n)

            for depth in range(1, self.config.max_depth + 1):
                actions = self.legal_actions_from_factors(factor_ids)
                if not actions:
                    break
                action = self.rng.choice(actions)
                factor_ids = factor_ids + [action]
                burau_matrix = append_factor_to_burau_matrix(
                    current_matrix=burau_matrix,
                    factor_id=action,
                    simple_table=self.simple_table,
                    p=self.config.p,
                )
                baseline.add(depth, polynomial_matrix_projlen(burau_matrix))

        return baseline

    def make_root_node(self) -> MCTSNode:
        return MCTSNode(
            node_id=0,
            factor_ids=[],
            parent_id=None,
            action_from_parent=None,
            burau_matrix=identity_burau_matrix(p=self.config.p, n=self.config.n),
            depth=0,
            untried_actions=self.first_actions(),
        )

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

    def legal_actions(self, node: MCTSNode) -> List[int]:
        return self.legal_actions_from_factors(node.factor_ids)

    def create_child(self, parent: MCTSNode, action: int) -> MCTSNode:
        child_factor_ids = parent.factor_ids + [action]
        child_matrix = append_factor_to_burau_matrix(
            current_matrix=parent.burau_matrix,
            factor_id=action,
            simple_table=self.simple_table,
            p=self.config.p,
        )
        child = MCTSNode(
            node_id=self.next_node_id,
            factor_ids=child_factor_ids,
            parent_id=parent.node_id,
            action_from_parent=action,
            burau_matrix=child_matrix,
            depth=parent.depth + 1,
            untried_actions=None,
        )
        child.untried_actions = self.legal_actions(child)
        self.nodes[child.node_id] = child
        parent.children[action] = child.node_id
        self.next_node_id += 1
        return child

    def ucb_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        if child.visits == 0:
            return float("inf")
        if parent.visits == 0:
            return child.average_value()
        exploration = self.config.exploration_weight * math.sqrt(
            math.log(parent.visits) / child.visits
        )
        return child.average_value() + exploration

    def select_child(self, node: MCTSNode) -> MCTSNode:
        if not node.children:
            raise ValueError(f"Node {node.node_id} has no expanded children")
        return max(
            (self.nodes[child_id] for child_id in node.children.values()),
            key=lambda child: self.ucb_score(node, child),
        )

    def select_path(self) -> List[int]:
        path = [self.root_id]
        current = self.nodes[self.root_id]
        while current.depth < self.config.max_depth:
            if current.untried_actions is None:
                current.untried_actions = self.legal_actions(current)

            if current.untried_actions and self.should_expand(current):
                break
            if not current.children:
                break
            current = self.select_child(current)
            path.append(current.node_id)
        return path

    def should_expand(self, node: MCTSNode) -> bool:
        """
        Progressive widening.

        The plain MCTS rule expands every untried child before going deeper.
        GNF branching is large enough that this makes the tree crawl sideways.
        This rule allows only about k * visits^alpha children at a node before
        selection starts revisiting the best existing children.
        """
        if not node.children:
            return True
        if node.visits <= 0:
            return False
        allowed_children = max(
            1,
            int(self.config.progressive_widening_k * (node.visits ** self.config.progressive_widening_alpha)),
        )
        return len(node.children) < allowed_children

    def expand(self, node: MCTSNode) -> MCTSNode:
        if node.depth >= self.config.max_depth:
            return node
        if node.untried_actions is None:
            node.untried_actions = self.legal_actions(node)
        if not node.untried_actions:
            return node
        action_index = self.rng.randrange(len(node.untried_actions))
        action = node.untried_actions.pop(action_index)
        return self.create_child(node, action)

    def score_prefix(self, factor_ids: List[int], burau_matrix) -> dict:
        projlen = polynomial_matrix_projlen(burau_matrix)
        depth = len(factor_ids)
        kernel_match = projective_kernel_match(
            burau_matrix,
            p=self.config.p,
            n=self.config.n,
        )
        typical_projlen = self.baseline.mean(depth)
        surprise = self.baseline.surprise(depth, projlen)
        surprise_z = self.baseline.surprise_z(depth, projlen)
        value = self.value_from_surprise(surprise, surprise_z, kernel_match, depth)
        state = serialize_prefix_state(
            factor_ids,
            poly_mat=burau_matrix,
            p=self.config.p,
            n=self.config.n,
        )
        state["gnf_d"] = 0
        if self.config.n == 2:
            state["note"] = "For n=2 the Delta/simple generator is allowed as the search move."
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

    def beam_frontier_playout(self, node: MCTSNode) -> dict:
        """
        Evaluate a node by beam/frontier expansion.

        At each depth we expand every current beam state by all legal suffixes,
        score each child by length-relative surprise, keep the best few, and
        feed all seen children into projlen reservoirs for logging.
        """
        reservoir = BucketReservoir(self.config.reservoir_size, self.rng)
        kernel_hits = []

        start_score = self.score_prefix(node.factor_ids, node.burau_matrix)
        if node.factor_ids:
            reservoir.add(start_score)
            if start_score["kernel_match"].get("matches"):
                kernel_hits.append(start_score)

        beam = [(list(node.factor_ids), node.burau_matrix)]
        while beam and len(beam[0][0]) < self.config.max_depth:
            candidates = []
            for factor_ids, burau_matrix in beam:
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
                    score = self.score_prefix(child_factor_ids, child_matrix)
                    reservoir.add(score)
                    candidates.append((score, child_factor_ids, child_matrix))
                    if score["kernel_match"].get("matches"):
                        kernel_hits.append(score)

            if not candidates:
                break

            candidates.sort(key=lambda item: item[0]["value"], reverse=True)
            beam = [
                (factor_ids, burau_matrix)
                for _, factor_ids, burau_matrix in candidates[: self.config.beam_width]
            ]

        best_prefix = reservoir.best_score() or start_score
        return {
            "best_prefix": best_prefix,
            "value": best_prefix["value"],
            "kernel_hits": kernel_hits,
            "reservoir_summary": reservoir.summary(),
        }

    def value_from_surprise(
        self,
        surprise: float,
        surprise_z: float,
        kernel_match: dict,
        depth: int,
    ) -> float:
        if depth <= 0:
            value = 0.0
        else:
            value = surprise_z
            value += 0.01 * surprise / max(1.0, float(depth))
            value += 0.001 * min(depth, self.config.max_depth) / self.config.max_depth
        if kernel_match.get("matches") and depth > 0:
            value += 1000.0
        return value

    def backpropagate(self, path: List[int], value: float) -> None:
        for node_id in path:
            node = self.nodes[node_id]
            node.visits += 1
            node.total_value += value

    def update_best(self, playout_result: dict) -> None:
        best_prefix = playout_result["best_prefix"]
        value = float(best_prefix["value"])
        projlen = int(best_prefix["projlen"])
        depth = int(best_prefix["depth"])

        if value > self.best_value:
            self.best_value = value
            self.best_projlen = projlen
            self.best_candidate = dict(best_prefix["state"])
            self.best_candidate["value"] = value
            self.best_candidate["projlen_per_length"] = best_prefix["projlen_per_length"]
            self.best_candidate["typical_projlen"] = best_prefix["typical_projlen"]
            self.best_candidate["surprise"] = best_prefix["surprise"]
            self.best_candidate["surprise_z"] = best_prefix["surprise_z"]

        current_depth_best = self.best_projlen_by_depth.get(depth)
        if current_depth_best is None or projlen < current_depth_best:
            candidate = dict(best_prefix["state"])
            candidate["value"] = value
            candidate["projlen_per_length"] = best_prefix["projlen_per_length"]
            candidate["typical_projlen"] = best_prefix["typical_projlen"]
            candidate["surprise"] = best_prefix["surprise"]
            candidate["surprise_z"] = best_prefix["surprise_z"]
            self.best_projlen_by_depth[depth] = projlen
            self.best_candidate_by_depth[depth] = candidate

        for hit_score in playout_result["kernel_hits"]:
            hit = dict(hit_score["state"])
            hit["value"] = hit_score["value"]
            hit["typical_projlen"] = hit_score["typical_projlen"]
            hit["surprise"] = hit_score["surprise"]
            hit["surprise_z"] = hit_score["surprise_z"]
            self.kernel_hits.append(hit)

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

    def save_tree_csv(self) -> None:
        with (self.run_dir / "tree_nodes.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "node_id",
                    "parent_id",
                    "action_from_parent",
                    "depth",
                    "visits",
                    "total_value",
                    "average_value",
                    "factor_ids",
                    "projlen",
                ],
            )
            writer.writeheader()
            for node in self.nodes.values():
                writer.writerow(
                    {
                        "node_id": node.node_id,
                        "parent_id": node.parent_id,
                        "action_from_parent": node.action_from_parent,
                        "depth": node.depth,
                        "visits": node.visits,
                        "total_value": node.total_value,
                        "average_value": node.average_value(),
                        "factor_ids": json.dumps(node.factor_ids),
                        "projlen": polynomial_matrix_projlen(node.burau_matrix),
                    }
                )

        with (self.run_dir / "tree_edges.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["parent_id", "action", "child_id"])
            writer.writeheader()
            for parent in self.nodes.values():
                for action, child_id in parent.children.items():
                    writer.writerow(
                        {
                            "parent_id": parent.node_id,
                            "action": action,
                            "child_id": child_id,
                        }
                    )

    def save_summary_json(self) -> None:
        with (self.run_dir / "best_candidate.json").open("w", encoding="utf-8") as f:
            json.dump(self.best_candidate, f, indent=2)
        with (self.run_dir / "best_candidate_by_depth.json").open("w", encoding="utf-8") as f:
            json.dump(self.best_candidate_by_depth, f, indent=2)
        with (self.run_dir / "kernel_hits.json").open("w", encoding="utf-8") as f:
            json.dump(self.kernel_hits, f, indent=2)

        summary = {
            "config": asdict(self.config),
            "num_nodes": len(self.nodes),
            "best_value": self.best_value,
            "best_projlen": self.best_projlen,
            "best_projlen_by_depth": self.best_projlen_by_depth,
            "num_kernel_hits": len(self.kernel_hits),
            "baseline_samples": self.config.baseline_samples,
            "beam_width": self.config.beam_width,
            "run_dir": str(self.run_dir),
        }
        with (self.run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def write_plots(self) -> None:
        """
        Save the same basic search diagnostics as the earlier MCTS scripts.

        Reservoir playouts do not have a single rollout endpoint, so the
        per-iteration line uses the best prefix kept by that iteration's
        reservoir.
        """
        if not HAS_MATPLOTLIB:
            with (self.run_dir / "plot_warning.txt").open("w", encoding="utf-8") as f:
                f.write("matplotlib is not installed; skipped PNG plot generation.\n")
            return

        records = []
        if not self.iterations_path.exists():
            return
        with self.iterations_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            return

        iterations = [record["iteration"] for record in records]
        best_prefix_projlen = [record["best_prefix_projlen"] for record in records]
        best_projlen = [record["best_projlen"] for record in records]
        best_value = [record["best_value"] for record in records]
        best_surprise = [record["best_prefix_surprise"] for record in records]
        best_surprise_z = [record["best_prefix_surprise_z"] for record in records]
        path_depth = [record["path_depth"] for record in records]
        best_prefix_depth = [record["best_prefix_depth"] for record in records]
        kernel_hits = [record["kernel_hits_this_iteration"] for record in records]

        def save_line_plot(y_values, title, ylabel, filename):
            plt.figure(figsize=(8, 4.5))
            plt.plot(iterations, y_values, linewidth=1.5)
            plt.title(title)
            plt.xlabel("Iteration")
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(self.figures_dir / filename, dpi=160)
            plt.close()

        save_line_plot(
            best_prefix_projlen,
            "Best reservoir prefix projective length per iteration",
            "Best reservoir prefix projlen",
            "best_prefix_projlen_per_iteration.png",
        )
        save_line_plot(
            best_projlen,
            "Best projective length over time",
            "Best projlen",
            "best_projlen_over_time.png",
        )
        save_line_plot(
            best_value,
            "Best value over time",
            "Best value",
            "best_value_over_time.png",
        )
        save_line_plot(
            best_surprise,
            "Best prefix surprise per iteration",
            "Typical projlen minus observed projlen",
            "best_prefix_surprise_per_iteration.png",
        )
        save_line_plot(
            best_surprise_z,
            "Best prefix surprise z-score per iteration",
            "Surprise z-score",
            "best_prefix_surprise_z_per_iteration.png",
        )
        save_line_plot(
            path_depth,
            "Selected tree depth over time",
            "Tree depth",
            "selected_depth_over_time.png",
        )
        save_line_plot(
            best_prefix_depth,
            "Best reservoir prefix depth per iteration",
            "Best prefix depth",
            "best_prefix_depth_per_iteration.png",
        )
        save_line_plot(
            kernel_hits,
            "Kernel hits per iteration",
            "Kernel hits",
            "kernel_hits_per_iteration.png",
        )

    def run(self) -> dict:
        start = time.time()
        for iteration in range(1, self.config.iterations + 1):
            path = self.select_path()
            leaf = self.nodes[path[-1]]

            if leaf.depth < self.config.max_depth:
                expanded = self.expand(leaf)
                if expanded.node_id != leaf.node_id:
                    path.append(expanded.node_id)
                    leaf = expanded

            playout_result = self.beam_frontier_playout(leaf)
            self.backpropagate(path, playout_result["value"])
            self.update_best(playout_result)
            self.log_iteration(iteration, path, playout_result)

        self.save_tree_csv()
        self.save_summary_json()
        self.write_plots()

        return {
            "run_dir": str(self.run_dir),
            "iterations": self.config.iterations,
            "num_nodes": len(self.nodes),
            "best_value": self.best_value,
            "best_projlen": self.best_projlen,
            "best_projlen_by_depth": self.best_projlen_by_depth,
            "num_kernel_hits": len(self.kernel_hits),
            "elapsed_sec": round(time.time() - start, 4),
        }


def parse_args() -> MCTSConfig:
    parser = argparse.ArgumentParser(
        description="Run MCTS with surprise scoring and beam/frontier playouts."
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
    return MCTSConfig(
        p=args.p,
        n=args.n,
        max_depth=args.max_depth,
        iterations=args.iterations,
        exploration_weight=args.exploration_weight,
        baseline_samples=args.baseline_samples,
        beam_width=args.beam_width,
        reservoir_size=args.reservoir_size,
        progressive_widening_k=args.progressive_widening_k,
        progressive_widening_alpha=args.progressive_widening_alpha,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def main() -> None:
    search = MonteCarloTreeSearch(parse_args())
    print(json.dumps(search.run(), indent=2))


if __name__ == "__main__":
    main()
