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

# Keep Matplotlib cache writes out of the home directory, which may not be
# writable in the Codex sandbox or on shared systems.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/braids_mcts_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from braid_data import (
    append_factor_to_burau_matrix,
    identity_burau_matrix,
    polynomial_matrix_projlen,
    projective_kernel_match,
    serialize_prefix_state,
    simple_factor_burau_table,
    valid_first_factor_ids,
    valid_suffix_factor_ids,
)

#!/usr/bin/env python3
"""
Monte Carlo tree search for promising positive GNF braid prefixes.

Prefix-scoring version:
- value function is based only on projective length
- no neural model / transformer
- scores every prefix visited during rollout, not only the final endpoint
- saves JSONL logs and basic plots
"""


@dataclass
class MCTSConfig:
    """
    Configuration for one MCTS run.
    """
    p: int = 7
    n: int = 4
    max_depth: int = 40
    iterations: int = 100000000
    exploration_weight: float = 1.4
    rollout_policy: str = "random"
    epsilon: float = 0.2
    seed: int = 1
    output_dir: str = "results"


@dataclass
class MCTSNode:
    """
    One node in the MCTS tree.

    Each node represents a GNF prefix, stored as simple factor IDs.
    """
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
        """
        Return the mean rollout value seen from this node.
        """
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

class MonteCarloTreeSearch:
    """
    MCTS driver for searching positive GNF braid prefixes.
    """

    def __init__(self, config: MCTSConfig):
        """
        Initialize RNG, Burau tables, root node, output directory, and logs.
        """
        self.config = config

        # Use a dedicated random generator so this search is reproducible and
        # does not disturb any other code using Python's global random module.
        self.rng = random.Random(config.seed)

        # Precompute the Burau image of every simple factor once. Every tree
        # expansion can then update matrices by multiplication instead of
        # recomputing from an Artin word.
        self.simple_table = simple_factor_burau_table(p=config.p, n=config.n)

        # Store every explored node by integer ID. Node 0 is always the root,
        # representing the empty braid prefix.
        self.nodes: Dict[int, MCTSNode] = {}
        self.next_node_id = 0
        root = self.make_root_node()
        self.nodes[root.node_id] = root
        self.root_id = root.node_id
        self.next_node_id = root.node_id + 1

        # Best candidates and kernel hits are kept in memory during the run and
        # later written to JSON files.
        self.best_candidate: Optional[dict] = None
        self.best_value = float("-inf")
        self.best_projlen: Optional[int] = None
        self.best_candidate_by_depth: Dict[int, dict] = {}
        self.best_projlen_by_depth: Dict[int, int] = {}
        self.kernel_hits: List[dict] = []

        # Create the output layout up front so every iteration can append to the
        # same evidence trail.
        self.run_dir = self.create_run_directory()
        self.figures_dir = self.run_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.iterations_path = self.run_dir / "iterations.jsonl"

        # Matplotlib tries to write a cache under the home directory on some
        # machines. Keep that cache inside the run directory for portability.
        os.environ.setdefault("MPLCONFIGDIR", str(self.run_dir / "matplotlib_cache"))

        with (self.run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2)

    def create_run_directory(self) -> Path:
        """
        Create a timestamped directory where all results for this run are saved.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = Path(self.config.output_dir)
        run_dir = base_dir / f"mcts_{timestamp}_seed{self.config.seed}"

        # If two runs start in the same second, add a small numeric suffix
        # instead of overwriting the first run.
        suffix = 1
        unique_run_dir = run_dir
        while unique_run_dir.exists():
            suffix += 1
            unique_run_dir = base_dir / f"{run_dir.name}_{suffix}"

        unique_run_dir.mkdir(parents=True, exist_ok=False)
        return unique_run_dir

    def make_root_node(self) -> MCTSNode:
        """
        Create the root node.

        The root represents the empty braid prefix, so its Burau matrix is the
        identity matrix.
        """
        return MCTSNode(
            node_id=0,
            factor_ids=[],
            parent_id=None,
            action_from_parent=None,
            burau_matrix=identity_burau_matrix(p=self.config.p, n=self.config.n),
            depth=0,
            untried_actions=valid_first_factor_ids(n=self.config.n),
        )
    
    def legal_actions(self, node: MCTSNode) -> List[int]:
        """
        Return valid simple factor IDs that can be appended to this node.
        """
        if node.depth == 0:
            return valid_first_factor_ids(n=self.config.n)
        return valid_suffix_factor_ids(node.factor_ids[-1], n=self.config.n)
    
    def create_child(self, parent: MCTSNode, action: int) -> MCTSNode:
        """
        Create a child node by appending `action` to `parent.factor_ids`.

        Also update the Burau matrix incrementally:
            child_matrix = parent_matrix * Burau(action)
        """
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
        """
        Compute the UCB score used during selection.

        Higher score means the child is more attractive.
        """
        if child.visits == 0:
            return float("inf")
        if parent.visits == 0:
            return child.average_value()
        exploration = self.config.exploration_weight * math.sqrt(
            math.log(parent.visits) / child.visits
        )
        return child.average_value() + exploration
        

    def select_child(self, node: MCTSNode) -> MCTSNode:
        """
        Choose the best already-expanded child using UCB.
        """
        if not node.children:
            raise ValueError(f"Node {node.node_id} has no expanded children to select")

        best_child = None
        best_score = float("-inf")

        # node.children maps: action factor ID -> child node ID.
        for child_id in node.children.values():
            child = self.nodes[child_id]
            score = self.ucb_score(node, child)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def select_path(self) -> List[int]:
        """
        Starting from root, follow UCB choices until reaching a node that can be
        expanded or a terminal node.

        Returns
        -------
        list[int]
            Node IDs along the selected path.
        """
        path = [self.root_id]
        current = self.nodes[self.root_id]

        while current.depth < self.config.max_depth:
            if current.untried_actions is None:
                current.untried_actions = self.legal_actions(current)

            # Stop selection as soon as there is a legal move that has not been
            # expanded yet. The expansion step will create that child.
            if current.untried_actions:
                break

            # If every action has been tried but there are no children, this is
            # a dead end in the GNF transition graph.
            if not current.children:
                break

            current = self.select_child(current)
            path.append(current.node_id)

        return path
        

    def expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expand one untried action from `node`.

        Returns the newly created child node.
        """
        if node.depth >= self.config.max_depth:
            return node

        if node.untried_actions is None:
            node.untried_actions = self.legal_actions(node)
        if not node.untried_actions:
            return node

        # Randomizing the expansion order prevents the tree from inheriting
        # artifacts from the fixed permutation ordering.
        action_index = self.rng.randrange(len(node.untried_actions))
        action = node.untried_actions.pop(action_index)
        return self.create_child(node, action)

    def score_prefix(self, factor_ids, burau_matrix) -> dict:
        """
        Score one braid prefix and return a JSON-friendly record.

        This is what lets the prefix-scoring MCTS notice a short kernel element
        even when the rollout continues to a larger max_depth.
        """
        projlen = polynomial_matrix_projlen(burau_matrix)
        kernel_match = projective_kernel_match(
            burau_matrix,
            p=self.config.p,
            n=self.config.n,
        )
        value = self.value_from_projlen(projlen, kernel_match)
        state = serialize_prefix_state(
            factor_ids,
            poly_mat=burau_matrix,
            p=self.config.p,
            n=self.config.n,
        )
        state["gnf_d"] = 0
        return {
            "factor_ids": list(factor_ids),
            "depth": len(factor_ids),
            "projlen": projlen,
            "projlen_per_length": projlen / max(1, len(factor_ids)),
            "kernel_match": kernel_match,
            "value": value,
            "state": state,
        }

    def rollout(self, node: MCTSNode) -> dict:
        """
        Complete a braid prefix from `node` up to max_depth.

        Returns a dictionary containing:
        - endpoint score for the final rollout prefix
        - best prefix score seen anywhere along this rollout
        - all kernel hits seen along this rollout
        - all prefix scores in compact form
        """
        factor_ids = list(node.factor_ids)
        burau_matrix = node.burau_matrix
        prefix_scores = []

        if factor_ids:
            prefix_scores.append(self.score_prefix(factor_ids, burau_matrix))

        while len(factor_ids) < self.config.max_depth:
            actions = (
                valid_first_factor_ids(n=self.config.n)
                if not factor_ids
                else valid_suffix_factor_ids(factor_ids[-1], n=self.config.n)
            )
            if not actions:
                break
            factor_ids, burau_matrix = self.rollout_step(factor_ids, burau_matrix)
            prefix_scores.append(self.score_prefix(factor_ids, burau_matrix))

        if not prefix_scores:
            prefix_scores.append(self.score_prefix(factor_ids, burau_matrix))

        endpoint = prefix_scores[-1]
        best_prefix = max(prefix_scores, key=lambda item: item["value"])
        rollout_kernel_hits = [
            score for score in prefix_scores if score["kernel_match"].get("matches")
        ]
        return {
            "factor_ids": endpoint["factor_ids"],
            "depth": endpoint["depth"],
            "projlen": endpoint["projlen"],
            "projlen_per_length": endpoint["projlen_per_length"],
            "kernel_match": endpoint["kernel_match"],
            "value": best_prefix["value"],
            "state": endpoint["state"],
            "endpoint": endpoint,
            "best_prefix": best_prefix,
            "prefix_scores": [
                {
                    "depth": score["depth"],
                    "projlen": score["projlen"],
                    "projlen_per_length": score["projlen_per_length"],
                    "value": score["value"],
                    "kernel_match": score["kernel_match"],
                }
                for score in prefix_scores
            ],
            "rollout_kernel_hits": rollout_kernel_hits,
        }

    def rollout_step(self, factor_ids, burau_matrix) -> tuple:
        """
        Take one rollout step according to the configured rollout policy.

        Returns
        -------
        tuple[list[int], matrix]
            Updated factor IDs and updated Burau matrix.
        """
        if not factor_ids:
            actions = valid_first_factor_ids(n=self.config.n)
        else:
            actions = valid_suffix_factor_ids(factor_ids[-1], n=self.config.n)
        if not actions:
            return factor_ids, burau_matrix

        if self.config.rollout_policy == "random":
            action = self.rng.choice(actions)
        elif self.config.rollout_policy == "greedy_projlen":
            action = self.choose_greedy_projlen_action(actions, burau_matrix)
        elif self.config.rollout_policy == "epsilon_greedy_projlen":
            if self.rng.random() < self.config.epsilon:
                action = self.rng.choice(actions)
            else:
                action = self.choose_greedy_projlen_action(actions, burau_matrix)
        else:
            raise ValueError(f"Unknown rollout policy: {self.config.rollout_policy}")

        next_factor_ids = list(factor_ids) + [action]
        next_matrix = append_factor_to_burau_matrix(
            current_matrix=burau_matrix,
            factor_id=action,
            simple_table=self.simple_table,
            p=self.config.p,
        )
        return next_factor_ids, next_matrix

    def choose_greedy_projlen_action(self, actions: List[int], burau_matrix) -> int:
        """
        Choose the action whose immediate child has the smallest projlen.

        This makes rollouts less noisy than fully random rollouts. Ties are
        broken randomly so repeated runs with different seeds can still explore
        different low-projlen continuations.
        """
        best_projlen = None
        best_actions = []

        for action in actions:
            child_matrix = append_factor_to_burau_matrix(
                current_matrix=burau_matrix,
                factor_id=action,
                simple_table=self.simple_table,
                p=self.config.p,
            )
            child_projlen = polynomial_matrix_projlen(child_matrix)
            if best_projlen is None or child_projlen < best_projlen:
                best_projlen = child_projlen
                best_actions = [action]
            elif child_projlen == best_projlen:
                best_actions.append(action)

        return self.rng.choice(best_actions)
    
    def value_from_projlen(self, projlen: int, kernel_match: dict) -> float:
        """
        Convert projlen into a reward.

        Smaller projlen should give larger value. If a projective kernel match
        is found, give a large bonus.
        """
        value = 1.0 / (1.0 + float(projlen))
        if kernel_match.get("matches"):
            value += 1000.0
        return value
    
    def backpropagate(self, path: List[int], value: float) -> None:
        """
        Update visit counts and total values for every node on the selected path.
        """
        for node_id in path:
            node = self.nodes[node_id]
            node.visits += 1
            node.total_value += value

    def update_best(self, rollout_result: dict) -> None:
        """
        Track the best candidate seen so far.
        """
        best_prefix = rollout_result["best_prefix"]
        value = float(best_prefix["value"])
        projlen = int(best_prefix["projlen"])
        depth = int(best_prefix["depth"])

        if value > self.best_value:
            self.best_value = value
            self.best_projlen = projlen
            self.best_candidate = best_prefix["state"]
            self.best_candidate["value"] = value
            self.best_candidate["projlen_per_length"] = best_prefix["projlen_per_length"]

        current_depth_best = self.best_projlen_by_depth.get(depth)
        if current_depth_best is None or projlen < current_depth_best:
            candidate = dict(best_prefix["state"])
            candidate["value"] = value
            candidate["projlen_per_length"] = best_prefix["projlen_per_length"]
            self.best_projlen_by_depth[depth] = projlen
            self.best_candidate_by_depth[depth] = candidate

        for hit_score in rollout_result["rollout_kernel_hits"]:
            hit = dict(hit_score["state"])
            hit["value"] = hit_score["value"]
            hit["projlen_per_length"] = hit_score["projlen_per_length"]
            self.kernel_hits.append(hit)

    def log_iteration(self, iteration: int, path: List[int], rollout_result: dict) -> None:
        """
        Append one JSON record to iterations.jsonl.

        This is your evidence trail for every tiny result produced by the run.
        """
        leaf = self.nodes[path[-1]]
        record = {
            "iteration": iteration,
            "path": path,
            "path_depth": leaf.depth,
            "expanded_or_selected_node_id": leaf.node_id,
            "rollout_depth": rollout_result["depth"],
            "rollout_projlen": rollout_result["projlen"],
            "rollout_projlen_per_length": rollout_result["projlen_per_length"],
            "rollout_value": rollout_result["endpoint"]["value"],
            "best_prefix_depth": rollout_result["best_prefix"]["depth"],
            "best_prefix_projlen": rollout_result["best_prefix"]["projlen"],
            "best_prefix_projlen_per_length": rollout_result["best_prefix"]["projlen_per_length"],
            "best_prefix_value": rollout_result["best_prefix"]["value"],
            "best_value": self.best_value,
            "best_projlen": self.best_projlen,
            "kernel_match": rollout_result["kernel_match"],
            "rollout_state": rollout_result["state"],
            "best_prefix_state": rollout_result["best_prefix"]["state"],
            "prefix_scores": rollout_result["prefix_scores"],
        }
        with self.iterations_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def save_tree_csv(self) -> None:
        """
        Save tree_nodes.csv and tree_edges.csv for later inspection/plotting.
        """
        nodes_path = self.run_dir / "tree_nodes.csv"
        with nodes_path.open("w", encoding="utf-8", newline="") as f:
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

        edges_path = self.run_dir / "tree_edges.csv"
        with edges_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["parent_id", "action", "child_id"],
            )
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
        """
        Save config, best candidate, and kernel hits as JSON files.
        """
        with (self.run_dir / "best_candidate.json").open("w", encoding="utf-8") as f:
            json.dump(self.best_candidate, f, indent=2)
        by_depth = {
            str(depth): candidate
            for depth, candidate in sorted(self.best_candidate_by_depth.items())
        }
        with (self.run_dir / "best_candidate_by_depth.json").open("w", encoding="utf-8") as f:
            json.dump(by_depth, f, indent=2)
        with (self.run_dir / "kernel_hits.json").open("w", encoding="utf-8") as f:
            json.dump(self.kernel_hits, f, indent=2)
        summary = {
            "config": asdict(self.config),
            "num_nodes": len(self.nodes),
            "best_value": self.best_value,
            "best_projlen": self.best_projlen,
            "best_projlen_by_depth": {
                str(depth): projlen
                for depth, projlen in sorted(self.best_projlen_by_depth.items())
            },
            "num_kernel_hits": len(self.kernel_hits),
            "run_dir": str(self.run_dir),
        }
        with (self.run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def write_plots(self) -> None:
        """
        Save basic PNG plots from the JSONL iteration log.

        First version plots:
        - best projlen over time
        - rollout projlen over time
        - best value over time
        - depth of selected/expanded nodes over time
        """
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
        rollout_projlen = [record["rollout_projlen"] for record in records]
        best_projlen = [record["best_projlen"] for record in records]
        best_prefix_projlen = [record["best_prefix_projlen"] for record in records]
        best_value = [record["best_value"] for record in records]
        path_depth = [record["path_depth"] for record in records]

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
            rollout_projlen,
            "Rollout projective length over time",
            "Rollout projlen",
            "rollout_projlen_over_time.png",
        )
        save_line_plot(
            best_projlen,
            "Best prefix projective length over time",
            "Best projlen",
            "best_projlen_over_time.png",
        )
        save_line_plot(
            best_prefix_projlen,
            "Best prefix within each rollout over time",
            "Best prefix projlen in rollout",
            "best_prefix_projlen_per_rollout.png",
        )
        save_line_plot(
            best_value,
            "Best value over time",
            "Best value",
            "best_value_over_time.png",
        )
        save_line_plot(
            path_depth,
            "Selected tree depth over time",
            "Depth",
            "selected_depth_over_time.png",
        )

    def run(self) -> dict:
        """
        Run the full MCTS loop.

        For each iteration:
        1. select path
        2. expand if possible
        3. rollout
        4. backpropagate
        5. log result
        """
        start = time.time()
        for iteration in range(1, self.config.iterations + 1):
            path = self.select_path()
            leaf = self.nodes[path[-1]]

            if leaf.depth < self.config.max_depth:
                expanded = self.expand(leaf)
                if expanded.node_id != leaf.node_id:
                    path.append(expanded.node_id)
                    leaf = expanded

            rollout_result = self.rollout(leaf)
            self.backpropagate(path, rollout_result["value"])
            self.update_best(rollout_result)
            self.log_iteration(iteration, path, rollout_result)

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
    """
    Parse command line arguments into an MCTSConfig.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run prefix-scoring projlen-only Monte Carlo tree search. "
            "This variant checks every rollout prefix for low projlen/kernel hits."
        )
    )
    parser.add_argument("--p", type=int, default=7, help="Modulus for Burau arithmetic")
    parser.add_argument("--n", type=int, default=4, help="Number of braid strands")
    parser.add_argument("--max-depth", type=int, default=40, help="Maximum Garside length")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of MCTS iterations")
    parser.add_argument(
        "--exploration-weight",
        type=float,
        default=1.4,
        help="UCB exploration weight",
    )
    parser.add_argument(
        "--rollout-policy",
        choices=["random", "greedy_projlen", "epsilon_greedy_projlen"],
        default="random",
        help="Rollout policy for completing prefixes",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="Random-action probability for epsilon_greedy_projlen rollouts",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--output-dir", default="results", help="Directory for run outputs")
    args = parser.parse_args()
    return MCTSConfig(
        p=args.p,
        n=args.n,
        max_depth=args.max_depth,
        iterations=args.iterations,
        exploration_weight=args.exploration_weight,
        rollout_policy=args.rollout_policy,
        epsilon=args.epsilon,
        seed=args.seed,
        output_dir=args.output_dir,
    )

def main() -> None:
    """
    Entry point when running this file directly.
    """
    config = parse_args()
    search = MonteCarloTreeSearch(config)
    summary = search.run()
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
