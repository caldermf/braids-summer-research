#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_SEED_WORD = "0:21,6,8,16,2,13,1,4,16,13,8,12"


def load_braid_gpt_module(braid_gpt_root: Path):
    module_path = braid_gpt_root / "braid_gpt.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find Braid-GPT script at {module_path}")
    spec = importlib.util.spec_from_file_location("braid_gpt_runtime", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Braid-GPT from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def metric_projlen(metrics: dict) -> float:
    return float(metrics.get("projlen", metrics.get("projective_width", 0.0)))


def clean_metrics(metrics: dict) -> dict:
    output = dict(metrics)
    if "projlen" not in output:
        output["projlen"] = output.get("projective_width", 0)
    output.pop("projective_width", None)
    return output


@dataclass(frozen=True)
class SearchState:
    power: int
    factors: tuple[int, ...]
    metrics: dict
    score: float


@dataclass
class ChildEdge:
    action: int
    prior: float
    child: "MCTSNode"


@dataclass
class MCTSNode:
    state: SearchState
    parent: "MCTSNode | None" = None
    action_from_parent: int | None = None
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, ChildEdge] = field(default_factory=dict)
    expanded: bool = False

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class PolicyGuidedMCTS:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(args.seed)
        self.bgpt = load_braid_gpt_module(Path(args.braid_gpt_root))

        import torch
        import torch.nn as nn

        self.torch = torch
        self.device = self.bgpt.resolve_device(torch, args.device)
        self.model, self.config, self.checkpoint = self.bgpt.load_checkpoint(
            torch,
            nn,
            Path(args.checkpoint),
            self.device,
        )
        self.model.eval()
        self.automaton = self.bgpt.GNFAutomaton(args.n)
        self.evaluator = self.bgpt.ExactEvaluator(
            author_repo=Path(args.author_repo),
            p=args.p,
            n=args.n,
            r=args.r,
            max_degree=args.matrix_max_degree,
            width_weight=args.projlen_weight,
            min_meaningful_length=args.min_meaningful_length,
            degeneracy_weight=args.degeneracy_weight,
        )

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.output_dir / "progress.jsonl"
        self.candidates_path = self.output_dir / "candidates.jsonl"
        self.targets_path = self.output_dir / "mcts_policy_targets.jsonl"
        self.progress_path.write_text("", encoding="utf-8")
        self.candidates_path.write_text("", encoding="utf-8")
        self.targets_path.write_text("", encoding="utf-8")

        self.best: dict[tuple[int, tuple[int, ...]], SearchState] = {}
        self.kernel_hits: list[SearchState] = []
        self.nodes: list[MCTSNode] = []
        self.evaluated_count = 0
        self.start_time = time.time()

    def state_context(self, state: SearchState) -> np.ndarray:
        if not state.factors:
            return self.bgpt.empty_context(state.power)
        metrics = dict(state.metrics)
        metrics.setdefault("projective_width", metrics.get("projlen", 0))
        return self.bgpt.normalized_context_features(
            power=state.power,
            factors=state.factors,
            metrics=metrics,
            score=state.score,
        )

    def legal_actions(self, factors: Sequence[int]) -> tuple[int, ...]:
        if not factors:
            return tuple(self.automaton.first_ids)
        return tuple(self.automaton.successors[factors[-1]])

    def policy_priors(self, state: SearchState) -> dict[int, float]:
        legal = self.legal_actions(state.factors)
        if not legal:
            return {}

        tokens, position = self.bgpt.encode_prefix(state.factors, self.config.max_factors)
        context = self.state_context(state)
        torch = self.torch
        with torch.no_grad():
            logits_all, _ = self.model(
                torch.tensor(tokens[None, :], dtype=torch.long, device=self.device),
                torch.tensor(context[None, :], dtype=torch.float32, device=self.device),
            )
            logits = logits_all[0, position].detach().clone()
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[list(legal)] = False
            logits = logits.masked_fill(mask, -1e9)
            probs = torch.softmax(logits / max(self.args.temperature, 1e-6), dim=-1)
            probs_np = probs.detach().cpu().numpy()

        priors = {int(action): float(probs_np[action]) for action in legal}
        total = sum(priors.values())
        if total <= 0:
            uniform = 1.0 / len(legal)
            return {int(action): uniform for action in legal}
        return {action: value / total for action, value in priors.items()}

    def selected_actions(self, priors: dict[int, float]) -> list[int]:
        ranked = sorted(priors, key=priors.get, reverse=True)[: self.args.expand_top_k]
        remaining = [action for action in priors if action not in set(ranked)]
        sampled: list[int] = []
        if remaining and self.args.expand_sample_k > 0:
            weights = np.array([priors[action] for action in remaining], dtype=np.float64)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            sample_count = min(self.args.expand_sample_k, len(remaining))
            sampled = [
                int(remaining[index])
                for index in self.rng.choices(range(len(remaining)), weights=weights, k=sample_count * 2)
            ]
        actions: list[int] = []
        seen: set[int] = set()
        for action in [*ranked, *sampled]:
            if action not in seen:
                seen.add(action)
                actions.append(action)
        return actions

    def mcts_cost(self, state: SearchState) -> float:
        metrics = state.metrics
        if not metrics:
            return 10_000.0
        cost = (
            self.args.identity_weight * float(metrics["identity_defect"])
            + self.args.projlen_weight * metric_projlen(metrics)
        )
        if self.args.degeneracy_weight:
            deg = self.bgpt.degeneracy_features(state.factors)
            penalty = 0.0
            penalty += max(0.0, deg["dominant_fraction"] - 0.45) * 80.0
            penalty += max(0.0, deg["top_two_fraction"] - 0.70) * 80.0
            penalty += max(0.0, deg["max_run_fraction"] - 0.25) * 80.0
            penalty += max(0.0, 0.35 - deg["unique_fraction"]) * 80.0
            penalty += max(0.0, deg["repeated_bigram_fraction"] - 0.20) * 60.0
            if deg["period_at_most_2"]:
                penalty += 40.0
            cost += self.args.degeneracy_weight * penalty
        if self.args.length_floor_weight and len(state.factors) < self.args.min_meaningful_length:
            cost += self.args.length_floor_weight * (self.args.min_meaningful_length - len(state.factors))
        if metrics.get("scalar_identity"):
            cost -= self.args.kernel_bonus
        return float(cost)

    def state_value(self, state: SearchState) -> float:
        return -self.mcts_cost(state) / max(self.args.value_scale, 1e-6)

    def update_best(self, states: Sequence[SearchState]) -> None:
        for state in states:
            key = (state.power % 2, state.factors)
            previous = self.best.get(key)
            if previous is None or self.rank_key(state) < self.rank_key(previous):
                self.best[key] = state
            if state.metrics.get("scalar_identity"):
                self.kernel_hits.append(state)

    def rank_key(self, state: SearchState) -> tuple:
        metrics = state.metrics
        return (
            0 if metrics.get("scalar_identity") else 1,
            int(metrics.get("identity_defect", 10**9)),
            self.mcts_cost(state),
            len(state.factors),
        )

    def evaluate_children(self, parent: MCTSNode, actions: Sequence[int]) -> list[MCTSNode]:
        words = [
            (parent.state.power, parent.state.factors + (int(action),))
            for action in actions
            if len(parent.state.factors) < self.args.max_length
        ]
        if not words:
            return []
        evaluated = self.evaluator.evaluate_batch(words, batch_size=self.args.eval_batch_size)
        nodes: list[MCTSNode] = []
        for (power, factors), item in zip(words, evaluated):
            state = SearchState(
                power=int(power),
                factors=tuple(int(value) for value in factors),
                metrics=clean_metrics(item.metrics),
                score=float(item.score),
            )
            action = state.factors[-1]
            node = MCTSNode(
                state=state,
                parent=parent,
                action_from_parent=action,
                prior=0.0,
            )
            nodes.append(node)
            self.nodes.append(node)
        self.evaluated_count += len(nodes)
        self.update_best([node.state for node in nodes])
        append_jsonl(
            self.candidates_path,
            [
                {
                    "parent_length": len(parent.state.factors),
                    "power": node.state.power,
                    "factor_ids": list(node.state.factors),
                    "length": len(node.state.factors),
                    "action": int(node.action_from_parent),
                    "metrics": node.state.metrics,
                    "score": node.state.score,
                    "mcts_cost": self.mcts_cost(node.state),
                }
                for node in nodes
            ],
        )
        return nodes

    def expand(self, node: MCTSNode) -> None:
        if node.expanded:
            return
        node.expanded = True
        if len(node.state.factors) >= self.args.max_length:
            return
        priors = self.policy_priors(node.state)
        actions = self.selected_actions(priors)
        children = self.evaluate_children(node, actions)
        for child in children:
            action = int(child.action_from_parent)
            child.prior = float(priors.get(action, 0.0))
            node.children[action] = ChildEdge(action=action, prior=child.prior, child=child)

    def select_child(self, node: MCTSNode) -> MCTSNode:
        parent_visits = max(1, node.visits)
        best_score = -float("inf")
        best_child: MCTSNode | None = None
        for edge in node.children.values():
            child = edge.child
            q = child.q_value if child.visits else 0.0
            u = self.args.puct_c * edge.prior * math.sqrt(parent_visits) / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        if best_child is None:
            raise RuntimeError("select_child called on a node without children")
        return best_child

    def run_simulation(self, root: MCTSNode) -> None:
        node = root
        path = [node]
        while True:
            if node.state.metrics.get("scalar_identity") or len(node.state.factors) >= self.args.max_length:
                value = self.state_value(node.state)
                break
            if not node.expanded:
                self.expand(node)
                if not node.children:
                    value = self.state_value(node.state)
                    break
                node = self.select_child(node)
                path.append(node)
                value = self.state_value(node.state)
                break
            node = self.select_child(node)
            path.append(node)

        for visited in path:
            visited.visits += 1
            visited.value_sum += value

    def make_roots(self) -> list[MCTSNode]:
        seed_words = list(self.args.seed_word)
        if not seed_words and self.args.use_default_seed:
            seed_words.append(DEFAULT_SEED_WORD)
        if not seed_words:
            raise ValueError("provide at least one --seed-word or keep --use-default-seed")

        parsed = [self.bgpt.parse_seed_word(value) for value in seed_words]
        evaluated = self.evaluator.evaluate_batch(parsed, batch_size=self.args.eval_batch_size)
        roots: list[MCTSNode] = []
        for item in evaluated:
            state = SearchState(
                power=item.power,
                factors=item.factors,
                metrics=clean_metrics(item.metrics),
                score=float(item.score),
            )
            root = MCTSNode(state=state)
            roots.append(root)
            self.nodes.append(root)
        self.update_best([root.state for root in roots])
        return roots

    def progress_row(self, simulation: int, roots: Sequence[MCTSNode]) -> dict:
        best_states = self.best_states(self.args.keep_best)
        best = best_states[0] if best_states else None
        best_metrics = best.metrics if best else {}
        return {
            "phase": "rl_mcts",
            "simulation": simulation,
            "elapsed_seconds": round(time.time() - self.start_time, 2),
            "tree_nodes": len(self.nodes),
            "evaluated": self.evaluated_count,
            "root_visits": [root.visits for root in roots],
            "best_identity_defect": best_metrics.get("identity_defect"),
            "best_projlen": best_metrics.get("projlen"),
            "best_mcts_cost": self.mcts_cost(best) if best else None,
            "best_length": len(best.factors) if best else None,
            "kernel_hits": len(self.kernel_hits),
        }

    def best_states(self, limit: int) -> list[SearchState]:
        return sorted(self.best.values(), key=self.rank_key)[:limit]

    def write_policy_targets(self) -> None:
        rows = []
        for node in self.nodes:
            if not node.children or node.visits < self.args.min_target_visits:
                continue
            total_child_visits = sum(edge.child.visits for edge in node.children.values())
            if total_child_visits <= 0:
                continue
            rows.append(
                {
                    "power": node.state.power,
                    "factor_ids": list(node.state.factors),
                    "length": len(node.state.factors),
                    "visits": node.visits,
                    "metrics": node.state.metrics,
                    "score": node.state.score,
                    "mcts_cost": self.mcts_cost(node.state),
                    "actions": [
                        {
                            "action": edge.action,
                            "prior": edge.prior,
                            "visits": edge.child.visits,
                            "target": edge.child.visits / total_child_visits,
                            "q": edge.child.q_value,
                            "child_identity_defect": edge.child.state.metrics.get("identity_defect"),
                            "child_projlen": edge.child.state.metrics.get("projlen"),
                            "child_mcts_cost": self.mcts_cost(edge.child.state),
                        }
                        for edge in sorted(
                            node.children.values(),
                            key=lambda item: item.child.visits,
                            reverse=True,
                        )
                    ],
                }
            )
        append_jsonl(self.targets_path, rows)

    def run(self) -> None:
        roots = self.make_roots()
        print(
            json.dumps(
                {
                    "phase": "start",
                    "roots": [
                        {
                            "power": root.state.power,
                            "factor_ids": list(root.state.factors),
                            "metrics": root.state.metrics,
                            "score": root.state.score,
                            "mcts_cost": self.mcts_cost(root.state),
                        }
                        for root in roots
                    ],
                    "checkpoint_stage": self.checkpoint.get("stage"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        for simulation in range(1, self.args.simulations + 1):
            root = roots[(simulation - 1) % len(roots)]
            self.run_simulation(root)
            if simulation % self.args.progress_every == 0 or simulation == 1:
                row = self.progress_row(simulation, roots)
                append_jsonl(self.progress_path, [row])
                print(json.dumps(row, sort_keys=True), flush=True)
            if self.kernel_hits and self.args.stop_at_kernel:
                break

        self.write_policy_targets()
        best_states = self.best_states(self.args.keep_best)
        summary = {
            "format": "braid-gpt-rl-mcts-summary-v1",
            "checkpoint": str(self.args.checkpoint),
            "checkpoint_stage": self.checkpoint.get("stage"),
            "config": {
                "p": self.args.p,
                "n": self.args.n,
                "r": self.args.r,
                "seed": self.args.seed,
                "simulations": self.args.simulations,
                "max_length": self.args.max_length,
                "expand_top_k": self.args.expand_top_k,
                "expand_sample_k": self.args.expand_sample_k,
                "temperature": self.args.temperature,
                "puct_c": self.args.puct_c,
                "identity_weight": self.args.identity_weight,
                "projlen_weight": self.args.projlen_weight,
                "degeneracy_weight": self.args.degeneracy_weight,
            },
            "kernel_hits": [self.state_record(state) for state in self.kernel_hits[: self.args.keep_best]],
            "best": [self.state_record(state) for state in best_states],
        }
        write_json(self.output_dir / "summary.json", summary)

    def state_record(self, state: SearchState) -> dict:
        return {
            "power": state.power,
            "factor_ids": list(state.factors),
            "length": len(state.factors),
            "metrics": state.metrics,
            "score": state.score,
            "mcts_cost": self.mcts_cost(state),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Policy-guided RL/MCTS on top of Braid-GPT.")
    parser.add_argument("--braid-gpt-root", default=str(Path(__file__).resolve().parents[1] / "Braid-GPT"))
    parser.add_argument("--author-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--p", type=int, default=7)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seed-word", action="append", default=[])
    parser.add_argument("--use-default-seed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--simulations", type=int, default=6000)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--expand-top-k", type=int, default=10)
    parser.add_argument("--expand-sample-k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--puct-c", type=float, default=1.5)
    parser.add_argument("--eval-batch-size", type=int, default=500)
    parser.add_argument("--matrix-max-degree", type=int, default=256)
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--projlen-weight", type=float, default=0.05)
    parser.add_argument("--degeneracy-weight", type=float, default=0.4)
    parser.add_argument("--min-meaningful-length", type=int, default=15)
    parser.add_argument("--length-floor-weight", type=float, default=0.0)
    parser.add_argument("--kernel-bonus", type=float, default=100000.0)
    parser.add_argument("--value-scale", type=float, default=100.0)
    parser.add_argument("--keep-best", type=int, default=200)
    parser.add_argument("--min-target-visits", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--stop-at-kernel", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    PolicyGuidedMCTS(args).run()


if __name__ == "__main__":
    main()
