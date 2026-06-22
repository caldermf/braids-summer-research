#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
STRUCTURAL_ROOT = REPO_ROOT / "structural-kernel-experiments"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STRUCTURAL_ROOT) not in sys.path:
    sys.path.insert(0, str(STRUCTURAL_ROOT))

from crispr_transformer.gnf import GNFAutomaton  # noqa: E402
from exact_transformer_policy.policy_experiment import (  # noqa: E402
    DEFAULT_AUTHOR_REPO,
    ExactEvaluator,
    EvaluatedWord,
    build_policy_model,
    normalized_context_features,
    parse_seed_word,
    read_json,
    resolve_device,
    write_json,
)


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


@dataclass(frozen=True)
class Action:
    kind: str
    payload: tuple[int, ...]

    def key(self) -> str:
        return f"{self.kind}:{','.join(str(value) for value in self.payload)}"


@dataclass
class Node:
    state: EvaluatedWord
    parent: "Node | None" = None
    action_from_parent: Action | None = None
    depth: int = 0
    visits: int = 0
    value_sum: float = 0.0
    actions: list[Action] = field(default_factory=list)
    priors: dict[str, float] = field(default_factory=dict)
    children: dict[str, "Node"] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class PolicyPrior:
    def __init__(
        self,
        *,
        checkpoint_path: Path | None,
        device_arg: str,
        evaluator: ExactEvaluator,
    ) -> None:
        self.evaluator = evaluator
        self.torch = None
        self.model = None
        self.device = None
        self.model_config = None
        if checkpoint_path is None:
            return
        import torch
        import torch.nn as nn

        self.torch = torch
        self.device = resolve_device(torch, device_arg)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model_config = checkpoint["model_config"]
        self.model = build_policy_model(torch, nn, self.model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def append_probs(self, state: EvaluatedWord, automaton: GNFAutomaton) -> dict[int, float]:
        legal = tuple(automaton.successors[state.factors[-1]])
        if self.model is None or self.torch is None or self.device is None:
            prob = 1.0 / len(legal)
            return {int(action): prob for action in legal}

        torch = self.torch
        x = torch.tensor(state.tensor[None, ...], dtype=torch.long, device=self.device)
        context = torch.tensor(
            normalized_context_features(
                power=state.power,
                factors=state.factors,
                metrics=state.metrics,
                score=state.score,
            )[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        last = torch.tensor([state.factors[-1]], dtype=torch.long, device=self.device)
        legal_mask = torch.zeros((1, 24), dtype=torch.bool, device=self.device)
        legal_mask[0, list(legal)] = True
        with torch.no_grad():
            logits, _ = self.model(x, context, last)
            logits = logits.masked_fill(~legal_mask, -1e9)
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        return {int(action): float(probs[action]) for action in legal}


class SearchEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(args.seed)
        self.automaton = GNFAutomaton(args.n)
        self.evaluator = ExactEvaluator(
            author_repo=Path(args.author_repo),
            p=args.p,
            n=args.n,
            r=args.r,
            max_degree=args.max_degree,
            width_weight=args.width_weight,
            min_meaningful_length=args.min_meaningful_length,
            degeneracy_weight=args.degeneracy_weight,
        )
        self.prior = PolicyPrior(
            checkpoint_path=Path(args.policy_checkpoint) if args.policy_checkpoint else None,
            device_arg=args.device,
            evaluator=self.evaluator,
        )
        self.eval_cache: dict[tuple[int, tuple[int, ...]], EvaluatedWord] = {}
        self.seen_output: set[tuple[int, tuple[int, ...]]] = set()
        self.generated_actions = 0
        self.duplicate_actions = 0

    def evaluate_word(self, power: int, factors: Sequence[int]) -> EvaluatedWord:
        key = (int(power) % 2, tuple(int(value) for value in factors))
        cached = self.eval_cache.get(key)
        if cached is not None:
            return cached
        evaluated = self.evaluator.evaluate_batch(
            [(int(power), tuple(int(value) for value in factors))],
            batch_size=1,
        )[0]
        self.eval_cache[key] = evaluated
        return evaluated

    def make_append_actions(self, state: EvaluatedWord) -> list[Action]:
        if self.args.action_mode == "replace":
            return []
        if len(state.factors) >= self.args.max_length:
            return []
        return [Action("append", (int(action),)) for action in self.automaton.successors[state.factors[-1]]]

    def sample_replace_actions(self, state: EvaluatedWord) -> list[Action]:
        if self.args.action_mode == "append":
            return []
        if len(state.factors) < min(self.args.replace_block_sizes):
            return []
        actions: list[Action] = []
        seen_payloads: set[tuple[int, ...]] = set()
        block_sizes = [size for size in self.args.replace_block_sizes if size <= len(state.factors)]
        attempts = max(self.args.mutation_actions_per_node * 8, 32)
        for _ in range(attempts):
            if len(actions) >= self.args.mutation_actions_per_node:
                break
            block = self.rng.choice(block_sizes)
            start = self.rng.randint(0, len(state.factors) - block)
            end = start + block
            left = state.factors[start - 1] if start > 0 else None
            right = state.factors[end] if end < len(state.factors) else None
            try:
                new_block = self.automaton.sample_bridge(left, right, block, self.rng)
            except ValueError:
                continue
            if tuple(new_block) == state.factors[start:end]:
                continue
            new_factors = state.factors[:start] + tuple(new_block) + state.factors[end:]
            if not self.automaton.is_legal(new_factors):
                continue
            payload = (start, block, *new_block)
            if payload in seen_payloads:
                continue
            seen_payloads.add(payload)
            actions.append(Action("replace", payload))
        return actions

    def next_factors(self, state: EvaluatedWord, action: Action) -> tuple[int, ...]:
        if action.kind == "append":
            return state.factors + (action.payload[0],)
        if action.kind == "replace":
            start, block, *new_block = action.payload
            end = start + block
            return state.factors[:start] + tuple(new_block) + state.factors[end:]
        raise ValueError(f"unknown action kind {action.kind!r}")

    def expand_node(self, node: Node) -> None:
        if node.actions:
            return
        append_actions = self.make_append_actions(node.state)
        replace_actions = self.sample_replace_actions(node.state)
        node.actions = append_actions + replace_actions
        if not node.actions:
            return

        append_probs = self.prior.append_probs(node.state, self.automaton) if append_actions else {}
        replace_fraction = self.args.replace_prior_fraction if replace_actions and append_actions else (1.0 if replace_actions else 0.0)
        append_fraction = 1.0 - replace_fraction if append_actions else 0.0
        priors: dict[str, float] = {}
        for action in append_actions:
            priors[action.key()] = append_fraction * append_probs.get(action.payload[0], 0.0)
        if replace_actions:
            uniform = replace_fraction / len(replace_actions)
            for action in replace_actions:
                priors[action.key()] = uniform
        total = sum(priors.values())
        if total <= 0:
            uniform = 1.0 / len(node.actions)
            priors = {action.key(): uniform for action in node.actions}
        else:
            priors = {key: value / total for key, value in priors.items()}
        node.priors = priors

    def select_action(self, node: Node) -> Action:
        log_parent = math.sqrt(max(1, node.visits))
        best_action = None
        best_value = -float("inf")
        for action in node.actions:
            key = action.key()
            child = node.children.get(key)
            q_value = child.mean_value if child is not None else 0.0
            child_visits = child.visits if child is not None else 0
            prior = node.priors.get(key, 1.0 / max(1, len(node.actions)))
            u_value = self.args.c_puct * prior * log_parent / (1 + child_visits)
            value = q_value + u_value
            if value > best_value:
                best_value = value
                best_action = action
        assert best_action is not None
        return best_action

    def shaped_value(self, root: EvaluatedWord, child: EvaluatedWord) -> float:
        scale = max(10.0, abs(root.score) * 0.25)
        value = (root.score - child.score) / scale
        value -= 0.01 * max(0, len(child.factors) - len(root.factors))
        if child.metrics.get("scalar_identity"):
            value += self.args.kernel_bonus
        return float(max(-self.args.value_clip, min(self.args.value_clip, value)))

    @staticmethod
    def backpropagate(path: Sequence[Node], value: float) -> None:
        for node in path:
            node.visits += 1
            node.value_sum += value

    def simulate(self, root: Node) -> EvaluatedWord:
        node = root
        path = [node]
        for _ in range(self.args.tree_depth):
            if node.state.metrics.get("scalar_identity"):
                value = self.args.kernel_bonus
                self.backpropagate(path, value)
                return node.state
            self.expand_node(node)
            if not node.actions:
                value = self.shaped_value(root.state, node.state)
                self.backpropagate(path, value)
                return node.state
            action = self.select_action(node)
            key = action.key()
            child = node.children.get(key)
            if child is None:
                factors = self.next_factors(node.state, action)
                child_key = (node.state.power % 2, factors)
                self.generated_actions += 1
                child_state = self.evaluate_word(node.state.power, factors)
                child = Node(
                    state=child_state,
                    parent=node,
                    action_from_parent=action,
                    depth=node.depth + 1,
                )
                node.children[key] = child
                path.append(child)
                value = self.shaped_value(root.state, child_state)
                self.backpropagate(path, value)
                return child_state
            node = child
            path.append(node)
        value = self.shaped_value(root.state, node.state)
        self.backpropagate(path, value)
        return node.state

    def collect_nodes(self, root: Node) -> list[Node]:
        output: list[Node] = []
        stack = [root]
        while stack:
            node = stack.pop()
            output.append(node)
            stack.extend(node.children.values())
        return output

    def initial_roots(self) -> list[EvaluatedWord]:
        roots: list[tuple[int, tuple[int, ...]]] = []
        for value in self.args.seed_word:
            roots.append(parse_seed_word(value))
        while len(roots) < self.args.root_count:
            length = self.rng.randint(self.args.root_min_length, self.args.root_max_length)
            power = self.rng.choice((0, 1))
            roots.append((power, self.automaton.sample_uniform(length, self.rng)))
        return [self.evaluate_word(power, factors) for power, factors in roots]

    def run(self) -> dict:
        start_time = time.time()
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_path = output_dir / "progress.jsonl"
        candidates_path = output_dir / "candidates.jsonl"
        replay_path = output_dir / "replay_targets.jsonl"
        progress_path.write_text("", encoding="utf-8")
        candidates_path.write_text("", encoding="utf-8")
        replay_path.write_text("", encoding="utf-8")

        frontier = self.initial_roots()
        best: list[EvaluatedWord] = sorted(frontier, key=lambda item: (item.score, len(item.factors)))[
            : self.args.keep_best
        ]
        kernel_hits: list[EvaluatedWord] = []

        for iteration in range(1, self.args.iterations + 1):
            roots = [Node(state=item) for item in frontier]
            iteration_candidates: list[EvaluatedWord] = []
            replay_rows: list[dict] = []
            for root in roots:
                for _ in range(self.args.simulations_per_root):
                    leaf_state = self.simulate(root)
                    iteration_candidates.append(leaf_state)
                    if leaf_state.metrics.get("scalar_identity"):
                        kernel_hits.append(leaf_state)
                        if self.args.stop_at_kernel:
                            break
                self.expand_node(root)
                if root.actions:
                    visits = {
                        action.key(): root.children[action.key()].visits
                        if action.key() in root.children
                        else 0
                        for action in root.actions
                    }
                    replay_rows.append(
                        {
                            "power": root.state.power,
                            "factor_ids": list(root.state.factors),
                            "length": len(root.state.factors),
                            "score": root.state.score,
                            "metrics": root.state.metrics,
                            "visit_counts": visits,
                            "root_value": root.mean_value,
                        }
                    )
                if kernel_hits and self.args.stop_at_kernel:
                    break

            unique_rows = []
            for item in iteration_candidates:
                key = (item.power % 2, item.factors)
                if key in self.seen_output:
                    continue
                self.seen_output.add(key)
                unique_rows.append(
                    {
                        "iteration": iteration,
                        "power": item.power,
                        "factor_ids": list(item.factors),
                        "length": len(item.factors),
                        "score": item.score,
                        "metrics": item.metrics,
                    }
                )
            append_jsonl(candidates_path, unique_rows)
            append_jsonl(replay_path, replay_rows)

            best = sorted(best + iteration_candidates, key=lambda item: (item.score, len(item.factors)))[
                : self.args.keep_best
            ]
            eligible = [item for item in best + iteration_candidates if len(item.factors) < self.args.max_length]
            frontier = sorted(eligible, key=lambda item: (item.score, len(item.factors)))[
                : self.args.frontier_size
            ]
            row = {
                "iteration": iteration,
                "frontier_size": len(frontier),
                "iteration_candidates": len(iteration_candidates),
                "unique_written": len(unique_rows),
                "best_score": best[0].score if best else None,
                "best_metrics": best[0].metrics if best else {},
                "best_length": len(best[0].factors) if best else None,
                "kernel_hits": len(kernel_hits),
                "eval_cache_size": len(self.eval_cache),
                "elapsed_seconds": round(time.time() - start_time, 2),
            }
            append_jsonl(progress_path, [row])
            print(json.dumps({"phase": "rl_mcts", **row}, sort_keys=True), flush=True)
            if kernel_hits and self.args.stop_at_kernel:
                break
            if not frontier:
                break

        summary = {
            "format": "shaped-rl-mcts-summary-v1",
            "metadata": {
                "p": self.args.p,
                "n": self.args.n,
                "r": self.args.r,
                "action_mode": self.args.action_mode,
                "policy_checkpoint": self.args.policy_checkpoint,
                "iterations": self.args.iterations,
                "simulations_per_root": self.args.simulations_per_root,
                "tree_depth": self.args.tree_depth,
                "max_length": self.args.max_length,
                "seed": self.args.seed,
                "elapsed_seconds": round(time.time() - start_time, 2),
            },
            "kernel_hits": [
                {
                    "power": item.power,
                    "factor_ids": list(item.factors),
                    "length": len(item.factors),
                    "score": item.score,
                    "metrics": item.metrics,
                }
                for item in kernel_hits[: self.args.keep_best]
            ],
            "best": [
                {
                    "power": item.power,
                    "factor_ids": list(item.factors),
                    "length": len(item.factors),
                    "score": item.score,
                    "metrics": item.metrics,
                }
                for item in best
            ],
            "stats": {
                "eval_cache_size": len(self.eval_cache),
                "generated_actions": self.generated_actions,
                "duplicate_actions": self.duplicate_actions,
                "seen_output": len(self.seen_output),
            },
        }
        write_json(output_dir / "summary.json", summary)
        return summary


def parse_block_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("replace block sizes must be positive")
    return sizes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shaped-reward policy/value MCTS over legal Garside actions."
    )
    parser.add_argument("--author-repo", default=str(DEFAULT_AUTHOR_REPO))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-checkpoint", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--p", type=int, default=7)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--max-degree", type=int, default=192)
    parser.add_argument("--width-weight", type=float, default=0.15)
    parser.add_argument("--min-meaningful-length", type=int, default=15)
    parser.add_argument("--degeneracy-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seed-word", action="append", default=[])
    parser.add_argument("--root-count", type=int, default=64)
    parser.add_argument("--root-min-length", type=int, default=12)
    parser.add_argument("--root-max-length", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--simulations-per-root", type=int, default=32)
    parser.add_argument("--tree-depth", type=int, default=6)
    parser.add_argument("--frontier-size", type=int, default=96)
    parser.add_argument("--keep-best", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=90)
    parser.add_argument("--action-mode", choices=("append", "replace", "mixed"), default="mixed")
    parser.add_argument("--mutation-actions-per-node", type=int, default=8)
    parser.add_argument("--replace-block-sizes", type=parse_block_sizes, default=(2, 3, 4))
    parser.add_argument("--replace-prior-fraction", type=float, default=0.35)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--kernel-bonus", type=float, default=50.0)
    parser.add_argument("--value-clip", type=float, default=50.0)
    parser.add_argument("--stop-at-kernel", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    SearchEngine(args).run()


if __name__ == "__main__":
    main()
