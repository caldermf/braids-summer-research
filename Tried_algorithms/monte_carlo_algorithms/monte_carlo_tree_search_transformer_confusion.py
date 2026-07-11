#!/usr/bin/env python3
"""
Late-breakout MCTS with transformer confusion as a secondary score.

The main search pressure is still the length-aware low-projlen surprise used by
`monte_carlo_tree_search_breakout_surprise.py`. This variant adds a small
secondary reward from a trained braidmod transformer:

    value = breakout_value + confusion_weight * transformer_confusion

The safest confusion metric is target cross-entropy: after appending a legal
Garside factor u, ask how surprised the model is that u is the final factor of
the current prefix. Entropy is also supported for comparison.
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMER_SUPPORT_ROOT = REPO_ROOT / "transformer_support"
for path in (REPO_ROOT, TRANSFORMER_SUPPORT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import torch.nn.functional as F

from peyl.braid_data import (
    append_factor_to_burau_matrix,
    factor_ids_to_perms,
    identity_burau_matrix,
    polynomial_matrix_degree_bounds,
)
from transformer_support.predict_garside_mlp import (
    build_model,
    confusion_score_from_logits,
    resolve_device,
)
from monte_carlo_algorithms.monte_carlo_tree_search_breakout_surprise import (
    BreakoutMCTSConfig,
    BreakoutSurpriseMCTS,
)
from monte_carlo_algorithms.monte_carlo_tree_search_surprise_beam import (
    BucketReservoir,
    MCTSNode,
)


@dataclass
class TransformerConfusionMCTSConfig(BreakoutMCTSConfig):
    checkpoint: str = ""
    device: str = "auto"
    confusion_weight: float = 0.1
    confusion_metric: str = "target_xent"
    confusion_min_depth: int = 1
    truncate_overflow: bool = True


class TransformerConfusionScorer:
    def __init__(self, config: TransformerConfusionMCTSConfig):
        if not config.checkpoint:
            raise ValueError("--checkpoint is required for transformer-confusion MCTS")
        if config.confusion_metric not in {"target_xent", "entropy"}:
            raise ValueError("--confusion-metric must be target_xent or entropy")

        self.config = config
        self.device = resolve_device(config.device)
        checkpoint = torch.load(config.checkpoint, map_location=self.device)
        self.model = build_model(checkpoint, self.device)
        self.model.eval()
        self.D = int(checkpoint["D"])
        self.p = int(checkpoint["p"])
        self.perm_classes = [tuple(item) for item in checkpoint.get("perm_classes", [])]
        if not self.perm_classes:
            from transformer_support.garside_transformer import PERMUTATIONS_S4

            self.perm_classes = [tuple(item) for item in PERMUTATIONS_S4]
        self.perm_to_class = {perm: idx for idx, perm in enumerate(self.perm_classes)}
        self.cache: Dict[Tuple[int, ...], dict] = {}

    def tensor_from_poly_matrix(self, poly_mat):
        min_exp, _ = polynomial_matrix_degree_bounds(poly_mat)
        tensor = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(self.D)]
        overflow_terms = 0
        for i in range(3):
            for j in range(3):
                for exp, coeff in poly_mat[i][j].items():
                    shifted = exp - min_exp
                    if 0 <= shifted < self.D:
                        tensor[shifted][i][j] = coeff % self.p
                    else:
                        overflow_terms += 1
        if overflow_terms and not self.config.truncate_overflow:
            raise ValueError(
                f"Burau support overflows transformer D={self.D}; "
                "rerun with --truncate-overflow to score truncated tensors."
            )
        return tensor, min_exp, overflow_terms

    def score(self, factor_ids: List[int], poly_mat) -> dict:
        if not factor_ids:
            return {
                "transformer_confusion": 0.0,
                "transformer_target_xent": 0.0,
                "transformer_entropy": 0.0,
                "transformer_target_prob": None,
                "transformer_predicted_class": None,
                "transformer_overflow_terms": 0,
            }
        key = tuple(int(item) for item in factor_ids)
        if key in self.cache:
            return self.cache[key]

        if self.config.p != self.p:
            raise ValueError(f"Search p={self.config.p}, but checkpoint was trained for p={self.p}")
        if self.config.n != 4:
            raise ValueError("The tracked braidmod transformer expects n=4 / 3x3 Burau tensors")

        tensor, min_degree, overflow_terms = self.tensor_from_poly_matrix(poly_mat)
        target_perm = factor_ids_to_perms([factor_ids[-1]], n=self.config.n)[0]
        target_class = self.perm_to_class[target_perm]

        with torch.no_grad():
            x = torch.tensor([tensor], dtype=torch.long, device=self.device)
            min_degree_tensor = torch.tensor([min_degree], dtype=torch.float32, device=self.device)
            length_tensor = torch.tensor([len(factor_ids)], dtype=torch.float32, device=self.device)
            logits, _ = self.model(
                x,
                min_degree=min_degree_tensor,
                garside_length=length_tensor,
            )
            entropy = float(confusion_score_from_logits(logits)[0].item())
            target_tensor = torch.tensor([target_class], dtype=torch.long, device=self.device)
            target_xent = float(F.cross_entropy(logits, target_tensor).item())
            probs = torch.softmax(logits[0], dim=-1)
            predicted_class = int(torch.argmax(probs).item())
            target_prob = float(probs[target_class].item())

        if self.config.confusion_metric == "target_xent":
            confusion = target_xent
        else:
            confusion = entropy

        result = {
            "transformer_confusion": confusion,
            "transformer_target_xent": target_xent,
            "transformer_entropy": entropy,
            "transformer_target_prob": target_prob,
            "transformer_predicted_class": predicted_class,
            "transformer_target_class": int(target_class),
            "transformer_overflow_terms": int(overflow_terms),
        }
        self.cache[key] = result
        return result


class TransformerConfusionMCTS(BreakoutSurpriseMCTS):
    def __init__(self, config: TransformerConfusionMCTSConfig):
        if config.confusion_weight < 0:
            raise ValueError("--confusion-weight must be nonnegative")
        if config.confusion_min_depth < 0:
            raise ValueError("--confusion-min-depth must be nonnegative")
        self.transformer_scorer = TransformerConfusionScorer(config)
        super().__init__(config)

    @property
    def config(self) -> TransformerConfusionMCTSConfig:
        return self._config

    @config.setter
    def config(self, value: TransformerConfusionMCTSConfig) -> None:
        self._config = value

    def create_run_directory(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = Path(self.config.output_dir)
        run_dir = base_dir / f"mcts_transformer_confusion_{timestamp}_seed{self.config.seed}"
        suffix = 1
        unique_run_dir = run_dir
        while unique_run_dir.exists():
            suffix += 1
            unique_run_dir = base_dir / f"{run_dir.name}_{suffix}"
        unique_run_dir.mkdir(parents=True, exist_ok=False)
        return unique_run_dir

    def score_prefix_with_history(self, factor_ids: List[int], burau_matrix, surprise_z_history: List[float]) -> dict:
        score = super().score_prefix_with_history(factor_ids, burau_matrix, surprise_z_history)
        depth = int(score["depth"])
        transformer_score = self.transformer_scorer.score(factor_ids, burau_matrix)

        weighted_confusion = 0.0
        if depth >= self.config.confusion_min_depth:
            weighted_confusion = self.config.confusion_weight * float(transformer_score["transformer_confusion"])
            score["value"] += weighted_confusion

        score.update(transformer_score)
        score["transformer_weighted_confusion"] = weighted_confusion
        score["state"].update(transformer_score)
        score["state"]["transformer_weighted_confusion"] = weighted_confusion
        return score

    def reconstruct_history_for_node(self, node: MCTSNode) -> Tuple[List[float], dict]:
        # Recompute through score_prefix_with_history so the saved node score
        # includes transformer fields as well as breakout fields.
        return super().reconstruct_history_for_node(node)

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
            "best_prefix_transformer_confusion": best_prefix.get("transformer_confusion"),
            "best_prefix_transformer_target_xent": best_prefix.get("transformer_target_xent"),
            "best_prefix_transformer_entropy": best_prefix.get("transformer_entropy"),
            "best_prefix_transformer_target_prob": best_prefix.get("transformer_target_prob"),
            "best_prefix_transformer_weighted_confusion": best_prefix.get("transformer_weighted_confusion"),
            "best_prefix_transformer_overflow_terms": best_prefix.get("transformer_overflow_terms"),
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
        summary.update(
            {
                "checkpoint": self.config.checkpoint,
                "device": self.config.device,
                "confusion_weight": self.config.confusion_weight,
                "confusion_metric": self.config.confusion_metric,
                "confusion_min_depth": self.config.confusion_min_depth,
                "truncate_overflow": self.config.truncate_overflow,
                "transformer_D": self.transformer_scorer.D,
                "transformer_p": self.transformer_scorer.p,
                "transformer_cache_size": len(self.transformer_scorer.cache),
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> TransformerConfusionMCTSConfig:
    parser = argparse.ArgumentParser(description="Run breakout MCTS with transformer confusion scoring.")
    parser.add_argument("--checkpoint", required=True, help="Path to braidmod transformer best_model.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=65)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--exploration-weight", type=float, default=1.4)
    parser.add_argument("--baseline-samples", type=int, default=512)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--reservoir-size", type=int, default=16)
    parser.add_argument("--breakout-weight", type=float, default=0.5)
    parser.add_argument("--depth-power", type=float, default=1.0)
    parser.add_argument("--confusion-weight", type=float, default=0.1)
    parser.add_argument("--confusion-metric", choices=["target_xent", "entropy"], default="target_xent")
    parser.add_argument("--confusion-min-depth", type=int, default=1)
    parser.add_argument("--no-truncate-overflow", action="store_true")
    parser.add_argument("--progressive-widening-k", type=float, default=0.5)
    parser.add_argument("--progressive-widening-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    return TransformerConfusionMCTSConfig(
        checkpoint=args.checkpoint,
        device=args.device,
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
        confusion_weight=args.confusion_weight,
        confusion_metric=args.confusion_metric,
        confusion_min_depth=args.confusion_min_depth,
        truncate_overflow=not args.no_truncate_overflow,
        progressive_widening_k=args.progressive_widening_k,
        progressive_widening_alpha=args.progressive_widening_alpha,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def main() -> None:
    search = TransformerConfusionMCTS(parse_args())
    print(json.dumps(search.run(), indent=2))


if __name__ == "__main__":
    main()
