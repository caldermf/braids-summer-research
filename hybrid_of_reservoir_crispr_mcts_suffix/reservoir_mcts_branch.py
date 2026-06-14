from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from crispr_algorithms.crispr_trajectory_search_v4.gnf import GNFAutomaton
except ModuleNotFoundError:
    from crispr_trajectory_search_v4.gnf import GNFAutomaton

from .candidates import Candidate, select_diverse_candidates
from .config import ReservoirMCTSConfig
from .exact import ExactEngine, ExactState
from .io_utils import append_jsonl, write_json


@dataclass
class DatabaseNode:
    factor_ids: tuple[int, ...]
    score: int
    visits: int = 0
    source: str = "paper_reservoir_depth35"


class UniformReservoir:
    def __init__(self, capacity: int, rng: random.Random):
        self.capacity = capacity
        self.rng = rng
        self.seen = 0
        self.items: list[ExactState] = []

    def add(self, item: ExactState) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        replacement = self.rng.randint(1, self.seen)
        if replacement <= self.capacity:
            self.items[replacement - 1] = item


def _select_whole_low_buckets(
    buckets: dict[int, UniformReservoir],
    use_best: int,
) -> list[ExactState]:
    selected = []
    for projlen in sorted(buckets):
        items = buckets[projlen].items
        if len(selected) + len(items) > use_best:
            break
        selected.extend(items)
    return selected


def _reservoir_playout(
    start: ExactState,
    *,
    max_depth: int,
    automaton: GNFAutomaton,
    engine: ExactEngine,
    bucket_size: int,
    use_best: int,
    rng: random.Random,
) -> tuple[ExactState, list[ExactState]]:
    frontier = [start]
    best = start
    kernels = [start] if start.has_kernel else []
    while frontier and len(frontier[0].factor_ids) < max_depth:
        buckets: dict[int, UniformReservoir] = {}
        for state in frontier:
            for factor_id in automaton.successors[state.factor_ids[-1]]:
                child = engine.extend(state, factor_id)
                if child.has_kernel:
                    kernels.append(child)
                if (
                    child.final_projlen,
                    len(child.factor_ids),
                ) < (
                    best.final_projlen,
                    len(best.factor_ids),
                ):
                    best = child
                reservoir = buckets.setdefault(
                    child.final_projlen,
                    UniformReservoir(bucket_size, rng),
                )
                reservoir.add(child)
        frontier = _select_whole_low_buckets(buckets, use_best)
    return best, kernels


def _weighted_sample(
    database: dict[tuple[int, ...], DatabaseNode],
    count: int,
    exploration_floor: float,
    rng: random.Random,
    max_depth: int,
) -> list[DatabaseNode]:
    eligible = [node for node in database.values() if len(node.factor_ids) < max_depth]
    if not eligible:
        return []
    ordered = sorted(
        eligible,
        key=lambda node: (node.score, -len(node.factor_ids), node.visits),
    )
    weights = [
        1.0 / math.sqrt(rank + 1) + exploration_floor / (1 + node.visits)
        for rank, node in enumerate(ordered)
    ]
    selected = []
    available = list(zip(ordered, weights))
    for _ in range(min(count, len(available))):
        nodes, current_weights = zip(*available)
        chosen = rng.choices(nodes, weights=current_weights, k=1)[0]
        selected.append(chosen)
        available = [(node, weight) for node, weight in available if node is not chosen]
    return selected


def run_reservoir_mcts_branch(
    candidates: list[Candidate],
    branch: ReservoirMCTSConfig,
    *,
    p: int,
    n: int,
    max_depth: int,
    output_dir: str | Path,
    stop_at_kernel: bool = True,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events_path = output / "iterations.jsonl"
    if events_path.exists():
        events_path.unlink()

    rng = random.Random(branch.seed)
    automaton = GNFAutomaton(n)
    engine = ExactEngine(p, n)
    selected = select_diverse_candidates(candidates, branch.pool_size, branch.seed)
    database = {
        candidate.factor_ids: DatabaseNode(
            factor_ids=candidate.factor_ids,
            score=max(0, candidate.author_projlen - 1),
        )
        for candidate in selected
    }
    kernel_hits: dict[tuple[int, ...], ExactState] = {}
    best_states: dict[tuple[int, ...], ExactState] = {}

    for iteration in range(1, branch.iterations + 1):
        roots = _weighted_sample(
            database,
            branch.selected_nodes_per_iteration,
            branch.exploration_floor,
            rng,
            max_depth,
        )
        expanded = 0
        for root in roots:
            root.visits += 1
            root_state = engine.evaluate(root.factor_ids)
            legal = list(automaton.successors[root.factor_ids[-1]])
            if branch.children_per_node > 0 and len(legal) > branch.children_per_node:
                legal = rng.sample(legal, branch.children_per_node)
            for factor_id in legal:
                expanded += 1
                child = engine.extend(root_state, factor_id)
                best, kernels = _reservoir_playout(
                    child,
                    max_depth=max_depth,
                    automaton=automaton,
                    engine=engine,
                    bucket_size=branch.playout_bucket_size,
                    use_best=branch.playout_use_best,
                    rng=rng,
                )
                score = min(child.final_projlen, best.final_projlen)
                existing = database.get(child.factor_ids)
                if existing is None or score < existing.score:
                    database[child.factor_ids] = DatabaseNode(
                        factor_ids=child.factor_ids,
                        score=score,
                        source="mcts_child",
                    )
                existing_best = database.get(best.factor_ids)
                if existing_best is None or best.final_projlen < existing_best.score:
                    database[best.factor_ids] = DatabaseNode(
                        factor_ids=best.factor_ids,
                        score=best.final_projlen,
                        source="reservoir_playout",
                    )
                best_states[best.factor_ids] = best
                for hit in kernels:
                    kernel_hits.setdefault(hit.factor_ids, hit)

        if len(database) > branch.database_limit:
            ordered = sorted(
                database.values(),
                key=lambda node: (node.score, -len(node.factor_ids), node.visits),
            )
            database = {node.factor_ids: node for node in ordered[: branch.database_limit]}

        event = {
            "iteration": iteration,
            "database_size": len(database),
            "roots": len(roots),
            "expanded_children": expanded,
            "best_score": min(node.score for node in database.values()),
            "maximum_depth": max(len(node.factor_ids) for node in database.values()),
            "kernel_hits": len(kernel_hits),
        }
        append_jsonl(events_path, event)
        print(
            f"[reservoir-mcts] iteration={iteration} database={len(database)} "
            f"best={event['best_score']} hits={len(kernel_hits)}",
            flush=True,
        )
        if kernel_hits and stop_at_kernel:
            break

    ranked_nodes = sorted(
        database.values(),
        key=lambda node: (node.score, -len(node.factor_ids), node.visits),
    )[:100]
    ranked = []
    for node in ranked_nodes:
        state = best_states.get(node.factor_ids) or engine.evaluate(node.factor_ids)
        ranked.append({**state.summary(), "mcts_score": node.score, "source": node.source})

    result = {
        "branch": "reservoir_mcts",
        "config": asdict(branch),
        "reservoir_seeds": len(selected),
        "database_size": len(database),
        "kernel_hits": [state.summary() for state in kernel_hits.values()],
        "best": ranked,
    }
    write_json(output / "result.json", result)
    return result
