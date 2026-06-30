#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRAID_GPT_ROOT = REPO_ROOT / "Braid-GPT"
DEFAULT_AUTHOR_REPO = REPO_ROOT / "hybrid_of_reservoir_crispr_mcts_suffix" / "third_party" / "braids_project"


@dataclass(frozen=True)
class EvaluatedBraid:
    power: int
    factors: tuple[int, ...]
    metrics: dict
    objective: float
    digest: str
    parent_digest: str
    operator: str
    generation: int
    source: str


def load_braid_gpt_module(braid_gpt_root: Path):
    module_path = braid_gpt_root / "braid_gpt.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find Braid-GPT script at {module_path}")
    spec = importlib.util.spec_from_file_location("braid_gpt_runtime_for_long_mutation", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Braid-GPT from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def patch_functools_cache_for_old_python() -> None:
    if sys.version_info >= (3, 10):
        return
    if getattr(functools.cache, "_peyl_staticmethod_compatible", False):
        return
    original_cache = functools.cache

    def cache_compat(user_function):
        if isinstance(user_function, staticmethod):
            return staticmethod(original_cache(user_function.__func__))
        if isinstance(user_function, classmethod):
            return classmethod(original_cache(user_function.__func__))
        return original_cache(user_function)

    cache_compat._peyl_staticmethod_compatible = True  # type: ignore[attr-defined]
    functools.cache = cache_compat  # type: ignore[assignment]


def read_json(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_int_list(value: str) -> tuple[int, ...]:
    output = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not output:
        raise ValueError("expected at least one integer")
    return output


def metric_projlen(metrics: dict) -> int:
    return int(metrics.get("projlen", metrics.get("projective_width", 0)))


def clean_metrics(metrics: dict) -> dict:
    output = dict(metrics)
    if "projlen" not in output:
        output["projlen"] = int(output.get("projective_width", 0))
    output.pop("projective_width", None)
    return output


def braid_digest(power: int, factors: Sequence[int]) -> str:
    payload = json.dumps([int(power), [int(x) for x in factors]], separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def row_to_serializable(row: EvaluatedBraid) -> dict:
    return {
        "power": row.power,
        "factors": list(row.factors),
        "length": len(row.factors),
        "metrics": row.metrics,
        "objective": row.objective,
        "digest": row.digest,
        "parent_digest": row.parent_digest,
        "operator": row.operator,
        "generation": row.generation,
        "source": row.source,
    }


def degeneracy_features(factors: Sequence[int]) -> dict:
    if not factors:
        return {
            "dominant_fraction": 0.0,
            "top_two_fraction": 0.0,
            "max_run_fraction": 0.0,
            "max_run_length": 0,
            "unique_fraction": 0.0,
            "repeated_bigram_fraction": 0.0,
            "period_at_most_2": False,
        }
    counts = Counter(factors)
    ordered_counts = sorted(counts.values(), reverse=True)
    max_run = 1
    run = 1
    for left, right in zip(factors, factors[1:]):
        if left == right:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    period_at_most_2 = False
    if len(factors) >= 4:
        period_at_most_2 = any(
            all(factors[index] == factors[index % period] for index in range(len(factors)))
            for period in (1, 2)
        )
    bigrams = Counter(zip(factors, factors[1:]))
    return {
        "dominant_fraction": ordered_counts[0] / len(factors),
        "top_two_fraction": sum(ordered_counts[:2]) / len(factors),
        "max_run_fraction": max_run / len(factors),
        "max_run_length": max_run,
        "unique_fraction": len(counts) / len(factors),
        "repeated_bigram_fraction": (
            max(bigrams.values()) / max(1, len(factors) - 1) if bigrams else 0.0
        ),
        "period_at_most_2": period_at_most_2,
    }


def degeneracy_penalty(factors: Sequence[int]) -> float:
    d = degeneracy_features(factors)
    penalty = 0.0
    penalty += max(0.0, d["dominant_fraction"] - 0.60) * 80.0
    penalty += max(0.0, d["top_two_fraction"] - 0.82) * 80.0
    penalty += max(0.0, d["max_run_fraction"] - 0.45) * 80.0
    penalty += max(0.0, d["max_run_length"] - 12) * 3.0
    penalty += max(0.0, 0.08 - d["unique_fraction"]) * 120.0
    penalty += max(0.0, d["repeated_bigram_fraction"] - 0.35) * 50.0
    if d["period_at_most_2"]:
        penalty += 20.0
    return float(penalty)


def objective_from_metrics(
    metrics: dict,
    factors: Sequence[int],
    *,
    projlen_weight: float,
    projlen_density_weight: float,
    identity_density_weight: float,
    degeneracy_weight: float,
    target_length: int,
    length_weight: float,
) -> float:
    length = max(1, len(factors))
    identity_defect = float(metrics["identity_defect"])
    projlen = float(metric_projlen(metrics))
    length_penalty = abs(length - target_length) / max(1, target_length)
    objective = (
        identity_defect
        + projlen_weight * projlen
        + projlen_density_weight * projlen / length
        + identity_density_weight * identity_defect / length
        + degeneracy_weight * degeneracy_penalty(factors)
        + length_weight * length_penalty
    )
    if metrics.get("scalar_identity"):
        objective -= 1_000_000.0
    return float(objective)


def reservoir_sort_key(row: EvaluatedBraid, kind: str) -> tuple:
    length = max(1, len(row.factors))
    projlen = metric_projlen(row.metrics)
    identity_defect = int(row.metrics["identity_defect"])
    if kind == "identity":
        return (identity_defect, projlen, row.objective, length)
    if kind == "density":
        return (identity_defect / length, projlen / length, identity_defect, row.objective)
    if kind == "long":
        return (projlen / length, identity_defect / length, row.objective, -length)
    if kind == "recent":
        return (row.generation * -1, row.objective, identity_defect, projlen)
    return (row.objective, identity_defect, projlen, length)


class ExactEvaluator:
    def __init__(
        self,
        *,
        bgpt,
        author_repo: Path,
        n: int,
        r: int,
        p: int,
        projlen_weight: float,
        projlen_density_weight: float,
        identity_density_weight: float,
        degeneracy_weight: float,
        target_length: int,
        length_weight: float,
    ) -> None:
        self.bgpt = bgpt
        patch_functools_cache_for_old_python()
        self.peyl, self.polymat, self.evaluate_braids = bgpt.setup_author_imports(author_repo)
        self.rep = self.peyl.JonesSummand(n=n, r=r, p=p)
        self.n = int(n)
        self.r = int(r)
        self.p = int(p)
        self.projlen_weight = float(projlen_weight)
        self.projlen_density_weight = float(projlen_density_weight)
        self.identity_density_weight = float(identity_density_weight)
        self.degeneracy_weight = float(degeneracy_weight)
        self.target_length = int(target_length)
        self.length_weight = float(length_weight)

    def braid(self, power: int, factors: Sequence[int]):
        return self.peyl.GNF(n=self.n, power=int(power), factors=tuple(int(x) for x in factors))

    def evaluate_braid_batch(
        self,
        braids: Sequence,
        *,
        parents: Sequence[str],
        operators: Sequence[str],
        generation: int,
        sources: Sequence[str],
        batch_size: int,
    ) -> list[EvaluatedBraid]:
        output: list[EvaluatedBraid] = []
        for start in range(0, len(braids), batch_size):
            chunk = braids[start : start + batch_size]
            images = self.evaluate_braids(self.rep, chunk)
            for local, (braid, image) in enumerate(zip(chunk, images)):
                index = start + local
                factors = tuple(int(x) for x in braid.factors)
                metrics = clean_metrics(self.bgpt.scalar_identity_metrics(self.polymat, image))
                objective = objective_from_metrics(
                    metrics,
                    factors,
                    projlen_weight=self.projlen_weight,
                    projlen_density_weight=self.projlen_density_weight,
                    identity_density_weight=self.identity_density_weight,
                    degeneracy_weight=self.degeneracy_weight,
                    target_length=self.target_length,
                    length_weight=self.length_weight,
                )
                output.append(
                    EvaluatedBraid(
                        power=int(braid.power),
                        factors=factors,
                        metrics=metrics,
                        objective=objective,
                        digest=braid_digest(int(braid.power), factors),
                        parent_digest=parents[index],
                        operator=operators[index],
                        generation=generation,
                        source=sources[index],
                    )
                )
        return output


def extract_word_from_row(row: dict) -> tuple[int, tuple[int, ...]] | None:
    candidates = (
        ("final_power", "final_factors"),
        ("powered_power", "powered_factors"),
        ("power", "factor_ids"),
        ("power", "factors"),
    )
    for power_key, factors_key in candidates:
        if power_key in row and factors_key in row:
            factors = tuple(int(x) for x in row[factors_key])
            if factors:
                return int(row[power_key]), factors
    return None


def walk_rows(obj) -> Iterable[dict]:
    if isinstance(obj, dict):
        if extract_word_from_row(obj) is not None:
            yield obj
        for value in obj.values():
            yield from walk_rows(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_rows(value)


def load_seed_words(paths: Sequence[Path], *, limit_per_path: int) -> list[tuple[int, tuple[int, ...], str]]:
    seeds: list[tuple[int, tuple[int, ...], str]] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for path in paths:
        if not path.exists():
            continue
        if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz") or path.suffix == ".txt":
            rows = list(iter_jsonl(path))
        else:
            rows = list(walk_rows(read_json(path)))
        sortable = []
        for index, row in enumerate(rows):
            word = extract_word_from_row(row)
            if word is None:
                continue
            metrics = row.get("exact_metrics", row.get("metrics", {}))
            defect = int(metrics.get("identity_defect", 10**9))
            projlen = int(metrics.get("projlen", metrics.get("projective_width", 10**9)))
            sortable.append((defect, projlen, index, word, row))
        sortable.sort()
        for defect, projlen, index, word, row in sortable[:limit_per_path]:
            key = (word[0], word[1])
            if key in seen:
                continue
            seen.add(key)
            seeds.append((word[0], word[1], f"{path.name}:{index}:defect_{defect}:projlen_{projlen}"))
    return seeds


def parse_seed_word(value: str) -> tuple[int, tuple[int, ...], str]:
    if ":" not in value:
        raise ValueError("seed words must have form POWER:f1,f2,...")
    power_text, factors_text = value.split(":", 1)
    factors = tuple(int(part.strip()) for part in factors_text.split(",") if part.strip())
    if not factors:
        raise ValueError("seed word has no factors")
    return int(power_text), factors, "cli_seed"


class Mutator:
    def __init__(
        self,
        *,
        evaluator: ExactEvaluator,
        automaton,
        min_length: int,
        max_length: int,
        max_window: int,
        max_growth: int,
        max_conjugator_length: int,
        max_commutator_length: int,
    ) -> None:
        self.evaluator = evaluator
        self.automaton = automaton
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.max_window = int(max_window)
        self.max_growth = int(max_growth)
        self.max_conjugator_length = int(max_conjugator_length)
        self.max_commutator_length = int(max_commutator_length)

    def random_positive_braid(self, length: int, rng: random.Random):
        return self.evaluator.braid(0, self.automaton.sample_uniform(length, rng))

    def in_length_bounds(self, braid) -> bool:
        length = len(tuple(braid.factors))
        return self.min_length <= length <= self.max_length

    def window_replace(self, parent, rng: random.Random):
        factors = tuple(int(x) for x in parent.factors)
        if not factors:
            return None
        width = rng.randint(1, min(self.max_window, len(factors)))
        start = rng.randint(0, len(factors) - width)
        left = factors[start - 1] if start > 0 else None
        right_index = start + width
        right = factors[right_index] if right_index < len(factors) else None
        for _ in range(12):
            try:
                replacement = self.automaton.sample_bridge(left, right, width, rng)
            except ValueError:
                return None
            if replacement != factors[start:right_index]:
                new_factors = factors[:start] + replacement + factors[right_index:]
                return self.evaluator.braid(int(parent.power), new_factors)
        return None

    def boundary_replace(self, parent, rng: random.Random):
        factors = tuple(int(x) for x in parent.factors)
        if not factors:
            return None
        side = rng.choice(("prefix", "suffix"))
        width = rng.randint(1, min(self.max_window, len(factors)))
        if side == "prefix":
            right = factors[width] if width < len(factors) else None
            try:
                replacement = self.automaton.sample_bridge(None, right, width, rng)
            except ValueError:
                return None
            new_factors = replacement + factors[width:]
        else:
            left = factors[-width - 1] if width < len(factors) else None
            try:
                replacement = self.automaton.sample_bridge(left, None, width, rng)
            except ValueError:
                return None
            new_factors = factors[: len(factors) - width] + replacement
        if tuple(new_factors) == factors:
            return None
        return self.evaluator.braid(int(parent.power), new_factors)

    def grow(self, parent, rng: random.Random):
        if len(tuple(parent.factors)) >= self.max_length:
            return None
        growth = rng.randint(1, min(self.max_growth, self.max_length - len(tuple(parent.factors))))
        block = self.random_positive_braid(growth, rng)
        if rng.random() < 0.5:
            return parent * block
        return block * parent

    def conjugate(self, parent, rng: random.Random):
        length = rng.randint(1, self.max_conjugator_length)
        u = self.random_positive_braid(length, rng)
        return u * parent * u.inv()

    def commutator_wrap(self, parent, rng: random.Random):
        length_u = rng.randint(1, self.max_commutator_length)
        length_v = rng.randint(1, self.max_commutator_length)
        u = self.random_positive_braid(length_u, rng)
        v = self.random_positive_braid(length_v, rng)
        commutator = u * v * u.inv() * v.inv()
        if rng.random() < 0.5:
            return commutator * parent
        return parent * commutator

    def random_restart(self, rng: random.Random):
        length = rng.randint(self.min_length, self.max_length)
        power = rng.choice((0, 1))
        return self.evaluator.braid(power, self.automaton.sample_uniform(length, rng))

    def burst(self, parent, rng: random.Random):
        current = parent
        steps = rng.randint(2, 5)
        for _ in range(steps):
            nxt = self.window_replace(current, rng) or self.boundary_replace(current, rng)
            if nxt is None:
                return None
            current = nxt
        return current

    def mutate(self, parent, rng: random.Random, weights: dict[str, float]) -> tuple[object | None, str]:
        names = list(weights)
        values = [float(weights[name]) for name in names]
        operator = rng.choices(names, weights=values, k=1)[0]
        try:
            if operator == "window_replace":
                child = self.window_replace(parent, rng)
            elif operator == "boundary_replace":
                child = self.boundary_replace(parent, rng)
            elif operator == "grow":
                child = self.grow(parent, rng)
            elif operator == "conjugate":
                child = self.conjugate(parent, rng)
            elif operator == "commutator_wrap":
                child = self.commutator_wrap(parent, rng)
            elif operator == "burst":
                child = self.burst(parent, rng)
            elif operator == "random_restart":
                child = self.random_restart(rng)
            else:
                raise ValueError(f"unknown mutation operator: {operator}")
        except Exception:
            return None, operator
        if child is None or not self.in_length_bounds(child):
            return None, operator
        return child, operator


class ReservoirBank:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.seen: dict[str, EvaluatedBraid] = {}
        self.best_identity: list[EvaluatedBraid] = []
        self.best_density: list[EvaluatedBraid] = []
        self.long_non_degenerate: list[EvaluatedBraid] = []
        self.recent_improvers: list[EvaluatedBraid] = []
        self.novelty: list[EvaluatedBraid] = []
        self.kernel_hits: list[EvaluatedBraid] = []

    def add_many(self, rows: Sequence[EvaluatedBraid], parent_lookup: dict[str, EvaluatedBraid]) -> dict:
        new_count = 0
        improved_count = 0
        kernel_count = 0
        operator_counts = Counter()
        for row in rows:
            if row.digest in self.seen:
                continue
            self.seen[row.digest] = row
            new_count += 1
            operator_counts[row.operator] += 1
            parent = parent_lookup.get(row.parent_digest)
            improved = False
            if parent is not None:
                improved = (
                    int(row.metrics["identity_defect"]) < int(parent.metrics["identity_defect"])
                    or row.objective < parent.objective
                )
            if row.metrics.get("scalar_identity") and len(row.factors) > 0:
                self.kernel_hits.append(row)
                kernel_count += 1
            if improved:
                improved_count += 1
                self.recent_improvers.append(row)
            self.best_identity.append(row)
            self.best_density.append(row)
            if len(row.factors) >= self.args.long_reservoir_min_length and degeneracy_penalty(row.factors) <= self.args.max_long_degeneracy_penalty:
                self.long_non_degenerate.append(row)
            if self.keep_for_novelty(row):
                self.novelty.append(row)
        self.prune()
        return {
            "new": new_count,
            "improved": improved_count,
            "kernel": kernel_count,
            "operators": dict(operator_counts),
        }

    def keep_for_novelty(self, row: EvaluatedBraid) -> bool:
        if not self.novelty:
            return True
        signature = self.novelty_signature(row)
        existing = {self.novelty_signature(item) for item in self.novelty[-self.args.novelty_capacity :]}
        return signature not in existing or random.random() < 0.08

    def novelty_signature(self, row: EvaluatedBraid) -> tuple:
        factors = row.factors
        if not factors:
            return (row.power % 2, 0, 0, 0, 0)
        counts = Counter(factors)
        top = tuple(sorted(counts, key=lambda k: (-counts[k], k))[:3])
        return (
            row.power % 2,
            len(factors) // 8,
            int(row.metrics["identity_defect"]) // 25,
            metric_projlen(row.metrics) // 25,
            top,
        )

    def prune(self) -> None:
        self.best_identity = sorted(
            self.best_identity,
            key=lambda row: reservoir_sort_key(row, "identity"),
        )[: self.args.best_capacity]
        self.best_density = sorted(
            self.best_density,
            key=lambda row: reservoir_sort_key(row, "density"),
        )[: self.args.density_capacity]
        self.long_non_degenerate = sorted(
            self.long_non_degenerate,
            key=lambda row: reservoir_sort_key(row, "long"),
        )[: self.args.long_capacity]
        self.recent_improvers = sorted(
            self.recent_improvers,
            key=lambda row: reservoir_sort_key(row, "recent"),
        )[: self.args.recent_capacity]
        if len(self.novelty) > self.args.novelty_capacity:
            self.novelty = self.novelty[-self.args.novelty_capacity :]
        self.kernel_hits = sorted(
            self.kernel_hits,
            key=lambda row: reservoir_sort_key(row, "identity"),
        )[: self.args.top_output]

    def parent_pool(self) -> list[EvaluatedBraid]:
        pool: list[EvaluatedBraid] = []
        pool.extend(self.best_identity * self.args.best_parent_weight)
        pool.extend(self.best_density * self.args.density_parent_weight)
        pool.extend(self.long_non_degenerate * self.args.long_parent_weight)
        pool.extend(self.recent_improvers * self.args.recent_parent_weight)
        pool.extend(self.novelty * self.args.novelty_parent_weight)
        if not pool:
            pool.extend(self.seen.values())
        return pool

    def summary(self) -> dict:
        return {
            "seen": len(self.seen),
            "best_identity": [row_to_serializable(row) for row in self.best_identity[: self.args.top_output]],
            "best_density": [row_to_serializable(row) for row in self.best_density[: self.args.top_output]],
            "long_non_degenerate": [row_to_serializable(row) for row in self.long_non_degenerate[: self.args.top_output]],
            "recent_improvers": [row_to_serializable(row) for row in self.recent_improvers[: self.args.top_output]],
            "novelty_sample": [row_to_serializable(row) for row in self.novelty[-min(len(self.novelty), self.args.top_output) :]],
            "kernel_hits": [row_to_serializable(row) for row in self.kernel_hits[: self.args.top_output]],
        }

    def load_checkpoint_rows(self, checkpoint: dict) -> list[tuple[int, tuple[int, ...], str]]:
        seeds: list[tuple[int, tuple[int, ...], str]] = []
        reservoirs = checkpoint.get("reservoirs", {})
        for key in ("best_identity", "best_density", "long_non_degenerate", "recent_improvers", "novelty_sample"):
            for row in reservoirs.get(key, []):
                word = extract_word_from_row(row)
                if word is not None:
                    seeds.append((word[0], word[1], f"resume:{key}:{row.get('digest', '')}"))
        return seeds


def build_initial_braids(
    *,
    args: argparse.Namespace,
    evaluator: ExactEvaluator,
    automaton,
    rng: random.Random,
) -> tuple[list, list[str]]:
    seed_paths = [Path(path) for path in args.seed_path]
    seed_words = load_seed_words(seed_paths, limit_per_path=args.seed_path_limit)
    seed_words.extend(parse_seed_word(value) for value in args.seed_word)

    checkpoint_path = Path(args.resume_from) if args.resume_from else None
    if checkpoint_path and checkpoint_path.exists():
        checkpoint = read_json(checkpoint_path)
        temp_bank = ReservoirBank(args)
        seed_words.extend(temp_bank.load_checkpoint_rows(checkpoint))

    seen: set[tuple[int, tuple[int, ...]]] = set()
    braids = []
    sources = []
    for power, factors, source in seed_words:
        key = (int(power), tuple(int(x) for x in factors))
        if key in seen:
            continue
        seen.add(key)
        try:
            braid = evaluator.braid(power, factors)
        except Exception:
            continue
        length = len(tuple(braid.factors))
        if args.min_length <= length <= args.max_length:
            braids.append(braid)
            sources.append(source)

    while len(braids) < args.initial_random_count:
        length = rng.randint(args.initial_min_length, args.initial_max_length)
        power = rng.choice((0, 1))
        braid = evaluator.braid(power, automaton.sample_uniform(length, rng))
        key = (int(braid.power), tuple(int(x) for x in braid.factors))
        if key in seen:
            continue
        seen.add(key)
        braids.append(braid)
        sources.append("initial_random")

    return braids, sources


def write_checkpoint(
    *,
    output_dir: Path,
    metadata: dict,
    generation: int,
    counts: dict,
    bank: ReservoirBank,
    elapsed: float,
) -> None:
    checkpoint = {
        "format": "long-mutational-reservoir-checkpoint-v1",
        "metadata": metadata,
        "generation": generation,
        "counts": counts,
        "reservoirs": bank.summary(),
        "elapsed_seconds": round(elapsed, 2),
    }
    write_json(output_dir / "checkpoint.json", checkpoint)
    write_json(output_dir / "summary.json", checkpoint)


def run_search(args: argparse.Namespace) -> None:
    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    accepted_path = output_dir / "accepted.jsonl"
    kernels_path = output_dir / "kernel_hits.jsonl"

    bgpt = load_braid_gpt_module(Path(args.braid_gpt_root))
    automaton = bgpt.GNFAutomaton(args.n)
    evaluator = ExactEvaluator(
        bgpt=bgpt,
        author_repo=Path(args.author_repo),
        n=args.n,
        r=args.r,
        p=args.p,
        projlen_weight=args.projlen_weight,
        projlen_density_weight=args.projlen_density_weight,
        identity_density_weight=args.identity_density_weight,
        degeneracy_weight=args.degeneracy_weight,
        target_length=args.target_length,
        length_weight=args.length_weight,
    )
    mutator = Mutator(
        evaluator=evaluator,
        automaton=automaton,
        min_length=args.min_length,
        max_length=args.max_length,
        max_window=args.max_window,
        max_growth=args.max_growth,
        max_conjugator_length=args.max_conjugator_length,
        max_commutator_length=args.max_commutator_length,
    )
    bank = ReservoirBank(args)

    metadata = {
        "format": "long-mutational-reservoir-search-v1",
        "n": args.n,
        "r": args.r,
        "p": args.p,
        "seed": args.seed,
        "generations": args.generations,
        "mutations_per_generation": args.mutations_per_generation,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "target_length": args.target_length,
        "operator_weights": {
            "window_replace": args.window_replace_weight,
            "boundary_replace": args.boundary_replace_weight,
            "grow": args.grow_weight,
            "conjugate": args.conjugate_weight,
            "commutator_wrap": args.commutator_weight,
            "burst": args.burst_weight,
            "random_restart": args.random_restart_weight,
        },
        "objective": {
            "primary": "identity_defect",
            "secondary": "projlen, projlen/length, identity_defect/length, degeneracy, length band",
            "projlen_weight": args.projlen_weight,
            "projlen_density_weight": args.projlen_density_weight,
            "identity_density_weight": args.identity_density_weight,
            "degeneracy_weight": args.degeneracy_weight,
            "length_weight": args.length_weight,
        },
        "seed_paths": args.seed_path,
        "resume_from": args.resume_from,
    }
    write_json(output_dir / "metadata.json", metadata)

    initial_rng = random.Random(args.seed)
    initial_braids, initial_sources = build_initial_braids(
        args=args,
        evaluator=evaluator,
        automaton=automaton,
        rng=initial_rng,
    )
    initial_rows = evaluator.evaluate_braid_batch(
        initial_braids,
        parents=[""] * len(initial_braids),
        operators=["initial"] * len(initial_braids),
        generation=0,
        sources=initial_sources,
        batch_size=args.exact_batch_size,
    )
    bank.add_many(initial_rows, {})
    append_jsonl(accepted_path, [row_to_serializable(row) for row in bank.best_identity[: args.top_output]])

    counts = {
        "evaluated": len(initial_rows),
        "accepted_unique": len(bank.seen),
        "improved": 0,
        "kernel_hits": len(bank.kernel_hits),
        "duplicate_or_invalid": 0,
    }
    print(
        json.dumps(
            {
                "phase": "initial",
                "evaluated": len(initial_rows),
                "seen": len(bank.seen),
                "best_identity_defect": int(bank.best_identity[0].metrics["identity_defect"]),
                "best_projlen": metric_projlen(bank.best_identity[0].metrics),
                "elapsed_seconds": round(time.time() - start, 2),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    op_weights = {
        "window_replace": args.window_replace_weight,
        "boundary_replace": args.boundary_replace_weight,
        "grow": args.grow_weight,
        "conjugate": args.conjugate_weight,
        "commutator_wrap": args.commutator_weight,
        "burst": args.burst_weight,
        "random_restart": args.random_restart_weight,
    }

    for generation in range(1, args.generations + 1):
        rng = random.Random(args.seed + generation * 1_000_003)
        parent_pool = bank.parent_pool()
        parent_lookup = {row.digest: row for row in parent_pool}
        child_braids = []
        child_parents = []
        child_ops = []
        child_sources = []
        attempts = 0
        max_attempts = max(args.mutations_per_generation * 30, 1000)

        while len(child_braids) < args.mutations_per_generation and attempts < max_attempts:
            attempts += 1
            parent_row = rng.choice(parent_pool)
            parent_braid = evaluator.braid(parent_row.power, parent_row.factors)
            child, operator = mutator.mutate(parent_braid, rng, op_weights)
            if child is None:
                counts["duplicate_or_invalid"] += 1
                continue
            key = braid_digest(int(child.power), tuple(int(x) for x in child.factors))
            if key in bank.seen:
                counts["duplicate_or_invalid"] += 1
                continue
            child_braids.append(child)
            child_parents.append(parent_row.digest)
            child_ops.append(operator)
            child_sources.append(f"generation_{generation}:{operator}")

        rows = evaluator.evaluate_braid_batch(
            child_braids,
            parents=child_parents,
            operators=child_ops,
            generation=generation,
            sources=child_sources,
            batch_size=args.exact_batch_size,
        )
        counts["evaluated"] += len(rows)
        add_stats = bank.add_many(rows, {row.digest: row for row in bank.seen.values()})
        counts["accepted_unique"] = len(bank.seen)
        counts["improved"] += int(add_stats["improved"])
        counts["kernel_hits"] += int(add_stats["kernel"])

        if add_stats["new"]:
            interesting = sorted(
                rows,
                key=lambda row: (
                    int(row.metrics["identity_defect"]),
                    metric_projlen(row.metrics),
                    row.objective,
                ),
            )[: args.write_top_per_generation]
            append_jsonl(accepted_path, [row_to_serializable(row) for row in interesting])
        if add_stats["kernel"]:
            append_jsonl(kernels_path, [row_to_serializable(row) for row in bank.kernel_hits])

        best = bank.best_identity[0]
        progress = {
            "phase": "generation",
            "generation": generation,
            "attempts": attempts,
            "evaluated_this_generation": len(rows),
            "new_this_generation": add_stats["new"],
            "improved_this_generation": add_stats["improved"],
            "kernel_this_generation": add_stats["kernel"],
            "operator_counts": add_stats["operators"],
            "seen": len(bank.seen),
            "best_identity_defect": int(best.metrics["identity_defect"]),
            "best_projlen": metric_projlen(best.metrics),
            "best_length": len(best.factors),
            "best_objective": best.objective,
            "elapsed_seconds": round(time.time() - start, 2),
        }
        append_jsonl(progress_path, [progress])
        print(json.dumps(progress, sort_keys=True), flush=True)

        if generation % args.checkpoint_every == 0 or add_stats["kernel"]:
            write_checkpoint(
                output_dir=output_dir,
                metadata=metadata,
                generation=generation,
                counts=counts,
                bank=bank,
                elapsed=time.time() - start,
            )
        if add_stats["kernel"] and args.stop_after_kernel:
            break

    write_checkpoint(
        output_dir=output_dir,
        metadata=metadata,
        generation=min(args.generations, generation if "generation" in locals() else 0),
        counts=counts,
        bank=bank,
        elapsed=time.time() - start,
    )
    print(
        json.dumps(
            {
                "phase": "done",
                **counts,
                "best_identity_defect": int(bank.best_identity[0].metrics["identity_defect"]),
                "best_projlen": metric_projlen(bank.best_identity[0].metrics),
                "elapsed_seconds": round(time.time() - start, 2),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Long-running exact-guided mutation search over Garside normal form "
            "braids.  It maintains multiple reservoirs and mutates candidates "
            "using exact Burau/Jones scalar-identity metrics as the search signal."
        )
    )
    parser.add_argument("--braid-gpt-root", default=str(DEFAULT_BRAID_GPT_ROOT))
    parser.add_argument("--author-repo", default=str(DEFAULT_AUTHOR_REPO))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seed-path", action="append", default=[])
    parser.add_argument("--seed-path-limit", type=int, default=200)
    parser.add_argument("--seed-word", action="append", default=[])
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--mutations-per-generation", type=int, default=512)
    parser.add_argument("--initial-random-count", type=int, default=512)
    parser.add_argument("--initial-min-length", type=int, default=24)
    parser.add_argument("--initial-max-length", type=int, default=96)
    parser.add_argument("--min-length", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=220)
    parser.add_argument("--target-length", type=int, default=96)
    parser.add_argument("--max-window", type=int, default=12)
    parser.add_argument("--max-growth", type=int, default=8)
    parser.add_argument("--max-conjugator-length", type=int, default=6)
    parser.add_argument("--max-commutator-length", type=int, default=4)
    parser.add_argument("--window-replace-weight", type=float, default=45.0)
    parser.add_argument("--boundary-replace-weight", type=float, default=15.0)
    parser.add_argument("--grow-weight", type=float, default=12.0)
    parser.add_argument("--conjugate-weight", type=float, default=10.0)
    parser.add_argument("--commutator-weight", type=float, default=6.0)
    parser.add_argument("--burst-weight", type=float, default=8.0)
    parser.add_argument("--random-restart-weight", type=float, default=4.0)
    parser.add_argument("--projlen-weight", type=float, default=0.15)
    parser.add_argument("--projlen-density-weight", type=float, default=4.0)
    parser.add_argument("--identity-density-weight", type=float, default=2.0)
    parser.add_argument("--degeneracy-weight", type=float, default=0.30)
    parser.add_argument("--length-weight", type=float, default=1.0)
    parser.add_argument("--best-capacity", type=int, default=512)
    parser.add_argument("--density-capacity", type=int, default=512)
    parser.add_argument("--long-capacity", type=int, default=512)
    parser.add_argument("--recent-capacity", type=int, default=512)
    parser.add_argument("--novelty-capacity", type=int, default=512)
    parser.add_argument("--long-reservoir-min-length", type=int, default=48)
    parser.add_argument("--max-long-degeneracy-penalty", type=float, default=40.0)
    parser.add_argument("--best-parent-weight", type=int, default=5)
    parser.add_argument("--density-parent-weight", type=int, default=3)
    parser.add_argument("--long-parent-weight", type=int, default=2)
    parser.add_argument("--recent-parent-weight", type=int, default=4)
    parser.add_argument("--novelty-parent-weight", type=int, default=2)
    parser.add_argument("--exact-batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--write-top-per-generation", type=int, default=32)
    parser.add_argument("--top-output", type=int, default=100)
    parser.add_argument("--stop-after-kernel", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
