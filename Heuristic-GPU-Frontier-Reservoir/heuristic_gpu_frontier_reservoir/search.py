from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from braidzero.core import BraidEnvironment, parse_int_list, sha256_file
from braidzero.frontier import iter_frontier_cache
from braidzero.frontier_bucket_reservoir import _best_target_metrics
from braidzero.search import SearchState, parse_completion_targets
from last_factor_confusion.data import collate_prefixes
from last_factor_confusion.factors import FactorTable
from last_factor_confusion.model_v3 import LastFactorTransformerV3, ModelV3Config


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def atomic_checkpoint(path: Path, payload: dict):
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


@dataclass
class Bucket:
    key: int
    capacity: int
    rng: random.Random
    states: list[SearchState] = field(default_factory=list)
    seen: int = 0

    def add(self, state):
        self.seen += 1
        if len(self.states) < self.capacity:
            self.states.append(state)
        else:
            replacement = self.rng.randint(1, self.seen)
            if replacement <= self.capacity:
                self.states[replacement - 1] = state


class Population:
    def __init__(self, heuristic, length, bucket_size, bin_width, max_score, rng):
        self.heuristic, self.length, self.bucket_size = heuristic, length, bucket_size
        self.bin_width, self.max_score, self.rng = bin_width, max_score, rng
        self.buckets = {}

    def key(self, state):
        if self.heuristic == "projlen":
            return int(state.metrics["projlen"])
        score = min(self.max_score, max(0.0, float(state.score)))
        return int(math.floor(score / self.bin_width))

    def add(self, state):
        key = self.key(state)
        self.buckets.setdefault(key, Bucket(key, self.bucket_size, self.rng)).add(state)

    def size(self): return sum(len(bucket.states) for bucket in self.buckets.values())

    def select(self, limit):
        reverse = self.heuristic == "confusion"
        selected, rows = [], []
        for key in sorted(self.buckets, reverse=reverse):
            bucket = self.buckets[key]
            remaining = limit - len(selected)
            if remaining <= 0: break
            count = min(remaining, len(bucket.states))
            chosen = bucket.states if count == len(bucket.states) else self.rng.sample(bucket.states, count)
            selected.extend(chosen)
            rows.append({"key": key, "seen": bucket.seen, "states": len(bucket.states), "selected": count})
        return selected, rows

    def summary(self):
        reverse = self.heuristic == "confusion"
        return {"states": self.size(), "buckets": len(self.buckets),
                "best": [{"key": key, "states": len(self.buckets[key].states), "seen": self.buckets[key].seen}
                         for key in sorted(self.buckets, reverse=reverse)[:8]]}


def print_banner(args, env, config):
    print("=" * 60)
    print("SEARCHING FOR KERNEL ELEMENTS")
    print("=" * 60)
    print(f"Braid group: B_{args.n}")
    print(f"Representation: ({args.n-args.r}, {args.r})")
    print(f"Dimension: {env.dim}")
    print(f"Number of simples: {math.factorial(args.n)}")
    print(f"Prime: {args.p}")
    print(f"Device: {args.device}")
    print(f"Heuristic: {args.heuristic}")
    print(f"Bucket size: {args.bucket_size}")
    print(f"Target length: {args.target_length}")
    print(f"BFS frontier length: {args.frontier_length}")
    print(f"Frontier shard: {args.frontier_shard_index + 1}/{args.frontier_shard_count}")
    print(f"Use best: {args.use_best}")
    print(f"Inference batch size: {args.inference_batch_size}")
    if args.heuristic == "confusion":
        print(f"Confusion bin width: {args.confusion_bin_width}")
        print(f"Maximum bucketed confusion: {args.max_confusion}")
    print(f"Seed: {args.seed}")
    print()
    print(f"Loaded transformer from {args.checkpoint}")
    print(f"  Checkpoint checksum: {config['checkpoint_checksum']}")
    print(f"  Calibration checksum: {config['calibration_checksum']}")
    print("  Model input: exact projectivized polynomial matrix")
    print("  Target: true final proper Garside factor")
    print("Storage: exact matrices=CPU NumPy, transformer batches=CUDA BF16")
    print("GPU-accelerated confusion scoring with exact peyl braid arithmetic", flush=True)


def print_population(population):
    label = "Confusion" if population.heuristic == "confusion" else "Projlen"
    print(f"  {label} bucket distribution:")
    reverse = population.heuristic == "confusion"
    for key in sorted(population.buckets, reverse=reverse):
        bucket = population.buckets[key]
        if population.heuristic == "confusion":
            low = key * population.bin_width
            high = min(population.max_score, (key + 1) * population.bin_width)
            print(f"    confusion=[{low:.2f},{high:.2f}): {bucket.seen} seen, {len(bucket.states)} kept")
        else:
            print(f"    projlen={key}: {bucket.seen} seen, {len(bucket.states)} kept")
    print(f"  Braids kept: {population.size()} (in {len(population.buckets)} buckets)", flush=True)


class ConfusionScorer:
    def __init__(self, checkpoint, calibration, env, device):
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved.get("architecture") != LastFactorTransformerV3.architecture:
            raise ValueError("checkpoint is not exact-degree v3")
        if int(saved["model_config"]["p"]) != env.p:
            raise ValueError("model prime does not match search prime")
        self.model = LastFactorTransformerV3(ModelV3Config(**saved["model_config"])).to(device).eval()
        self.model.load_state_dict(saved["state_dict"])
        self.temperature = float(json.loads(Path(calibration).read_text())["temperature"])
        self.env, self.device = env, device
        table = FactorTable.from_peyl(env.n)
        self.factor_class = {}
        for factor_id, permutation in enumerate(env.nf_table.divs):
            try:
                self.factor_class[factor_id] = table.class_id(permutation)
            except ValueError:
                # Identity and Delta are not among the 22 proper-factor targets.
                pass
        if len(self.factor_class) != 22:
            raise RuntimeError(f"expected 22 proper factor classes, found {len(self.factor_class)}")

    def matrix(self, image):
        normalized = self.env.polymat.projectivise(np.asarray(image)) % self.env.p
        return np.moveaxis(normalized, -1, 0).tolist()

    @torch.no_grad()
    def score(self, states):
        records = [{"matrix": self.matrix(state.exact), "target_class": self.factor_class[state.factors[-1]],
                    "target_descents": [0] * 6, "trajectory_id": "search", "status": "clean"} for state in states]
        x, mask, degrees, targets, _, _ = collate_prefixes(records, sparse=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits, _ = self.model(x.to(self.device), mask.to(self.device), degrees.to(self.device))
        return F.cross_entropy(logits.float() / self.temperature, targets.to(self.device), reduction="none").cpu().tolist()


def run(args):
    started = time.time(); rng = random.Random(args.seed); output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    atomic_json(status_path, {"status": "truncated", "reason": "search has not completed"})
    env = BraidEnvironment(author_repo=Path(args.author_repo), n=args.n, r=args.r, p=args.p,
                           t_values=parse_int_list(args.t_values, default=tuple(range(1, args.p))))
    targets = parse_completion_targets(args.completion_targets)
    device = torch.device(args.device)
    scorer = ConfusionScorer(Path(args.checkpoint), Path(args.calibration), env, device) if args.heuristic == "confusion" else None
    population = Population(args.heuristic, args.frontier_length, args.bucket_size,
                            args.confusion_bin_width, args.max_confusion, rng)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"checkpoint_checksum": sha256_file(Path(args.checkpoint)),
                                               "calibration_checksum": sha256_file(Path(args.calibration))})
    atomic_json(output / "config.json", config)
    print_banner(args, env, config)
    exact_evaluations = expanded = frontier_loaded = kernel_candidates = 0
    best_projlen = None; buffer = []
    checkpoint_path = output / "checkpoint.pt"
    candidate_path = output / "kernel_candidates.jsonl"
    candidate_seen = set()
    if candidate_path.exists():
        for line in candidate_path.open():
            if line.strip(): candidate_seen.add(tuple(json.loads(line)["factors"]))

    def flush(states, destination):
        nonlocal exact_evaluations, best_projlen
        if not states: return
        scores = scorer.score(states) if scorer else [-float(state.metrics["projlen"]) for state in states]
        for state, score in zip(states, scores):
            state.score = float(score); destination.add(state)
            pl = int(state.metrics["projlen"])
            best_projlen = pl if best_projlen is None else min(best_projlen, pl)
        exact_evaluations += len(states); states.clear()

    completed_length = args.frontier_length
    if checkpoint_path.exists() and not args.no_resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["config_fingerprint"] != config:
            raise ValueError("checkpoint configuration differs from this invocation")
        population = saved["population"]
        completed_length = int(saved["completed_length"])
        exact_evaluations = int(saved["exact_evaluations"]); expanded = int(saved["expanded"])
        frontier_loaded = int(saved["frontier_loaded"]); kernel_candidates = int(saved["kernel_candidates"])
        best_projlen = saved["best_projlen"]; rng.setstate(saved["rng_state"])
        population.rng = rng
        for bucket in population.buckets.values(): bucket.rng = rng
        print()
        print("=" * 60); print("RESUMING SEARCH"); print("=" * 60)
        print(f"Completed length: {completed_length}")
        print_population(population)
    else:
        frontier_start = time.time()
        print(); print("=" * 60); print(f"Level {args.frontier_length} - BFS FRONTIER"); print("=" * 60)
        print(f"  Loading exhaustive frontier cache {args.frontier_path}...")
        for record in iter_frontier_cache(env=env, path=Path(args.frontier_path),
                shard_count=args.frontier_shard_count, shard_index=args.frontier_shard_index,
                shard_by=args.frontier_shard_by, max_records=args.frontier_max_records):
            exact = env.exact_evaluate(record.factors)
            metrics, _ = _best_target_metrics(env, exact, targets)
            buffer.append(SearchState(record.factors, None, None, exact, metrics, 0.0)); frontier_loaded += 1
            if len(buffer) >= args.inference_batch_size: flush(buffer, population)
        flush(buffer, population)
        atomic_json(output / "frontier_summary.json", {"loaded": frontier_loaded, **population.summary()})
        print(f"  Assigned to this shard: {frontier_loaded:,} braids")
        print_population(population)
        print(f"  Timing: total={time.time() - frontier_start:.2f}s", flush=True)
        atomic_checkpoint(checkpoint_path, {"config_fingerprint": config, "completed_length": completed_length,
            "population": population, "exact_evaluations": exact_evaluations, "expanded": expanded,
            "frontier_loaded": frontier_loaded, "kernel_candidates": kernel_candidates,
            "best_projlen": best_projlen, "rng_state": rng.getstate()})
    progress = output / "progress.jsonl"
    for length in range(completed_length + 1, args.target_length + 1):
        level_start = time.time()
        parents, selected_buckets = population.select(args.use_best)
        candidate_count = sum(len(env.legal_next(parent.factors)) for parent in parents)
        print("=" * 60); print(f"Level {length} - SAMPLING"); print("=" * 60)
        print(f"  Starting braids: {len(parents):,}")
        print(f"  Candidates to generate: {candidate_count:,}")
        chunks = math.ceil(candidate_count / args.inference_batch_size)
        if chunks > 1: print(f"  Processing transformer inference in {chunks:,} batches...", flush=True)
        nxt = Population(args.heuristic, length, args.bucket_size, args.confusion_bin_width, args.max_confusion, rng)
        depth_expanded = 0
        for parent in parents:
            for action in env.legal_next(parent.factors):
                factors = parent.factors + (int(action),)
                exact = env.exact_append(parent.exact, int(action))
                metrics, by_target = _best_target_metrics(env, exact, targets)
                if (any(item.get("scalar_identity") or item.get("target_match") for item in by_target.values())
                        and factors not in candidate_seen):
                    with candidate_path.open("a") as handle:
                        handle.write(json.dumps({"length": length, "factors": list(factors), "metrics": metrics}) + "\n")
                    candidate_seen.add(factors)
                    kernel_candidates += 1
                buffer.append(SearchState(factors, None, None, exact, metrics, 0.0))
                expanded += 1; depth_expanded += 1
                if len(buffer) >= args.inference_batch_size: flush(buffer, nxt)
        flush(buffer, nxt); population = nxt
        row = {"length": length, "heuristic": args.heuristic, "selected_parents": len(parents),
               "selected_buckets": selected_buckets, "expanded": depth_expanded,
               "population": population.summary(), "best_projlen": best_projlen,
               "kernel_candidates": kernel_candidates, "elapsed_seconds": time.time() - started}
        with progress.open("a") as handle: handle.write(json.dumps(row) + "\n")
        print_population(population)
        print(f"  Best projlen observed: {best_projlen}")
        print(f"  Kernel/target candidates: {kernel_candidates}")
        print(f"  Timing: total={time.time() - level_start:.2f}s", flush=True)
        atomic_checkpoint(checkpoint_path, {"config_fingerprint": config, "completed_length": length,
            "population": population, "exact_evaluations": exact_evaluations, "expanded": expanded,
            "frontier_loaded": frontier_loaded, "kernel_candidates": kernel_candidates,
            "best_projlen": best_projlen, "rng_state": rng.getstate()})
        if args.stop_after_candidate and kernel_candidates: break
    summary = {"schema_version": 1, "status": "clean", "method": "heuristic_gpu_frontier_reservoir",
               "heuristic": args.heuristic, "prime": args.p, "seed": args.seed,
               "frontier_length": args.frontier_length, "target_length": args.target_length,
               "frontier_loaded": frontier_loaded, "expanded_states": expanded,
               "exact_evaluations": exact_evaluations, "best_projlen": best_projlen,
               "kernel_candidates": kernel_candidates, "elapsed_seconds": time.time() - started,
               "artifact_path": str(output.resolve()), "verifier_version": env.verifier_version}
    atomic_json(output / "summary.json", summary); atomic_json(status_path, summary)
    print(); print("=" * 60); print("SEARCH COMPLETE"); print("=" * 60)
    print(f"Final level: {min(args.target_length, length if 'length' in locals() else completed_length)}")
    print(f"Total time: {time.time() - started:.2f}s")
    print(f"Best projlen observed: {best_projlen}")
    print(f"Kernel/target candidates: {kernel_candidates}")
    print(f"Summary: {output / 'summary.json'}", flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--heuristic", choices=("confusion", "projlen"), required=True)
    p.add_argument("--author-repo", required=True); p.add_argument("--frontier-path", required=True)
    p.add_argument("--checkpoint", required=True); p.add_argument("--calibration", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--n", type=int, default=4)
    p.add_argument("--r", type=int, default=1); p.add_argument("--p", type=int, default=5)
    p.add_argument("--t-values", default=""); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--frontier-length", type=int, default=8); p.add_argument("--target-length", type=int, default=66)
    p.add_argument("--frontier-shard-count", type=int, default=16); p.add_argument("--frontier-shard-index", type=int, default=0)
    p.add_argument("--frontier-shard-by", choices=("record", "key", "none"), default="record")
    p.add_argument("--frontier-max-records", type=int, default=0); p.add_argument("--bucket-size", type=int, default=3000)
    p.add_argument("--use-best", type=int, default=50000); p.add_argument("--inference-batch-size", type=int, default=256)
    p.add_argument("--confusion-bin-width", type=float, default=.25); p.add_argument("--max-confusion", type=float, default=20.)
    p.add_argument("--completion-targets", default="identity,delta"); p.add_argument("--device", default="cuda")
    p.add_argument("--stop-after-candidate", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    return p


def main(): run(parser().parse_args())
if __name__ == "__main__": main()
