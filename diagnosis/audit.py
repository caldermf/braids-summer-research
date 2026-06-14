from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

from peyl.braid_data import (
    GNF,
    delta_burau_matrix,
    simple_factor_burau_table,
    simple_factor_id_maps,
)

from .crispr import sampled_crispr_ranks
from .metrics import (
    build_target_trajectory,
    estimate_baseline,
    legal_actions,
    make_child,
    root_candidate,
)
from .models import AuditConfig, Candidate, KernelCase
from .plotting import render_plots
from .policies import (
    CandidateSample,
    PeriodicBucket,
    RankCounter,
    UniformReservoir,
    paper_select_buckets,
    periodic_select,
)


class PrefixSurvivalAudit:
    def __init__(self, config: AuditConfig):
        config.validate()
        self.config = config

    def run(self, kernel: KernelCase) -> dict:
        max_depth = min(
            len(kernel.factor_ids),
            self.config.max_depth or len(kernel.factor_ids),
        )
        if self.config.bootstrap_depth > max_depth:
            raise ValueError("bootstrap_depth cannot exceed the audited depth")

        output_dir = self._create_output_dir(kernel)
        rng = random.Random(self.config.seed)
        baseline_rng = random.Random(self.config.seed + 10_000)
        simple_table = simple_factor_burau_table(p=self.config.p, n=self.config.n)
        delta_factor_id = simple_factor_id_maps(self.config.n)[0][GNF.delta_perm(self.config.n)]
        delta_target = delta_burau_matrix(p=self.config.p, n=self.config.n)
        baseline = estimate_baseline(
            self.config,
            max_depth=max_depth,
            simple_table=simple_table,
            rng=baseline_rng,
        )
        targets = build_target_trajectory(
            kernel.factor_ids,
            config=self.config,
            max_depth=max_depth,
            baseline=baseline,
            simple_table=simple_table,
        )

        final_target = targets[-1]
        if max_depth == len(kernel.factor_ids) and not final_target.kernel_match.get("matches"):
            raise ValueError(
                f"{kernel.name} does not verify as a projective kernel match "
                f"for p={self.config.p}, n={self.config.n}"
            )

        self._write_json(output_dir / "config.json", self._config_json(kernel, max_depth))
        self._write_csv(output_dir / "baseline.csv", baseline.rows())

        frontier = [root_candidate(self.config)]
        rows: list[dict] = []
        cumulative_log10: float | None = 0.0
        started = time.perf_counter()

        for depth in range(1, max_depth + 1):
            target = targets[depth - 1]
            paper_buckets: dict[int, UniformReservoir] = {}
            periodic_buckets: dict[int, PeriodicBucket] = {}
            crispr_sample = CandidateSample(
                self.config.crispr_sample_size,
                random.Random(rng.randrange(2**63)),
            )
            mcts_rank = RankCounter(target.mcts_value)
            breakout_rank = RankCounter(target.breakout_value)
            periodic_score_rank = RankCounter(target.periodic_score)
            periodic_descent_rank = RankCounter(target.descent_score)
            generated = 0
            target_seen = 0
            exhaustive_children: list[Candidate] = []

            for parent in frontier:
                actions = legal_actions(parent.factor_ids, self.config.n, delta_factor_id)
                for action in actions:
                    child = make_child(
                        parent,
                        action,
                        config=self.config,
                        max_depth=max_depth,
                        baseline=baseline,
                        simple_table=simple_table,
                        delta_target=delta_target,
                    )
                    generated += 1
                    target_seen += child.factor_ids == target.factor_ids
                    if depth < self.config.bootstrap_depth:
                        exhaustive_children.append(child)

                    paper_bucket = paper_buckets.get(child.projlen)
                    if paper_bucket is None:
                        paper_bucket = UniformReservoir(
                            self.config.bucket_size,
                            random.Random(rng.randrange(2**63)),
                        )
                        paper_buckets[child.projlen] = paper_bucket
                    paper_bucket.add(child)

                    periodic_bucket = periodic_buckets.get(child.projlen)
                    if periodic_bucket is None:
                        periodic_bucket = PeriodicBucket(
                            capacity=self.config.periodic_bucket_size,
                            elite_fraction=self.config.periodic_elite_fraction,
                            descent_fraction=self.config.periodic_descent_fraction,
                            random_keep_rate=self.config.periodic_random_keep_rate,
                            rng=random.Random(rng.randrange(2**63)),
                        )
                        periodic_buckets[child.projlen] = periodic_bucket
                    periodic_bucket.add(child)

                    crispr_sample.add(child)
                    mcts_rank.add(child.mcts_value)
                    breakout_rank.add(child.breakout_value)
                    periodic_score_rank.add(child.periodic_score)
                    periodic_descent_rank.add(child.descent_score)

            if target_seen != 1:
                raise RuntimeError(
                    f"expected target prefix exactly once at depth {depth}, saw {target_seen}"
                )

            target_bucket = paper_buckets[target.projlen]
            natural_reservoir_contains = target_bucket.contains(target.factor_ids)
            bucket_probability = min(
                1.0,
                self.config.bucket_size / target_bucket.seen,
            )

            paper_selected, selected_projlens = paper_select_buckets(
                paper_buckets,
                self.config.use_best,
            )
            global_bucket_selected = (
                depth < self.config.bootstrap_depth
                or target.projlen in selected_projlens
            )
            if depth < self.config.bootstrap_depth:
                step_probability = 1.0
            else:
                step_probability = bucket_probability if global_bucket_selected else 0.0
            natural_paper_selected = (
                depth < self.config.bootstrap_depth
                or (natural_reservoir_contains and global_bucket_selected)
            )
            if step_probability == 0.0:
                cumulative_log10 = None
            elif cumulative_log10 is not None:
                cumulative_log10 += math.log10(step_probability)

            periodic_selected = periodic_select(
                periodic_buckets,
                self.config.periodic_use_best,
            )
            periodic_target_in_bucket = periodic_buckets[target.projlen].contains(
                target.factor_ids
            )
            periodic_target_selected = any(
                candidate.factor_ids == target.factor_ids
                for candidate in periodic_selected
            )

            crispr = sampled_crispr_ranks(
                target,
                crispr_sample.items,
                self.config,
            )
            crispr["crispr_sample_population_seen"] = crispr_sample.seen
            row = {
                "kernel_name": kernel.name,
                "depth": depth,
                "target_factor_id": target.factor_ids[-1],
                "target_projlen": target.projlen,
                "target_typical_projlen": target.typical_projlen,
                "target_surprise": target.surprise,
                "target_surprise_z": target.surprise_z,
                "target_periodic_distance": target.periodic_distance,
                "target_periodic_score": target.periodic_score,
                "target_descent_score": target.descent_score,
                "target_mcts_value": target.mcts_value,
                "target_breakout_value": target.breakout_value,
                "target_kernel_match": bool(target.kernel_match.get("matches")),
                "generated_children": generated,
                "driver_frontier_size": len(frontier),
                "paper_num_buckets": len(paper_buckets),
                "paper_bucket_arrivals": target_bucket.seen,
                "paper_bucket_kept": len(target_bucket.items),
                "paper_bucket_survival_probability": bucket_probability,
                "paper_natural_reservoir_contains_target": natural_reservoir_contains,
                "paper_global_bucket_selected": global_bucket_selected,
                "paper_natural_selected": natural_paper_selected,
                "paper_step_survival_probability": step_probability,
                "paper_cumulative_log10_survival": cumulative_log10,
                "paper_selected_frontier_size": len(paper_selected),
                "periodic_bucket_kept": len(periodic_buckets[target.projlen].items),
                "periodic_target_in_bucket": periodic_target_in_bucket,
                "periodic_target_selected": periodic_target_selected,
                "periodic_selected_frontier_size": len(periodic_selected),
                "periodic_score_best_rank": periodic_score_rank.best_rank,
                "periodic_score_worst_rank": periodic_score_rank.worst_rank,
                "periodic_descent_best_rank": periodic_descent_rank.best_rank,
                "periodic_descent_worst_rank": periodic_descent_rank.worst_rank,
                "mcts_value_best_rank": mcts_rank.best_rank,
                "mcts_value_worst_rank": mcts_rank.worst_rank,
                "mcts_proxy_selected_at_beam_width": (
                    mcts_rank.best_rank <= self.config.mcts_beam_width
                ),
                "breakout_value_best_rank": breakout_rank.best_rank,
                "breakout_value_worst_rank": breakout_rank.worst_rank,
                "breakout_proxy_selected_at_beam_width": (
                    breakout_rank.best_rank <= self.config.mcts_beam_width
                ),
                **crispr,
            }

            if depth < self.config.bootstrap_depth:
                next_frontier = exhaustive_children
            else:
                next_frontier = paper_selected

            forced = (
                depth < max_depth
                and not any(
                    candidate.factor_ids == target.factor_ids
                    for candidate in next_frontier
                )
            )
            if forced:
                next_frontier.append(target)
            row["paper_forced_for_continuation"] = forced
            row["driver_next_frontier_size"] = len(next_frontier)
            rows.append(row)
            self._append_jsonl(output_dir / "depth_rows.jsonl", row)
            frontier = self._deduplicate(next_frontier)

            print(
                f"depth={depth:02d} generated={generated:,} "
                f"frontier={len(frontier):,} target_projlen={target.projlen} "
                f"bucket_seen={target_bucket.seen:,} "
                f"step_p={step_probability:.3g} forced={forced}"
            )

        self._write_csv(output_dir / "prefix_survival.csv", rows)
        summary = self._summary(kernel, rows, output_dir, time.perf_counter() - started)
        self._write_json(output_dir / "summary.json", summary)
        if self.config.render_plots:
            render_plots(rows, output_dir)
        return summary

    def _create_output_dir(self, kernel: KernelCase) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / (
            f"{kernel.name}_p{self.config.p}_seed{self.config.seed}_{stamp}"
        )
        suffix = 1
        candidate = output_dir
        while candidate.exists():
            suffix += 1
            candidate = output_dir.with_name(f"{output_dir.name}_{suffix}")
        candidate.mkdir(parents=True)
        return candidate

    def _config_json(self, kernel: KernelCase, max_depth: int) -> dict:
        payload = asdict(self.config)
        payload["output_dir"] = str(payload["output_dir"])
        payload["kernel"] = {
            "name": kernel.name,
            "source": kernel.source,
            "factor_ids": list(kernel.factor_ids),
        }
        payload["audited_depth"] = max_depth
        payload["policy_notes"] = {
            "paper": "Exact bucket survival probability and whole-bucket global selection.",
            "periodic": "Realized selection on the same generated candidate stream.",
            "mcts": "Score rank proxy only; not a claim about UCT tree-selection probability.",
            "crispr": "Rank percentile in a uniform candidate sample, not a full generation.",
        }
        return payload

    @staticmethod
    def _deduplicate(candidates: list[Candidate]) -> list[Candidate]:
        unique = {}
        for candidate in candidates:
            unique.setdefault(candidate.factor_ids, candidate)
        return list(unique.values())

    @staticmethod
    def _append_jsonl(path: Path, row: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _first_depth(rows: list[dict], key: str, expected=False):
        return next(
            (row["depth"] for row in rows if bool(row[key]) is expected),
            None,
        )

    def _summary(
        self,
        kernel: KernelCase,
        rows: list[dict],
        output_dir: Path,
        elapsed: float,
    ) -> dict:
        final_log10 = rows[-1]["paper_cumulative_log10_survival"]
        return {
            "kernel_name": kernel.name,
            "output_dir": str(output_dir),
            "depths_audited": len(rows),
            "elapsed_sec": round(elapsed, 4),
            "paper_cumulative_log10_survival": final_log10,
            "paper_cumulative_survival_probability": (
                10.0**final_log10
                if final_log10 is not None and final_log10 > -308
                else 0.0
            ),
            "paper_first_global_rejection_depth": self._first_depth(
                rows,
                "paper_global_bucket_selected",
                expected=False,
            ),
            "paper_first_realized_rejection_depth": self._first_depth(
                rows,
                "paper_natural_selected",
                expected=False,
            ),
            "paper_first_probability_below_one_depth": next(
                (
                    row["depth"]
                    for row in rows
                    if row["paper_step_survival_probability"] < 1.0
                ),
                None,
            ),
            "paper_first_forced_depth": self._first_depth(
                rows,
                "paper_forced_for_continuation",
                expected=True,
            ),
            "periodic_first_rejection_depth": self._first_depth(
                rows,
                "periodic_target_selected",
                expected=False,
            ),
            "mcts_proxy_first_rejection_depth": self._first_depth(
                rows,
                "mcts_proxy_selected_at_beam_width",
                expected=False,
            ),
            "breakout_proxy_first_rejection_depth": self._first_depth(
                rows,
                "breakout_proxy_selected_at_beam_width",
                expected=False,
            ),
            "crispr_endpoint_proxy_first_rejection_depth": self._first_depth(
                rows,
                "crispr_endpoint_proxy_selected",
                expected=False,
            ),
            "crispr_envelope_proxy_first_rejection_depth": self._first_depth(
                rows,
                "crispr_envelope_proxy_selected",
                expected=False,
            ),
            "crispr_collapse_proxy_first_rejection_depth": self._first_depth(
                rows,
                "crispr_collapse_proxy_selected",
                expected=False,
            ),
            "crispr_suffix_proxy_first_rejection_depth": self._first_depth(
                rows,
                "crispr_suffix_proxy_selected",
                expected=False,
            ),
            "final_target_projlen": rows[-1]["target_projlen"],
            "final_target_kernel_match": rows[-1]["target_kernel_match"],
        }
