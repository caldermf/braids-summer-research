from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from .bootstrap import ensure_author_peyl
from .datta import analyze_factor_ids

ensure_author_peyl()

from peyl import polymat  # type: ignore  # noqa: E402
from peyl.braidsearch import Tracker  # type: ignore  # noqa: E402


class DattaNormalTracker(Tracker):
    """The paper's tracker with a graded Datta-defect bucket axis."""

    def __init__(self, rep, bucket_size: int, rand: random.Random):
        super().__init__(
            rep=rep,
            bucket_size=bucket_size,
            bucket_keys=("length", "projlen", "datta_defect_count"),
            criterion=lambda frame: frame["length"] >= 1,
            rand=rand,
        )

    @staticmethod
    def datta_defect_count(braid) -> int:
        if braid.power != 0:
            raise ValueError("Datta reservoir expects Delta-power-zero positive parts")
        return len(
            analyze_factor_ids(tuple(int(value) for value in braid.factors)).defects
        )

    def add_braids_images(self, braids, images):
        stats = self.braid_image_stats(braids, images)
        keep = self.criterion(stats)
        indices = [index for index in range(len(keep)) if keep[index]]
        lengths = [braid.garside_length() for braid in braids]
        projlens = polymat.projlen(images)

        for index in indices:
            braid, image = braids[index], images[index]
            defect_count = self.datta_defect_count(braid)
            bucket = (int(lengths[index]), int(projlens[index]), int(defect_count))

            if bucket not in self.buckets:
                self.buckets.add(bucket)
                self.bucket_braids[bucket] = [braid]
                self.bucket_braid_set[bucket] = {braid}
                self.bucket_reservoir_counts[bucket] = 1
                image = polymat.projectivise(image)
                self.bucket_images[bucket] = np.zeros(
                    shape=(self.bucket_size, *image.shape), dtype=image.dtype
                )
                self.bucket_images[bucket][0] = image
                continue

            self.bucket_reservoir_counts[bucket] += 1
            if len(self.bucket_braids[bucket]) == self.bucket_size:
                replacement = self.rand.randint(1, self.bucket_reservoir_counts[bucket])
                if replacement <= self.bucket_size:
                    self.bucket_braids[bucket][replacement - 1] = braid
                    self.bucket_images[bucket][replacement - 1] = polymat.projectivise(image)
                continue

            if braid in self.bucket_braid_set[bucket]:
                continue
            storage_index = len(self.bucket_braids[bucket])
            self.bucket_braids[bucket].append(braid)
            self.bucket_braid_set[bucket].add(braid)
            self.bucket_images[bucket][storage_index] = polymat.projectivise(image)

    def stats(self):
        frame = pd.DataFrame(
            columns=[
                "bucket",
                "count",
                "length",
                "projlen",
                "datta_defect_count",
                "reservoir_count",
            ],
            data=[
                (
                    bucket,
                    len(self.bucket_braids[bucket]),
                    *bucket,
                    self.bucket_reservoir_counts[bucket],
                )
                for bucket in self.buckets
            ],
        )
        return frame


def select_stratified_buckets(stats, use_best: int, structural_fraction: float):
    """Split expansion budget between low projlen and high defect severity."""
    if not 0.0 <= structural_fraction <= 1.0:
        raise ValueError("structural_fraction must lie in [0, 1]")
    structural_budget = round(use_best * structural_fraction)
    projlen_budget = use_best - structural_budget
    selected: list[tuple] = []

    lanes = (
        (stats.sort_values(["projlen", "datta_defect_count"], ascending=[True, False]), projlen_budget),
        (stats.sort_values(["datta_defect_count", "projlen"], ascending=[False, True]), structural_budget),
    )
    for lane, budget in lanes:
        if budget <= 0 or lane.empty:
            continue
        used_lane = 0
        for row in lane.itertuples(index=False):
            bucket = tuple(row.bucket)
            if bucket in selected or used_lane + int(row.count) > budget:
                continue
            selected.append(bucket)
            used_lane += int(row.count)

    selected_set = set(selected)
    remaining = stats[~stats["bucket"].isin(selected_set)].sort_values(
        "projlen", ignore_index=True
    )
    used = sum(int(stats.loc[stats["bucket"] == bucket, "count"].iloc[0]) for bucket in selected)
    for row in remaining.itertuples(index=False):
        if used + int(row.count) > use_best:
            continue
        selected.append(tuple(row.bucket))
        used += int(row.count)
    return selected
