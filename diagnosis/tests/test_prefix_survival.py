from __future__ import annotations

import csv
import random
import tempfile
import unittest
from pathlib import Path

from diagnosis.audit import PrefixSurvivalAudit
from diagnosis.known_examples import EMBEDDED_CASES
from diagnosis.models import AuditConfig
from diagnosis.policies import UniformReservoir, paper_select_buckets


class DummyCandidate:
    def __init__(self, factor_ids):
        self.factor_ids = factor_ids


class PolicyTests(unittest.TestCase):
    def test_uniform_reservoir_tracks_seen_and_capacity(self):
        reservoir = UniformReservoir(3, random.Random(1))
        for value in range(20):
            reservoir.add(DummyCandidate((value,)))
        self.assertEqual(reservoir.seen, 20)
        self.assertEqual(len(reservoir.items), 3)

    def test_paper_selection_does_not_take_partial_bucket(self):
        buckets = {}
        for projlen, count in ((2, 2), (3, 4), (4, 1)):
            reservoir = UniformReservoir(10, random.Random(projlen))
            for index in range(count):
                reservoir.add(DummyCandidate((projlen, index)))
            buckets[projlen] = reservoir
        selected, projlens = paper_select_buckets(buckets, use_best=5)
        self.assertEqual(len(selected), 2)
        self.assertEqual(projlens, {2})


class AuditSmokeTest(unittest.TestCase):
    def test_small_audit_writes_one_row_per_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AuditConfig(
                max_depth=4,
                bootstrap_depth=2,
                bucket_size=20,
                use_best=100,
                baseline_samples=8,
                periodic_bucket_size=10,
                periodic_use_best=100,
                crispr_sample_size=16,
                output_dir=Path(tmp),
                render_plots=False,
            )
            summary = PrefixSurvivalAudit(config).run(EMBEDDED_CASES["p5_length54"])
            output_dir = Path(summary["output_dir"])
            with (output_dir / "prefix_survival.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "baseline.csv").exists())


if __name__ == "__main__":
    unittest.main()
