from __future__ import annotations

import unittest

from annealed_reservoir_search.annealing import (
    allocate_annealed_quotas,
    allocate_core_annealed_quotas,
    boltzmann_bucket_weights,
    cooled_temperature,
)


class AnnealingTests(unittest.TestCase):
    def test_quota_uses_exact_budget_without_exceeding_counts(self):
        counts = [10, 20, 30, 40]
        quotas = allocate_annealed_quotas(
            energies=[5, 6, 7, 8],
            counts=counts,
            budget=67,
            temperature=2.0,
            minimum_per_bucket=2,
        )
        self.assertEqual(sum(quotas), 67)
        self.assertTrue(all(0 <= quota <= count for quota, count in zip(quotas, counts)))

    def test_low_temperature_concentrates_on_low_projlen(self):
        low_temperature = allocate_annealed_quotas(
            energies=[10, 11, 12],
            counts=[1_000, 1_000, 1_000],
            budget=900,
            temperature=0.5,
        )
        high_temperature = allocate_annealed_quotas(
            energies=[10, 11, 12],
            counts=[1_000, 1_000, 1_000],
            budget=900,
            temperature=20.0,
        )
        self.assertGreater(low_temperature[0], high_temperature[0])
        self.assertLess(low_temperature[2], high_temperature[2])

    def test_minimum_quota_preserves_every_available_bucket(self):
        quotas = allocate_annealed_quotas(
            energies=[1, 50, 100],
            counts=[100, 100, 3],
            budget=50,
            temperature=0.1,
            minimum_per_bucket=4,
        )
        self.assertGreaterEqual(quotas[0], 4)
        self.assertGreaterEqual(quotas[1], 4)
        self.assertEqual(quotas[2], 3)

    def test_weights_depend_only_on_projlen_gap(self):
        weights = boltzmann_bucket_weights([20, 22, 24], temperature=2.0)
        self.assertAlmostEqual(weights[0], 1.0)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    def test_temperature_cools_and_reheat_boost_is_bounded(self):
        start = cooled_temperature(6.0, 0.75, 0.97, 0)
        later = cooled_temperature(6.0, 0.75, 0.97, 50)
        reheated = cooled_temperature(6.0, 0.75, 0.97, 50, boost=3.0)
        bounded = cooled_temperature(
            6.0,
            0.75,
            0.97,
            50,
            boost=100.0,
            maximum_boost=4.0,
        )
        self.assertLess(later, start)
        self.assertGreater(reheated, later)
        self.assertLessEqual(bounded, 24.0)

    def test_core_annealing_protects_low_projlen_capacity(self):
        total, core, spillover = allocate_core_annealed_quotas(
            energies=[10, 11, 12, 13],
            counts=[15_000, 15_000, 15_000, 15_000],
            budget=30_000,
            temperature=6.0,
            core_fraction=0.8,
            minimum_per_bucket=16,
        )
        self.assertEqual(sum(total), 30_000)
        self.assertEqual(sum(core), 24_000)
        self.assertEqual(sum(spillover), 6_000)
        self.assertEqual(core[0], 15_000)
        self.assertEqual(core[1], 9_000)
        self.assertEqual(core[2:], [0, 0])
        self.assertEqual(spillover[0], 0)
        self.assertTrue(all(value > 0 for value in spillover[1:]))

    def test_zero_core_reduces_to_pure_annealed_allocation(self):
        expected = allocate_annealed_quotas(
            energies=[4, 5, 6],
            counts=[100, 100, 100],
            budget=120,
            temperature=3.0,
            minimum_per_bucket=2,
        )
        total, core, spillover = allocate_core_annealed_quotas(
            energies=[4, 5, 6],
            counts=[100, 100, 100],
            budget=120,
            temperature=3.0,
            core_fraction=0.0,
            minimum_per_bucket=2,
        )
        self.assertEqual(total, expected)
        self.assertEqual(core, [0, 0, 0])
        self.assertEqual(spillover, expected)


if __name__ == "__main__":
    unittest.main()
