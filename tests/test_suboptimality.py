"""Tests for rlplot.suboptimality. Run with `python -m unittest discover tests`."""

import unittest

import numpy as np

from rlplot import metrics
from rlplot import suboptimality as so


class SuboptimalityTest(unittest.TestCase):

    def test_experience_optimal_return(self):
        returns = np.arange(100)
        # Top 5% of 100 episodes: 95..99.
        self.assertEqual(so.experience_optimal_return(returns), 97)
        self.assertEqual(so.experience_optimal_return(returns, top_fraction=0.1), 94.5)
        # At least one episode is always used.
        self.assertEqual(so.experience_optimal_return([3, 1, 2]), 3)
        with self.assertRaises(ValueError):
            so.experience_optimal_return([])
        with self.assertRaises(ValueError):
            so.experience_optimal_return(returns, top_fraction=0)

    def test_experience_normalized_scores(self):
        scores = np.array([[5., 10.], [0., 2.]])
        optimal = np.array([[10., 10.], [4., 2.]])
        normalized = so.experience_normalized_scores(scores, optimal, 0.)
        np.testing.assert_allclose(normalized, [[0.5, 1.], [0., 1.]])
        # Practical sub-optimality gap with the existing optimality gap metric.
        self.assertAlmostEqual(metrics.aggregate_optimality_gap(normalized), 0.375)
        # No return above the minimum: no measurable gap.
        np.testing.assert_allclose(so.experience_normalized_scores([[1.]], [[1.]], [[1.]]), [[1.]])

    def test_topk_variance(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(size=5000)
        v_k, var_closed = so.topk_mean_and_variance(returns, 250)
        self.assertAlmostEqual(v_k, so.experience_optimal_return(returns))
        _, var_boot = so.bootstrap_topk_variance(returns, 250, n_bootstrap=500, rng=rng)
        # Closed form and bootstrap agree to within a factor of two.
        self.assertLess(abs(np.log(var_closed / var_boot)), np.log(2))

    def test_compute_optimal_k(self):
        rng = np.random.default_rng(0)
        returns = 10 + rng.normal(size=2000)
        result = so.compute_optimal_k(returns, epsilon=0.01, method="closed_form")
        self.assertTrue(result["found"])
        self.assertEqual(result["k_star"], min(result["table"]))
        self.assertAlmostEqual(result["alpha_star"], result["k_star"] / 2000)
        # An unreachable precision falls back to the largest k.
        result = so.compute_optimal_k(returns, epsilon=0., method="closed_form")
        self.assertFalse(result["found"])
        self.assertEqual(result["k_star"], max(result["table"]))

    def test_tracker(self):
        tracker = so.ExperienceGapTracker(buffer_size=20, top_fraction=0.1)
        self.assertEqual(tracker.stats(), {})
        for r in range(100):
            tracker.add(r)
        stats = tracker.stats()
        self.assertEqual(stats["best_trajectory_return"], 99)
        # Global: top 2 over all episodes; recent: top 2 of the last 20 (80..99).
        self.assertEqual(stats["avg_top_returns_global"], 98.5)
        self.assertEqual(stats["avg_top_returns_local"], 98.5)
        self.assertEqual(stats["local_optimality_gap"], 98.5 - 89.5)
        # Once returns drop, the recent estimate follows but the global one does not.
        for _ in range(20):
            tracker.add(0)
        stats = tracker.stats()
        self.assertEqual(stats["avg_top_returns_global"], 98.5)
        self.assertEqual(stats["avg_top_returns_local"], 0)
        self.assertEqual(stats["global_optimality_gap"], 98.5)


if __name__ == "__main__":
    unittest.main()
