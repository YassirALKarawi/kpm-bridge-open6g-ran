import unittest

import numpy as np

from kpm_bridge.calibration import QualityBudget, admit, split_conformal_radius
from kpm_bridge.shiftbench import (
    ImplementationProfile,
    apply_affine_bridge,
    fit_affine_bridge,
    latent_ran_process,
    observe_implementation,
)


class CalibrationTests(unittest.TestCase):
    def test_conformal_radius_is_order_invariant(self):
        scores = np.array([0.4, 0.1, 0.3, 0.2, 0.7, 0.5])
        self.assertEqual(split_conformal_radius(scores, 0.2), split_conformal_radius(scores[::-1], 0.2))

    def test_unresolvable_conformal_level_fails_closed(self):
        scores = np.array([0.2, 0.5, 0.7])
        self.assertTrue(np.isinf(split_conformal_radius(scores, 0.01)))

    def test_quality_gate_fails_closed(self):
        budget = QualityBudget(max_radius=1.0, max_age_ms=500, min_support=0.8)
        self.assertTrue(admit(0.5, 100, 0.9, False, budget))
        self.assertFalse(admit(1.1, 100, 0.9, False, budget))
        self.assertFalse(admit(0.5, 100, 0.9, True, budget))

    def test_shiftbench_is_deterministic(self):
        z = latent_ran_process(300, seed=7)
        profile = ImplementationProfile(
            scale=np.array([1000.0, 100.0, 100.0, 1.0, 1.0, 0.001]),
            bias=np.zeros(6),
            noise_std=np.array([80.0, 0.6, 0.4, 0.4, 0.2, 0.0004]),
            window=3,
            lag=1,
            missing_rate=0.01,
            quantisation=0.01,
        )
        y1, m1 = observe_implementation(z, profile, seed=11)
        y2, m2 = observe_implementation(z, profile, seed=11)
        np.testing.assert_equal(m1, m2)
        np.testing.assert_allclose(y1, y2, equal_nan=True)
        slopes, offsets = fit_affine_bridge(y1[:150], z[:150])
        estimate = apply_affine_bridge(y1[150:], slopes, offsets)
        self.assertEqual(estimate.shape, (150, 6))


if __name__ == "__main__":
    unittest.main()
