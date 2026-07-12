import unittest

import numpy as np

from kpm_bridge.certificates import (
    TelemetryCertificate,
    calibrate_anchor_mean_shift,
    detect_anchor_drift,
    detect_anchor_mean_shift,
)


class CertificateTests(unittest.TestCase):
    def test_certificate_is_fixed_48_bytes(self):
        certificate = TelemetryCertificate(b"a" * 16, 7, 9, 0.9, 125.0, 0.4, 0, 8)
        self.assertEqual(len(certificate.encode()), 48)

    def test_drift_detector_fails_closed_after_two_anchors(self):
        scores = np.full(60, 0.1)
        scores[20] = 2.0
        scores[40] = 2.1
        drift, detected = detect_anchor_drift(scores, threshold=1.0, anchor_stride=20, consecutive=2)
        self.assertEqual(detected, 40)
        self.assertFalse(drift[39])
        self.assertTrue(drift[40])

    def test_multivariate_detector_invalidates_persistent_shift(self):
        rng = np.random.default_rng(4)
        calibration_blocks = [rng.normal(0, 0.1, (200, 3)) for _ in range(20)]
        calibration = calibrate_anchor_mean_shift(
            calibration_blocks, window=3, quantile=0.95, anchor_stride=5
        )
        residuals = rng.normal(0, 0.1, (200, 3))
        residuals[100:] += 2.0
        drift, detected = detect_anchor_mean_shift(residuals, calibration)
        self.assertIsNotNone(detected)
        self.assertGreaterEqual(detected, 100)
        self.assertTrue(drift[detected])


if __name__ == "__main__":
    unittest.main()
