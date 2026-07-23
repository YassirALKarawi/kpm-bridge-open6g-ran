import unittest

import numpy as np

from kpm_bridge.dataset import (
    CanonicalTrace,
    attach_qos_risk_labels,
    robust_feature_stats,
)
from kpm_bridge.splits import (
    PREDICTION_HORIZON,
    calibration_mask,
    fit_stop,
    mapper_fit_mask,
    xapp_fit_mask,
)


class SplitProtocolTests(unittest.TestCase):
    def test_all_fitted_rows_are_disjoint_from_calibration(self):
        length = 101
        calibration = calibration_mask(length)
        anchors = mapper_fit_mask(length, 0.10)
        xapp = xapp_fit_mask(length, PREDICTION_HORIZON)
        self.assertFalse(np.any(calibration & anchors))
        self.assertFalse(np.any(calibration & xapp))
        self.assertLess(np.flatnonzero(xapp)[-1], fit_stop(length) - PREDICTION_HORIZON)

    def test_feature_statistics_ignore_calibration_suffix(self):
        rng = np.random.default_rng(7)
        values = rng.normal(size=(100, 8))
        changed = values.copy()
        changed[fit_stop(len(values)) :] += 1_000_000.0

        def trace(block):
            return CanonicalTrace(
                trace_id="trace",
                scheduler="sched",
                training_config="tr0",
                experiment="exp1",
                base_station="bs1",
                user_equipment="ue2",
                traffic_class="URLLC",
                time_ms=np.arange(len(block), dtype=float) * 250.0,
                values=block,
                risk=np.zeros(len(block), dtype=int),
            )

        original = robust_feature_stats([trace(values)])
        modified = robust_feature_stats([trace(changed)])
        np.testing.assert_allclose(modified.location, original.location)
        np.testing.assert_allclose(modified.scale, original.scale)

        # Task thresholds are likewise frozen before the calibration suffix,
        # including the complete four-sample future-label horizon.
        classes = (("eMBB", "ue3"), ("MTC", "ue4"), ("URLLC", "ue2"))
        original_traces = []
        changed_traces = []
        for index, (traffic, ue) in enumerate(classes):
            block = rng.normal(size=(100, 8)) + index
            changed_block = block.copy()
            changed_block[fit_stop(100 - PREDICTION_HORIZON) :] += 1_000_000.0
            base = trace(block)
            original_traces.append(
                CanonicalTrace(
                    **{
                        **base.__dict__,
                        "trace_id": f"trace-{traffic}",
                        "user_equipment": ue,
                        "traffic_class": traffic,
                    }
                )
            )
            changed_traces.append(
                CanonicalTrace(
                    **{
                        **base.__dict__,
                        "trace_id": f"trace-{traffic}",
                        "user_equipment": ue,
                        "traffic_class": traffic,
                        "values": changed_block,
                    }
                )
            )
        _, original_thresholds = attach_qos_risk_labels(original_traces)
        _, changed_thresholds = attach_qos_risk_labels(changed_traces)
        self.assertEqual(changed_thresholds, original_thresholds)


if __name__ == "__main__":
    unittest.main()
