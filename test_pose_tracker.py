import unittest

from pose_tracker import (
    MetricMeasurement,
    MetricPose,
    PoseHoldTracker,
    PoseState,
)


def measurement(sequence=1, captured=10.0, x=100.0):
    return MetricMeasurement(
        sequence=sequence,
        received_monotonic_s=captured,
        pose=MetricPose(x, 200.0, 90.0),
        decision_margin=45.0,
        calibration_id="cal-1",
        calibration_rms_error_mm=2.0,
    )


class PoseTrackerTest(unittest.TestCase):
    def test_initial_missing_is_lost(self):
        estimate = PoseHoldTracker().update(10.0, None)
        self.assertEqual(estimate.state, PoseState.LOST)
        self.assertIsNone(estimate.pose)
        self.assertIsNone(estimate.measurement_age_ms)

    def test_fresh_measurement_is_measured(self):
        estimate = PoseHoldTracker().update(10.0, measurement())
        self.assertEqual(estimate.state, PoseState.MEASURED)
        self.assertTrue(estimate.fresh)
        self.assertEqual(estimate.pose, MetricPose(100.0, 200.0, 90.0))

    def test_measurement_at_fresh_age_boundary_is_measured(self):
        tracker = PoseHoldTracker(maximum_fresh_age_seconds=0.10)
        estimate = tracker.update(10.10, measurement(captured=10.0))
        self.assertEqual(estimate.state, PoseState.MEASURED)
        self.assertTrue(estimate.fresh)

    def test_stale_initial_measurement_is_lost_and_not_retained(self):
        tracker = PoseHoldTracker(
            hold_seconds=1.0,
            maximum_fresh_age_seconds=0.10,
        )
        estimate = tracker.update(10.11, measurement(captured=10.0))
        self.assertEqual(estimate.state, PoseState.LOST)
        self.assertIsNone(estimate.pose)
        self.assertFalse(estimate.fresh)

        estimate = tracker.update(10.12, None)
        self.assertEqual(estimate.state, PoseState.LOST)
        self.assertIsNone(estimate.pose)

    def test_stale_measurement_falls_back_to_last_accepted_hold(self):
        tracker = PoseHoldTracker(
            hold_seconds=0.50,
            maximum_fresh_age_seconds=0.05,
        )
        accepted = measurement(sequence=1, captured=10.0, x=100.0)
        tracker.update(10.0, accepted)

        stale = measurement(sequence=2, captured=10.1, x=999.0)
        estimate = tracker.update(10.2, stale)
        self.assertEqual(estimate.state, PoseState.HELD)
        self.assertEqual(estimate.pose, accepted.pose)
        self.assertEqual(estimate.source_sequence, accepted.sequence)
        self.assertFalse(estimate.fresh)

    def test_short_dropout_holds_exact_pose(self):
        tracker = PoseHoldTracker(hold_seconds=0.20)
        original = measurement()
        tracker.update(10.0, original)
        for offset in (0.033, 0.067, 0.100, 0.167, 0.200):
            estimate = tracker.update(10.0 + offset, None)
            self.assertEqual(estimate.state, PoseState.HELD)
            self.assertEqual(estimate.pose, original.pose)
            self.assertFalse(estimate.fresh)
            self.assertAlmostEqual(estimate.measurement_age_ms, offset * 1000.0)

    def test_expired_measurement_is_lost_and_pose_removed(self):
        tracker = PoseHoldTracker(hold_seconds=0.20)
        tracker.update(10.0, measurement())
        estimate = tracker.update(10.201, None)
        self.assertEqual(estimate.state, PoseState.LOST)
        self.assertIsNone(estimate.pose)
        self.assertGreater(estimate.measurement_age_ms, 200.0)

    def test_reacquisition_after_lost_is_measured(self):
        tracker = PoseHoldTracker(hold_seconds=0.20)
        tracker.update(10.0, measurement())
        tracker.update(10.5, None)
        updated = measurement(sequence=2, captured=10.6, x=150.0)
        estimate = tracker.update(10.6, updated)
        self.assertEqual(estimate.state, PoseState.MEASURED)
        self.assertEqual(estimate.pose.x_mm, 150.0)
        self.assertEqual(estimate.source_sequence, 2)

    def test_nonfinite_pose_is_rejected(self):
        with self.assertRaises(ValueError):
            MetricPose(float("nan"), 0.0, 0.0)

    def test_time_regression_is_rejected(self):
        tracker = PoseHoldTracker()
        tracker.update(10.0, measurement())
        with self.assertRaisesRegex(ValueError, "monotonic"):
            tracker.update(9.9, None)

    def test_measurement_time_regression_is_rejected_even_after_stale_input(self):
        tracker = PoseHoldTracker(maximum_fresh_age_seconds=0.01)
        tracker.update(10.0, measurement(sequence=1, captured=9.0))
        with self.assertRaisesRegex(ValueError, "monotonic"):
            tracker.update(10.1, measurement(sequence=2, captured=8.9))

    def test_invalid_age_configuration_is_rejected(self):
        for value in (-0.01, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "maximum_fresh_age_seconds"):
                    PoseHoldTracker(maximum_fresh_age_seconds=value)

    def test_nonfinite_tracker_time_is_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    PoseHoldTracker().update(value, None)


if __name__ == "__main__":
    unittest.main()
