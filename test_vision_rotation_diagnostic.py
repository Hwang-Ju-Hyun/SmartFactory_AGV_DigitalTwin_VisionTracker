import json
import math
import unittest

import numpy as np

from vision_rotation_diagnostic import (
    RotationSample,
    diagnose_stationary_rotation,
    measured_rotation_samples,
)


def sample(
    heading_deg,
    *,
    fixed_center=(500.0, 200.0),
    true_offset=(80.0, -15.0),
    applied_offset=(60.0, 0.0),
    physical_shift=(0.0, 0.0),
):
    angle = math.radians(heading_deg)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    actual_center = np.asarray(fixed_center) + np.asarray(physical_shift)
    raw_tag = actual_center - rotation @ np.asarray(true_offset)
    reported_body = raw_tag + rotation @ np.asarray(applied_offset)
    return RotationSample(
        calibration_id="cal-id",
        raw_tag_x_mm=float(raw_tag[0]),
        raw_tag_z_mm=float(raw_tag[1]),
        body_heading_deg=float(heading_deg),
        applied_forward_mm=float(applied_offset[0]),
        applied_left_mm=float(applied_offset[1]),
        body_x_mm=float(reported_body[0]),
        body_z_mm=float(reported_body[1]),
    )


def record_line(state, rotation_sample=None):
    record = {"state": state, "calibration_id": "cal-id"}
    if rotation_sample is not None:
        record["transform_diagnostic"] = {
            "raw_tag_center_mm": {
                "x_mm": rotation_sample.raw_tag_x_mm,
                "z_mm": rotation_sample.raw_tag_z_mm,
            },
            "body_heading_deg": rotation_sample.body_heading_deg,
            "applied_tag_to_body_offset_body_mm": {
                "forward_mm": rotation_sample.applied_forward_mm,
                "left_mm": rotation_sample.applied_left_mm,
            },
            "body_center_mm": {
                "x_mm": rotation_sample.body_x_mm,
                "z_mm": rotation_sample.body_z_mm,
            },
        }
    return "[POSE] " + json.dumps(record)


class VisionRotationDiagnosticTest(unittest.TestCase):
    def test_parser_uses_only_measured_records(self):
        measured = sample(0.0)
        lines = [
            record_line("MEASURED", measured),
            record_line("HELD", sample(45.0)),
            record_line("LOST"),
            record_line("MEASURED", sample(90.0)).replace("cal-id", "other"),
        ]
        parsed = measured_rotation_samples(lines, calibration_id="cal-id")
        self.assertEqual(parsed, [measured])

    def test_fixed_rotation_recovers_offset_candidate(self):
        samples = [sample(heading) for heading in range(0, 360, 45)]
        result = diagnose_stationary_rotation(samples)
        self.assertEqual(result.sample_count, 8)
        self.assertAlmostEqual(result.candidate_forward_mm, 80.0, places=9)
        self.assertAlmostEqual(result.candidate_left_mm, -15.0, places=9)
        self.assertAlmostEqual(result.fitted_center_x_mm, 500.0, places=9)
        self.assertAlmostEqual(result.fitted_center_z_mm, 200.0, places=9)
        self.assertLess(result.candidate_center_rms_residual_mm, 1e-9)
        self.assertEqual(result.classification, "OFFSET_ERROR_LIKELY")

    def test_irregular_physical_motion_is_not_explained_by_offset(self):
        shifts = [
            (0.0, 0.0),
            (28.0, 5.0),
            (-20.0, 27.0),
            (16.0, -31.0),
            (-29.0, -8.0),
            (7.0, 34.0),
            (31.0, -21.0),
            (-12.0, 18.0),
        ]
        samples = [
            sample(heading, true_offset=(60.0, 0.0), physical_shift=shift)
            for heading, shift in zip(range(0, 360, 45), shifts)
        ]
        result = diagnose_stationary_rotation(samples)
        self.assertGreater(result.candidate_center_rms_residual_mm, 8.0)
        self.assertEqual(result.classification, "IRREGULAR_MOTION_LIKELY")

    def test_requires_four_measured_samples(self):
        with self.assertRaisesRegex(ValueError, "at least four MEASURED"):
            diagnose_stationary_rotation([sample(0.0), sample(90.0)])


if __name__ == "__main__":
    unittest.main()
