import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from vision_calibration import (
    CalibrationError,
    CalibrationUseState,
    LockedCalibrationGuard,
    PlanarCalibration,
    build_planar_calibration,
    validate_calibration_compatibility,
    verify_locked_calibration,
)
from vision_geometry import project_point


class VisionCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.map_points = {
            1: np.array([0.0, 0.0]),
            2: np.array([800.0, 0.0]),
            3: np.array([800.0, 400.0]),
            4: np.array([0.0, 400.0]),
            5: np.array([400.0, 0.0]),
            6: np.array([400.0, 400.0]),
        }
        self.map_to_pixel = np.array(
            [[1.1, 0.12, 180.0], [0.05, 1.35, 90.0], [0.0002, 0.0001, 1.0]]
        )
        rng = np.random.default_rng(1234)
        self.samples = {}
        for tag_id, map_point in self.map_points.items():
            center = project_point(self.map_to_pixel, map_point)
            self.samples[tag_id] = [
                center + rng.normal(0.0, 0.15, size=2) for _ in range(25)
            ]

    def build(self):
        return build_planar_calibration(
            self.samples,
            self.map_points,
            map_name="TestCase0",
            map_contract_id="map-contract-test",
            pose_contract_id="pose-contract-test",
            image_size_px=(1280, 720),
            reference_plane_height_mm=120.0,
            robot_tag_height_mm=120.0,
            minimum_samples_per_tag=20,
            minimum_reference_tags=5,
            minimum_inliers=5,
            map_bounds_mm=(0.0, 800.0, 0.0, 400.0),
        )

    def test_build_recovers_metric_coordinates(self):
        calibration = self.build()
        test_map_point = np.array([250.0, 175.0])
        pixel = project_point(self.map_to_pixel, test_map_point)
        restored = project_point(calibration.homography, pixel)
        np.testing.assert_allclose(restored, test_map_point, atol=1.0)
        self.assertGreaterEqual(calibration.quality.inlier_count, 5)

    def test_outlier_reference_is_rejected_by_ransac(self):
        self.samples[6] = [sample + [180.0, -130.0] for sample in self.samples[6]]
        with self.assertRaisesRegex(CalibrationError, "outliers"):
            self.build()

    def test_consistent_six_anchor_layout_builds_without_ransac_false_rejection(self):
        map_points = {
            1: np.array([-150.0, -150.0]),
            3: np.array([1550.0, -150.0]),
            5: np.array([1550.0, 850.0]),
            6: np.array([700.0, 850.0]),
            7: np.array([-150.0, 850.0]),
            8: np.array([700.0, -150.0]),
        }
        measured_pixels = {
            1: np.array([130.81, 520.99]),
            3: np.array([1147.72, 582.73]),
            5: np.array([1079.35, 130.21]),
            6: np.array([652.35, 100.97]),
            7: np.array([239.18, 73.76]),
            8: np.array([632.01, 550.74]),
        }
        samples = {
            tag_id: [pixel.copy() for _ in range(20)]
            for tag_id, pixel in measured_pixels.items()
        }

        calibration = build_planar_calibration(
            samples,
            map_points,
            map_name="TestCase0",
            map_contract_id="map-contract-live-regression",
            pose_contract_id="pose-contract-live-regression",
            image_size_px=(1280, 720),
            reference_plane_height_mm=90.0,
            robot_tag_height_mm=90.0,
            minimum_samples_per_tag=20,
            minimum_reference_tags=5,
            minimum_inliers=5,
            ransac_threshold_mm=10.0,
            maximum_inlier_error_mm=15.0,
            map_bounds_mm=(-150.0, 1550.0, -150.0, 850.0),
            minimum_map_coverage_ratio=0.5,
        )

        self.assertEqual(calibration.quality.reference_count, 6)
        self.assertEqual(calibration.quality.inlier_count, 6)
        self.assertLessEqual(calibration.quality.max_error_mm, 10.0)

    def test_save_load_and_runtime_verification(self):
        calibration = self.build()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            calibration.save(path)
            loaded = PlanarCalibration.load(path)
        np.testing.assert_allclose(loaded.homography, calibration.homography)
        self.assertEqual(loaded.calibration_id, calibration.calibration_id)

        current = {
            tag_id: project_point(self.map_to_pixel, point)
            for tag_id, point in self.map_points.items()
        }
        verification = verify_locked_calibration(loaded, current)
        self.assertTrue(verification.available)
        self.assertTrue(verification.passed)

    def test_modified_calibration_file_fails_integrity_check(self):
        calibration = self.build()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            calibration.save(path)
            with path.open("r", encoding="utf-8") as source:
                raw = json.load(source)
            raw["homography_pixel_to_local_mm"][0][2] += 100.0
            with path.open("w", encoding="utf-8") as output:
                json.dump(raw, output)
            with self.assertRaisesRegex(CalibrationError, "integrity"):
                PlanarCalibration.load(path)

    def test_changed_resolution_is_rejected(self):
        calibration = self.build()
        with self.assertRaisesRegex(CalibrationError, "resolution"):
            validate_calibration_compatibility(
                calibration,
                map_name="TestCase0",
                map_contract_id="map-contract-test",
                pose_contract_id="pose-contract-test",
                image_size_px=(1920, 1080),
                reference_plane_height_mm=120.0,
                robot_tag_height_mm=120.0,
                reference_map_mm=self.map_points,
                minimum_reference_tags=5,
                minimum_inliers=5,
                maximum_error_mm=15.0,
                minimum_map_coverage_ratio=0.5,
            )

    def test_changed_robot_pose_contract_is_rejected(self):
        calibration = self.build()
        with self.assertRaisesRegex(CalibrationError, "pose alignment changed"):
            validate_calibration_compatibility(
                calibration,
                map_name="TestCase0",
                map_contract_id="map-contract-test",
                pose_contract_id="different-pose-contract",
                image_size_px=(1280, 720),
                reference_plane_height_mm=120.0,
                robot_tag_height_mm=120.0,
                reference_map_mm=self.map_points,
                minimum_reference_tags=5,
                minimum_inliers=5,
                maximum_error_mm=15.0,
                minimum_map_coverage_ratio=0.5,
            )

    def test_different_tag_planes_are_rejected(self):
        with self.assertRaisesRegex(CalibrationError, "different height planes"):
            build_planar_calibration(
                self.samples,
                self.map_points,
                map_name="TestCase0",
                map_contract_id="map-contract-test",
                pose_contract_id="pose-contract-test",
                image_size_px=(1280, 720),
                reference_plane_height_mm=0.0,
                robot_tag_height_mm=120.0,
                minimum_samples_per_tag=20,
                minimum_reference_tags=5,
                map_bounds_mm=(0.0, 800.0, 0.0, 400.0),
            )

    def test_collinear_references_are_rejected(self):
        map_points = {
            1: np.array([0.0, 0.0]),
            2: np.array([100.0, 0.0]),
            3: np.array([200.0, 0.0]),
            4: np.array([300.0, 0.0]),
            5: np.array([400.0, 0.0]),
        }
        samples = {
            tag_id: [np.array([point[0], 100.0])] * 20
            for tag_id, point in map_points.items()
        }
        with self.assertRaisesRegex(CalibrationError, "collinear"):
            build_planar_calibration(
                samples,
                map_points,
                map_name="TestCase0",
                map_contract_id="map-contract-test",
                pose_contract_id="pose-contract-test",
                image_size_px=(1280, 720),
                reference_plane_height_mm=100.0,
                robot_tag_height_mm=100.0,
                minimum_reference_tags=5,
                minimum_inliers=5,
                map_bounds_mm=(0.0, 800.0, 0.0, 400.0),
            )

    def test_clustered_calibration_anchors_are_rejected(self):
        clustered_map = {
            1: np.array([0.0, 0.0]),
            2: np.array([200.0, 0.0]),
            3: np.array([200.0, 100.0]),
            4: np.array([0.0, 100.0]),
            5: np.array([100.0, 50.0]),
        }
        clustered_samples = {
            tag_id: [project_point(self.map_to_pixel, point)] * 20
            for tag_id, point in clustered_map.items()
        }
        with self.assertRaisesRegex(CalibrationError, "coverage"):
            build_planar_calibration(
                clustered_samples,
                clustered_map,
                map_name="TestCase0",
                map_contract_id="map-contract-test",
                pose_contract_id="pose-contract-test",
                image_size_px=(1280, 720),
                reference_plane_height_mm=120.0,
                robot_tag_height_mm=120.0,
                minimum_samples_per_tag=20,
                minimum_reference_tags=5,
                minimum_inliers=5,
                map_bounds_mm=(0.0, 800.0, 0.0, 400.0),
                minimum_map_coverage_ratio=0.5,
            )

    def test_reference_hull_outside_map_is_rejected(self):
        outside_map = {
            1: np.array([1000.0, 0.0]),
            2: np.array([1800.0, 0.0]),
            3: np.array([1800.0, 400.0]),
            4: np.array([1000.0, 400.0]),
            5: np.array([1400.0, 200.0]),
        }
        outside_samples = {
            tag_id: [project_point(self.map_to_pixel, point)] * 20
            for tag_id, point in outside_map.items()
        }
        with self.assertRaisesRegex(CalibrationError, "coverage"):
            build_planar_calibration(
                outside_samples,
                outside_map,
                map_name="TestCase0",
                map_contract_id="map-contract-test",
                pose_contract_id="pose-contract-test",
                image_size_px=(3000, 2000),
                reference_plane_height_mm=120.0,
                robot_tag_height_mm=120.0,
                map_bounds_mm=(0.0, 800.0, 0.0, 400.0),
            )

    def test_reference_hull_enclosing_map_has_bounded_full_coverage(self):
        enclosing_map = {
            1: np.array([-100.0, -100.0]),
            2: np.array([900.0, -100.0]),
            3: np.array([900.0, 500.0]),
            4: np.array([-100.0, 500.0]),
            5: np.array([400.0, 200.0]),
        }
        enclosing_samples = {
            tag_id: [project_point(self.map_to_pixel, point)] * 20
            for tag_id, point in enclosing_map.items()
        }
        calibration = build_planar_calibration(
            enclosing_samples,
            enclosing_map,
            map_name="TestCase0",
            map_contract_id="map-contract-test",
            pose_contract_id="pose-contract-test",
            image_size_px=(3000, 2000),
            reference_plane_height_mm=120.0,
            robot_tag_height_mm=120.0,
            map_bounds_mm=(0.0, 800.0, 0.0, 400.0),
        )
        self.assertAlmostEqual(calibration.quality.map_coverage_ratio, 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            calibration.save(path)
            loaded = PlanarCalibration.load(path)
        self.assertAlmostEqual(loaded.quality.map_coverage_ratio, 1.0)

    def test_loaded_calibration_must_be_verified_and_expires(self):
        calibration = self.build()
        guard = LockedCalibrationGuard(
            calibration,
            verification_max_age_seconds=2.0,
            bad_frames_to_invalidate=3,
        )
        self.assertEqual(guard.state(10.0), CalibrationUseState.AWAITING_VERIFICATION)
        current = {
            tag_id: project_point(self.map_to_pixel, point)
            for tag_id, point in self.map_points.items()
        }
        result = verify_locked_calibration(calibration, current)
        guard.update(10.0, result)
        self.assertEqual(guard.state(11.9), CalibrationUseState.VERIFIED)
        self.assertIsNotNone(guard.usable_homography(11.9))
        self.assertEqual(guard.state(12.01), CalibrationUseState.STALE)
        self.assertIsNone(guard.usable_homography(12.01))

    def test_repeated_bad_anchor_frames_invalidate_calibration(self):
        calibration = self.build()
        guard = LockedCalibrationGuard(
            calibration,
            verification_max_age_seconds=2.0,
            bad_frames_to_invalidate=3,
            verified_at_s=10.0,
        )
        moved = {
            tag_id: project_point(self.map_to_pixel, point) + [100.0, 50.0]
            for tag_id, point in self.map_points.items()
        }
        result = verify_locked_calibration(calibration, moved)
        self.assertFalse(result.passed)
        guard.update(10.1, result)
        self.assertEqual(guard.state(10.1), CalibrationUseState.MISMATCH)
        self.assertIsNone(guard.usable_homography(10.1))
        guard.update(10.2, result)
        self.assertEqual(guard.state(10.2), CalibrationUseState.MISMATCH)
        guard.update(10.3, result)
        self.assertEqual(guard.state(10.3), CalibrationUseState.INVALID)

    def test_missing_current_references_blocks_fresh_metric_output(self):
        calibration = self.build()
        guard = LockedCalibrationGuard(
            calibration,
            verification_max_age_seconds=2.0,
            bad_frames_to_invalidate=3,
            verified_at_s=10.0,
        )
        missing = verify_locked_calibration(calibration, {})
        self.assertFalse(missing.available)
        guard.update(10.01, missing)
        self.assertEqual(
            guard.state(10.01), CalibrationUseState.REFERENCES_MISSING
        )
        self.assertIsNone(guard.usable_homography(10.01))

    def test_verification_requires_spatial_coverage(self):
        calibration = self.build()
        clustered_ids = (1, 2, 5, 6)
        current = {
            tag_id: project_point(self.map_to_pixel, self.map_points[tag_id])
            for tag_id in clustered_ids
        }
        verification = verify_locked_calibration(
            calibration,
            current,
            minimum_tags=4,
            minimum_coverage_ratio=0.75,
        )
        self.assertTrue(verification.available)
        self.assertFalse(verification.passed)
        self.assertLess(verification.coverage_ratio, 0.75)

    def test_four_reference_policy_is_rejected(self):
        with self.assertRaisesRegex(CalibrationError, "five reference"):
            build_planar_calibration(
                self.samples,
                self.map_points,
                map_name="TestCase0",
                map_contract_id="map-contract-test",
                pose_contract_id="pose-contract-test",
                image_size_px=(1280, 720),
                reference_plane_height_mm=120.0,
                robot_tag_height_mm=120.0,
                minimum_reference_tags=4,
                minimum_inliers=4,
                map_bounds_mm=(0.0, 800.0, 0.0, 400.0),
            )

    def test_metadata_mutation_fails_integrity_check(self):
        calibration = self.build()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            calibration.save(path)
            with path.open("r", encoding="utf-8") as source:
                raw = json.load(source)
            raw["reference_pixels"]["1"][0] += 1.0
            with path.open("w", encoding="utf-8") as output:
                json.dump(raw, output)
            with self.assertRaisesRegex(CalibrationError, "integrity"):
                PlanarCalibration.load(path)

    def test_malformed_numeric_file_is_cleanly_rejected(self):
        calibration = self.build()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            calibration.save(path)
            with path.open("r", encoding="utf-8") as source:
                raw = json.load(source)
            raw["image_size_px"] = ["not-a-number", 720]
            with path.open("w", encoding="utf-8") as output:
                json.dump(raw, output)
            with self.assertRaisesRegex(CalibrationError, "cannot load"):
                PlanarCalibration.load(path)

    def test_guard_rejects_nonfinite_or_regressing_time(self):
        calibration = self.build()
        guard = LockedCalibrationGuard(
            calibration,
            verification_max_age_seconds=2.0,
            bad_frames_to_invalidate=3,
            verified_at_s=10.0,
        )
        self.assertEqual(guard.state(10.1), CalibrationUseState.VERIFIED)
        with self.assertRaisesRegex(ValueError, "monotonic"):
            guard.state(10.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            LockedCalibrationGuard(
                calibration,
                verification_max_age_seconds=2.0,
                bad_frames_to_invalidate=3,
                verified_at_s=float("nan"),
            )

    def test_loaded_calibration_must_meet_current_quality_policy(self):
        calibration = self.build()
        with self.assertRaisesRegex(CalibrationError, "too few references"):
            validate_calibration_compatibility(
                calibration,
                map_name="TestCase0",
                map_contract_id="map-contract-test",
                pose_contract_id="pose-contract-test",
                image_size_px=(1280, 720),
                reference_plane_height_mm=120.0,
                robot_tag_height_mm=120.0,
                reference_map_mm=self.map_points,
                minimum_reference_tags=7,
                minimum_inliers=5,
                maximum_error_mm=15.0,
                minimum_map_coverage_ratio=0.5,
            )

    def test_library_rejects_nonfinite_error_thresholds(self):
        with self.assertRaisesRegex(CalibrationError, "positive and finite"):
            build_planar_calibration(
                self.samples,
                self.map_points,
                map_name="TestCase0",
                map_contract_id="map-contract-test",
                pose_contract_id="pose-contract-test",
                image_size_px=(1280, 720),
                reference_plane_height_mm=120.0,
                robot_tag_height_mm=120.0,
                minimum_samples_per_tag=20,
                map_bounds_mm=(0.0, 800.0, 0.0, 400.0),
                maximum_inlier_error_mm=float("inf"),
            )
        calibration = self.build()
        current = {
            tag_id: project_point(self.map_to_pixel, point)
            for tag_id, point in self.map_points.items()
        }
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            verify_locked_calibration(
                calibration,
                current,
                maximum_error_mm=float("inf"),
            )


if __name__ == "__main__":
    unittest.main()
