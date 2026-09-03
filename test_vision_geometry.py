import unittest

import numpy as np

from vision_geometry import (
    fit_pixel_to_map_homography,
    image_heading_degrees,
    map_pose_from_pixel_axis,
    map_pose_from_tag,
    project_point,
    tag_center_to_robot_origin,
    tag_axis_point,
    tag_front_midpoint,
    trace_tag_center_to_robot_origin,
)


class VisionGeometryTest(unittest.TestCase):
    def setUp(self):
        # pupil_apriltags ordering for an upright generated tag:
        # bottom-left, bottom-right, top-right, top-left.
        self.corners = np.array(
            [[0.0, 100.0], [100.0, 100.0], [100.0, 0.0], [0.0, 0.0]]
        )
        self.center = np.array([50.0, 50.0])

    def test_front_edge_and_image_heading(self):
        front = tag_front_midpoint(self.corners, (2, 3))
        np.testing.assert_allclose(front, [50.0, 0.0])
        self.assertAlmostEqual(image_heading_degrees(self.center, front), 90.0)

    def test_homography_maps_reference_square(self):
        pixels = [[0, 100], [100, 100], [100, 0], [0, 0]]
        map_mm = [[0, 0], [200, 0], [200, 100], [0, 100]]
        homography, inliers = fit_pixel_to_map_homography(pixels, map_mm)
        self.assertTrue(np.all(inliers))
        np.testing.assert_allclose(
            project_point(homography, [50, 50]), [100, 50], atol=1e-6
        )

    def test_consistent_six_anchor_grid_is_not_lost_to_small_sample_ransac(self):
        # Real C270 measurements from the 3x2 TestCase0 reference layout.
        # OpenCV RANSAC selects only four anchors for this small structured set,
        # although a direct fit keeps every residual below the 10 mm threshold.
        pixels = np.array(
            [
                [130.81, 520.99],
                [1147.72, 582.73],
                [1079.35, 130.21],
                [652.35, 100.97],
                [239.18, 73.76],
                [632.01, 550.74],
            ]
        )
        map_mm = np.array(
            [
                [-150.0, -150.0],
                [1550.0, -150.0],
                [1550.0, 850.0],
                [700.0, 850.0],
                [-150.0, 850.0],
                [700.0, -150.0],
            ]
        )

        homography, inliers = fit_pixel_to_map_homography(
            pixels, map_mm, ransac_threshold_mm=10.0
        )

        self.assertTrue(np.all(inliers))
        projected = np.asarray(
            [project_point(homography, pixel) for pixel in pixels]
        )
        errors = np.linalg.norm(projected - map_mm, axis=1)
        self.assertLessEqual(float(np.max(errors)), 10.0)

    def test_canonical_tag_axis_points_toward_top(self):
        tag_homography = np.array(
            [[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]]
        )
        front = tag_axis_point(tag_homography, (0.0, -0.8))
        np.testing.assert_allclose(front, [50.0, 10.0])
        self.assertAlmostEqual(image_heading_degrees(self.center, front), 90.0)

    def test_map_pose_uses_projected_front_direction(self):
        pixel_to_map = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 100.0], [0.0, 0.0, 1.0]]
        )
        x_mm, z_mm, heading = map_pose_from_tag(
            pixel_to_map, self.center, self.corners, (2, 3)
        )
        self.assertAlmostEqual(x_mm, 50.0)
        self.assertAlmostEqual(z_mm, 50.0)
        self.assertAlmostEqual(heading, 90.0)

    def test_map_pose_from_pixel_axis(self):
        pixel_to_map = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 100.0], [0.0, 0.0, 1.0]]
        )
        x_mm, z_mm, heading = map_pose_from_pixel_axis(
            pixel_to_map, self.center, [50.0, 10.0]
        )
        self.assertAlmostEqual(x_mm, 50.0)
        self.assertAlmostEqual(z_mm, 50.0)
        self.assertAlmostEqual(heading, 90.0)

    def test_project_point_rejects_projective_horizon(self):
        homography = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, -10.0]]
        )
        with self.assertRaisesRegex(ValueError, "horizon"):
            project_point(homography, [10.0, 5.0])

    def test_tag_center_to_robot_origin_rotates_body_offset(self):
        expected = {
            0.0: (110.0, 205.0),
            90.0: (95.0, 210.0),
            180.0: (90.0, 195.0),
            -90.0: (105.0, 190.0),
        }
        for heading, point in expected.items():
            with self.subTest(heading=heading):
                actual = tag_center_to_robot_origin(
                    100.0, 200.0, heading, 10.0, 5.0
                )
                np.testing.assert_allclose(actual, point, atol=1e-9)

    def test_zero_body_offset_preserves_tag_center_at_every_heading(self):
        for heading in (0.0, 90.0, 180.0, -90.0):
            with self.subTest(heading=heading):
                actual = tag_center_to_robot_origin(
                    100.0, 200.0, heading, 0.0, 0.0
                )
                np.testing.assert_allclose(actual, (100.0, 200.0), atol=1e-9)

    def test_robot_origin_trace_preserves_raw_and_applied_values(self):
        trace = trace_tag_center_to_robot_origin(
            100.0, 200.0, -100.0, 10.0, 10.0, 5.0
        )
        self.assertEqual(trace.raw_tag_x_mm, 100.0)
        self.assertEqual(trace.raw_tag_z_mm, 200.0)
        self.assertEqual(trace.raw_tag_heading_deg, -100.0)
        self.assertEqual(trace.heading_offset_deg, 10.0)
        self.assertEqual(trace.body_heading_deg, -90.0)
        self.assertEqual(trace.forward_offset_mm, 10.0)
        self.assertEqual(trace.left_offset_mm, 5.0)
        np.testing.assert_allclose(
            (trace.body_x_mm, trace.body_z_mm), (105.0, 190.0), atol=1e-9
        )


if __name__ == "__main__":
    unittest.main()
