import unittest
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import cv2
import numpy as np

from vision_map import MapContract
from vision_tracker_preview import (
    CameraProbeResult,
    LatestFrameReader,
    TagObservation,
    _event_rate,
    _probe_capture,
    annotate_frame,
    load_config,
    open_camera,
    require_frame_size,
    validate_configuration,
)


BASE_DIR = Path(__file__).resolve().parent


class FiniteFakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.finished = threading.Event()
        self.released = False

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        self.finished.set()
        return False, None

    def release(self):
        self.released = True


class ConfigurableFakeCapture:
    def __init__(self, opened=True, backend_name="DSHOW"):
        self.opened = opened
        self.backend_name = backend_name
        self.released = False
        self.set_calls = []

    def isOpened(self):
        return self.opened

    def set(self, property_id, value):
        self.set_calls.append((property_id, value))
        return True

    def get(self, property_id):
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"MJPG")),
        }
        return values.get(property_id, 0.0)

    def getBackendName(self):
        return self.backend_name

    def release(self):
        self.released = True


def robot_observation(hamming=0, decision_margin=40.0, center=(100.0, 100.0)):
    center_x, center_y = center
    return TagObservation(
        tag_id=0,
        family="tag36h11",
        center_px=np.array([center_x, center_y]),
        corners_px=np.array(
            [
                [center_x - 40.0, center_y + 40.0],
                [center_x + 40.0, center_y + 40.0],
                [center_x + 40.0, center_y - 40.0],
                [center_x - 40.0, center_y - 40.0],
            ]
        ),
        tag_homography=np.array(
            [
                [50.0, 0.0, center_x],
                [0.0, 50.0, center_y],
                [0.0, 0.0, 1.0],
            ]
        ),
        decision_margin=decision_margin,
        hamming=hamming,
    )


class VisionPreviewGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(BASE_DIR / "vision_config.json")
        cls.map_contract = MapContract.load(BASE_DIR / "testcase0_map.json")
        cls.frame = np.zeros((240, 320, 3), dtype=np.uint8)

    def annotate(self, observations, homography=None):
        return annotate_frame(
            self.frame,
            observations,
            self.config,
            self.map_contract,
            homography,
            30.0,
            "NO_CALIBRATION",
            None,
            0,
            None,
        )

    def test_pixel_detection_does_not_become_metric_pose_without_calibration(self):
        _, record = self.annotate([robot_observation()], np.eye(3))
        self.assertIsNotNone(record)
        self.assertNotIn("map_pose", record)

    def test_hamming_rejected_robot_does_not_create_record(self):
        _, record = self.annotate([robot_observation(hamming=1)])
        self.assertIsNone(record)

    def test_duplicate_robot_id_does_not_create_record(self):
        _, record = self.annotate([robot_observation(), robot_observation()])
        self.assertIsNone(record)

    def test_low_decision_margin_does_not_create_record(self):
        _, record = self.annotate([robot_observation(decision_margin=10.0)])
        self.assertIsNone(record)

    def test_pose_alignment_is_required_for_metric_output(self):
        config = deepcopy(self.config)
        config["tags"]["robot_heading_offset_deg"] = None
        config["tags"]["tag_center_to_robot_origin_body_mm"] = None
        calibration = SimpleNamespace(
            calibration_id="test",
            quality=SimpleNamespace(rms_error_mm=1.0),
        )
        _, record = annotate_frame(
            self.frame,
            [robot_observation()],
            config,
            self.map_contract,
            np.eye(3),
            30.0,
            "VERIFIED",
            calibration,
            5,
            0.0,
        )
        self.assertIsNotNone(record)
        self.assertNotIn("map_pose", record)

    def test_measured_body_offset_is_applied_to_metric_pose(self):
        config = deepcopy(self.config)
        config["tags"]["robot_heading_offset_deg"] = 0.0
        config["tags"]["tag_center_to_robot_origin_body_mm"] = [10.0, 5.0]
        calibration = SimpleNamespace(
            calibration_id="test",
            quality=SimpleNamespace(rms_error_mm=1.0),
        )
        _, record = annotate_frame(
            self.frame,
            [robot_observation()],
            config,
            self.map_contract,
            np.eye(3),
            30.0,
            "VERIFIED",
            calibration,
            5,
            0.0,
        )
        self.assertIsNotNone(record)
        self.assertAlmostEqual(record["map_pose"]["x_mm"], 105.0)
        self.assertAlmostEqual(record["map_pose"]["z_mm"], 90.0)
        self.assertAlmostEqual(record["map_pose"]["heading_deg"], -90.0)

    def test_outside_map_roi_is_not_metric_output(self):
        config = deepcopy(self.config)
        config["tags"]["robot_heading_offset_deg"] = 0.0
        config["tags"]["tag_center_to_robot_origin_body_mm"] = [0.0, 0.0]
        calibration = SimpleNamespace(
            calibration_id="test",
            quality=SimpleNamespace(rms_error_mm=1.0),
        )
        _, record = annotate_frame(
            self.frame,
            [robot_observation(center=(2000.0, 2000.0))],
            config,
            self.map_contract,
            np.eye(3),
            30.0,
            "VERIFIED",
            calibration,
            5,
            0.0,
        )
        self.assertEqual(record["metric_rejection"], "OUTSIDE_MAP_ROI")
        self.assertNotIn("map_pose", record)

    def test_runtime_frame_size_change_is_rejected(self):
        self.assertEqual(require_frame_size(self.frame, None), (320, 240))
        with self.assertRaisesRegex(RuntimeError, "changed"):
            require_frame_size(self.frame, (1280, 720))

    def test_nonfinite_or_disabled_calibration_policy_is_rejected(self):
        invalid_values = (
            ("ransac_threshold_mm", float("inf")),
            ("maximum_inlier_error_mm", float("inf")),
            ("verification_maximum_error_mm", float("inf")),
            ("verification_max_age_seconds", float("inf")),
            ("verification_bad_frames_to_invalidate", 0),
        )
        for key, value in invalid_values:
            with self.subTest(key=key):
                config = deepcopy(self.config)
                config["calibration"][key] = value
                with self.assertRaisesRegex(SystemExit, "Invalid config"):
                    validate_configuration(config)

    def test_wrong_local_coordinate_declaration_is_rejected(self):
        config = deepcopy(self.config)
        config["map"]["coordinate_system"] = "heading_cw"
        with self.assertRaisesRegex(SystemExit, "coordinate_system"):
            validate_configuration(config)

    def test_camera_reader_discards_backlog_and_returns_latest_frame(self):
        frames = [
            np.full((2, 2, 3), value, dtype=np.uint8)
            for value in (1, 2, 3)
        ]
        capture = FiniteFakeCapture(frames)
        reader = LatestFrameReader(capture)
        reader.start()
        self.assertTrue(capture.finished.wait(1.0))
        received = reader.read_after(0)
        self.assertEqual(received.source_sequence, 3)
        self.assertTrue(np.all(received.frame == 3))
        reader.close()
        self.assertTrue(capture.released)

    def test_committed_camera_profile_targets_c270_native_mode(self):
        camera = self.config["camera"]
        self.assertEqual(
            (camera["width"], camera["height"], camera["fps"], camera["fourcc"]),
            (1280, 720, 30, "MJPG"),
        )

    def test_open_camera_negotiates_mjpg_before_720p30(self):
        capture = ConfigurableFakeCapture()
        probe = CameraProbeResult((1280, 720), 29.8)
        with patch(
            "vision_tracker_preview.cv2.VideoCapture", return_value=capture
        ) as constructor, patch(
            "vision_tracker_preview._probe_capture", return_value=probe
        ):
            selected, actual = open_camera(deepcopy(self.config["camera"]), None)

        self.assertIs(selected, capture)
        constructor.assert_called_once_with(0, cv2.CAP_DSHOW)
        property_order = [property_id for property_id, _ in capture.set_calls]
        self.assertEqual(
            property_order[:4],
            [
                cv2.CAP_PROP_FOURCC,
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS,
            ],
        )
        self.assertEqual((actual["width"], actual["height"]), (1280, 720))
        self.assertAlmostEqual(actual["measured_fps"], 29.8)
        self.assertEqual(actual["fourcc"], "MJPG")

    def test_open_camera_releases_failed_dshow_and_uses_msmf(self):
        failed = ConfigurableFakeCapture(opened=False)
        accepted = ConfigurableFakeCapture(backend_name="MSMF")
        probe = CameraProbeResult((1280, 720), 30.0)
        with patch(
            "vision_tracker_preview.cv2.VideoCapture",
            side_effect=[failed, accepted],
        ) as constructor, patch(
            "vision_tracker_preview._probe_capture", return_value=probe
        ):
            selected, actual = open_camera(deepcopy(self.config["camera"]), None)

        self.assertIs(selected, accepted)
        self.assertTrue(failed.released)
        self.assertFalse(accepted.released)
        self.assertEqual(
            constructor.call_args_list,
            [call(0, cv2.CAP_DSHOW), call(0, cv2.CAP_MSMF)],
        )
        self.assertEqual(actual["requested_backend"], "msmf")
        self.assertEqual(
            [property_id for property_id, _ in accepted.set_calls][:4],
            [
                cv2.CAP_PROP_FOURCC,
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS,
            ],
        )

    def test_open_camera_rejects_slow_mode_from_every_backend(self):
        captures = [
            ConfigurableFakeCapture(backend_name=name)
            for name in ("DSHOW", "MSMF", "AUTO")
        ]
        slow = CameraProbeResult((1280, 720), 7.5)
        with patch(
            "vision_tracker_preview.cv2.VideoCapture", side_effect=captures
        ), patch(
            "vision_tracker_preview._probe_capture",
            side_effect=[slow, slow, slow],
        ):
            with self.assertRaisesRegex(SystemExit, "Could not negotiate"):
                open_camera(deepcopy(self.config["camera"]), None)

        self.assertTrue(all(capture.released for capture in captures))

    def test_capture_rate_counts_frames_skipped_by_processing(self):
        self.assertAlmostEqual(_event_rate(4, 4.0 / 30.0), 30.0)
        self.assertEqual(_event_rate(0, 1.0), 0.0)
        self.assertEqual(_event_rate(1, 0.0), 0.0)

    def test_camera_probe_measures_delivery_rate_from_real_frames(self):
        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(6)]
        capture = FiniteFakeCapture(frames)
        timestamps = [1.0, 1.0 + 1.0 / 30.0, 1.0 + 2.0 / 30.0]
        with patch("vision_tracker_preview.time.perf_counter", side_effect=timestamps):
            result = _probe_capture(capture, 3)
        self.assertEqual(result.frame_size_px, (1280, 720))
        self.assertAlmostEqual(result.measured_fps, 30.0)


if __name__ == "__main__":
    unittest.main()
