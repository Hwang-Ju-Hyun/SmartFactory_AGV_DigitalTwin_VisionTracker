import unittest
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import cv2
import numpy as np

from vision_map import MapContract
from pose_tracker import MetricMeasurement, MetricPose, PoseEstimate, PoseState
from vision_calibration import CalibrationUseState, VerificationResult
from vision_server_client import (
    QUALITY_CALIBRATION_RMS_ERROR,
    QUALITY_DECISION_MARGIN,
    QUALITY_VERIFICATION_AGE,
    QUALITY_VERIFICATION_COVERAGE,
    QUALITY_VERIFICATION_MAX_ERROR,
    QUALITY_VERIFICATION_REFERENCE_COUNT,
    QUALITY_VERIFICATION_RMS_ERROR,
    VisionTrackingState,
    VisionVerificationState,
)
from vision_tracker_preview import (
    CameraProbeResult,
    LatestFrameReader,
    TagObservation,
    _event_rate,
    _probe_capture,
    annotate_frame,
    build_server_observation,
    configured_robot_pose_contract,
    load_config,
    open_camera,
    require_frame_size,
    run_camera,
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

    def test_production_pose_contract_matches_measured_robot_origin(self):
        contract = configured_robot_pose_contract(self.config)
        self.assertEqual(contract.forward_offset_mm, 0.0)
        self.assertEqual(contract.left_offset_mm, 0.0)
        self.assertEqual(contract.contract_id, "f84eb43ebb6cf7ff")

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

    def test_invalid_server_transport_config_is_rejected(self):
        config = deepcopy(self.config)
        config["server"]["source_id"] = 0
        with self.assertRaisesRegex(SystemExit, "source_id"):
            validate_configuration(config)

    def test_measured_pose_maps_to_verified_server_observation(self):
        pose = MetricPose(123.0, -45.0, 90.0)
        measurement = MetricMeasurement(
            sequence=8,
            received_monotonic_s=10.0,
            pose=pose,
            decision_margin=52.0,
            calibration_id="cal-test",
            calibration_rms_error_mm=2.0,
        )
        estimate = PoseEstimate(
            state=PoseState.MEASURED,
            pose=pose,
            measurement_age_ms=8.4,
            measured_at_s=10.0,
            source_sequence=8,
            calibration_id="cal-test",
            fresh=True,
        )
        calibration = SimpleNamespace(
            calibration_id="cal-test",
            quality=SimpleNamespace(rms_error_mm=2.0),
        )
        verification = VerificationResult(
            available=True,
            passed=True,
            matched_count=6,
            rms_error_mm=1.5,
            max_error_mm=3.0,
            coverage_ratio=0.8,
        )
        output = build_server_observation(
            estimate,
            calibration=calibration,
            calibration_state=CalibrationUseState.VERIFIED,
            verification=verification,
            verification_age_s=0.02,
            measurement=measurement,
        )
        self.assertEqual(output.state, VisionTrackingState.MEASURED)
        self.assertEqual(output.verification_state, VisionVerificationState.VERIFIED)
        self.assertEqual(output.source_timestamp_us, 10_000_000)
        self.assertEqual(output.reported_age_ms, 8)
        self.assertEqual((output.x_mm, output.z_mm, output.heading_deg), (123.0, -45.0, 90.0))
        expected_fields = (
            QUALITY_DECISION_MARGIN
            | QUALITY_CALIBRATION_RMS_ERROR
            | QUALITY_VERIFICATION_REFERENCE_COUNT
            | QUALITY_VERIFICATION_RMS_ERROR
            | QUALITY_VERIFICATION_MAX_ERROR
            | QUALITY_VERIFICATION_COVERAGE
            | QUALITY_VERIFICATION_AGE
        )
        self.assertEqual(output.quality.quality_fields, expected_fields)
        self.assertEqual(output.quality.verification_age_ms, 20)

    def test_initial_lost_uses_locked_calibration_without_pose(self):
        estimate = PoseEstimate(
            state=PoseState.LOST,
            pose=None,
            measurement_age_ms=None,
            measured_at_s=None,
            source_sequence=None,
            calibration_id=None,
            fresh=False,
        )
        calibration = SimpleNamespace(
            calibration_id="cal-test",
            quality=SimpleNamespace(rms_error_mm=2.0),
        )
        output = build_server_observation(
            estimate,
            calibration=calibration,
            calibration_state=CalibrationUseState.AWAITING_VERIFICATION,
            verification=None,
            verification_age_s=None,
            measurement=None,
        )
        self.assertEqual(output.state, VisionTrackingState.LOST)
        self.assertEqual(
            output.verification_state,
            VisionVerificationState.AWAITING_VERIFICATION,
        )
        self.assertEqual(output.calibration_id, "cal-test")
        self.assertEqual(output.source_timestamp_us, 0)
        self.assertEqual(output.reported_age_ms, 0)
        self.assertIsNone(output.x_mm)

    def test_camera_loop_publishes_only_through_compatible_calibration_client(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        capture = FiniteFakeCapture([frame])
        calibration = SimpleNamespace(
            calibration_id="cal-test",
            pose_contract_id="pose-contract-test",
            homography=np.eye(3),
            quality=SimpleNamespace(rms_error_mm=2.0),
        )

        class FakeGuard:
            def __init__(self):
                self.calibration = calibration

            def update(self, _now_s, _verification):
                pass

            def state(self, _now_s):
                return CalibrationUseState.VERIFIED

            def usable_homography(self, _now_s):
                return np.eye(3)

            def verification_age_s(self, _now_s):
                return 0.0

        class FakeServerClient:
            instances = []

            def __init__(self, _config, **identities):
                self.calibration_id = identities["calibration_id"]
                self.identities = identities
                self.started = False
                self.closed = False
                self.published = []
                self.instances.append(self)

            def start(self):
                self.started = True

            def publish(self, observation):
                self.published.append(observation)

            def close(self):
                self.closed = True

        verification = VerificationResult(
            available=True,
            passed=True,
            matched_count=6,
            rms_error_mm=1.0,
            max_error_mm=2.0,
            coverage_ratio=1.0,
        )
        reported_camera = {
            "width": 320,
            "height": 240,
            "measured_fps": 30.0,
        }
        with (
            patch("vision_tracker_preview.open_camera", return_value=(capture, reported_camera)),
            patch(
                "vision_tracker_preview.load_compatible_calibration_guard",
                return_value=FakeGuard(),
            ),
            patch("vision_tracker_preview.verify_locked_calibration", return_value=verification),
            patch("vision_tracker_preview.observations_from_frame", return_value=[]),
            patch("vision_tracker_preview.VisionServerClient", FakeServerClient),
            patch.object(cv2, "namedWindow"),
            patch.object(cv2, "resizeWindow"),
            patch.object(cv2, "imshow"),
            patch.object(cv2, "waitKey", return_value=ord("q")),
            patch.object(cv2, "destroyAllWindows"),
        ):
            run_camera(object(), deepcopy(self.config), None)

        self.assertEqual(len(FakeServerClient.instances), 1)
        client = FakeServerClient.instances[0]
        self.assertTrue(client.started)
        self.assertTrue(client.closed)
        self.assertEqual(client.identities["map_contract_id"], self.map_contract.contract_id)
        self.assertEqual(client.identities["pose_contract_id"], "pose-contract-test")
        self.assertEqual(len(client.published), 1)
        self.assertEqual(client.published[0].state, VisionTrackingState.LOST)

    def test_camera_loop_never_starts_sender_without_compatible_calibration(self):
        capture = FiniteFakeCapture([np.zeros((240, 320, 3), dtype=np.uint8)])
        reported_camera = {
            "width": 320,
            "height": 240,
            "measured_fps": 30.0,
        }
        with (
            patch("vision_tracker_preview.open_camera", return_value=(capture, reported_camera)),
            patch(
                "vision_tracker_preview.load_compatible_calibration_guard",
                return_value=None,
            ),
            patch("vision_tracker_preview.observations_from_frame", return_value=[]),
            patch("vision_tracker_preview.VisionServerClient") as server_client,
            patch.object(cv2, "namedWindow"),
            patch.object(cv2, "resizeWindow"),
            patch.object(cv2, "imshow"),
            patch.object(cv2, "waitKey", return_value=ord("q")),
            patch.object(cv2, "destroyAllWindows"),
        ):
            run_camera(object(), deepcopy(self.config), None)
        server_client.assert_not_called()

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
        self.assertFalse(camera["auto_exposure"])
        self.assertEqual(camera["exposure"], -4.0)

    def test_open_camera_keeps_mjpg_as_final_stream_setting(self):
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
            property_order[:6],
            [
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS,
                cv2.CAP_PROP_FOURCC,
                cv2.CAP_PROP_BUFFERSIZE,
                cv2.CAP_PROP_AUTO_EXPOSURE,
            ],
        )
        self.assertEqual(capture.set_calls[6], (cv2.CAP_PROP_EXPOSURE, -4.0))
        self.assertEqual((actual["width"], actual["height"]), (1280, 720))
        self.assertAlmostEqual(actual["measured_fps"], 29.8)
        self.assertEqual(actual["fourcc"], "MJPG")
        self.assertTrue(actual["minimum_capture_fps_met"])

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
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS,
                cv2.CAP_PROP_FOURCC,
            ],
        )

    def test_open_camera_warns_but_keeps_slow_correct_size_mode(self):
        capture = ConfigurableFakeCapture(backend_name="DSHOW")
        slow = CameraProbeResult((1280, 720), 7.5)
        with patch(
            "vision_tracker_preview.cv2.VideoCapture", return_value=capture
        ) as constructor, patch(
            "vision_tracker_preview._probe_capture", return_value=slow
        ), patch("builtins.print") as printer:
            selected, actual = open_camera(deepcopy(self.config["camera"]), None)

        self.assertIs(selected, capture)
        self.assertFalse(capture.released)
        self.assertFalse(actual["minimum_capture_fps_met"])
        constructor.assert_called_once_with(0, cv2.CAP_DSHOW)
        self.assertTrue(
            any(
                str(arguments[0]).startswith("[CAMERA] WARNING slow capture")
                for arguments, _ in printer.call_args_list
            )
        )

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
