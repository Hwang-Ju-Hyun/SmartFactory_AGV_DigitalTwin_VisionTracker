"""AprilTag VisionTracker preview for the SmartFactory AGV project.

This program publishes calibrated observations to the Server when the optional
transport is enabled. It cannot send commands to the ESP32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pupil_apriltags import Detector

from pose_tracker import MetricMeasurement, MetricPose, PoseEstimate, PoseHoldTracker
from vision_calibration import (
    CalibrationCollector,
    CalibrationError,
    CalibrationUseState,
    LockedCalibrationGuard,
    PlanarCalibration,
    VerificationResult,
    build_planar_calibration,
    require_same_measurement_plane,
    validate_calibration_compatibility,
    verify_locked_calibration,
)
from vision_server_client import (
    QUALITY_CALIBRATION_RMS_ERROR,
    QUALITY_DECISION_MARGIN,
    QUALITY_VERIFICATION_AGE,
    QUALITY_VERIFICATION_COVERAGE,
    QUALITY_VERIFICATION_MAX_ERROR,
    QUALITY_VERIFICATION_REFERENCE_COUNT,
    QUALITY_VERIFICATION_RMS_ERROR,
    VisionObservation,
    VisionQuality,
    VisionServerClient,
    VisionServerConfig,
    VisionTrackingState,
    VisionVerificationState,
)
from vision_geometry import (
    image_heading_degrees,
    map_pose_from_pixel_axis,
    tag_axis_point,
    trace_tag_center_to_robot_origin,
)
from vision_map import MapContract


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "vision_config.json"
WINDOW_TITLE = "AGV AprilTag VisionTracker Preview"


@dataclass(frozen=True)
class TagObservation:
    tag_id: int
    family: str
    center_px: np.ndarray
    corners_px: np.ndarray
    tag_homography: np.ndarray
    decision_margin: float
    hamming: int


@dataclass(frozen=True)
class RobotPoseContract:
    front_axis_tip: tuple[float, float]
    heading_offset_deg: float
    forward_offset_mm: float
    left_offset_mm: float
    contract_id: str


@dataclass(frozen=True)
class ReceivedFrame:
    source_sequence: int
    received_monotonic_s: float
    frame: np.ndarray


@dataclass(frozen=True)
class CameraProbeResult:
    frame_size_px: tuple[int, int]
    measured_fps: float


class LatestFrameReader:
    """Continuously drain the camera and retain only its newest received frame."""

    def __init__(self, capture: Any):
        self._capture = capture
        self._condition = threading.Condition()
        self._latest: ReceivedFrame | None = None
        self._error: RuntimeError | None = None
        self._stop_requested = False
        self._started = False
        self._thread = threading.Thread(
            target=self._read_loop,
            name="vision-camera-reader",
            daemon=True,
        )

    def start(self) -> None:
        if self._started:
            raise RuntimeError("latest-frame reader was already started")
        self._started = True
        self._thread.start()

    def _read_loop(self) -> None:
        sequence = 0
        while True:
            success, frame = self._capture.read()
            received_at = time.perf_counter()
            with self._condition:
                if self._stop_requested:
                    return
                if not success:
                    self._error = RuntimeError("camera returned no frame")
                    self._condition.notify_all()
                    return
                sequence += 1
                self._latest = ReceivedFrame(sequence, received_at, frame)
                self._condition.notify_all()

    def read_after(
        self, source_sequence: int, timeout_seconds: float = 2.0
    ) -> ReceivedFrame:
        if not self._started:
            raise RuntimeError("latest-frame reader has not been started")
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("frame wait timeout must be positive and finite")
        deadline = time.perf_counter() + timeout
        with self._condition:
            while True:
                if (
                    self._latest is not None
                    and self._latest.source_sequence > int(source_sequence)
                ):
                    latest = self._latest
                    return ReceivedFrame(
                        source_sequence=latest.source_sequence,
                        received_monotonic_s=latest.received_monotonic_s,
                        frame=latest.frame.copy(),
                    )
                if self._error is not None:
                    raise self._error
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    raise RuntimeError("timed out waiting for a new camera frame")
                self._condition.wait(remaining)

    def close(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()
        if self._started:
            self._thread.join(timeout=1.0)
        self._capture.release()
        if self._started and self._thread.is_alive():
            self._thread.join(timeout=1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="JSON configuration path"
    )
    parser.add_argument("--camera", type=int, help="override the configured camera index")
    parser.add_argument(
        "--image", type=Path, help="inspect one saved image instead of opening a camera"
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    try:
        with path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"[VISION] Cannot load config {path}: {error}") from error

    for section in (
        "camera",
        "detector",
        "tags",
        "map",
        "calibration",
        "tracking",
        "display",
        "server",
    ):
        if section not in config:
            raise SystemExit(f"[VISION] Config is missing section: {section}")
    config["_config_dir"] = str(path.parent)
    validate_configuration(config)
    return config


def validate_configuration(config: dict[str, Any]) -> None:
    """Reject unsafe or internally inconsistent tracking policy values."""

    try:
        camera = config["camera"]
        detector = config["detector"]
        tags = config["tags"]
        calibration = config["calibration"]
        tracking = config["tracking"]
        server = config["server"]

        if int(camera["index"]) < 0:
            raise ValueError("camera index must be non-negative")
        _backend_code(str(camera.get("backend", "auto")))
        if int(camera["width"]) <= 0 or int(camera["height"]) <= 0:
            raise ValueError("camera width and height must be positive")
        requested_fps = float(camera["fps"])
        minimum_capture_fps = float(
            camera.get("minimum_capture_fps", requested_fps * 0.8)
        )
        if not math.isfinite(requested_fps) or requested_fps <= 0.0:
            raise ValueError("camera fps must be positive and finite")
        if (
            not math.isfinite(minimum_capture_fps)
            or minimum_capture_fps <= 0.0
            or minimum_capture_fps > requested_fps
        ):
            raise ValueError(
                "minimum_capture_fps must be positive and no greater than fps"
            )
        if int(camera.get("probe_frames", 20)) < 3:
            raise ValueError("camera probe_frames must be at least three")
        fourcc = str(camera.get("fourcc", ""))
        if fourcc and len(fourcc) != 4:
            raise ValueError("camera fourcc must be empty or exactly four characters")
        if "auto_exposure" in camera and not isinstance(
            camera["auto_exposure"], bool
        ):
            raise ValueError("camera auto_exposure must be true or false")
        if "exposure" in camera and not math.isfinite(float(camera["exposure"])):
            raise ValueError("camera exposure must be finite")

        robot_id = int(tags["robot_id"])
        reference_ids = [int(value) for value in tags["reference_ids"]]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("reference tag IDs must be unique")
        if robot_id in reference_ids:
            raise ValueError("robot tag ID must not also be a reference tag")
        if (
            str(config["map"]["coordinate_system"])
            != "local_x_mm_z_mm_heading_deg_0_is_positive_x_ccw"
        ):
            raise ValueError("unsupported local map coordinate_system")
        tag_size = float(tags["tag_size_mm"])
        minimum_margin = float(detector["minimum_decision_margin"])
        if not math.isfinite(tag_size) or tag_size <= 0.0:
            raise ValueError("tag_size_mm must be positive and finite")
        if not math.isfinite(minimum_margin) or minimum_margin < 0.0:
            raise ValueError("minimum_decision_margin must be finite and non-negative")
        if int(detector["max_hamming"]) < 0:
            raise ValueError("max_hamming must be non-negative")

        minimum_references = int(calibration["minimum_reference_tags"])
        minimum_inliers = int(calibration["minimum_inliers"])
        verification_tags = int(calibration["verification_minimum_tags"])
        if minimum_references < 5 or minimum_inliers < 5:
            raise ValueError("calibration requires at least five references/inliers")
        if minimum_inliers > minimum_references:
            raise ValueError("minimum_inliers cannot exceed minimum_reference_tags")
        if verification_tags < 4 or verification_tags > minimum_references:
            raise ValueError("verification tag count must be between four and calibration references")
        for key in (
            "minimum_map_coverage_ratio",
            "verification_minimum_coverage_ratio",
        ):
            ratio = float(calibration[key])
            if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
                raise ValueError(f"{key} must be between zero and one")
        if int(calibration["samples_per_reference"]) <= 0:
            raise ValueError("samples_per_reference must be positive")
        for key in (
            "ransac_threshold_mm",
            "maximum_inlier_error_mm",
            "verification_maximum_error_mm",
        ):
            value = float(calibration[key])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{key} must be positive and finite")
        verification_age = float(calibration["verification_max_age_seconds"])
        if not math.isfinite(verification_age) or verification_age < 0.0:
            raise ValueError(
                "verification_max_age_seconds must be finite and non-negative"
            )
        if int(calibration["verification_bad_frames_to_invalidate"]) < 1:
            raise ValueError(
                "verification_bad_frames_to_invalidate must be at least one"
            )

        hold_seconds = float(tracking["hold_seconds"])
        maximum_fresh_age = float(tracking["maximum_fresh_age_seconds"])
        map_margin = float(tracking["allowed_map_margin_mm"])
        if not all(
            math.isfinite(value)
            for value in (hold_seconds, maximum_fresh_age, map_margin)
        ):
            raise ValueError("tracking timing and margin values must be finite")
        if maximum_fresh_age < 0.0 or hold_seconds < maximum_fresh_age:
            raise ValueError("hold_seconds must be at least the non-negative fresh age")
        if map_margin < 0.0:
            raise ValueError("allowed_map_margin_mm must be non-negative")

        if not isinstance(server["enabled"], bool):
            raise ValueError("server enabled must be true or false")
        VisionServerConfig(
            host=str(server["host"]),
            port=int(server["port"]),
            source_id=int(server["source_id"]),
            agv_id=int(server["agv_id"]),
            connect_timeout_seconds=float(server["connect_timeout_seconds"]),
            reconnect_delay_seconds=float(server["reconnect_delay_seconds"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise SystemExit(f"[VISION] Invalid config: {error}") from error


def build_detector(config: dict[str, Any]) -> Detector:
    detector = config["detector"]
    try:
        return Detector(
            families=str(detector["families"]),
            nthreads=int(detector["nthreads"]),
            quad_decimate=float(detector["quad_decimate"]),
            quad_sigma=float(detector["quad_sigma"]),
            refine_edges=bool(detector["refine_edges"]),
            decode_sharpening=float(detector["decode_sharpening"]),
            debug=False,
        )
    except (OSError, FileNotFoundError) as error:
        raise SystemExit(
            "[VISION] AprilTag detector DLL could not be loaded. "
            "Run this project with Python 3.12: "
            "py -3.12 vision_tracker_preview.py"
        ) from error


def _backend_code(name: str) -> int | None:
    normalized = name.strip().lower()
    if normalized == "auto":
        return None
    choices = {
        "dshow": getattr(cv2, "CAP_DSHOW", None),
        "msmf": getattr(cv2, "CAP_MSMF", None),
    }
    if normalized not in choices or choices[normalized] is None:
        raise SystemExit(f"[VISION] Unsupported camera backend: {name}")
    return int(choices[normalized])


def _backend_candidates(preferred_name: str) -> list[tuple[str, int | None]]:
    preferred = preferred_name.strip().lower()
    # DSHOW has historically exposed UVC MJPG modes reliably; MSMF is the
    # modern Windows fallback. Auto remains last because its selected backend
    # otherwise varies across OpenCV and Windows releases.
    names = [preferred, "dshow", "msmf", "auto"]
    candidates: list[tuple[str, int | None]] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        candidates.append((name, _backend_code(name)))
    return candidates


def _fourcc_text(raw_value: float) -> str:
    raw = int(round(raw_value))
    if raw <= 0:
        return ""
    return "".join(chr((raw >> (8 * index)) & 0xFF) for index in range(4))


def _event_rate(event_count: int, elapsed_seconds: float) -> float:
    if event_count <= 0 or not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        return 0.0
    return float(event_count) / elapsed_seconds


def _configure_capture(capture: Any, camera_config: dict[str, Any]) -> dict[str, bool]:
    results: dict[str, bool] = {}
    # DirectShow rebuilds its stream whenever one of these properties changes.
    # Apply FOURCC last so the final rebuild keeps MJPG instead of falling back
    # to an uncompressed 720p mode that the C270 delivers at roughly 7-10 FPS.
    results["width"] = bool(
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_config["width"]))
    )
    results["height"] = bool(
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_config["height"]))
    )
    results["fps"] = bool(
        capture.set(cv2.CAP_PROP_FPS, float(camera_config["fps"]))
    )
    fourcc = str(camera_config.get("fourcc", ""))
    if len(fourcc) == 4:
        results["fourcc"] = bool(
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        )
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        results["buffer_size"] = bool(capture.set(cv2.CAP_PROP_BUFFERSIZE, 1))
    if "auto_exposure" in camera_config:
        results["auto_exposure"] = bool(
            capture.set(
                cv2.CAP_PROP_AUTO_EXPOSURE,
                1.0 if bool(camera_config["auto_exposure"]) else 0.0,
            )
        )
    if "exposure" in camera_config:
        results["exposure"] = bool(
            capture.set(cv2.CAP_PROP_EXPOSURE, float(camera_config["exposure"]))
        )
    return results


def _probe_capture(capture: Any, frame_count: int) -> CameraProbeResult:
    # Discard initial frames so device startup and an old driver buffer do not
    # inflate or depress the steady-state measurement.
    for _ in range(3):
        success, _ = capture.read()
        if not success:
            raise RuntimeError("camera returned no frame during warm-up")

    timestamps: list[float] = []
    frame_size: tuple[int, int] | None = None
    for _ in range(frame_count):
        success, frame = capture.read()
        received_at = time.perf_counter()
        if not success:
            raise RuntimeError("camera returned no frame during FPS probe")
        current_size = frame_size_px(frame)
        if frame_size is not None and current_size != frame_size:
            raise RuntimeError(
                f"camera frame size changed during probe: {frame_size} -> {current_size}"
            )
        frame_size = current_size
        timestamps.append(received_at)

    if frame_size is None or len(timestamps) < 2:
        raise RuntimeError("camera FPS probe did not receive enough frames")
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0.0:
        raise RuntimeError("camera FPS probe clock did not advance")
    return CameraProbeResult(
        frame_size_px=frame_size,
        measured_fps=(len(timestamps) - 1) / elapsed,
    )


def open_camera(camera_config: dict[str, Any], override_index: int | None):
    index = int(camera_config["index"] if override_index is None else override_index)
    preferred_backend = str(camera_config.get("backend", "auto"))
    requested_width = int(camera_config["width"])
    requested_height = int(camera_config["height"])
    requested_fps = float(camera_config["fps"])
    minimum_capture_fps = float(
        camera_config.get("minimum_capture_fps", requested_fps * 0.8)
    )
    probe_frames = int(camera_config.get("probe_frames", 20))
    failures: list[str] = []

    for backend_name, backend in _backend_candidates(preferred_backend):
        print(f"[CAMERA] Trying {backend_name} for camera {index}...")
        capture = (
            cv2.VideoCapture(index)
            if backend is None
            else cv2.VideoCapture(index, backend)
        )
        if not capture.isOpened():
            capture.release()
            failures.append(f"{backend_name}: open failed")
            continue

        set_results = _configure_capture(capture, camera_config)
        try:
            probe = _probe_capture(capture, probe_frames)
        except RuntimeError as error:
            capture.release()
            failures.append(f"{backend_name}: {error}")
            continue

        actual = {
            "index": index,
            "requested_backend": backend_name,
            "backend": capture.getBackendName()
            if hasattr(capture, "getBackendName")
            else "unknown",
            "width": probe.frame_size_px[0],
            "height": probe.frame_size_px[1],
            "reported_fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "measured_fps": probe.measured_fps,
            "fourcc": _fourcc_text(capture.get(cv2.CAP_PROP_FOURCC)),
            "auto_exposure": float(capture.get(cv2.CAP_PROP_AUTO_EXPOSURE)),
            "exposure": float(capture.get(cv2.CAP_PROP_EXPOSURE)),
            "set_ok": set_results,
        }
        size_matches = probe.frame_size_px == (requested_width, requested_height)
        speed_matches = probe.measured_fps >= minimum_capture_fps
        actual["minimum_capture_fps_met"] = speed_matches
        if size_matches:
            prefix = "[CAMERA]" if speed_matches else "[CAMERA] WARNING slow capture"
            print(prefix + " " + json.dumps(actual, ensure_ascii=False))
            return capture, actual

        capture.release()
        reason = (
            f"{backend_name}: actual={probe.frame_size_px[0]}x"
            f"{probe.frame_size_px[1]}@{probe.measured_fps:.1f}, "
            f"required={requested_width}x{requested_height}@"
            f">={minimum_capture_fps:.1f}"
        )
        failures.append(reason)
        print(f"[CAMERA] Rejected {reason}")

    details = "; ".join(failures)
    raise SystemExit(
        f"[CAMERA] Could not negotiate camera {index} at "
        f"{requested_width}x{requested_height} {camera_config.get('fourcc', '')} "
        f"near {requested_fps:.1f} FPS. {details}. Close other camera apps and "
        "add lighting if auto exposure is lowering the frame rate."
    )


def observations_from_frame(detector: Detector, frame: np.ndarray) -> list[TagObservation]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    observations: list[TagObservation] = []
    for detection in detections:
        family = detection.tag_family
        if isinstance(family, bytes):
            family = family.decode("ascii", errors="replace")
        observations.append(
            TagObservation(
                tag_id=int(detection.tag_id),
                family=str(family),
                center_px=np.asarray(detection.center, dtype=np.float64).copy(),
                corners_px=np.asarray(detection.corners, dtype=np.float64).copy(),
                tag_homography=np.asarray(detection.homography, dtype=np.float64).copy(),
                decision_margin=float(detection.decision_margin),
                hamming=int(detection.hamming),
            )
        )
    return observations


def observation_quality_accepted(
    observation: TagObservation, config: dict[str, Any]
) -> bool:
    detector_config = config["detector"]
    maximum_hamming = int(detector_config.get("max_hamming", 0))
    minimum_margin = float(detector_config.get("minimum_decision_margin", 0.0))
    return (
        observation.hamming <= maximum_hamming
        and math.isfinite(observation.decision_margin)
        and observation.decision_margin >= minimum_margin
    )


def accepted_observations(
    observations: list[TagObservation], config: dict[str, Any]
) -> list[TagObservation]:
    return [
        observation
        for observation in observations
        if observation_quality_accepted(observation, config)
    ]


def configured_robot_pose_contract(config: dict[str, Any]) -> RobotPoseContract:
    tags = config["tags"]
    axis = np.asarray(tags.get("robot_front_axis_tip"), dtype=np.float64)
    if axis.shape != (2,) or not np.all(np.isfinite(axis)):
        raise CalibrationError("robot_front_axis_tip must contain two finite values")
    if float(np.linalg.norm(axis)) <= 1e-9:
        raise CalibrationError("robot_front_axis_tip must not be zero")

    raw_heading = tags.get("robot_heading_offset_deg")
    raw_offset = tags.get("tag_center_to_robot_origin_body_mm")
    if raw_heading is None or raw_offset is None:
        raise CalibrationError(
            "robot heading alignment and tag-center-to-origin offset must be measured"
        )
    heading_offset = float(raw_heading)
    offset = np.asarray(raw_offset, dtype=np.float64)
    if not math.isfinite(heading_offset):
        raise CalibrationError("robot_heading_offset_deg must be finite")
    if offset.shape != (2,) or not np.all(np.isfinite(offset)):
        raise CalibrationError(
            "tag_center_to_robot_origin_body_mm must be [forward_mm, left_mm]"
        )
    payload = {
        "front_axis_tip": np.round(axis, 9).tolist(),
        "heading_offset_deg": heading_offset,
        "tag_center_to_robot_origin_body_mm": np.round(offset, 9).tolist(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return RobotPoseContract(
        front_axis_tip=(float(axis[0]), float(axis[1])),
        heading_offset_deg=heading_offset,
        forward_offset_mm=float(offset[0]),
        left_offset_mm=float(offset[1]),
        contract_id=hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
    )


def frame_size_px(frame: np.ndarray) -> tuple[int, int]:
    if not isinstance(frame, np.ndarray) or frame.ndim < 2:
        raise ValueError("camera frame must be an image array")
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("camera frame size must be positive")
    return int(width), int(height)


def require_frame_size(
    frame: np.ndarray, expected_size_px: tuple[int, int] | None
) -> tuple[int, int]:
    actual = frame_size_px(frame)
    if expected_size_px is not None and actual != tuple(expected_size_px):
        raise RuntimeError(
            f"camera frame size changed from {tuple(expected_size_px)} to {actual}"
        )
    return actual


def load_map_contract(config: dict[str, Any]) -> MapContract:
    path = Path(config["_config_dir"]) / str(config["map"]["contract_file"])
    try:
        return MapContract.load(path)
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        raise SystemExit(f"[MAP] {error}") from error


def configured_reference_positions(
    config: dict[str, Any], map_contract: MapContract
) -> dict[int, np.ndarray]:
    try:
        return map_contract.reference_positions_mm(
            config["map"]["reference_tag_anchors"]
        )
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        raise SystemExit(f"[MAP] {error}") from error


def unique_reference_centers(
    observations: list[TagObservation],
    reference_ids: set[int],
    config: dict[str, Any],
) -> dict[int, np.ndarray]:
    accepted = [
        observation
        for observation in observations
        if observation_quality_accepted(observation, config)
        and observation.tag_id in reference_ids
    ]
    counts = Counter(observation.tag_id for observation in accepted)
    return {
        observation.tag_id: observation.center_px
        for observation in accepted
        if counts[observation.tag_id] == 1
    }


def draw_observation(
    frame: np.ndarray,
    observation: TagObservation,
    robot_id: int,
    reference_ids: set[int],
    axis_tip: list[float],
    quality_accepted: bool,
) -> tuple[np.ndarray | None, float | None]:
    is_robot = observation.tag_id == robot_id
    is_reference = observation.tag_id in reference_ids
    color = (
        (0, 0, 255)
        if not quality_accepted
        else (0, 255, 255)
        if is_robot
        else (255, 180, 0)
        if is_reference
        else (0, 255, 0)
    )
    corners_int = np.rint(observation.corners_px).astype(int)

    for index in range(4):
        point_a = tuple(corners_int[index])
        point_b = tuple(corners_int[(index + 1) % 4])
        cv2.line(frame, point_a, point_b, color, 2)
        cv2.circle(frame, point_a, 4, (255, 0, 255), -1)
        cv2.putText(
            frame,
            f"C{index}",
            (point_a[0] + 5, point_a[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    center = tuple(np.rint(observation.center_px).astype(int))
    cv2.circle(frame, center, 5, (0, 0, 255), -1)
    role = (
        "REJECT"
        if not quality_accepted
        else "ROBOT"
        if is_robot
        else "REF"
        if is_reference
        else "TAG"
    )
    cv2.putText(
        frame,
        f"{role} ID={observation.tag_id} H={observation.hamming} M={observation.decision_margin:.1f}",
        (center[0] + 10, center[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )

    if not is_robot or not quality_accepted:
        return None, None

    front = tag_axis_point(observation.tag_homography, axis_tip)
    heading = image_heading_degrees(observation.center_px, front)
    front_int = tuple(np.rint(front).astype(int))
    cv2.arrowedLine(frame, center, front_int, (0, 0, 255), 3, tipLength=0.25)
    cv2.putText(
        frame,
        f"pixel=({observation.center_px[0]:.1f},{observation.center_px[1]:.1f}) image_heading={heading:.1f} deg",
        (center[0] + 10, center[1] + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return front, heading


def annotate_frame(
    frame: np.ndarray,
    observations: list[TagObservation],
    config: dict[str, Any],
    map_contract: MapContract,
    map_homography: np.ndarray | None,
    measured_fps: float,
    calibration_state: str,
    calibration: PlanarCalibration | None,
    verification_count: int,
    verification_age_s: float | None,
    capture_fps: float | None = None,
    detection_ms: float | None = None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    output = frame.copy()
    tags_config = config["tags"]
    robot_id = int(tags_config["robot_id"])
    reference_ids = {int(tag_id) for tag_id in tags_config["reference_ids"]}
    try:
        pose_contract = configured_robot_pose_contract(config)
        axis_tip = list(pose_contract.front_axis_tip)
    except CalibrationError:
        pose_contract = None
        axis_tip = [float(value) for value in tags_config["robot_front_axis_tip"]]
    accepted = accepted_observations(observations, config)
    counts = Counter(observation.tag_id for observation in accepted)
    duplicate_ids = sorted(tag_id for tag_id, count in counts.items() if count > 1)
    robot_record: dict[str, Any] | None = None

    for observation in observations:
        front_pixel, image_heading = draw_observation(
            output,
            observation,
            robot_id,
            reference_ids,
            axis_tip,
            observation_quality_accepted(observation, config),
        )
        if (
            not observation_quality_accepted(observation, config)
            or observation.tag_id != robot_id
            or counts[robot_id] != 1
        ):
            continue
        robot_record = {
            "tag_id": robot_id,
            "center_px": [float(value) for value in observation.center_px],
            "image_heading_deg": float(image_heading),
            "hamming": observation.hamming,
            "decision_margin": observation.decision_margin,
        }
        if (
            map_homography is not None
            and front_pixel is not None
            and calibration is not None
            and pose_contract is not None
        ):
            tag_x_mm, tag_z_mm, raw_tag_heading_deg = map_pose_from_pixel_axis(
                map_homography, observation.center_px, front_pixel
            )
            transform = trace_tag_center_to_robot_origin(
                tag_x_mm,
                tag_z_mm,
                raw_tag_heading_deg,
                pose_contract.heading_offset_deg,
                pose_contract.forward_offset_mm,
                pose_contract.left_offset_mm,
            )
            heading_deg = transform.body_heading_deg
            x_mm, z_mm = transform.body_x_mm, transform.body_z_mm
            allowed_margin = float(config["tracking"]["allowed_map_margin_mm"])
            if not map_contract.contains_local_mm(x_mm, z_mm, allowed_margin):
                robot_record["metric_rejection"] = "OUTSIDE_MAP_ROI"
                continue
            server_pose = map_contract.local_mm_to_server(x_mm, z_mm)
            robot_record["map_pose"] = {
                "x_mm": x_mm,
                "z_mm": z_mm,
                "server_x": server_pose.x,
                "server_z": server_pose.z,
                "heading_deg": heading_deg,
                "calibration_id": calibration.calibration_id
                if calibration is not None
                else None,
                "calibration_rms_error_mm": calibration.quality.rms_error_mm
                if calibration is not None
                else None,
                "verification_age_s": float(verification_age_s or 0.0),
            }
            robot_record["transform_diagnostic"] = {
                "image_heading_deg": float(image_heading),
                "raw_tag_center_mm": {
                    "x_mm": transform.raw_tag_x_mm,
                    "z_mm": transform.raw_tag_z_mm,
                },
                "raw_tag_heading_deg": transform.raw_tag_heading_deg,
                "heading_offset_deg": transform.heading_offset_deg,
                "body_heading_deg": transform.body_heading_deg,
                "applied_tag_to_body_offset_body_mm": {
                    "forward_mm": transform.forward_offset_mm,
                    "left_mm": transform.left_offset_mm,
                },
                "body_center_mm": {
                    "x_mm": transform.body_x_mm,
                    "z_mm": transform.body_z_mm,
                },
            }
            center = tuple(np.rint(observation.center_px).astype(int))
            cv2.putText(
                output,
                f"MAP x={x_mm:.1f} z={z_mm:.1f} heading={heading_deg:.1f} deg",
                (center[0] + 10, center[1] + 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    age_text = (
        f" age={verification_age_s:.2f}s"
        if verification_age_s is not None
        else ""
    )
    performance_text = f"FPS PROC={measured_fps:.1f}"
    if capture_fps is not None:
        performance_text = f"FPS CAM={capture_fps:.1f} PROC={measured_fps:.1f}"
    if detection_ms is not None:
        performance_text += f" DET={detection_ms:.1f}ms"
    cv2.putText(
        output,
        f"{performance_text} | accepted {len(accepted)}/{len(observations)} | calibration {calibration_state}{age_text} refs={verification_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "q/ESC: quit | s: save frames | c: collect and lock calibration",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "LENS: NO INTRINSICS - VERIFY KNOWN-NODE ERROR ACROSS MAP",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )

    if duplicate_ids:
        warning = f"DUPLICATE TAG IDS: {duplicate_ids}"
        cv2.putText(
            output,
            warning,
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
    return output, robot_record


def draw_pose_estimate(
    frame: np.ndarray,
    estimate: PoseEstimate,
    map_contract: MapContract,
) -> None:
    color = (
        (0, 255, 0)
        if estimate.state.value == "MEASURED"
        else (0, 200, 255)
        if estimate.state.value == "HELD"
        else (0, 0, 255)
    )
    if estimate.pose is None:
        text = f"POSE {estimate.state.value}"
    else:
        server = map_contract.local_mm_to_server(
            estimate.pose.x_mm, estimate.pose.z_mm
        )
        text = (
            f"POSE {estimate.state.value} x={estimate.pose.x_mm:.1f}mm "
            f"z={estimate.pose.z_mm:.1f}mm heading={estimate.pose.heading_deg:.1f} "
            f"server=({server.x:.2f},{server.z:.2f})"
        )
    if estimate.measurement_age_ms is not None:
        text += f" age={estimate.measurement_age_ms:.0f}ms"
    cv2.putText(
        frame,
        text,
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def pose_estimate_record(
    estimate: PoseEstimate,
    map_contract: MapContract,
    transform_diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "state": estimate.state.value,
        "fresh": estimate.fresh,
        "measurement_age_ms": estimate.measurement_age_ms,
        "source_sequence": estimate.source_sequence,
        "calibration_id": estimate.calibration_id,
    }
    if estimate.pose is not None:
        server = map_contract.local_mm_to_server(
            estimate.pose.x_mm, estimate.pose.z_mm
        )
        record["pose"] = {
            "x_mm": estimate.pose.x_mm,
            "z_mm": estimate.pose.z_mm,
            "server_x": server.x,
            "server_z": server.z,
            "heading_deg": estimate.pose.heading_deg,
        }
    else:
        record["pose"] = None
    if estimate.state.value == "MEASURED" and transform_diagnostic is not None:
        record["transform_diagnostic"] = transform_diagnostic
    return record


def build_server_observation(
    estimate: PoseEstimate,
    *,
    calibration: PlanarCalibration,
    calibration_state: CalibrationUseState,
    verification: VerificationResult | None,
    verification_age_s: float | None,
    measurement: MetricMeasurement | None,
) -> VisionObservation:
    """Map a fail-closed tracker result to the canonical Server wire model."""

    tracking_state = VisionTrackingState[estimate.state.value]
    verification_state = VisionVerificationState[calibration_state.value]
    source_timestamp_us = (
        0
        if estimate.measured_at_s is None
        else max(0, int(round(estimate.measured_at_s * 1_000_000.0)))
    )
    reported_age_ms = (
        0
        if estimate.measurement_age_ms is None
        else min(0xFFFFFFFF, max(0, int(round(estimate.measurement_age_ms))))
    )

    fields = QUALITY_CALIBRATION_RMS_ERROR
    decision_margin = 0.0
    if tracking_state == VisionTrackingState.MEASURED and measurement is not None:
        fields |= QUALITY_DECISION_MARGIN
        decision_margin = float(measurement.decision_margin)

    reference_count = 0
    verification_rms = 0.0
    verification_max = 0.0
    verification_coverage = 0.0
    if verification is not None:
        fields |= QUALITY_VERIFICATION_REFERENCE_COUNT
        reference_count = int(verification.matched_count)
        if verification.rms_error_mm is not None:
            fields |= QUALITY_VERIFICATION_RMS_ERROR
            verification_rms = float(verification.rms_error_mm)
        if verification.max_error_mm is not None:
            fields |= QUALITY_VERIFICATION_MAX_ERROR
            verification_max = float(verification.max_error_mm)
        if verification.coverage_ratio is not None:
            fields |= QUALITY_VERIFICATION_COVERAGE
            verification_coverage = float(verification.coverage_ratio)

    verification_age_ms = 0
    if verification_age_s is not None:
        fields |= QUALITY_VERIFICATION_AGE
        verification_age_ms = min(
            0xFFFFFFFF,
            max(0, int(round(float(verification_age_s) * 1000.0))),
        )

    pose_values: dict[str, float | None] = {
        "x_mm": None,
        "z_mm": None,
        "heading_deg": None,
    }
    if estimate.pose is not None:
        pose_values = {
            "x_mm": float(estimate.pose.x_mm),
            "z_mm": float(estimate.pose.z_mm),
            "heading_deg": float(estimate.pose.heading_deg),
        }

    return VisionObservation(
        source_timestamp_us=source_timestamp_us,
        reported_age_ms=reported_age_ms,
        state=tracking_state,
        calibration_id=calibration.calibration_id,
        verification_state=verification_state,
        quality=VisionQuality(
            quality_fields=fields,
            decision_margin=decision_margin,
            calibration_rms_error_mm=float(calibration.quality.rms_error_mm),
            verification_reference_count=reference_count,
            verification_rms_error_mm=verification_rms,
            verification_max_error_mm=verification_max,
            verification_coverage_ratio=verification_coverage,
            verification_age_ms=verification_age_ms,
        ),
        **pose_values,
    )


def save_frames(raw_frame: np.ndarray, annotated_frame: np.ndarray) -> None:
    output_dir = BASE_DIR / "captures"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    raw_path = output_dir / f"{timestamp}_raw.png"
    annotated_path = output_dir / f"{timestamp}_annotated.png"
    raw_ok = cv2.imwrite(str(raw_path), raw_frame)
    annotated_ok = cv2.imwrite(str(annotated_path), annotated_frame)
    if not raw_ok or not annotated_ok:
        raise RuntimeError("capture image could not be written")
    print(f"[CAPTURE] {raw_path.name}, {annotated_path.name}")


def calibration_file_path(config: dict[str, Any]) -> Path:
    return Path(config["_config_dir"]) / str(config["calibration"]["file"])


def load_compatible_calibration_guard(
    config: dict[str, Any],
    map_contract: MapContract,
    reference_positions: dict[int, np.ndarray],
    image_size_px: tuple[int, int],
) -> LockedCalibrationGuard | None:
    path = calibration_file_path(config)
    if not path.exists():
        print(f"[CALIBRATION] No locked calibration at {path.name}")
        return None
    try:
        pose_contract = configured_robot_pose_contract(config)
        calibration = PlanarCalibration.load(path)
        validate_calibration_compatibility(
            calibration,
            map_name=map_contract.name,
            map_contract_id=map_contract.contract_id,
            pose_contract_id=pose_contract.contract_id,
            image_size_px=image_size_px,
            reference_plane_height_mm=config["map"]["reference_plane_height_mm"],
            robot_tag_height_mm=config["map"]["robot_tag_height_mm"],
            reference_map_mm=reference_positions,
            minimum_reference_tags=int(
                config["calibration"]["minimum_reference_tags"]
            ),
            minimum_inliers=int(config["calibration"]["minimum_inliers"]),
            maximum_error_mm=float(
                config["calibration"]["maximum_inlier_error_mm"]
            ),
            minimum_map_coverage_ratio=float(
                config["calibration"]["minimum_map_coverage_ratio"]
            ),
        )
    except CalibrationError as error:
        print(f"[CALIBRATION] Locked file rejected: {error}")
        return None
    print(
        f"[CALIBRATION] Loaded id={calibration.calibration_id}; "
        "waiting for fixed-reference verification"
    )
    calibration_config = config["calibration"]
    return LockedCalibrationGuard(
        calibration,
        verification_max_age_seconds=float(
            calibration_config["verification_max_age_seconds"]
        ),
        bad_frames_to_invalidate=int(
            calibration_config["verification_bad_frames_to_invalidate"]
        ),
    )


def run_single_image(
    image_path: Path, detector: Detector, config: dict[str, Any]
) -> None:
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"[VISION] Image could not be read: {image_path}")
    map_contract = load_map_contract(config)
    reference_positions = configured_reference_positions(config, map_contract)
    image_size = (frame.shape[1], frame.shape[0])
    guard = load_compatible_calibration_guard(
        config, map_contract, reference_positions, image_size
    )
    observations = observations_from_frame(detector, frame)
    reference_centers = unique_reference_centers(
        observations,
        set(reference_positions),
        config,
    )
    verification_age = None
    calibration = guard.calibration if guard is not None else None
    if guard is not None:
        verification = verify_locked_calibration(
            guard.calibration,
            reference_centers,
            minimum_tags=int(
                config["calibration"]["verification_minimum_tags"]
            ),
            maximum_error_mm=float(
                config["calibration"]["verification_maximum_error_mm"]
            ),
            minimum_coverage_ratio=float(
                config["calibration"]["verification_minimum_coverage_ratio"]
            ),
        )
        guard.update(0.0, verification)
        calibration_state = guard.state(0.0).value
        map_homography = guard.usable_homography(0.0)
        verification_age = guard.verification_age_s(0.0)
    else:
        calibration_state = "NO_CALIBRATION"
        map_homography = None

    annotated, robot_record = annotate_frame(
        frame,
        observations,
        config,
        map_contract,
        map_homography,
        0.0,
        calibration_state,
        calibration,
        len(reference_centers),
        verification_age,
    )
    print(f"[VISION] detections={len(observations)}")
    if robot_record is not None:
        print("[ROBOT] " + json.dumps(robot_record, ensure_ascii=False))
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_TITLE, annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_camera(
    detector: Detector, config: dict[str, Any], override_index: int | None
) -> None:
    capture, reported_camera = open_camera(config["camera"], override_index)
    frame_reader = LatestFrameReader(capture)
    display = config["display"]
    calibration_config = config["calibration"]
    map_contract = load_map_contract(config)
    reference_positions = configured_reference_positions(config, map_contract)
    server_settings = config["server"]
    server_config = VisionServerConfig(
        host=str(server_settings["host"]),
        port=int(server_settings["port"]),
        source_id=int(server_settings["source_id"]),
        agv_id=int(server_settings["agv_id"]),
        connect_timeout_seconds=float(server_settings["connect_timeout_seconds"]),
        reconnect_delay_seconds=float(server_settings["reconnect_delay_seconds"]),
    )
    server_enabled = bool(server_settings["enabled"])
    server_client: VisionServerClient | None = None
    image_size: tuple[int, int] | None = None
    guard: LockedCalibrationGuard | None = None
    collector: CalibrationCollector | None = None
    pose_tracker = PoseHoldTracker(
        float(config["tracking"]["hold_seconds"]),
        float(config["tracking"]["maximum_fresh_age_seconds"]),
    )
    console_period = float(display["console_period_seconds"])
    previous_capture_time: float | None = None
    previous_processing_time: float | None = None
    smoothed_capture_fps = float(reported_camera.get("measured_fps", 0.0))
    smoothed_processing_fps = 0.0
    last_console_time = 0.0
    last_detection_state: tuple[tuple[int, int], ...] | None = None
    last_pose_state: str | None = None
    last_source_sequence = 0

    try:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            WINDOW_TITLE,
            int(display["window_width"]),
            int(display["window_height"]),
        )
        frame_reader.start()
        while True:
            previous_source_sequence = last_source_sequence
            received = frame_reader.read_after(last_source_sequence)
            last_source_sequence = received.source_sequence
            frame = received.frame
            received_at_s = received.received_monotonic_s
            processing_started_s = time.perf_counter()

            current_size = require_frame_size(frame, image_size)
            if image_size is None:
                image_size = current_size
                reported_size = (
                    int(reported_camera["width"]),
                    int(reported_camera["height"]),
                )
                if reported_size != image_size:
                    print(
                        f"[CAMERA] Driver reported {reported_size}, "
                        f"actual frame is {image_size}; using actual frame size"
                    )
                guard = load_compatible_calibration_guard(
                    config, map_contract, reference_positions, image_size
                )

            if previous_capture_time is not None and previous_source_sequence > 0:
                capture_elapsed = received_at_s - previous_capture_time
                captured_frames = received.source_sequence - previous_source_sequence
                instantaneous_capture_fps = _event_rate(
                    captured_frames, capture_elapsed
                )
                smoothed_capture_fps = (
                    instantaneous_capture_fps
                    if smoothed_capture_fps == 0.0
                    else smoothed_capture_fps * 0.9
                    + instantaneous_capture_fps * 0.1
                )
            previous_capture_time = received_at_s

            if previous_processing_time is not None:
                processing_elapsed = processing_started_s - previous_processing_time
                instantaneous_processing_fps = _event_rate(1, processing_elapsed)
                smoothed_processing_fps = (
                    instantaneous_processing_fps
                    if smoothed_processing_fps == 0.0
                    else smoothed_processing_fps * 0.9
                    + instantaneous_processing_fps * 0.1
                )
            previous_processing_time = processing_started_s

            detection_started_s = time.perf_counter()
            observations = observations_from_frame(detector, frame)
            detection_ms = (time.perf_counter() - detection_started_s) * 1000.0
            reference_centers = unique_reference_centers(
                observations,
                set(reference_positions),
                config,
            )

            if collector is not None:
                collector.add(reference_centers)
                ready_count = len(collector.ready_tag_ids())
                if ready_count >= int(
                    calibration_config["minimum_reference_tags"]
                ):
                    try:
                        pose_contract = configured_robot_pose_contract(config)
                        calibration = build_planar_calibration(
                            collector.samples(),
                            reference_positions,
                            map_name=map_contract.name,
                            map_contract_id=map_contract.contract_id,
                            pose_contract_id=pose_contract.contract_id,
                            image_size_px=image_size,
                            reference_plane_height_mm=config["map"][
                                "reference_plane_height_mm"
                            ],
                            robot_tag_height_mm=config["map"][
                                "robot_tag_height_mm"
                            ],
                            minimum_samples_per_tag=int(
                                calibration_config["samples_per_reference"]
                            ),
                            minimum_reference_tags=int(
                                calibration_config["minimum_reference_tags"]
                            ),
                            minimum_inliers=int(
                                calibration_config["minimum_inliers"]
                            ),
                            ransac_threshold_mm=float(
                                calibration_config["ransac_threshold_mm"]
                            ),
                            maximum_inlier_error_mm=float(
                                calibration_config["maximum_inlier_error_mm"]
                            ),
                            map_bounds_mm=map_contract.local_bounds_mm(),
                            minimum_map_coverage_ratio=float(
                                calibration_config[
                                    "minimum_map_coverage_ratio"
                                ]
                            ),
                        )
                        lock_verification = verify_locked_calibration(
                            calibration,
                            reference_centers,
                            minimum_tags=int(
                                calibration_config[
                                    "verification_minimum_tags"
                                ]
                            ),
                            maximum_error_mm=float(
                                calibration_config[
                                    "verification_maximum_error_mm"
                                ]
                            ),
                            minimum_coverage_ratio=float(
                                calibration_config[
                                    "verification_minimum_coverage_ratio"
                                ]
                            ),
                        )
                        if not lock_verification.available or not lock_verification.passed:
                            raise CalibrationError(
                                "new calibration did not pass an independent "
                                "current-frame reference verification"
                            )
                        calibration.save(calibration_file_path(config))
                        guard = LockedCalibrationGuard(
                            calibration,
                            verification_max_age_seconds=float(
                                calibration_config[
                                    "verification_max_age_seconds"
                                ]
                            ),
                            bad_frames_to_invalidate=int(
                                calibration_config[
                                    "verification_bad_frames_to_invalidate"
                                ]
                            ),
                        )
                        guard.update(time.perf_counter(), lock_verification)
                        print(
                            f"[CALIBRATION] LOCKED id={calibration.calibration_id} "
                            f"refs={calibration.quality.reference_count} "
                            f"inliers={calibration.quality.inlier_count} "
                            f"rms={calibration.quality.rms_error_mm:.2f}mm "
                            f"max={calibration.quality.max_error_mm:.2f}mm"
                        )
                    except CalibrationError as error:
                        print(f"[CALIBRATION] Rejected: {error}")
                    collector = None

            verification: VerificationResult | None = None
            verification_age = None
            calibration_use_state: CalibrationUseState | None = None
            calibration = guard.calibration if guard is not None else None
            verification_now_s = time.perf_counter()
            if guard is not None:
                verification = verify_locked_calibration(
                    guard.calibration,
                    reference_centers,
                    minimum_tags=int(
                        calibration_config["verification_minimum_tags"]
                    ),
                    maximum_error_mm=float(
                        calibration_config["verification_maximum_error_mm"]
                    ),
                    minimum_coverage_ratio=float(
                        calibration_config[
                            "verification_minimum_coverage_ratio"
                        ]
                    ),
                )
                guard.update(verification_now_s, verification)
                calibration_use_state = guard.state(verification_now_s)
                calibration_state = calibration_use_state.value
                map_homography = guard.usable_homography(verification_now_s)
                verification_age = guard.verification_age_s(verification_now_s)
            elif collector is not None:
                ready = len(collector.ready_tag_ids())
                calibration_state = (
                    f"COLLECTING {ready}/"
                    f"{calibration_config['minimum_reference_tags']}"
                )
                map_homography = None
            elif len(reference_positions) < int(
                calibration_config["minimum_reference_tags"]
            ):
                calibration_state = "ANCHORS_NOT_CONFIGURED"
                map_homography = None
            else:
                calibration_state = "NO_CALIBRATION"
                map_homography = None

            active_calibration_id = (
                calibration.calibration_id if calibration is not None else None
            )
            if server_client is not None and (
                not server_enabled
                or active_calibration_id != server_client.calibration_id
            ):
                server_client.close()
                server_client = None
            if (
                server_enabled
                and calibration is not None
                and server_client is None
            ):
                server_client = VisionServerClient(
                    server_config,
                    map_contract_id=map_contract.contract_id,
                    pose_contract_id=calibration.pose_contract_id,
                    calibration_id=calibration.calibration_id,
                )
                server_client.start()

            annotated, robot_record = annotate_frame(
                frame,
                observations,
                config,
                map_contract,
                map_homography,
                smoothed_processing_fps,
                calibration_state,
                calibration,
                len(reference_centers),
                verification_age,
                capture_fps=smoothed_capture_fps,
                detection_ms=detection_ms,
            )

            measurement = None
            evaluation_now_s = time.perf_counter()
            if robot_record is not None and "map_pose" in robot_record:
                map_pose = robot_record["map_pose"]
                measurement = MetricMeasurement(
                    sequence=received.source_sequence,
                    received_monotonic_s=received_at_s,
                    pose=MetricPose(
                        float(map_pose["x_mm"]),
                        float(map_pose["z_mm"]),
                        float(map_pose["heading_deg"]),
                    ),
                    decision_margin=float(robot_record["decision_margin"]),
                    calibration_id=str(map_pose["calibration_id"]),
                    calibration_rms_error_mm=float(
                        map_pose["calibration_rms_error_mm"]
                    ),
                )
            estimate = pose_tracker.update(evaluation_now_s, measurement)
            draw_pose_estimate(annotated, estimate, map_contract)
            if (
                server_client is not None
                and calibration is not None
                and calibration_use_state is not None
            ):
                server_client.publish(
                    build_server_observation(
                        estimate,
                        calibration=calibration,
                        calibration_state=calibration_use_state,
                        verification=verification,
                        verification_age_s=verification_age,
                        measurement=measurement,
                    )
                )

            counts = Counter(
                observation.tag_id
                for observation in accepted_observations(observations, config)
            )
            detection_state = tuple(sorted(counts.items()))
            if detection_state != last_detection_state:
                last_detection_state = detection_state
                duplicate_ids = sorted(
                    tag_id for tag_id, count in counts.items() if count > 1
                )
                if duplicate_ids:
                    print(f"[VISION] DUPLICATE TAG IDS: {duplicate_ids}")

            pose_state_changed = estimate.state.value != last_pose_state
            if (
                pose_state_changed
                or evaluation_now_s - last_console_time >= console_period
            ):
                last_console_time = evaluation_now_s
                last_pose_state = estimate.state.value
                transform_diagnostic = (
                    robot_record.get("transform_diagnostic")
                    if robot_record is not None
                    else None
                )
                record = pose_estimate_record(
                    estimate,
                    map_contract,
                    transform_diagnostic,
                )
                record["monotonic_s"] = evaluation_now_s
                record["calibration_state"] = calibration_state
                print("[POSE] " + json.dumps(record, ensure_ascii=False))

            cv2.imshow(WINDOW_TITLE, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                save_frames(frame, annotated)
            if key == ord("c"):
                minimum_references = int(
                    calibration_config["minimum_reference_tags"]
                )
                if len(reference_positions) < minimum_references:
                    print(
                        "[CALIBRATION] Configure at least "
                        f"{minimum_references} physical reference anchors first"
                    )
                    continue
                try:
                    require_same_measurement_plane(
                        config["map"]["reference_plane_height_mm"],
                        config["map"]["robot_tag_height_mm"],
                    )
                    configured_robot_pose_contract(config)
                except CalibrationError as error:
                    print(f"[CALIBRATION] Cannot start: {error}")
                    continue
                collector = CalibrationCollector(
                    set(reference_positions),
                    int(calibration_config["samples_per_reference"]),
                )
                guard = None
                pose_tracker = PoseHoldTracker(
                    float(config["tracking"]["hold_seconds"]),
                    float(config["tracking"]["maximum_fresh_age_seconds"]),
                )
                print(
                    "[CALIBRATION] COLLECTING fixed references; "
                    "do not move camera or tags"
                )
    except KeyboardInterrupt:
        print("\n[VISION] Interrupted")
    finally:
        if server_client is not None:
            server_client.close()
        frame_reader.close()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    detector = build_detector(config)
    print(
        f"[VISION] family={config['detector']['families']} robot_id={config['tags']['robot_id']} "
        f"tag_size_mm={config['tags']['tag_size_mm']} "
        f"server={'ENABLED' if config['server']['enabled'] else 'DISABLED'}"
    )
    if args.image is not None:
        run_single_image(args.image.resolve(), detector, config)
    else:
        run_camera(detector, config, args.camera)


if __name__ == "__main__":
    main()
