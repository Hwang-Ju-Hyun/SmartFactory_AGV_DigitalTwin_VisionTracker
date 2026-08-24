"""Locked planar registration for metric AprilTag tracking.

Calibration and runtime tracking are intentionally separate. A fixed camera is
calibrated from several frames, the resulting transform is saved, and runtime
frames only verify that locked transform. This avoids injecting reference-tag
jitter into every robot measurement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import cv2

from vision_geometry import fit_pixel_to_map_homography, project_point


class CalibrationError(ValueError):
    pass


def _finite_point(value: Any, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise CalibrationError(f"{name} must contain two finite values")
    return point


def require_same_measurement_plane(
    reference_plane_height_mm: float | None,
    robot_tag_height_mm: float | None,
    tolerance_mm: float = 1.0,
) -> tuple[float, float]:
    """Fail closed when a floor homography would be applied to a raised tag."""

    if reference_plane_height_mm is None or robot_tag_height_mm is None:
        raise CalibrationError(
            "reference_plane_height_mm and robot_tag_height_mm must be measured"
        )
    reference = float(reference_plane_height_mm)
    robot = float(robot_tag_height_mm)
    if not np.isfinite(reference) or not np.isfinite(robot):
        raise CalibrationError("tag-plane heights must be finite")
    if abs(reference - robot) > float(tolerance_mm):
        raise CalibrationError(
            "reference tags and robot tag are on different height planes; "
            "parallax compensation is not calibrated"
        )
    return reference, robot


@dataclass(frozen=True)
class CalibrationQuality:
    rms_error_mm: float
    max_error_mm: float
    reference_count: int
    inlier_count: int
    map_coverage_ratio: float


@dataclass(frozen=True)
class VerificationResult:
    available: bool
    passed: bool
    matched_count: int
    rms_error_mm: float | None
    max_error_mm: float | None
    coverage_ratio: float | None


class CalibrationUseState(str, Enum):
    AWAITING_VERIFICATION = "AWAITING_VERIFICATION"
    VERIFIED = "VERIFIED"
    REFERENCES_MISSING = "REFERENCES_MISSING"
    MISMATCH = "MISMATCH"
    STALE = "STALE"
    INVALID = "INVALID"


class LockedCalibrationGuard:
    """Authorize a locked transform only while fixed references agree with it."""

    def __init__(
        self,
        calibration: "PlanarCalibration",
        *,
        verification_max_age_seconds: float,
        bad_frames_to_invalidate: int,
        verified_at_s: float | None = None,
    ):
        max_age = float(verification_max_age_seconds)
        if not np.isfinite(max_age) or max_age < 0.0:
            raise ValueError("verification max age must be finite and non-negative")
        if int(bad_frames_to_invalidate) <= 0:
            raise ValueError("bad_frames_to_invalidate must be positive")
        self.calibration = calibration
        self.max_age_seconds = max_age
        self.bad_frames_to_invalidate = int(bad_frames_to_invalidate)
        if verified_at_s is not None and not np.isfinite(float(verified_at_s)):
            raise ValueError("initial verification time must be finite")
        self.last_verified_s = (
            None if verified_at_s is None else float(verified_at_s)
        )
        self._last_clock_s = self.last_verified_s
        self.bad_frame_count = 0
        self.invalid = False
        self.last_result: VerificationResult | None = None

    def update(self, now_s: float, result: VerificationResult) -> None:
        now = self._validated_time(now_s)
        self.last_result = result
        if not result.available:
            return
        if result.passed:
            self.last_verified_s = now
            self.bad_frame_count = 0
            return
        self.bad_frame_count += 1
        if self.bad_frame_count >= self.bad_frames_to_invalidate:
            self.invalid = True

    def state(self, now_s: float) -> CalibrationUseState:
        now = self._validated_time(now_s)
        if self.invalid:
            return CalibrationUseState.INVALID
        if self.last_result is not None and not self.last_result.available:
            return CalibrationUseState.REFERENCES_MISSING
        if (
            self.last_result is not None
            and self.last_result.available
            and not self.last_result.passed
        ):
            return CalibrationUseState.MISMATCH
        if self.last_verified_s is None:
            return CalibrationUseState.AWAITING_VERIFICATION
        if now - self.last_verified_s > self.max_age_seconds:
            return CalibrationUseState.STALE
        return CalibrationUseState.VERIFIED

    def usable_homography(self, now_s: float) -> np.ndarray | None:
        if self.state(now_s) != CalibrationUseState.VERIFIED:
            return None
        return self.calibration.homography

    def verification_age_s(self, now_s: float) -> float | None:
        now = self._validated_time(now_s)
        if self.last_verified_s is None:
            return None
        return max(0.0, now - self.last_verified_s)

    def _validated_time(self, now_s: float) -> float:
        now = float(now_s)
        if not np.isfinite(now):
            raise ValueError("verification time must be finite")
        if self._last_clock_s is not None and now < self._last_clock_s:
            raise ValueError("verification time must be monotonic")
        self._last_clock_s = now
        return now


@dataclass(frozen=True)
class PlanarCalibration:
    calibration_id: str
    created_utc: str
    map_name: str
    map_contract_id: str
    pose_contract_id: str
    image_width: int
    image_height: int
    reference_plane_height_mm: float
    robot_tag_height_mm: float
    homography: np.ndarray
    reference_pixels: dict[int, np.ndarray]
    reference_map_mm: dict[int, np.ndarray]
    inlier_tag_ids: tuple[int, ...]
    quality: CalibrationQuality

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "calibration_id": self.calibration_id,
            "created_utc": self.created_utc,
            "map_name": self.map_name,
            "map_contract_id": self.map_contract_id,
            "pose_contract_id": self.pose_contract_id,
            "image_size_px": [self.image_width, self.image_height],
            "reference_plane_height_mm": self.reference_plane_height_mm,
            "robot_tag_height_mm": self.robot_tag_height_mm,
            "homography_pixel_to_local_mm": self.homography.tolist(),
            "reference_pixels": {
                str(tag_id): point.tolist()
                for tag_id, point in sorted(self.reference_pixels.items())
            },
            "reference_map_mm": {
                str(tag_id): point.tolist()
                for tag_id, point in sorted(self.reference_map_mm.items())
            },
            "inlier_tag_ids": list(self.inlier_tag_ids),
            "quality": {
                "rms_error_mm": self.quality.rms_error_mm,
                "max_error_mm": self.quality.max_error_mm,
                "reference_count": self.quality.reference_count,
                "inlier_count": self.quality.inlier_count,
                "map_coverage_ratio": self.quality.map_coverage_ratio,
            },
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlanarCalibration":
        if int(raw.get("schema_version", 0)) != 2:
            raise CalibrationError("unsupported calibration schema")
        image_size = np.asarray(raw["image_size_px"], dtype=np.int64)
        if image_size.shape != (2,) or np.any(image_size <= 0):
            raise CalibrationError("calibration image size is invalid")
        homography = np.asarray(
            raw["homography_pixel_to_local_mm"], dtype=np.float64
        )
        if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
            raise CalibrationError("calibration homography is invalid")
        if np.linalg.matrix_rank(homography) < 3:
            raise CalibrationError("calibration homography is singular")

        reference_pixels = {
            int(tag_id): _finite_point(point, f"reference pixel {tag_id}")
            for tag_id, point in raw["reference_pixels"].items()
        }
        reference_map_mm = {
            int(tag_id): _finite_point(point, f"reference map point {tag_id}")
            for tag_id, point in raw["reference_map_mm"].items()
        }
        if set(reference_pixels) != set(reference_map_mm):
            raise CalibrationError("calibration reference IDs do not match")

        quality_raw = raw["quality"]
        quality = CalibrationQuality(
            rms_error_mm=float(quality_raw["rms_error_mm"]),
            max_error_mm=float(quality_raw["max_error_mm"]),
            reference_count=int(quality_raw["reference_count"]),
            inlier_count=int(quality_raw["inlier_count"]),
            map_coverage_ratio=float(quality_raw["map_coverage_ratio"]),
        )
        if (
            not np.isfinite(quality.rms_error_mm)
            or not np.isfinite(quality.max_error_mm)
            or quality.rms_error_mm < 0.0
            or quality.max_error_mm < 0.0
            or quality.reference_count < 5
            or quality.inlier_count < 5
            or quality.inlier_count > quality.reference_count
            or quality.rms_error_mm > quality.max_error_mm + 1e-9
            or not 0.0 <= quality.map_coverage_ratio <= 1.0
        ):
            raise CalibrationError("calibration quality values are invalid")
        inlier_tag_ids = tuple(int(value) for value in raw["inlier_tag_ids"])
        if len(set(inlier_tag_ids)) != len(inlier_tag_ids):
            raise CalibrationError("calibration inlier IDs are duplicated")
        if not set(inlier_tag_ids).issubset(reference_map_mm):
            raise CalibrationError("calibration inlier IDs are unknown")
        if quality.reference_count != len(reference_map_mm):
            raise CalibrationError("calibration reference count is inconsistent")
        if quality.inlier_count != len(inlier_tag_ids):
            raise CalibrationError("calibration inlier count is inconsistent")
        if set(inlier_tag_ids) != set(reference_map_mm):
            raise CalibrationError("calibration contains rejected reference anchors")
        _assert_non_collinear(
            np.asarray(list(reference_pixels.values())), "reference pixels"
        )
        _assert_non_collinear(
            np.asarray(list(reference_map_mm.values())), "reference map points"
        )
        required_text = {
            "calibration ID": raw["calibration_id"],
            "creation time": raw["created_utc"],
            "map name": raw["map_name"],
            "map contract ID": raw["map_contract_id"],
            "pose contract ID": raw["pose_contract_id"],
        }
        for name, value in required_text.items():
            if not isinstance(value, str) or not value:
                raise CalibrationError(f"{name} must not be empty")
        heights = np.asarray(
            [raw["reference_plane_height_mm"], raw["robot_tag_height_mm"]],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(heights)):
            raise CalibrationError("calibration tag-plane heights are invalid")
        calibration = cls(
            calibration_id=str(raw["calibration_id"]),
            created_utc=str(raw["created_utc"]),
            map_name=str(raw["map_name"]),
            map_contract_id=str(raw["map_contract_id"]),
            pose_contract_id=str(raw["pose_contract_id"]),
            image_width=int(image_size[0]),
            image_height=int(image_size[1]),
            reference_plane_height_mm=float(raw["reference_plane_height_mm"]),
            robot_tag_height_mm=float(raw["robot_tag_height_mm"]),
            homography=homography,
            reference_pixels=reference_pixels,
            reference_map_mm=reference_map_mm,
            inlier_tag_ids=inlier_tag_ids,
            quality=quality,
        )
        expected_id = _calibration_identifier(
            _calibration_identity_payload(
                created_utc=calibration.created_utc,
                map_name=calibration.map_name,
                map_contract_id=calibration.map_contract_id,
                pose_contract_id=calibration.pose_contract_id,
                image_size_px=(calibration.image_width, calibration.image_height),
                reference_plane_height_mm=calibration.reference_plane_height_mm,
                robot_tag_height_mm=calibration.robot_tag_height_mm,
                homography=calibration.homography,
                reference_pixels=calibration.reference_pixels,
                reference_map_mm=calibration.reference_map_mm,
                inlier_tag_ids=calibration.inlier_tag_ids,
                quality=calibration.quality,
            )
        )
        if calibration.calibration_id != expected_id:
            raise CalibrationError("calibration integrity check failed")
        return calibration

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(self.to_dict(), output, ensure_ascii=False, indent=2)
            output.write("\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "PlanarCalibration":
        try:
            with path.open("r", encoding="utf-8") as source:
                return cls.from_dict(json.load(source))
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
        ) as error:
            raise CalibrationError(f"cannot load calibration {path}: {error}") from error


class CalibrationCollector:
    def __init__(self, reference_tag_ids: set[int], samples_per_tag: int):
        if samples_per_tag <= 0:
            raise ValueError("samples_per_tag must be positive")
        self.reference_tag_ids = {int(tag_id) for tag_id in reference_tag_ids}
        self.samples_per_tag = int(samples_per_tag)
        self._samples: dict[int, list[np.ndarray]] = {
            tag_id: [] for tag_id in self.reference_tag_ids
        }

    def add(self, centers_by_tag_id: dict[int, np.ndarray]) -> None:
        for tag_id, center in centers_by_tag_id.items():
            if tag_id not in self._samples:
                continue
            values = _finite_point(center, f"tag {tag_id} center")
            samples = self._samples[tag_id]
            if len(samples) < self.samples_per_tag:
                samples.append(values.copy())

    def ready_tag_ids(self) -> set[int]:
        return {
            tag_id
            for tag_id, samples in self._samples.items()
            if len(samples) >= self.samples_per_tag
        }

    def sample_counts(self) -> dict[int, int]:
        return {tag_id: len(samples) for tag_id, samples in self._samples.items()}

    def samples(self) -> dict[int, list[np.ndarray]]:
        return {
            tag_id: [sample.copy() for sample in samples]
            for tag_id, samples in self._samples.items()
        }


def _assert_non_collinear(points: np.ndarray, name: str) -> None:
    centered = points - np.mean(points, axis=0)
    if np.linalg.matrix_rank(centered, tol=1e-6) < 2:
        raise CalibrationError(f"{name} are collinear")


def _map_intersection_coverage_ratio(
    reference_points: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> float:
    min_x, max_x, min_z, max_z = bounds
    map_area = (max_x - min_x) * (max_z - min_z)
    reference_hull = cv2.convexHull(
        np.asarray(reference_points, dtype=np.float32)
    )
    map_rectangle = np.asarray(
        [
            [min_x, min_z],
            [max_x, min_z],
            [max_x, max_z],
            [min_x, max_z],
        ],
        dtype=np.float32,
    )
    intersection_area, _ = cv2.intersectConvexConvex(
        reference_hull, map_rectangle
    )
    ratio = float(intersection_area) / float(map_area)
    return min(1.0, max(0.0, ratio))


def _calibration_identifier(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _calibration_identity_payload(
    *,
    created_utc: str,
    map_name: str,
    map_contract_id: str,
    pose_contract_id: str,
    image_size_px: tuple[int, int],
    reference_plane_height_mm: float,
    robot_tag_height_mm: float,
    homography: np.ndarray,
    reference_pixels: dict[int, np.ndarray],
    reference_map_mm: dict[int, np.ndarray],
    inlier_tag_ids: tuple[int, ...],
    quality: CalibrationQuality,
) -> dict[str, Any]:
    return {
        "created_utc": str(created_utc),
        "map_name": str(map_name),
        "map_contract_id": str(map_contract_id),
        "pose_contract_id": str(pose_contract_id),
        "image_size_px": [int(image_size_px[0]), int(image_size_px[1])],
        "reference_plane_height_mm": float(reference_plane_height_mm),
        "robot_tag_height_mm": float(robot_tag_height_mm),
        "homography": np.round(homography, decimals=12).tolist(),
        "reference_pixels": {
            str(tag_id): np.round(np.asarray(point), decimals=9).tolist()
            for tag_id, point in sorted(reference_pixels.items())
        },
        "reference_map_mm": {
            str(tag_id): np.round(np.asarray(point), decimals=9).tolist()
            for tag_id, point in sorted(reference_map_mm.items())
        },
        "inlier_tag_ids": [int(tag_id) for tag_id in inlier_tag_ids],
        "quality": {
            "rms_error_mm": float(quality.rms_error_mm),
            "max_error_mm": float(quality.max_error_mm),
            "reference_count": int(quality.reference_count),
            "inlier_count": int(quality.inlier_count),
            "map_coverage_ratio": float(quality.map_coverage_ratio),
        },
    }


def build_planar_calibration(
    samples_by_tag_id: dict[int, list[np.ndarray]],
    reference_map_mm: dict[int, np.ndarray],
    *,
    map_name: str,
    map_contract_id: str,
    pose_contract_id: str,
    image_size_px: tuple[int, int],
    reference_plane_height_mm: float | None,
    robot_tag_height_mm: float | None,
    minimum_samples_per_tag: int = 20,
    minimum_reference_tags: int = 5,
    minimum_inliers: int = 5,
    ransac_threshold_mm: float = 10.0,
    maximum_inlier_error_mm: float = 15.0,
    map_bounds_mm: tuple[float, float, float, float],
    minimum_map_coverage_ratio: float = 0.5,
) -> PlanarCalibration:
    if int(minimum_samples_per_tag) <= 0:
        raise CalibrationError("minimum samples per tag must be positive")
    if int(minimum_reference_tags) < 5:
        raise CalibrationError(
            "at least five reference tags are required to detect a bad anchor"
        )
    if int(minimum_inliers) < 5:
        raise CalibrationError("at least five calibration inliers are required")
    if not str(map_contract_id) or not str(pose_contract_id):
        raise CalibrationError("map and pose contract IDs must not be empty")
    maximum_error = float(maximum_inlier_error_mm)
    if not np.isfinite(maximum_error) or maximum_error <= 0.0:
        raise CalibrationError("maximum inlier error must be positive and finite")
    bounds = np.asarray(map_bounds_mm, dtype=np.float64)
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
        raise CalibrationError("map bounds must contain four finite values")
    min_x, max_x, min_z, max_z = (float(value) for value in bounds)
    map_area = (max_x - min_x) * (max_z - min_z)
    if map_area <= 0.0:
        raise CalibrationError("map bounds must enclose a positive area")
    minimum_coverage = float(minimum_map_coverage_ratio)
    if not np.isfinite(minimum_coverage) or not 0.0 <= minimum_coverage <= 1.0:
        raise CalibrationError("minimum map coverage ratio must be between 0 and 1")
    reference_height, robot_height = require_same_measurement_plane(
        reference_plane_height_mm, robot_tag_height_mm
    )
    width, height = (int(image_size_px[0]), int(image_size_px[1]))
    if width <= 0 or height <= 0:
        raise CalibrationError("image size must be positive")

    medians: dict[int, np.ndarray] = {}
    for tag_id, map_point in reference_map_mm.items():
        _finite_point(map_point, f"reference map point {tag_id}")
        raw_samples = samples_by_tag_id.get(tag_id, [])
        if len(raw_samples) < minimum_samples_per_tag:
            continue
        samples = np.asarray(raw_samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1:] != (2,):
            raise CalibrationError(f"tag {tag_id} samples must have shape (N, 2)")
        if not np.all(np.isfinite(samples)):
            raise CalibrationError(f"tag {tag_id} samples are not finite")
        medians[tag_id] = np.median(samples, axis=0)

    tag_ids = sorted(medians)
    if len(tag_ids) < int(minimum_reference_tags):
        raise CalibrationError(
            f"need {minimum_reference_tags} calibrated references; got {len(tag_ids)}"
        )
    pixel_points = np.asarray([medians[tag_id] for tag_id in tag_ids])
    map_points = np.asarray([reference_map_mm[tag_id] for tag_id in tag_ids])
    _assert_non_collinear(pixel_points, "reference pixels")
    _assert_non_collinear(map_points, "reference map points")
    map_coverage_ratio = _map_intersection_coverage_ratio(
        map_points, (min_x, max_x, min_z, max_z)
    )
    if map_coverage_ratio < minimum_coverage:
        raise CalibrationError(
            f"reference map coverage {map_coverage_ratio:.3f} is below "
            f"{minimum_coverage:.3f}; spread anchors across the map"
        )

    try:
        homography, inliers = fit_pixel_to_map_homography(
            pixel_points, map_points, ransac_threshold_mm
        )
    except ValueError as error:
        raise CalibrationError(str(error)) from error
    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < int(minimum_inliers):
        raise CalibrationError(
            f"need {minimum_inliers} calibration inliers; got {inlier_count}"
        )
    if inlier_count != len(tag_ids):
        rejected = [
            tag_id for tag_id, accepted in zip(tag_ids, inliers) if not accepted
        ]
        raise CalibrationError(
            f"reference anchor outliers detected {rejected}; fix placement and recalibrate"
        )

    projected = np.asarray(
        [project_point(homography, point) for point in pixel_points]
    )
    errors = np.linalg.norm(projected - map_points, axis=1)
    inlier_errors = errors[inliers]
    rms_error = float(np.sqrt(np.mean(np.square(inlier_errors))))
    max_error = float(np.max(inlier_errors))
    if max_error > maximum_error:
        raise CalibrationError(
            f"calibration max inlier error {max_error:.2f} mm exceeds "
            f"{maximum_error:.2f} mm"
        )

    homography_scale = float(np.max(np.abs(homography)))
    if not np.isfinite(homography_scale) or homography_scale <= 1e-15:
        raise CalibrationError("calibration homography scale is invalid")
    normalized_h = homography / homography_scale
    inlier_tag_ids = tuple(
        tag_id for tag_id, accepted in zip(tag_ids, inliers) if accepted
    )
    calibration_reference_map = {
        tag_id: np.asarray(reference_map_mm[tag_id], dtype=np.float64).copy()
        for tag_id in tag_ids
    }
    calibration_reference_pixels = {
        tag_id: medians[tag_id].copy() for tag_id in tag_ids
    }
    quality = CalibrationQuality(
        rms_error_mm=rms_error,
        max_error_mm=max_error,
        reference_count=len(tag_ids),
        inlier_count=inlier_count,
        map_coverage_ratio=map_coverage_ratio,
    )
    created_utc = datetime.now(timezone.utc).isoformat()
    identity_payload = _calibration_identity_payload(
        created_utc=created_utc,
        map_name=str(map_name),
        map_contract_id=str(map_contract_id),
        pose_contract_id=str(pose_contract_id),
        image_size_px=(width, height),
        reference_plane_height_mm=reference_height,
        robot_tag_height_mm=robot_height,
        homography=normalized_h,
        reference_pixels=calibration_reference_pixels,
        reference_map_mm=calibration_reference_map,
        inlier_tag_ids=inlier_tag_ids,
        quality=quality,
    )
    return PlanarCalibration(
        calibration_id=_calibration_identifier(identity_payload),
        created_utc=created_utc,
        map_name=str(map_name),
        map_contract_id=str(map_contract_id),
        pose_contract_id=str(pose_contract_id),
        image_width=width,
        image_height=height,
        reference_plane_height_mm=reference_height,
        robot_tag_height_mm=robot_height,
        homography=normalized_h,
        reference_pixels=calibration_reference_pixels,
        reference_map_mm=calibration_reference_map,
        inlier_tag_ids=inlier_tag_ids,
        quality=quality,
    )


def validate_calibration_compatibility(
    calibration: PlanarCalibration,
    *,
    map_name: str,
    map_contract_id: str,
    pose_contract_id: str,
    image_size_px: tuple[int, int],
    reference_plane_height_mm: float | None,
    robot_tag_height_mm: float | None,
    reference_map_mm: dict[int, np.ndarray],
    minimum_reference_tags: int,
    minimum_inliers: int,
    maximum_error_mm: float,
    minimum_map_coverage_ratio: float,
) -> None:
    reference_height, robot_height = require_same_measurement_plane(
        reference_plane_height_mm, robot_tag_height_mm
    )
    policy_error = float(maximum_error_mm)
    policy_coverage = float(minimum_map_coverage_ratio)
    if not np.isfinite(policy_error) or policy_error <= 0.0:
        raise CalibrationError("current maximum error policy is invalid")
    if not np.isfinite(policy_coverage) or not 0.0 <= policy_coverage <= 1.0:
        raise CalibrationError("current map coverage policy is invalid")
    if calibration.map_name != str(map_name):
        raise CalibrationError("calibration belongs to a different map")
    if calibration.map_contract_id != str(map_contract_id):
        raise CalibrationError("map contract changed after calibration")
    if calibration.pose_contract_id != str(pose_contract_id):
        raise CalibrationError("robot pose alignment changed after calibration")
    if (calibration.image_width, calibration.image_height) != tuple(image_size_px):
        raise CalibrationError("camera resolution differs from the calibration")
    if abs(calibration.reference_plane_height_mm - reference_height) > 1.0:
        raise CalibrationError("reference-tag height differs from the calibration")
    if abs(calibration.robot_tag_height_mm - robot_height) > 1.0:
        raise CalibrationError("robot-tag height differs from the calibration")
    if calibration.quality.reference_count < int(minimum_reference_tags):
        raise CalibrationError("calibration has too few references for current policy")
    if calibration.quality.inlier_count < int(minimum_inliers):
        raise CalibrationError("calibration has too few inliers for current policy")
    if calibration.quality.inlier_count != calibration.quality.reference_count:
        raise CalibrationError("calibration contains rejected reference anchors")
    if calibration.quality.map_coverage_ratio < policy_coverage:
        raise CalibrationError("calibration map coverage is below current policy")
    if (
        calibration.quality.rms_error_mm > policy_error
        or calibration.quality.max_error_mm > policy_error
    ):
        raise CalibrationError("calibration error exceeds current policy")
    for tag_id, saved_point in calibration.reference_map_mm.items():
        if tag_id not in reference_map_mm:
            raise CalibrationError(f"reference tag {tag_id} is no longer configured")
        if not np.allclose(saved_point, reference_map_mm[tag_id], atol=1e-6):
            raise CalibrationError(f"reference tag {tag_id} map position changed")


def verify_locked_calibration(
    calibration: PlanarCalibration,
    current_reference_pixels: dict[int, np.ndarray],
    *,
    minimum_tags: int = 4,
    maximum_error_mm: float = 20.0,
    minimum_coverage_ratio: float = 0.25,
) -> VerificationResult:
    if int(minimum_tags) < 4:
        raise ValueError("verification requires at least four reference tags")
    coverage_threshold = float(minimum_coverage_ratio)
    if not np.isfinite(coverage_threshold) or not 0.0 <= coverage_threshold <= 1.0:
        raise ValueError("verification coverage ratio must be between 0 and 1")
    error_threshold = float(maximum_error_mm)
    if not np.isfinite(error_threshold) or error_threshold <= 0.0:
        raise ValueError("verification maximum error must be positive and finite")
    verification_ids = set(calibration.inlier_tag_ids)
    matched = sorted(
        set(current_reference_pixels).intersection(verification_ids)
    )
    if len(matched) < int(minimum_tags):
        return VerificationResult(False, False, len(matched), None, None, None)

    full_points = np.asarray(
        [calibration.reference_map_mm[tag_id] for tag_id in calibration.inlier_tag_ids],
        dtype=np.float32,
    )
    matched_points = np.asarray(
        [calibration.reference_map_mm[tag_id] for tag_id in matched],
        dtype=np.float32,
    )
    full_area = float(cv2.contourArea(cv2.convexHull(full_points)))
    matched_area = float(cv2.contourArea(cv2.convexHull(matched_points)))
    if full_area <= 1e-6:
        raise CalibrationError("saved calibration references have no map coverage")
    coverage_ratio = matched_area / full_area
    if matched_area <= 1e-6 or coverage_ratio < coverage_threshold:
        return VerificationResult(
            available=True,
            passed=False,
            matched_count=len(matched),
            rms_error_mm=None,
            max_error_mm=None,
            coverage_ratio=coverage_ratio,
        )

    errors = []
    for tag_id in matched:
        projected = project_point(
            calibration.homography, current_reference_pixels[tag_id]
        )
        errors.append(
            float(np.linalg.norm(projected - calibration.reference_map_mm[tag_id]))
        )
    rms_error = float(np.sqrt(np.mean(np.square(errors))))
    max_error = float(np.max(errors))
    return VerificationResult(
        available=True,
        passed=max_error <= error_threshold,
        matched_count=len(matched),
        rms_error_mm=rms_error,
        max_error_mm=max_error,
        coverage_ratio=coverage_ratio,
    )
