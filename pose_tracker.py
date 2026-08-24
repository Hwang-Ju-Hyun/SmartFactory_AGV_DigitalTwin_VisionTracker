"""Fresh/stale/lost semantics for metric VisionTracker measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class PoseState(str, Enum):
    MEASURED = "MEASURED"
    HELD = "HELD"
    LOST = "LOST"


@dataclass(frozen=True)
class MetricPose:
    x_mm: float
    z_mm: float
    heading_deg: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.x_mm, self.z_mm, self.heading_deg)
        ):
            raise ValueError("metric pose values must be finite")


@dataclass(frozen=True)
class MetricMeasurement:
    sequence: int
    received_monotonic_s: float
    pose: MetricPose
    decision_margin: float
    calibration_id: str
    calibration_rms_error_mm: float

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("measurement sequence must not be negative")
        if not math.isfinite(self.received_monotonic_s):
            raise ValueError("measurement timestamp must be finite")
        if not math.isfinite(self.decision_margin):
            raise ValueError("decision margin must be finite")
        if not math.isfinite(self.calibration_rms_error_mm):
            raise ValueError("calibration error must be finite")
        if not self.calibration_id:
            raise ValueError("calibration ID must not be empty")


@dataclass(frozen=True)
class PoseEstimate:
    state: PoseState
    pose: MetricPose | None
    measurement_age_ms: float | None
    measured_at_s: float | None
    source_sequence: int | None
    calibration_id: str | None
    fresh: bool


class PoseHoldTracker:
    """Hold a measurement briefly without pretending it is fresh.

    No interpolation or motion prediction is performed. A held pose is only a
    display continuity aid; downstream code can always distinguish it from a
    current camera measurement through ``state`` and ``fresh``.
    """

    def __init__(
        self,
        hold_seconds: float = 0.20,
        maximum_fresh_age_seconds: float = 0.10,
    ):
        if not math.isfinite(hold_seconds) or hold_seconds < 0.0:
            raise ValueError("hold_seconds must be finite and non-negative")
        if (
            not math.isfinite(maximum_fresh_age_seconds)
            or maximum_fresh_age_seconds < 0.0
        ):
            raise ValueError(
                "maximum_fresh_age_seconds must be finite and non-negative"
            )
        self.hold_seconds = float(hold_seconds)
        self.maximum_fresh_age_seconds = float(maximum_fresh_age_seconds)
        self._last_measurement: MetricMeasurement | None = None
        self._last_observed_measurement_s: float | None = None
        self._last_update_s: float | None = None

    def update(
        self,
        now_s: float,
        measurement: MetricMeasurement | None,
    ) -> PoseEstimate:
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError("tracker time must be finite")
        if self._last_update_s is not None and now < self._last_update_s:
            raise ValueError("tracker time must be monotonic")
        self._last_update_s = now

        if measurement is not None:
            if measurement.received_monotonic_s > now + 1e-9:
                raise ValueError("measurement timestamp cannot be in the future")
            if (
                self._last_observed_measurement_s is not None
                and measurement.received_monotonic_s
                < self._last_observed_measurement_s
            ):
                raise ValueError("measurement timestamps must be monotonic")
            self._last_observed_measurement_s = measurement.received_monotonic_s

            measurement_age_s = max(
                0.0, now - measurement.received_monotonic_s
            )
            if (
                measurement_age_s
                <= self.maximum_fresh_age_seconds + 1e-12
            ):
                self._last_measurement = measurement
                return PoseEstimate(
                    state=PoseState.MEASURED,
                    pose=measurement.pose,
                    measurement_age_ms=measurement_age_s * 1000.0,
                    measured_at_s=measurement.received_monotonic_s,
                    source_sequence=measurement.sequence,
                    calibration_id=measurement.calibration_id,
                    fresh=True,
                )

        if self._last_measurement is None:
            return PoseEstimate(
                state=PoseState.LOST,
                pose=None,
                measurement_age_ms=None,
                measured_at_s=None,
                source_sequence=None,
                calibration_id=None,
                fresh=False,
            )

        age_s = max(0.0, now - self._last_measurement.received_monotonic_s)
        age_ms = age_s * 1000.0
        if age_s <= self.hold_seconds + 1e-12:
            return PoseEstimate(
                state=PoseState.HELD,
                pose=self._last_measurement.pose,
                measurement_age_ms=age_ms,
                measured_at_s=self._last_measurement.received_monotonic_s,
                source_sequence=self._last_measurement.sequence,
                calibration_id=self._last_measurement.calibration_id,
                fresh=False,
            )

        return PoseEstimate(
            state=PoseState.LOST,
            pose=None,
            measurement_age_ms=age_ms,
            measured_at_s=self._last_measurement.received_monotonic_s,
            source_sequence=self._last_measurement.sequence,
            calibration_id=self._last_measurement.calibration_id,
            fresh=False,
        )
