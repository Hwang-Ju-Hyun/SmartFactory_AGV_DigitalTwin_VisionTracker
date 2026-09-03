"""Estimate a tag-to-axle offset from stationary-rotation Vision poses.

This tool is intentionally read-only. It consumes ``[POSE]`` console records
produced by ``vision_tracker_preview.py`` and never writes vision_config.json or
the locked calibration. Only fresh MEASURED records with transform diagnostics
participate in the calculation; HELD and LOST records are ignored.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class RotationSample:
    calibration_id: str
    raw_tag_x_mm: float
    raw_tag_z_mm: float
    body_heading_deg: float
    applied_forward_mm: float
    applied_left_mm: float
    body_x_mm: float
    body_z_mm: float


@dataclass(frozen=True)
class RotationDiagnostic:
    sample_count: int
    calibration_id: str
    heading_span_deg: float
    applied_forward_mm: float
    applied_left_mm: float
    candidate_forward_mm: float
    candidate_left_mm: float
    candidate_offset_change_mm: float
    fitted_center_x_mm: float
    fitted_center_z_mm: float
    current_center_rms_movement_mm: float
    current_center_max_movement_mm: float
    candidate_center_rms_residual_mm: float
    candidate_center_max_residual_mm: float
    residual_ratio: float
    classification: str
    interpretation: str


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def pose_record_from_line(line: str) -> dict[str, Any] | None:
    """Parse a plain JSON line or a preview ``[POSE] {...}`` line."""

    text = line.strip()
    if not text:
        return None
    if text.startswith("[POSE]"):
        text = text[len("[POSE]") :].strip()
    elif not text.startswith("{"):
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def measured_rotation_samples(
    lines: Iterable[str], calibration_id: str | None = None
) -> list[RotationSample]:
    """Extract only MEASURED transform records, excluding HELD and LOST."""

    samples: list[RotationSample] = []
    for line in lines:
        record = pose_record_from_line(line)
        if record is None or record.get("state") != "MEASURED":
            continue
        record_calibration_id = str(record.get("calibration_id") or "")
        if calibration_id is not None and record_calibration_id != calibration_id:
            continue
        transform = record.get("transform_diagnostic")
        if not isinstance(transform, dict):
            continue
        raw_center = transform.get("raw_tag_center_mm")
        applied = transform.get("applied_tag_to_body_offset_body_mm")
        body_center = transform.get("body_center_mm")
        if not all(isinstance(value, dict) for value in (raw_center, applied, body_center)):
            continue
        samples.append(
            RotationSample(
                calibration_id=record_calibration_id,
                raw_tag_x_mm=_finite_float(raw_center.get("x_mm"), "raw tag x"),
                raw_tag_z_mm=_finite_float(raw_center.get("z_mm"), "raw tag z"),
                body_heading_deg=_finite_float(
                    transform.get("body_heading_deg"), "body heading"
                ),
                applied_forward_mm=_finite_float(
                    applied.get("forward_mm"), "applied forward offset"
                ),
                applied_left_mm=_finite_float(
                    applied.get("left_mm"), "applied left offset"
                ),
                body_x_mm=_finite_float(body_center.get("x_mm"), "body x"),
                body_z_mm=_finite_float(body_center.get("z_mm"), "body z"),
            )
        )
    return samples


def _heading_span_degrees(headings_deg: np.ndarray) -> float:
    angles = np.mod(headings_deg, 360.0)
    ordered = np.sort(angles)
    if len(ordered) < 2:
        return 0.0
    gaps = np.diff(np.concatenate((ordered, ordered[:1] + 360.0)))
    return float(360.0 - np.max(gaps))


def _center_scatter(points: np.ndarray) -> tuple[float, float]:
    center = np.mean(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    rms = float(np.sqrt(np.mean(np.square(radii))))
    pairwise = points[:, None, :] - points[None, :, :]
    maximum = float(np.max(np.linalg.norm(pairwise, axis=2)))
    return rms, maximum


def diagnose_stationary_rotation(
    samples: list[RotationSample], minimum_heading_span_deg: float = 45.0
) -> RotationDiagnostic:
    """Fit fixed center C and body-frame offset O in ``C = tag + R O``."""

    if len(samples) < 4:
        raise ValueError("at least four MEASURED samples are required")
    calibration_ids = {sample.calibration_id for sample in samples}
    if len(calibration_ids) != 1 or not next(iter(calibration_ids)):
        raise ValueError("samples must share one non-empty calibration ID")
    applied_offsets = np.asarray(
        [[sample.applied_forward_mm, sample.applied_left_mm] for sample in samples]
    )
    if not np.allclose(applied_offsets, applied_offsets[0], atol=1e-9, rtol=0.0):
        raise ValueError("samples use inconsistent applied offsets")

    headings = np.asarray([sample.body_heading_deg for sample in samples])
    radians = np.deg2rad(headings)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    design = np.empty((len(samples) * 2, 4), dtype=np.float64)
    target = np.empty(len(samples) * 2, dtype=np.float64)
    for index, (sample, cos_heading, sin_heading) in enumerate(
        zip(samples, cosine, sine)
    ):
        design[index * 2] = (1.0, 0.0, -cos_heading, sin_heading)
        design[index * 2 + 1] = (0.0, 1.0, -sin_heading, -cos_heading)
        target[index * 2] = sample.raw_tag_x_mm
        target[index * 2 + 1] = sample.raw_tag_z_mm
    if np.linalg.matrix_rank(design) < 4:
        raise ValueError("headings do not provide enough rotation diversity")
    solution, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    center_x, center_z, candidate_forward, candidate_left = solution

    raw_centers = np.asarray(
        [[sample.raw_tag_x_mm, sample.raw_tag_z_mm] for sample in samples]
    )
    candidate_offsets_world = np.column_stack(
        (
            candidate_forward * cosine - candidate_left * sine,
            candidate_forward * sine + candidate_left * cosine,
        )
    )
    candidate_centers = raw_centers + candidate_offsets_world
    current_centers = np.asarray(
        [[sample.body_x_mm, sample.body_z_mm] for sample in samples]
    )
    current_rms, current_max = _center_scatter(current_centers)
    candidate_rms, candidate_max = _center_scatter(candidate_centers)
    residual_ratio = candidate_rms / current_rms if current_rms > 1e-9 else 0.0
    offset_change = float(
        np.linalg.norm(
            np.asarray([candidate_forward, candidate_left]) - applied_offsets[0]
        )
    )
    heading_span = _heading_span_degrees(headings)

    if heading_span < float(minimum_heading_span_deg):
        classification = "INSUFFICIENT_HEADING_SPAN"
        interpretation = "회전각 범위가 부족하여 offset과 실제 이동을 구분할 수 없음"
    elif current_rms <= 3.0 and candidate_rms <= 3.0:
        classification = "CURRENT_OFFSET_CONSISTENT"
        interpretation = "현재 offset으로 계산한 중심 이동이 이미 측정 잡음 수준임"
    elif offset_change >= 3.0 and residual_ratio <= 0.4 and candidate_rms <= 10.0:
        classification = "OFFSET_ERROR_LIKELY"
        interpretation = "회전각에 따른 원/호 성분이 단일 offset 보정으로 크게 감소함"
    elif candidate_rms >= 8.0 and residual_ratio >= 0.65:
        classification = "IRREGULAR_MOTION_LIKELY"
        interpretation = "단일 offset으로 설명되지 않는 불규칙 이동; 슬립/caster/하중 영향 가능"
    else:
        classification = "INCONCLUSIVE"
        interpretation = "offset 오차와 실제 이동이 함께 있거나 표본이 충분히 분리되지 않음"

    return RotationDiagnostic(
        sample_count=len(samples),
        calibration_id=next(iter(calibration_ids)),
        heading_span_deg=heading_span,
        applied_forward_mm=float(applied_offsets[0, 0]),
        applied_left_mm=float(applied_offsets[0, 1]),
        candidate_forward_mm=float(candidate_forward),
        candidate_left_mm=float(candidate_left),
        candidate_offset_change_mm=offset_change,
        fitted_center_x_mm=float(center_x),
        fitted_center_z_mm=float(center_z),
        current_center_rms_movement_mm=current_rms,
        current_center_max_movement_mm=current_max,
        candidate_center_rms_residual_mm=candidate_rms,
        candidate_center_max_residual_mm=candidate_max,
        residual_ratio=residual_ratio,
        classification=classification,
        interpretation=interpretation,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Vision preview console log")
    parser.add_argument(
        "--calibration-id",
        help="Use only records produced by this calibration ID",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    return parser


def main() -> int:
    args = _parser().parse_args()
    with args.log.open("r", encoding="utf-8", errors="replace") as handle:
        samples = measured_rotation_samples(handle, args.calibration_id)
    try:
        result = diagnose_stationary_rotation(samples)
    except ValueError as error:
        raise SystemExit(f"[ROTATION_DIAGNOSTIC] {error}") from error
    values = asdict(result)
    if args.json:
        print(json.dumps(values, ensure_ascii=False, indent=2))
    else:
        print(f"samples_used={result.sample_count} state=MEASURED_ONLY")
        print(
            f"calibration_id={result.calibration_id} "
            f"heading_span_deg={result.heading_span_deg:.2f}"
        )
        print(
            "applied_offset_body_mm="
            f"[{result.applied_forward_mm:.3f}, {result.applied_left_mm:.3f}]"
        )
        print(
            "candidate_offset_body_mm="
            f"[{result.candidate_forward_mm:.3f}, {result.candidate_left_mm:.3f}] "
            f"change_mm={result.candidate_offset_change_mm:.3f}"
        )
        print(
            f"fixed_center_mm=[{result.fitted_center_x_mm:.3f}, "
            f"{result.fitted_center_z_mm:.3f}]"
        )
        print(
            f"current_center_movement_mm=rms:{result.current_center_rms_movement_mm:.3f} "
            f"max:{result.current_center_max_movement_mm:.3f}"
        )
        print(
            f"candidate_residual_mm=rms:{result.candidate_center_rms_residual_mm:.3f} "
            f"max:{result.candidate_center_max_residual_mm:.3f} "
            f"ratio:{result.residual_ratio:.3f}"
        )
        print(f"classification={result.classification}")
        print(f"interpretation={result.interpretation}")
        print("config_updated=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
