"""Non-blocking VisionTracker -> Server observation transport.

The camera loop only publishes immutable snapshots.  A background thread owns
the TCP connection, performs the Vision HELLO handshake, coalesces pending
snapshots to the newest one, and reconnects without blocking frame processing.
"""

from __future__ import annotations

import math
import secrets
import socket
import struct
import threading
import time
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Callable


PROTOCOL_VERSION = 1
VISION_HELLO_PACKET_ID = 600
VISION_HELLO_ACK_PACKET_ID = 601
VISION_OBSERVATION_PACKET_ID = 602
MAX_IDENTITY_BYTES = 64
MAX_FRAME_BYTES = 65535


class VisionTrackingState(IntEnum):
    MEASURED = 1
    HELD = 2
    LOST = 3


class VisionVerificationState(IntEnum):
    UNKNOWN = 0
    VERIFIED = 1
    AWAITING_VERIFICATION = 2
    REFERENCES_MISSING = 3
    MISMATCH = 4
    STALE = 5
    INVALID = 6


QUALITY_DECISION_MARGIN = 1 << 0
QUALITY_CALIBRATION_RMS_ERROR = 1 << 1
QUALITY_VERIFICATION_REFERENCE_COUNT = 1 << 2
QUALITY_VERIFICATION_RMS_ERROR = 1 << 3
QUALITY_VERIFICATION_MAX_ERROR = 1 << 4
QUALITY_VERIFICATION_COVERAGE = 1 << 5
QUALITY_VERIFICATION_AGE = 1 << 6
KNOWN_QUALITY_FIELDS = (
    QUALITY_DECISION_MARGIN
    | QUALITY_CALIBRATION_RMS_ERROR
    | QUALITY_VERIFICATION_REFERENCE_COUNT
    | QUALITY_VERIFICATION_RMS_ERROR
    | QUALITY_VERIFICATION_MAX_ERROR
    | QUALITY_VERIFICATION_COVERAGE
    | QUALITY_VERIFICATION_AGE
)
KNOWN_HELLO_REJECTION_REASONS = frozenset(range(8))


@dataclass(frozen=True)
class VisionServerConfig:
    host: str
    port: int
    source_id: int
    agv_id: int
    connect_timeout_seconds: float = 1.0
    reconnect_delay_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not str(self.host).strip():
            raise ValueError("Server host must not be empty")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("Server port must be between 1 and 65535")
        if not 1 <= int(self.source_id) <= 0xFFFFFFFF:
            raise ValueError("Vision source_id must be a non-zero uint32")
        if not 1 <= int(self.agv_id) <= 0xFFFFFFFF:
            raise ValueError("Vision agv_id must be a non-zero uint32")
        for name, value in (
            ("connect_timeout_seconds", self.connect_timeout_seconds),
            ("reconnect_delay_seconds", self.reconnect_delay_seconds),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class VisionQuality:
    quality_fields: int = 0
    decision_margin: float = 0.0
    calibration_rms_error_mm: float = 0.0
    verification_reference_count: int = 0
    verification_rms_error_mm: float = 0.0
    verification_max_error_mm: float = 0.0
    verification_coverage_ratio: float = 0.0
    verification_age_ms: int = 0

    def __post_init__(self) -> None:
        fields = int(self.quality_fields)
        if fields < 0 or fields > 0xFFFF or fields & ~KNOWN_QUALITY_FIELDS:
            raise ValueError("quality_fields contains an unknown bit")
        if not 0 <= int(self.verification_reference_count) <= 0xFFFF:
            raise ValueError("verification_reference_count must be a uint16")
        if not 0 <= int(self.verification_age_ms) <= 0xFFFFFFFF:
            raise ValueError("verification_age_ms must be a uint32")

        non_negative_values = (
            self.decision_margin,
            self.calibration_rms_error_mm,
            self.verification_rms_error_mm,
            self.verification_max_error_mm,
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in non_negative_values):
            raise ValueError("quality errors and margins must be finite and non-negative")
        coverage = float(self.verification_coverage_ratio)
        if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ValueError("verification_coverage_ratio must be between 0 and 1")
        if (
            fields & QUALITY_VERIFICATION_RMS_ERROR
            and fields & QUALITY_VERIFICATION_MAX_ERROR
            and float(self.verification_max_error_mm)
            < float(self.verification_rms_error_mm)
        ):
            raise ValueError("verification maximum error cannot be below RMS error")

        optional_values = (
            (QUALITY_DECISION_MARGIN, float(self.decision_margin)),
            (QUALITY_CALIBRATION_RMS_ERROR, float(self.calibration_rms_error_mm)),
            (
                QUALITY_VERIFICATION_REFERENCE_COUNT,
                int(self.verification_reference_count),
            ),
            (QUALITY_VERIFICATION_RMS_ERROR, float(self.verification_rms_error_mm)),
            (QUALITY_VERIFICATION_MAX_ERROR, float(self.verification_max_error_mm)),
            (QUALITY_VERIFICATION_COVERAGE, coverage),
            (QUALITY_VERIFICATION_AGE, int(self.verification_age_ms)),
        )
        if any(not fields & bit and value != 0 for bit, value in optional_values):
            raise ValueError("quality value without its quality_fields bit must be zero")


@dataclass(frozen=True)
class VisionObservation:
    source_timestamp_us: int
    reported_age_ms: int
    state: VisionTrackingState
    calibration_id: str
    verification_state: VisionVerificationState
    quality: VisionQuality = VisionQuality()
    x_mm: float | None = None
    z_mm: float | None = None
    heading_deg: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= int(self.source_timestamp_us) <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("source_timestamp_us must be a uint64")
        if not 0 <= int(self.reported_age_ms) <= 0xFFFFFFFF:
            raise ValueError("reported_age_ms must be a uint32")
        try:
            state = VisionTrackingState(self.state)
            VisionVerificationState(self.verification_state)
        except ValueError as error:
            raise ValueError("unknown Vision state") from error
        _encode_identity(self.calibration_id)

        pose_values = (self.x_mm, self.z_mm, self.heading_deg)
        has_pose = all(value is not None for value in pose_values)
        has_partial_pose = any(value is not None for value in pose_values) and not has_pose
        if has_partial_pose:
            raise ValueError("Vision pose must contain x, z, and heading together")
        if (state in (VisionTrackingState.MEASURED, VisionTrackingState.HELD)) != has_pose:
            raise ValueError("MEASURED/HELD require a pose and LOST forbids one")
        if has_pose and not all(math.isfinite(float(value)) for value in pose_values):
            raise ValueError("Vision pose values must be finite")
        if has_pose and not -180.0 <= float(self.heading_deg) < 180.0:
            raise ValueError("heading_deg must be in [-180, 180)")


@dataclass(frozen=True)
class VisionHelloAck:
    accepted: bool
    rejection_reason: int
    source_id: int
    session_id: int


@dataclass(frozen=True)
class _QueuedObservation:
    observation: VisionObservation
    published_monotonic_s: float

    def observation_for_send(self, now_s: float) -> VisionObservation:
        # Round queue dwell upward so transport delay is never hidden by
        # millisecond conversion, including when this item is retried.
        queued_age_ms = int(
            math.ceil(max(0.0, float(now_s) - self.published_monotonic_s) * 1000.0)
        )
        reported_age_ms = min(
            0xFFFFFFFF,
            int(self.observation.reported_age_ms) + queued_age_ms,
        )
        if reported_age_ms == self.observation.reported_age_ms:
            return self.observation
        return replace(self.observation, reported_age_ms=reported_age_ms)


def _encode_identity(value: str) -> bytes:
    try:
        encoded = str(value).encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("Vision identity must contain visible ASCII") from error
    if not 1 <= len(encoded) <= MAX_IDENTITY_BYTES:
        raise ValueError("Vision identity must contain 1-64 bytes")
    if any(byte < 0x21 or byte > 0x7E for byte in encoded):
        raise ValueError("Vision identity must contain visible ASCII without spaces")
    return struct.pack("<H", len(encoded)) + encoded


def _encode_frame(packet_id: int, agv_id: int, sequence: int, payload: bytes) -> bytes:
    body = struct.pack("<HII", int(packet_id), int(agv_id), int(sequence)) + payload
    frame_size = 2 + len(body)
    if frame_size > MAX_FRAME_BYTES:
        raise ValueError("Vision frame is too large")
    return struct.pack("<H", frame_size) + body


def encode_vision_hello_frame(
    *,
    source_id: int,
    session_id: int,
    map_contract_id: str,
    pose_contract_id: str,
) -> bytes:
    if not 1 <= int(source_id) <= 0xFFFFFFFF:
        raise ValueError("source_id must be a non-zero uint32")
    if not 1 <= int(session_id) <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("session_id must be a non-zero uint64")
    payload = struct.pack("<HIQ", PROTOCOL_VERSION, source_id, session_id)
    payload += _encode_identity(map_contract_id)
    payload += _encode_identity(pose_contract_id)
    # HELLO sequence is intentionally zero. Observation transport sequence is
    # independent and starts at one after this handshake is accepted.
    return _encode_frame(VISION_HELLO_PACKET_ID, 0, 0, payload)


def encode_vision_observation_frame(
    *, agv_id: int, sequence: int, observation: VisionObservation
) -> bytes:
    if not 1 <= int(agv_id) <= 0xFFFFFFFF:
        raise ValueError("agv_id must be a non-zero uint32")
    if not 1 <= int(sequence) <= 0xFFFFFFFF:
        raise ValueError("observation sequence must be a non-zero uint32")
    state = VisionTrackingState(observation.state)
    payload = struct.pack(
        "<QIB",
        int(observation.source_timestamp_us),
        int(observation.reported_age_ms),
        int(state),
    )
    if state in (VisionTrackingState.MEASURED, VisionTrackingState.HELD):
        payload += struct.pack(
            "<fff",
            float(observation.x_mm),
            float(observation.z_mm),
            float(observation.heading_deg),
        )
    payload += _encode_identity(observation.calibration_id)
    quality = observation.quality
    payload += struct.pack(
        "<BHffHfffI",
        int(observation.verification_state),
        int(quality.quality_fields),
        float(quality.decision_margin),
        float(quality.calibration_rms_error_mm),
        int(quality.verification_reference_count),
        float(quality.verification_rms_error_mm),
        float(quality.verification_max_error_mm),
        float(quality.verification_coverage_ratio),
        int(quality.verification_age_ms),
    )
    return _encode_frame(VISION_OBSERVATION_PACKET_ID, agv_id, sequence, payload)


def decode_vision_hello_ack_frame(frame: bytes) -> VisionHelloAck:
    if len(frame) < 2:
        raise ValueError("Vision HELLO_ACK frame is truncated")
    (declared_size,) = struct.unpack_from("<H", frame, 0)
    if declared_size != len(frame):
        raise ValueError("Vision HELLO_ACK frame size does not match")
    expected_size = 2 + struct.calcsize("<HIIHBHIQ")
    if len(frame) != expected_size:
        raise ValueError("Vision HELLO_ACK has an unexpected payload size")
    (
        packet_id,
        agv_id,
        sequence,
        protocol_version,
        accepted,
        rejection_reason,
        source_id,
        session_id,
    ) = struct.unpack_from("<HIIHBHIQ", frame, 2)
    if packet_id != VISION_HELLO_ACK_PACKET_ID or agv_id != 0 or sequence != 0:
        raise ValueError("unexpected Vision HELLO_ACK header")
    if protocol_version != PROTOCOL_VERSION:
        raise ValueError("Vision HELLO_ACK protocol version mismatch")
    if accepted not in (0, 1):
        raise ValueError("Vision HELLO_ACK accepted flag is invalid")
    if rejection_reason not in KNOWN_HELLO_REJECTION_REASONS:
        raise ValueError("Vision HELLO_ACK rejection reason is invalid")
    if (accepted == 1 and rejection_reason != 0) or (
        accepted == 0 and rejection_reason == 0
    ):
        raise ValueError("Vision HELLO_ACK rejection state is inconsistent")
    return VisionHelloAck(bool(accepted), rejection_reason, source_id, session_id)


def _receive_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Server closed the Vision connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _receive_frame(sock: socket.socket) -> bytes:
    size_prefix = _receive_exact(sock, 2)
    (frame_size,) = struct.unpack("<H", size_prefix)
    if frame_size < 12:
        raise ConnectionError("Server sent an invalid Vision frame size")
    return size_prefix + _receive_exact(sock, frame_size - 2)


class VisionServerClient:
    """Latest-only background sender for one locked calibration identity."""

    def __init__(
        self,
        config: VisionServerConfig,
        *,
        map_contract_id: str,
        pose_contract_id: str,
        calibration_id: str,
        session_id: int | None = None,
        log: Callable[[str], None] = print,
    ):
        self.config = config
        _encode_identity(map_contract_id)
        _encode_identity(pose_contract_id)
        _encode_identity(calibration_id)
        self.map_contract_id = map_contract_id
        self.pose_contract_id = pose_contract_id
        self.calibration_id = calibration_id
        generated_session = secrets.randbits(64) if session_id is None else int(session_id)
        self.session_id = generated_session or 1
        if not 1 <= self.session_id <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("session_id must be a non-zero uint64")
        self._log = log
        self._condition = threading.Condition()
        self._latest: _QueuedObservation | None = None
        self._stop_requested = False
        self._started = False
        self._next_sequence = 1
        self._thread = threading.Thread(
            target=self._run,
            name="vision-server-sender",
            daemon=True,
        )

    def start(self) -> None:
        with self._condition:
            if self._started:
                raise RuntimeError("Vision Server client was already started")
            self._started = True
        self._thread.start()

    def publish(self, observation: VisionObservation) -> None:
        if observation.calibration_id != self.calibration_id:
            raise ValueError("observation calibration ID differs from sender lock")
        with self._condition:
            if not self._started:
                raise RuntimeError("Vision Server client has not been started")
            if self._stop_requested:
                return
            # Overwrite instead of queueing camera history. The Server needs the
            # latest state, not stale frames accumulated during a disconnect.
            # Retaining this timestamp makes any surviving latest item age while
            # the connection or sender is delayed.
            self._latest = _QueuedObservation(observation, time.perf_counter())
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()
        if self._started:
            self._thread.join(timeout=max(2.0, self.config.connect_timeout_seconds + 1.0))

    def _take_latest(self) -> _QueuedObservation | None:
        with self._condition:
            while self._latest is None and not self._stop_requested:
                self._condition.wait(timeout=0.5)
            if self._stop_requested:
                return None
            latest = self._latest
            self._latest = None
            return latest

    def _retry_wait(self) -> bool:
        with self._condition:
            self._condition.wait_for(
                lambda: self._stop_requested,
                timeout=self.config.reconnect_delay_seconds,
            )
            return self._stop_requested

    def _connect(self) -> socket.socket:
        sock = socket.create_connection(
            (self.config.host, self.config.port),
            timeout=self.config.connect_timeout_seconds,
        )
        sock.settimeout(self.config.connect_timeout_seconds)
        try:
            sock.sendall(
                encode_vision_hello_frame(
                    source_id=self.config.source_id,
                    session_id=self.session_id,
                    map_contract_id=self.map_contract_id,
                    pose_contract_id=self.pose_contract_id,
                )
            )
            ack = decode_vision_hello_ack_frame(_receive_frame(sock))
            if ack.source_id != self.config.source_id or ack.session_id != self.session_id:
                raise ConnectionError("Server ACK identity does not match this Vision session")
            if not ack.accepted:
                raise ConnectionError(
                    f"Server rejected Vision HELLO (reason={ack.rejection_reason})"
                )
            return sock
        except Exception:
            sock.close()
            raise

    def _run(self) -> None:
        sock: socket.socket | None = None
        connected_logged = False
        while True:
            with self._condition:
                if self._stop_requested:
                    break
            if sock is None:
                try:
                    sock = self._connect()
                    connected_logged = True
                    self._log(
                        f"[SERVER] Vision source connected to "
                        f"{self.config.host}:{self.config.port} "
                        f"source={self.config.source_id} agv={self.config.agv_id}"
                    )
                except (OSError, ValueError, ConnectionError) as error:
                    if connected_logged:
                        self._log(f"[SERVER] Vision connection lost: {error}")
                    else:
                        self._log(f"[SERVER] Vision connection pending: {error}")
                    connected_logged = False
                    if self._retry_wait():
                        break
                    continue

            queued = self._take_latest()
            if queued is None:
                break
            sequence = self._next_sequence
            self._next_sequence += 1
            if self._next_sequence > 0xFFFFFFFF:
                # A process would need to sustain 30 FPS for over four years to
                # reach this. Reusing a sequence is less safe than stopping.
                self._log("[SERVER] Vision transport sequence exhausted; stopping sender")
                break
            try:
                observation = queued.observation_for_send(time.perf_counter())
                sock.sendall(
                    encode_vision_observation_frame(
                        agv_id=self.config.agv_id,
                        sequence=sequence,
                        observation=observation,
                    )
                )
            except (OSError, ValueError) as error:
                self._log(f"[SERVER] Vision send failed; reconnecting: {error}")
                try:
                    sock.close()
                finally:
                    sock = None
                # Preserve the newest unsent state. If a newer frame arrived
                # while sendall failed, it remains preferable to this one.
                with self._condition:
                    if self._latest is None:
                        self._latest = queued

        if sock is not None:
            sock.close()
