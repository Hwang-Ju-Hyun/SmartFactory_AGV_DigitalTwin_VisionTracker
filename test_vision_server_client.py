import socket
import struct
import threading
import unittest
from dataclasses import replace
from unittest.mock import patch

from vision_server_client import (
    PROTOCOL_VERSION,
    QUALITY_CALIBRATION_RMS_ERROR,
    QUALITY_DECISION_MARGIN,
    VisionObservation,
    VisionQuality,
    VisionServerClient,
    VisionServerConfig,
    VisionTrackingState,
    VisionVerificationState,
    decode_vision_hello_ack_frame,
    encode_vision_hello_frame,
    encode_vision_observation_frame,
)


def receive_exact(sock: socket.socket, byte_count: int) -> bytes:
    output = b""
    while len(output) < byte_count:
        chunk = sock.recv(byte_count - len(output))
        if not chunk:
            raise ConnectionError("test peer closed")
        output += chunk
    return output


def receive_frame(sock: socket.socket) -> bytes:
    prefix = receive_exact(sock, 2)
    (size,) = struct.unpack("<H", prefix)
    return prefix + receive_exact(sock, size - 2)


def accepted_ack(source_id: int, session_id: int) -> bytes:
    body = struct.pack(
        "<HIIHBHIQ",
        601,
        0,
        0,
        PROTOCOL_VERSION,
        1,
        0,
        source_id,
        session_id,
    )
    return struct.pack("<H", len(body) + 2) + body


def observation(x_mm: float = 12.5) -> VisionObservation:
    return VisionObservation(
        source_timestamp_us=1_234_567,
        reported_age_ms=7,
        state=VisionTrackingState.MEASURED,
        calibration_id="calibration-test",
        verification_state=VisionVerificationState.VERIFIED,
        quality=VisionQuality(
            quality_fields=(
                QUALITY_DECISION_MARGIN | QUALITY_CALIBRATION_RMS_ERROR
            ),
            decision_margin=55.0,
            calibration_rms_error_mm=2.5,
        ),
        x_mm=x_mm,
        z_mm=-3.25,
        heading_deg=90.0,
    )


class VisionProtocolEncodingTest(unittest.TestCase):
    def test_hello_matches_server_field_order_and_little_endian_u64(self):
        frame = encode_vision_hello_frame(
            source_id=7,
            session_id=0x0102030405060708,
            map_contract_id="map-id",
            pose_contract_id="pose-id",
        )
        self.assertEqual(struct.unpack_from("<H", frame, 0)[0], len(frame))
        packet_id, agv_id, sequence = struct.unpack_from("<HII", frame, 2)
        self.assertEqual((packet_id, agv_id, sequence), (600, 0, 0))
        version, source_id, low, high = struct.unpack_from("<HIII", frame, 12)
        self.assertEqual(version, 1)
        self.assertEqual(source_id, 7)
        self.assertEqual(low, 0x05060708)
        self.assertEqual(high, 0x01020304)

    def test_measured_observation_contains_pose_and_fixed_quality_scalars(self):
        frame = encode_vision_observation_frame(
            agv_id=1,
            sequence=9,
            observation=observation(),
        )
        packet_id, agv_id, sequence = struct.unpack_from("<HII", frame, 2)
        self.assertEqual((packet_id, agv_id, sequence), (602, 1, 9))
        timestamp, age, state = struct.unpack_from("<QIB", frame, 12)
        self.assertEqual((timestamp, age, state), (1_234_567, 7, 1))
        x_mm, z_mm, heading = struct.unpack_from("<fff", frame, 25)
        self.assertAlmostEqual(x_mm, 12.5)
        self.assertAlmostEqual(z_mm, -3.25)
        self.assertAlmostEqual(heading, 90.0)

    def test_lost_observation_omits_pose_bytes(self):
        lost = VisionObservation(
            source_timestamp_us=0,
            reported_age_ms=0,
            state=VisionTrackingState.LOST,
            calibration_id="calibration-test",
            verification_state=VisionVerificationState.AWAITING_VERIFICATION,
        )
        measured_frame = encode_vision_observation_frame(
            agv_id=1, sequence=1, observation=observation()
        )
        lost_frame = encode_vision_observation_frame(
            agv_id=1, sequence=2, observation=lost
        )
        measured_identity_offset = 25 + 12
        lost_identity_offset = 25
        self.assertEqual(
            struct.unpack_from("<H", measured_frame, measured_identity_offset)[0],
            len("calibration-test"),
        )
        self.assertEqual(
            struct.unpack_from("<H", lost_frame, lost_identity_offset)[0],
            len("calibration-test"),
        )
        self.assertEqual(len(measured_frame) - len(lost_frame), 12)

    def test_held_observation_keeps_pose_and_uses_held_wire_state(self):
        held = replace(
            observation(),
            reported_age_ms=150,
            state=VisionTrackingState.HELD,
            verification_state=VisionVerificationState.STALE,
        )
        frame = encode_vision_observation_frame(
            agv_id=1, sequence=3, observation=held
        )
        self.assertEqual(struct.unpack_from("<B", frame, 24)[0], 2)
        self.assertAlmostEqual(struct.unpack_from("<f", frame, 25)[0], 12.5)

    def test_ack_rejects_identity_mismatch_at_client_boundary(self):
        ack = decode_vision_hello_ack_frame(accepted_ack(1, 99))
        self.assertTrue(ack.accepted)
        self.assertEqual(ack.source_id, 1)
        self.assertEqual(ack.session_id, 99)

    def test_ack_rejects_unknown_rejection_reason(self):
        body = struct.pack(
            "<HIIHBHIQ", 601, 0, 0, PROTOCOL_VERSION, 0, 99, 1, 2
        )
        frame = struct.pack("<H", len(body) + 2) + body
        with self.assertRaisesRegex(ValueError, "reason"):
            decode_vision_hello_ack_frame(frame)

    def test_quality_value_without_presence_bit_is_rejected(self):
        with self.assertRaises(ValueError):
            VisionQuality(decision_margin=1.0)


class VisionServerClientIntegrationTest(unittest.TestCase):
    def test_fragmented_ack_and_latest_only_background_send(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        allow_ack = threading.Event()
        received = []
        failure = []

        def fake_server() -> None:
            try:
                connection, _ = listener.accept()
                with connection:
                    hello = receive_frame(connection)
                    packet_id, agv_id, sequence = struct.unpack_from(
                        "<HII", hello, 2
                    )
                    self.assertEqual((packet_id, agv_id, sequence), (600, 0, 0))
                    allow_ack.wait(2.0)
                    ack = accepted_ack(1, 0x1234)
                    connection.sendall(ack[:1])
                    connection.sendall(ack[1:8])
                    connection.sendall(ack[8:])
                    received.append(receive_frame(connection))
            except Exception as error:  # surfaced in the main test thread
                failure.append(error)
            finally:
                listener.close()

        server_thread = threading.Thread(target=fake_server, daemon=True)
        server_thread.start()
        logs = []
        client = VisionServerClient(
            VisionServerConfig(
                host="127.0.0.1",
                port=port,
                source_id=1,
                agv_id=1,
                connect_timeout_seconds=1.0,
                reconnect_delay_seconds=0.05,
            ),
            map_contract_id="map-contract",
            pose_contract_id="pose-contract",
            calibration_id="calibration-test",
            session_id=0x1234,
            log=logs.append,
        )
        try:
            client.start()
            client.publish(observation(10.0))
            client.publish(observation(20.0))
            client.publish(observation(30.0))
            allow_ack.set()
            server_thread.join(3.0)
            self.assertFalse(server_thread.is_alive())
            self.assertFalse(failure, failure)
            self.assertEqual(len(received), 1)
            packet_id, agv_id, sequence = struct.unpack_from(
                "<HII", received[0], 2
            )
            self.assertEqual((packet_id, agv_id, sequence), (602, 1, 1))
            x_mm = struct.unpack_from("<f", received[0], 25)[0]
            self.assertAlmostEqual(x_mm, 30.0)
            self.assertTrue(any("connected" in message for message in logs))
        finally:
            client.close()

    def test_reconnect_never_reuses_failed_transport_sequence(self):
        delivered = []
        delivered_event = threading.Event()

        class FakeSocket:
            def __init__(self, fail_first_observation: bool):
                self.fail_first_observation = fail_first_observation
                self.ack_bytes = accepted_ack(1, 0x4321)
                self.closed = False

            def settimeout(self, _timeout):
                pass

            def sendall(self, frame):
                packet_id = struct.unpack_from("<H", frame, 2)[0]
                if packet_id == 600:
                    return
                if self.fail_first_observation:
                    self.fail_first_observation = False
                    raise OSError("injected disconnect")
                delivered.append(frame)
                delivered_event.set()

            def recv(self, byte_count):
                chunk = self.ack_bytes[:byte_count]
                self.ack_bytes = self.ack_bytes[byte_count:]
                return chunk

            def close(self):
                self.closed = True

        first_socket = FakeSocket(True)
        second_socket = FakeSocket(False)
        client = VisionServerClient(
            VisionServerConfig(
                host="127.0.0.1",
                port=6666,
                source_id=1,
                agv_id=1,
                connect_timeout_seconds=0.1,
                reconnect_delay_seconds=0.01,
            ),
            map_contract_id="map-contract",
            pose_contract_id="pose-contract",
            calibration_id="calibration-test",
            session_id=0x4321,
            log=lambda _message: None,
        )
        with patch(
            "vision_server_client.socket.create_connection",
            side_effect=[first_socket, second_socket],
        ) as create_connection:
            try:
                client.start()
                client.publish(observation())
                self.assertTrue(delivered_event.wait(2.0))
                self.assertEqual(len(delivered), 1)
                packet_id, agv_id, sequence = struct.unpack_from(
                    "<HII", delivered[0], 2
                )
                self.assertEqual((packet_id, agv_id, sequence), (602, 1, 2))
                self.assertEqual(create_connection.call_count, 2)
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
