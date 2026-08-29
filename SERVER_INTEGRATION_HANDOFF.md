# VisionTracker -> Server handoff

## Where this code lives

- GitHub (private):
  `https://github.com/Hwang-Ju-Hyun/SmartFactory_AGV_DigitalTwin_VisionTracker`
- Windows path: `C:\Users\HwangJuhyun\Desktop\MyPython`
- The same directory from WSL:
  `/mnt/c/Users/HwangJuhyun/Desktop/MyPython`
- Use `git rev-parse HEAD` for the exact reviewed VisionTracker revision.

The SmartFactory Server repository and the ESP32 repository are separate. This
handoff does not authorize changing either one without the user's request.

## Why VisionTracker exists

The physical differential-drive AGV accumulates position and heading error even
when its encoders reach their commanded counts. AprilTag tracking is intended to
measure the real AGV's absolute `x/z/heading` in TestCase0 coordinates. The
Server can later compare this observation with planned and encoder-derived state,
and Unity can visualize the difference.

AprilTag data must initially be observation-only. It must not directly command
the ESP32, overwrite Server-authoritative route state, or declare arrival.

## Current data flow

```text
Windows webcam
  -> tag36h11 detection
  -> fixed, verified pixel-to-map homography
  -> ID 0 tag-centre pose
  -> rigid offset to the AGV origin
  -> MEASURED / HELD / LOST observation
```

`vision_server_client.py` implements the Server's canonical RobotProtocol
Vision channel. A background thread performs `VISION_HELLO`/ACK, coalesces
pending camera results to the latest snapshot, preserves a strictly increasing
observation sequence across reconnects, and sends `MEASURED`/`HELD`/`LOST`.
It is created only for a statically compatible locked calibration. The camera
thread never blocks on socket I/O.

## Files to read first

1. `README.md` - operator workflow, safety gates, and limitations
2. `VISION_TRACKER_STATUS.md` - implemented and remaining work
3. `vision_config.json` - active physical/configuration contract
4. `testcase0_map.json` and `vision_map.py` - coordinate contract
5. `vision_calibration.py` - calibration lock and compatibility checks
6. `pose_tracker.py` - MEASURED/HELD/LOST semantics
7. `vision_tracker_preview.py` - camera and observation pipeline
8. Tracked `test_pose_tracker.py` and `test_vision_*.py` files - executable
   behavioral contract

## TestCase0 coordinate contract

- Canonical source snapshot: `testcase0_map.json`
- Server axes: `x` east, `z` north
- Scale: `50 mm` per Server map unit
- Local metric origin: Server node 1 at `(50, -36)`
- Conversion:

```text
local_x_mm = (server_x - 50) * 50
local_z_mm = (server_z + 36) * 50
server_x   = local_x_mm / 50 + 50
server_z   = local_z_mm / 50 - 36
```

- Local heading: `0 deg = +x`, positive counter-clockwise
- The map semantic digest is stored with calibration data. A calibration for a
  different map contract is rejected.

The JSON snapshot is not a substitute for checking the active Server map.
`VISION_HELLO` carries the map and pose contract IDs; the Server validates its
active canonical map at startup and rejects a mismatched HELLO.

## Current physical measurements

- AprilTag family: `tag36h11`
- Robot tag ID: `0`
- Printed tag size: `60.1 mm`
- Robot tag surface height: approximately `90 mm`
- Printed front aligned with chassis front: heading correction `0 deg`
- ID 0 centre is over the drive-wheel axle midpoint:
  `[forward_mm, left_mm] = [0, 0]`

These are recorded in `vision_config.json`.

## Deliberately incomplete physical calibration

The 35 cm configuration now contains six reference anchors and records both
tag planes as 90 mm. A local, ignored `vision_calibration.json` can be used only
when its static contract checks and live fixed-reference verification both
pass; moving the camera or changing the map rejects it. The configured 90 mm
reference height must match the physical installation, not merely the JSON.
Floor references and a 90 mm robot tag are different planes and must not be
calibrated as though they were equal.

The planar V1 requires fixed reference tags and the robot tag to be at the same
physical height. Floor references combined with a robot-roof tag introduce
parallax and are rejected. At least five well-spread, non-collinear reference
tags are required; six to eight are preferred.

## Observation semantics for Server integration

- `MEASURED`: a calibrated observation received by the host within the maximum
  fresh-age limit
- `HELD`: a short, explicitly stale carry-forward used only for display
- `LOST`: no pose; consumers must not reuse an old coordinate as current

The transport carries:

```text
agv_id
source/session identity
strictly increasing sequence
source observation timestamp
x_mm, z_mm, heading_deg (only when a pose exists)
tracking state: MEASURED / HELD / LOST
calibration_id
map contract identity
quality/verification metadata
```

A process-local monotonic timestamp is not compared across the Windows sender
and WSL Server. The Server uses its own receive time for freshness timeout.

The implemented wire layout matches the Server's canonical
`Shared/Protocol.hpp` and `Shared/PacketSerializer.*`; Python serializes each
little-endian field explicitly and never transmits a native struct.

## Recommended Server ownership boundary

Keep three concepts separate:

1. planned/authoritative route state;
2. ESP32 encoder/motion-controller report;
3. VisionTracker physical observation.

The first integration should store VisionTracker data separately and expose it
to Unity for comparison. It should reject non-finite values, unknown AGVs,
wrong map/calibration identities, stale or reordered sequences, out-of-bounds
poses, and invalid state/pose combinations. Do not use a Vision observation to
alter motion or arrival state until real-camera accuracy has been measured.

## Required physical validation before control use or final demonstration

1. Mount and lock the final camera.
2. Mount at least five references at the measured robot-tag height.
3. Enter their exact TestCase0 anchors and reference-plane height.
4. Lock and independently verify calibration.
5. Measure static position and heading error at several known nodes/headings.
6. Measure detection loss and latency while the robot moves.

The observation-only path can be exercised before those measurements, but no
Vision pose should affect arrival, replanning, or motor control until they pass.

The runtime sender consumes only the real calibrated preview pipeline. The
local fake TCP peer in unit tests verifies framing, fragmented ACK handling,
latest-only delivery, reconnect, and non-reused transport sequences; it is not
a runtime pose source.

## Verification

From Windows PowerShell in the VisionTracker repository:

```powershell
py -3.12 -m unittest test_pose_tracker test_vision_calibration test_vision_geometry test_vision_map test_vision_preview test_vision_server_client
```

The runtime transport sends observations only and cannot send robot commands.
