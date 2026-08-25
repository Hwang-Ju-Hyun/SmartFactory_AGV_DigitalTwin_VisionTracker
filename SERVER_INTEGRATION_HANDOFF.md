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

There is currently no Server socket transport in this repository. The preview
explicitly runs with Server output disabled.

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
- Scale: `50.0 mm` per Server map unit
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

The JSON snapshot is not a substitute for checking the active Server map. A
future integration should compare an explicit Server-provided map/config
identity rather than assuming local files stayed synchronized.

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

The final webcam has not arrived or been mounted. Therefore:

- reference tag anchors ID 1-8 are still `null`;
- `reference_plane_height_mm` is still `null`;
- `vision_calibration.json` has not been generated;
- no trusted metric pose should be emitted yet.

The planar V1 requires fixed reference tags and the robot tag to be at the same
physical height. Floor references combined with a robot-roof tag introduce
parallax and are rejected. At least five well-spread, non-collinear reference
tags are required; six to eight are preferred.

## Observation semantics for future Server integration

- `MEASURED`: a calibrated observation received by the host within the maximum
  fresh-age limit
- `HELD`: a short, explicitly stale carry-forward used only for display
- `LOST`: no pose; consumers must not reuse an old coordinate as current

Any future transport should carry, at minimum:

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

A process-local monotonic timestamp is not automatically comparable across the
Windows sender and WSL Server. Specify timestamp semantics explicitly and use
the Server's receive time for its own freshness timeout.

Before defining packet numbers or layouts, inspect the Server's canonical
`Shared/Protocol.hpp` and `Shared/PacketSerializer.*`. Preserve field-by-field
serialization; do not transmit raw C++ structs.

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

## Required physical validation before network integration

1. Mount and lock the final camera.
2. Mount at least five references at the measured robot-tag height.
3. Enter their exact TestCase0 anchors and reference-plane height.
4. Lock and independently verify calibration.
5. Measure static position and heading error at several known nodes/headings.
6. Measure detection loss and latency while the robot moves.

Only after those measurements should the Python-to-Server packet contract be
finalized and Unity display work begin.

The user explicitly does not want a standalone fake/mock Vision sender merely
because the final camera has not arrived. Wait for the real calibrated output;
ordinary isolated unit-test fixtures are still appropriate.

## Verification

From Windows PowerShell in the VisionTracker repository:

```powershell
py -3.12 -m unittest test_pose_tracker test_vision_calibration test_vision_geometry test_vision_map test_vision_preview
```

The reviewed baseline has 61 unit tests and does not connect to the Server or
send robot commands.
