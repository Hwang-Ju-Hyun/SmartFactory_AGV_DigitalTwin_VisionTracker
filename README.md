# SmartFactory AGV VisionTracker

Windows overhead-camera process for the SmartFactory AGV Digital Twin.

The goal is not AprilTag detection by itself. VisionTracker must measure the
physical AGV's absolute `x/z/heading` in the same TestCase0 coordinate system
used by the Server, so encoder drift can be measured and visualized in Unity.

```text
overhead camera -> VisionTracker -> Server -> Unity
                        ^
                        `-- Server transport is deliberately disabled for now
```

## Current safety boundary

- Camera preview and AprilTag ID/direction detection work.
- TestCase0 coordinate conversion is fixed at `50 mm/server unit`.
- Metric registration is fail-closed until physical anchors and tag-plane
  heights are configured.
- Calibration collects several frames, locks one transform, saves it, and only
  uses it while fixed references still verify that transform.
- A camera move, actual frame-size change, changed map/anchor/robot-pose
  contract, stale verification, poor tag margin, or different tag heights
  blocks fresh metric output.
- The reported point is the configured Server AGV origin, not blindly the
  centre of ID 0. Its measured body-frame offset rotates with the chassis.
- Poses outside the TestCase0 region plus a small configured margin are rejected.
- A dedicated camera thread continuously drains the webcam and retains only the
  newest host-received frame; processing never walks through an old FIFO backlog.
- The program does not connect to the Server and cannot command the ESP32.

## TestCase0 coordinates

Node 1 is the physical local origin. The outer node-centre rectangle is
`800 x 400 mm`.

```text
local_X_mm = (server_x - 50) * 50
local_Z_mm = (server_z + 36) * 50
```

The inverse conversion is included in every metric pose so later Server
integration does not guess units.

`testcase0_map.json` is a reviewed snapshot of the current Server map, not a
live synchronization mechanism. Its semantic digest is bound into every
calibration, so editing the local map invalidates that calibration. A later
Server transport must also compare a Server-owned map/version identifier.

## Before metric calibration

1. Fix the final camera rigidly and confirm its actual resolution.
2. Assign at least five well-spread, non-collinear reference tags in
   `vision_config.json`. Any anchor rejected as an outlier prevents locking;
   it is not silently ignored.
3. Measure `reference_plane_height_mm` and `robot_tag_height_mm`.
4. For this first planar version, put the fixed references at the same height
   as the robot's ID 0 tag. Floor references and a raised robot tag cause
   parallax and are intentionally rejected until 3D height compensation exists.
   In practice, mount the fixed tags on rigid spacers at the ID 0 height; merely
   typing equal numbers for physically unequal planes produces bad coordinates.
5. Measure the printed-top-to-chassis heading correction and the vector from
   the ID 0 centre to the Server AGV origin. Enter `[forward_mm, left_mm]` in
   `tag_center_to_robot_origin_body_mm`. The current chassis measurement is a
   90 mm robot-tag height, 0 degree heading correction, and `[0, 0]` because
   ID 0 is centred over the drive-wheel axle midpoint.
6. Run the preview and press `c`. Do not move the camera or tags while 30
   samples per reference are collected.
7. Validate position and heading at several known nodes before adding any
   Server network transport.

## Run

Python 3.12 is the verified interpreter.

```powershell
py -3.12 -m pip install -r requirements-vision.txt
py -3.12 vision_tracker_preview.py
```

Controls:

- `q` or Escape: quit
- `s`: save raw and annotated frames under ignored `captures/`
- `c`: collect and lock a new metric calibration

The short dropout state is explicit:

- `MEASURED`: calibrated frame received by the host no more than 0.1 seconds ago
- `HELD`: last measurement retained for at most 0.2 seconds, never marked fresh
- `LOST`: no pose is returned

Decision margin and these timing limits are initial conservative gates, not
final accuracy claims. Tune them only from captured data after the final camera
is rigidly mounted.

The common Windows webcam API does not expose a trustworthy sensor-exposure
timestamp here. `fresh` therefore means fresh at host receipt, aided by
continuous draining and a requested one-frame driver buffer; it does not claim
hardware-synchronized exposure time.

## Offline verification

```powershell
py -3.12 -m py_compile vision_tracker_preview.py vision_geometry.py vision_map.py vision_calibration.py pose_tracker.py
py -3.12 -m unittest discover -v -p "test_*.py"
```
