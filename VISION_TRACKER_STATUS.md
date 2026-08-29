# VisionTracker status

## Objective

Measure the real AGV's absolute TestCase0 pose (`x`, `z`, heading) from an
overhead camera. This measurement will later let the Server compare planned
motion with the physical robot and let Unity display both. It is not yet a
closed-loop motor correction source.

## Implemented

- `tag36h11` observation with Hamming, decision-margin, and duplicate-ID gates
- ID 0 printed-top direction projected through the tag homography
- TestCase0 Server/local millimetre coordinate contract and round-trip tests
- multi-frame reference-centre collection using per-tag medians
- RANSAC calibration with inlier and reprojection-error checks
- locked calibration persistence with an integrity ID
- startup compatibility checks for map digest, actual frame size, anchors,
  robot pose contract, calibration quality policy, and tag heights
- live fixed-reference verification without per-frame homography refitting
- distributed-reference coverage verification and anchor-outlier rejection
- fail-closed metric output on stale/moved references or plane mismatch
- measured tag-centre-to-AGV-origin rigid-body correction and map-ROI rejection
- latest-frame-only camera reader plus explicit host-receipt freshness timing
- C270-native 1280x720 MJPG/30 request with DirectShow-safe property ordering,
  bounded manual exposure, and measured DSHOW/MSMF/auto backend negotiation
- separate camera-delivery, full-processing, and AprilTag-detection performance
  measurements so skipped frames are counted correctly
- `MEASURED`, explicitly stale `HELD`, and pose-free `LOST` states
- observation-only Server TCP sender with HELLO/ACK, latest-only coalescing,
  reconnect, and explicit MEASURED/HELD/LOST packets
- no direct Unity, ESP32, arrival, replanning, or motor-control connection

## Physical information still required

- confirmation that the latest camera placement is rigid and final
- confirmation that fixed references really share the robot tag's 90 mm plane
- live fixed-reference verification of the local locked 35 cm calibration
- measured node-position and heading error across the map
- moving-robot loss and end-to-end latency measurements

`vision_config.json` records the current chassis measurement: the ID 0 surface
is approximately 90 mm above the floor, its printed front is aligned with the
chassis front, and its centre is over the drive-wheel axle midpoint. It also
contains six anchors and a 90 mm reference plane. Software cannot verify that
physical height, so floor references must be raised or a later 3D model must
compensate the plane difference.

## Known limitation

The webcam lens is not intrinsically calibrated. A planar homography cannot
fully remove radial lens distortion. Server integration must remain
observation-only until known-node error is measured. Checkerboard assets are
included for camera-specific calibration.

Floor reference tags and a robot tag on top of the chassis are different
planes. The first version requires equal heights. A later version may instead
use camera intrinsics/extrinsics plus measured robot-tag height for parallax
compensation.

The committed TestCase0 JSON is a local Server-map snapshot. Its digest protects
calibrations from local edits. The Vision HELLO also carries map and pose
contract IDs, which the Server checks against its active canonical map.

## Next gate

After the Logitech camera is fixed:

1. enter physical anchors and heights;
2. create and verify a locked calibration;
3. measure errors at multiple nodes and headings;
4. run the Server with the newly locked calibration ID and verify that
   MEASURED/HELD/LOST observations do not affect robot control;
5. verify Unity planned-vs-actual display end to end.
