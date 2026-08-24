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
- `MEASURED`, explicitly stale `HELD`, and pose-free `LOST` states
- no Server, Unity, ESP32, or motor-control connection

## Physical information still required

- final camera and rigid mounting
- actual accepted resolution/FPS
- final tag ID to map-anchor assignments
- measured reference and robot tag-plane heights
- measured ID 0 heading alignment and tag-centre-to-Server-origin offset
- a locked calibration created after installation
- measured node-position and heading error across the map

`vision_config.json` records the current chassis measurement: the ID 0 surface
is approximately 90 mm above the floor, its printed front is aligned with the
chassis front, and its centre is over the drive-wheel axle midpoint. Reference
anchors and the reference-plane height remain null until the final camera and
physical reference mounts are installed.
Therefore the current committed configuration cannot produce a trusted metric
pose by accident.

## Known limitation

The webcam lens is not intrinsically calibrated. A planar homography cannot
fully remove radial lens distortion. The current version must remain
Server-disabled until known-node error is measured. Checkerboard assets are
included for the camera-specific calibration step after the final camera
arrives.

Floor reference tags and a robot tag on top of the chassis are different
planes. The first version requires equal heights. A later version may instead
use camera intrinsics/extrinsics plus measured robot-tag height for parallax
compensation.

The committed TestCase0 JSON is a local Server-map snapshot. Its digest protects
calibrations from local edits, but it cannot detect a future Server repository
map change until the network protocol carries a Server-owned map/version ID.

## Next gate

After the Logitech camera is fixed:

1. enter physical anchors and heights;
2. create and verify a locked calibration;
3. measure errors at multiple nodes and headings;
4. only if those errors are acceptable, define the VisionTracker-to-Server
   observation packet and add Unity planned-vs-actual display.
