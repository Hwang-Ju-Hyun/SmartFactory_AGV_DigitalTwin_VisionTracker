"""TestCase map-coordinate contract for the Windows VisionTracker."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_COORDINATE_SYSTEM = "server_x_z_units_x_east_z_north"


@dataclass(frozen=True)
class ServerPoint:
    x: float
    z: float


@dataclass(frozen=True)
class LocalPointMillimetres:
    x_mm: float
    z_mm: float


@dataclass(frozen=True)
class MapContract:
    name: str
    coordinate_system: str
    mm_per_server_unit: float
    origin: ServerPoint
    nodes: dict[int, ServerPoint]

    @property
    def contract_id(self) -> str:
        """Return a stable ID for the coordinate contract's semantic content."""

        payload = {
            "name": self.name,
            "coordinate_system": self.coordinate_system,
            "mm_per_server_unit": float(self.mm_per_server_unit),
            "origin": [float(self.origin.x), float(self.origin.z)],
            "nodes": [
                [int(node_id), float(point.x), float(point.z)]
                for node_id, point in sorted(self.nodes.items())
            ],
        }
        canonical_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical_json).hexdigest()[:16]

    @classmethod
    def load(cls, path: Path) -> "MapContract":
        try:
            with path.open("r", encoding="utf-8") as source:
                raw: dict[str, Any] = json.load(source)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load map contract {path}: {error}") from error

        scale = float(raw["mm_per_server_unit"])
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("mm_per_server_unit must be positive and finite")

        coordinate_system = str(raw["coordinate_system"])
        if coordinate_system != SUPPORTED_COORDINATE_SYSTEM:
            raise ValueError(
                f"unsupported map coordinate_system: {coordinate_system}"
            )

        origin_values = np.asarray(
            raw["local_origin_server_units"], dtype=np.float64
        )
        if origin_values.shape != (2,) or not np.all(np.isfinite(origin_values)):
            raise ValueError("local_origin_server_units must contain two finite values")

        nodes: dict[int, ServerPoint] = {}
        for raw_node in raw["nodes"]:
            node_id = int(raw_node["id"])
            point = ServerPoint(float(raw_node["x"]), float(raw_node["z"]))
            if node_id in nodes:
                raise ValueError(f"duplicate map node ID {node_id}")
            if not np.isfinite(point.x) or not np.isfinite(point.z):
                raise ValueError(f"map node {node_id} is not finite")
            nodes[node_id] = point

        if not nodes:
            raise ValueError("map contract must contain at least one node")

        return cls(
            name=str(raw["name"]),
            coordinate_system=coordinate_system,
            mm_per_server_unit=scale,
            origin=ServerPoint(float(origin_values[0]), float(origin_values[1])),
            nodes=nodes,
        )

    def server_to_local_mm(self, x: float, z: float) -> LocalPointMillimetres:
        values = np.asarray([x, z], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("server coordinates must be finite")
        return LocalPointMillimetres(
            x_mm=(float(x) - self.origin.x) * self.mm_per_server_unit,
            z_mm=(float(z) - self.origin.z) * self.mm_per_server_unit,
        )

    def local_mm_to_server(self, x_mm: float, z_mm: float) -> ServerPoint:
        values = np.asarray([x_mm, z_mm], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("local millimetre coordinates must be finite")
        return ServerPoint(
            x=self.origin.x + float(x_mm) / self.mm_per_server_unit,
            z=self.origin.z + float(z_mm) / self.mm_per_server_unit,
        )

    def node_local_mm(self, node_id: int) -> LocalPointMillimetres:
        try:
            node = self.nodes[int(node_id)]
        except KeyError as error:
            raise ValueError(f"unknown map node ID {node_id}") from error
        return self.server_to_local_mm(node.x, node.z)

    def local_bounds_mm(self) -> tuple[float, float, float, float]:
        """Return node bounds as ``(min_x, max_x, min_z, max_z)`` in mm."""

        local_points = [
            self.server_to_local_mm(node.x, node.z)
            for node in self.nodes.values()
        ]
        return (
            min(point.x_mm for point in local_points),
            max(point.x_mm for point in local_points),
            min(point.z_mm for point in local_points),
            max(point.z_mm for point in local_points),
        )

    def contains_local_mm(
        self,
        x_mm: float,
        z_mm: float,
        margin_mm: float = 0.0,
    ) -> bool:
        """Return whether a local point lies inside the map node bounds."""

        values = np.asarray([x_mm, z_mm, margin_mm], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("local coordinates and margin must be finite")
        if margin_mm < 0.0:
            raise ValueError("map margin must be non-negative")

        min_x, max_x, min_z, max_z = self.local_bounds_mm()
        return (
            min_x - margin_mm <= float(x_mm) <= max_x + margin_mm
            and min_z - margin_mm <= float(z_mm) <= max_z + margin_mm
        )

    def reference_positions_mm(
        self, raw_anchors: dict[str, Any]
    ) -> dict[int, np.ndarray]:
        """Resolve tag anchors without assuming tag IDs are node IDs.

        Each configured value is either null, ``{"node_id": N}``, or
        ``{"server_units": [x, z]}``. Null keeps metric registration disabled
        until the physical placement is deliberately chosen.
        """

        resolved: dict[int, np.ndarray] = {}
        for raw_tag_id, raw_anchor in raw_anchors.items():
            if raw_anchor is None:
                continue
            tag_id = int(raw_tag_id)
            if tag_id in resolved:
                raise ValueError(f"duplicate reference tag ID {tag_id}")
            if not isinstance(raw_anchor, dict):
                raise ValueError(
                    f"reference tag {tag_id} must be an object or null"
                )

            has_node_id = "node_id" in raw_anchor
            has_server_units = "server_units" in raw_anchor
            if has_node_id == has_server_units:
                raise ValueError(
                    f"reference tag {tag_id} needs exactly one of "
                    "node_id or server_units"
                )

            if has_node_id:
                local = self.node_local_mm(int(raw_anchor["node_id"]))
            else:
                values = np.asarray(raw_anchor["server_units"], dtype=np.float64)
                if values.shape != (2,) or not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"reference tag {tag_id} server_units must contain two finite values"
                    )
                local = self.server_to_local_mm(float(values[0]), float(values[1]))

            position = np.array([local.x_mm, local.z_mm], dtype=np.float64)
            for other_tag_id, other_position in resolved.items():
                if np.allclose(position, other_position, rtol=0.0, atol=1e-9):
                    raise ValueError(
                        f"reference tags {other_tag_id} and {tag_id} resolve "
                        "to the same physical position"
                    )
            resolved[tag_id] = position
        return resolved
