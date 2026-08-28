import unittest
import json
import tempfile
from dataclasses import replace
from pathlib import Path

from vision_map import MapContract, ServerPoint


BASE_DIR = Path(__file__).resolve().parent


class VisionMapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = MapContract.load(BASE_DIR / "testcase0_map.json")

    def assertLocal(self, node_id, expected_x_mm, expected_z_mm):
        point = self.contract.node_local_mm(node_id)
        self.assertAlmostEqual(point.x_mm, expected_x_mm)
        self.assertAlmostEqual(point.z_mm, expected_z_mm)

    def test_testcase0_corner_nodes(self):
        self.assertLocal(1, 0.0, 0.0)
        self.assertLocal(5, 1400.0, 0.0)
        self.assertLocal(15, 1400.0, 700.0)
        self.assertLocal(11, 0.0, 700.0)

    def test_round_trip_server_and_local_coordinates(self):
        for node in self.contract.nodes.values():
            local = self.contract.server_to_local_mm(node.x, node.z)
            restored = self.contract.local_mm_to_server(local.x_mm, local.z_mm)
            self.assertAlmostEqual(restored.x, node.x)
            self.assertAlmostEqual(restored.z, node.z)

    def test_local_bounds_and_containment(self):
        self.assertEqual(
            self.contract.local_bounds_mm(),
            (0.0, 1400.0, 0.0, 700.0),
        )
        self.assertTrue(self.contract.contains_local_mm(0.0, 0.0))
        self.assertTrue(self.contract.contains_local_mm(1400.0, 700.0))
        self.assertFalse(self.contract.contains_local_mm(-0.01, 350.0))
        self.assertTrue(self.contract.contains_local_mm(-50.0, 750.0, 50.0))
        self.assertFalse(self.contract.contains_local_mm(-50.01, 750.0, 50.0))

    def test_contract_id_is_stable_and_order_independent(self):
        contract_id = self.contract.contract_id
        self.assertEqual(len(contract_id), 16)
        int(contract_id, 16)

        reversed_nodes = dict(reversed(list(self.contract.nodes.items())))
        reordered = replace(self.contract, nodes=reversed_nodes)
        self.assertEqual(reordered.contract_id, contract_id)

    def test_contract_id_changes_with_coordinate_contract(self):
        node_id = next(iter(self.contract.nodes))
        original_node = self.contract.nodes[node_id]
        changed_nodes = dict(self.contract.nodes)
        changed_nodes[node_id] = ServerPoint(
            x=original_node.x + 0.01,
            z=original_node.z,
        )

        variants = (
            replace(self.contract, nodes=changed_nodes),
            replace(self.contract, coordinate_system="different_axes"),
            replace(
                self.contract,
                mm_per_server_unit=self.contract.mm_per_server_unit + 0.01,
            ),
            replace(
                self.contract,
                origin=ServerPoint(
                    self.contract.origin.x + 0.01,
                    self.contract.origin.z,
                ),
            ),
        )
        for variant in variants:
            with self.subTest(contract=variant):
                self.assertNotEqual(variant.contract_id, self.contract.contract_id)

    def test_unsupported_coordinate_system_is_rejected_on_load(self):
        source_path = BASE_DIR / "testcase0_map.json"
        with source_path.open("r", encoding="utf-8") as source:
            raw = json.load(source)
        raw["coordinate_system"] = "z_east_x_south_heading_cw"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            with path.open("w", encoding="utf-8") as output:
                json.dump(raw, output)
            with self.assertRaisesRegex(ValueError, "coordinate_system"):
                MapContract.load(path)

    def test_containment_rejects_invalid_values(self):
        for point in (
            (float("nan"), 0.0, 0.0),
            (0.0, float("inf"), 0.0),
            (0.0, 0.0, float("nan")),
            (0.0, 0.0, -1.0),
        ):
            with self.subTest(point=point):
                with self.assertRaises(ValueError):
                    self.contract.contains_local_mm(*point)

    def test_tag_id_and_node_id_are_independent(self):
        positions = self.contract.reference_positions_mm(
            {
                "1": {"node_id": 1},
                "2": {"node_id": 5},
                "3": {"node_id": 15},
                "4": {"node_id": 11},
            }
        )
        self.assertEqual(sorted(positions), [1, 2, 3, 4])
        self.assertEqual(positions[2].tolist(), [1400.0, 0.0])

    def test_reference_requires_exactly_one_position_source(self):
        invalid_anchors = (
            {"1": {}},
            {
                "1": {
                    "node_id": 1,
                    "server_units": [50.0, -36.0],
                }
            },
        )
        for anchors in invalid_anchors:
            with self.subTest(anchors=anchors):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    self.contract.reference_positions_mm(anchors)

    def test_duplicate_physical_reference_positions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same physical position"):
            self.contract.reference_positions_mm(
                {
                    "1": {"node_id": 1},
                    "8": {"server_units": [50.0, -36.0]},
                }
            )


if __name__ == "__main__":
    unittest.main()
