from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

TOOLS_DIR = os.path.dirname(__file__)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from semantics import CollectiveSemantics
from uni_runner import AlgoType, DataType, UniRunnerWorkflow, build_groupedag, build_slicedar
from utils import Collective, RedOp


class UniRunnerDslTest(unittest.TestCase):
    def test_groupedag_example_semantics_and_runtime_json(self):
        workflow = build_groupedag(
            name="groupedag_test",
            world_size=4,
            count=8,
            group_size=2,
            datatype=DataType.float32,
        )

        workflow.semantic_check().raise_for_error()
        entry = workflow.runtime_entry(rank=0)
        self.assertEqual(entry["key"]["algo"], "grouped_ag")
        self.assertEqual(entry["key"]["comm_op"], "all_gather")
        self.assertEqual(entry["dag"]["num_nodes"], 3)
        self.assertEqual(entry["dag"]["nodes"][-1]["node_type"], "cpy")

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = workflow.write_rank_files(tmpdir)
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(os.path.basename(path).startswith("dag_hash_") for path in paths))
            with open(paths[0], "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["hash"], entry["hash"])
            self.assertEqual(payload["dag"]["nodes"][0]["node_type"], "p2p")

    def test_slicedar_example_semantics_and_runtime_json(self):
        workflow = build_slicedar(
            name="slicedar_test",
            world_size=4,
            count=17,
            datatype=DataType.float32,
            red_op=RedOp.sum,
            num_slices=2,
            num_red_slices=2,
        )

        workflow.semantic_check().raise_for_error()
        entry = workflow.runtime_entry(rank=0)
        node_types = [node["node_type"] for node in entry["dag"]["nodes"]]
        self.assertEqual(entry["key"]["algo"], "sliced_ar")
        self.assertEqual(entry["key"]["comm_op"], "all_reduce")
        self.assertEqual(entry["key"]["num_red_slices"], 2)
        self.assertIn("p2p", node_types)
        self.assertIn("red", node_types)
        self.assertEqual(entry["dag"]["num_nodes"], 24)

    def test_custom_semantics_for_identity_copy(self):
        expected = [
            [(RedOp.nop, [0]), (RedOp.nop, [1])],
            [(RedOp.nop, [2]), (RedOp.nop, [3])],
        ]
        semantic = CollectiveSemantics.custom(
            "identity_copy",
            world_size=2,
            input_count=2,
            output_count=2,
            expected_output=expected,
        )

        with UniRunnerWorkflow(
            "identity_copy",
            collective=Collective.Custom,
            world_size=2,
            count=2,
            datatype=DataType.float32,
            algo=AlgoType.dummy,
            comm_op="noop",
            input_count=2,
            output_count=2,
            semantic=semantic,
        ) as workflow:
            for rank in workflow.ranks():
                with workflow.rank(rank) as rb:
                    rb.cpy(workflow.input(0), workflow.output(0), 2)

        workflow.semantic_check().raise_for_error()
        self.assertEqual(workflow.runtime_entry(0)["key"]["algo"], "dummy")


if __name__ == "__main__":
    unittest.main()
