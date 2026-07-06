from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from typing import List, Optional, Tuple, Union

TOOLS_DIR = os.path.dirname(__file__)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from semantics import CollectiveSemantics
from uni_runner import P2pOp, UniRunnerWorkflow
from utils import (
    DEFAULT_NTHREADS,
    DEFAULT_NUM_SLICES,
    DEFAULT_RED_SLICE_SIZE,
    AlgoType,
    Collective,
    DataType,
    RedOp,
    _effective_red_slices,
    _rank_slice,
    _red_op_from_value,
    _slice_chunk,
)


def build_groupedag(
    name: str = "groupedag",
    world_size: int = 4,
    count: int = 8,
    group_size: int = 2,
    datatype: Union[DataType, int] = DataType.float32,
    num_slices: int = DEFAULT_NUM_SLICES,
    nthreads: int = DEFAULT_NTHREADS,
) -> UniRunnerWorkflow:
    if group_size <= 0 or world_size % group_size != 0:
        raise ValueError("group_size must divide world_size")
    with UniRunnerWorkflow(
        name,
        collective=Collective.AllGather,
        world_size=world_size,
        count=count,
        datatype=datatype,
        red_op=RedOp.nop,
        algo=AlgoType.grouped_ag,
        group_size=group_size,
        num_slices=num_slices,
        num_red_slices=0,
        nthreads=nthreads,
        input_count=count,
        output_count=count * world_size,
    ) as workflow:
        n_groups = world_size // group_size
        group_chunk_count = count * group_size

        for rank in workflow.ranks():
            group_idx = rank // group_size
            loc_rank = rank % group_size
            local_base = group_idx * group_chunk_count
            previous_p2p: Optional[int] = None

            with workflow.rank(rank) as rb:
                if world_size == 1:
                    rb.cpy(workflow.input(0), workflow.output(0), count, name="self_copy")
                    continue

                for step in range(n_groups):
                    ops: List[P2pOp] = []
                    is_last_step = step == n_groups - 1
                    for peer_index in range(group_size - 1):
                        loc_send_peer = (loc_rank + peer_index + 1) % group_size
                        loc_recv_peer = (loc_rank - peer_index - 1 + group_size) % group_size
                        send_ref = (
                            workflow.input(0)
                            if step == 0
                            else workflow.output(local_base + loc_rank * count)
                        )
                        recv_ref = workflow.output(local_base + loc_recv_peer * count)
                        ops.append(workflow.send(group_idx * group_size + loc_send_peer, send_ref, count))
                        ops.append(workflow.recv(group_idx * group_size + loc_recv_peer, recv_ref, count))

                    if not is_last_step:
                        send_group_idx = (group_idx + step + 1) % n_groups
                        recv_group_idx = (group_idx - step - 1 + n_groups) % n_groups
                        send_peer = send_group_idx * group_size + loc_rank
                        recv_peer = recv_group_idx * group_size + loc_rank
                        send_ref = (
                            workflow.input(0)
                            if step == 0
                            else workflow.output(local_base + loc_rank * count)
                        )
                        ops.append(workflow.send(send_peer, send_ref, count))
                        ops.append(workflow.recv(recv_peer, workflow.output(recv_peer * count), count))
                        local_base = recv_group_idx * group_chunk_count

                    parents = [] if previous_p2p is None else [previous_p2p]
                    previous_p2p = rb.p2p(ops, parents=parents, name=f"grouped_ag_step_{step}")

                rb.cpy(workflow.input(0), workflow.output(rank * count), count, name="local_chunk_copy")
    return workflow


def build_slicedar(
    name: str = "slicedar",
    world_size: int = 4,
    count: int = 16,
    datatype: Union[DataType, int] = DataType.float32,
    red_op: Union[RedOp, int] = RedOp.sum,
    num_slices: int = 2,
    num_red_slices: int = 0,
    red_slice_size: int = DEFAULT_RED_SLICE_SIZE,
    nthreads: int = DEFAULT_NTHREADS,
) -> UniRunnerWorkflow:
    red_op_enum = _red_op_from_value(red_op)
    effective_red_slices = _effective_red_slices(
        count, world_size, num_slices, num_red_slices, red_slice_size
    )
    with UniRunnerWorkflow(
        name,
        collective=Collective.AllReduce,
        world_size=world_size,
        count=count,
        datatype=datatype,
        red_op=red_op_enum,
        algo=AlgoType.sliced_ar,
        num_slices=num_slices,
        num_red_slices=effective_red_slices,
        red_slice_size=red_slice_size,
        nthreads=nthreads,
        input_count=count,
        output_count=count,
    ) as workflow:
        if world_size == 1:
            for rank in workflow.ranks():
                with workflow.rank(rank) as rb:
                    rb.cpy(workflow.input(0), workflow.output(0), count, name="self_copy")
            return workflow

        nodes_per_slice = (effective_red_slices + 2) * (world_size - 1)
        num_nodes = num_slices * nodes_per_slice

        for rank in workflow.ranks():
            next_rank = (rank + 1) % world_size
            prev_rank = (rank - 1 + world_size) % world_size
            with workflow.rank(rank) as rb:
                global_node_idx = 0

                for slice_idx in range(num_slices):
                    for step in range(world_size - 1):
                        p2p_node_idx = global_node_idx
                        tx_chunk = (rank - step + world_size) % world_size
                        rx_chunk = (rank - step - 1 + world_size) % world_size
                        tx_offset, tx_count = _rank_slice(count, world_size, num_slices, tx_chunk, slice_idx)
                        rx_offset, rx_count = _rank_slice(count, world_size, num_slices, rx_chunk, slice_idx)

                        send_ref = workflow.input(tx_offset) if step == 0 else workflow.output(tx_offset)
                        recv_ref = workflow.output(rx_offset)
                        p2p_parents: List[int] = []
                        if p2p_node_idx != 0:
                            parent_idx = p2p_node_idx - nodes_per_slice
                            if step > 0 and slice_idx == 0:
                                parent_idx = (
                                    (num_slices - 1) * nodes_per_slice
                                    + (step - 1) * (1 + effective_red_slices)
                                )
                            p2p_parents.append(parent_idx)
                            if step > 0:
                                p2p_parents.extend(
                                    p2p_node_idx - effective_red_slices + red_slice
                                    for red_slice in range(effective_red_slices)
                                )
                        rb.p2p(
                            [
                                workflow.send(next_rank, send_ref, tx_count),
                                workflow.recv(prev_rank, recv_ref, rx_count),
                            ],
                            parents=p2p_parents,
                            name=f"rs_p2p_s{slice_idx}_step{step}",
                        )
                        global_node_idx += 1

                        red_base_offset, red_base_count = rx_offset, rx_count
                        red_start_idx = global_node_idx
                        for red_slice in range(effective_red_slices):
                            sub_offset, sub_count = _slice_chunk(red_base_count, effective_red_slices, red_slice)
                            red_offset = red_base_offset + sub_offset
                            rb.red(
                                workflow.output(red_offset),
                                workflow.input(red_offset),
                                workflow.output(red_offset),
                                sub_count,
                                red_op=red_op_enum,
                                parents=[p2p_node_idx],
                                name=f"rs_red_s{slice_idx}_step{step}_r{red_slice}",
                            )
                            global_node_idx += 1
                        assert red_start_idx + effective_red_slices == global_node_idx

                    for step in range(world_size - 1):
                        p2p_node_idx = global_node_idx
                        tx_chunk = (rank - step + 1 + world_size) % world_size
                        rx_chunk = (rank - step + world_size) % world_size
                        tx_offset, tx_count = _rank_slice(count, world_size, num_slices, tx_chunk, slice_idx)
                        rx_offset, rx_count = _rank_slice(count, world_size, num_slices, rx_chunk, slice_idx)

                        p2p_parents = []
                        parent_idx = p2p_node_idx - nodes_per_slice
                        if slice_idx == 0:
                            if step == 0:
                                parent_idx = (
                                    (num_slices - 1) * nodes_per_slice
                                    + (world_size - 2) * (1 + effective_red_slices)
                                )
                            else:
                                parent_idx = (
                                    (num_slices - 1) * nodes_per_slice
                                    + (world_size - 1) * (1 + effective_red_slices)
                                    + step
                                    - 1
                                )
                        p2p_parents.append(parent_idx)
                        if step == 0:
                            p2p_parents.extend(
                                p2p_node_idx - effective_red_slices + red_slice
                                for red_slice in range(effective_red_slices)
                            )

                        rb.p2p(
                            [
                                workflow.send(next_rank, workflow.output(tx_offset), tx_count),
                                workflow.recv(prev_rank, workflow.output(rx_offset), rx_count),
                            ],
                            parents=p2p_parents,
                            name=f"ag_p2p_s{slice_idx}_step{step}",
                        )
                        global_node_idx += 1

                if global_node_idx != num_nodes:
                    raise AssertionError(f"rank {rank} built {global_node_idx} nodes, expected {num_nodes}")
    return workflow


def build_groupedag_example():
    return build_groupedag(
        name="groupedag_example",
        world_size=4,
        count=8,
        group_size=2,
        datatype=DataType.float32,
    )


def build_slicedar_example():
    return build_slicedar(
        name="slicedar_example",
        world_size=4,
        count=17,
        datatype=DataType.float32,
        red_op=RedOp.sum,
        num_slices=2,
        num_red_slices=2,
    )


def write_examples(output_dir: str) -> Tuple[str, str]:
    groupedag_dir = os.path.join(output_dir, "groupedag")
    slicedar_dir = os.path.join(output_dir, "slicedar")

    groupedag = build_groupedag_example()
    groupedag.semantic_check().raise_for_error()
    groupedag.write_rank_files(groupedag_dir)
    groupedag.write_dag_json(os.path.join(groupedag_dir, "groupedag_dag.json"))

    slicedar = build_slicedar_example()
    slicedar.semantic_check().raise_for_error()
    slicedar.write_rank_files(slicedar_dir)
    slicedar.write_dag_json(os.path.join(slicedar_dir, "slicedar_dag.json"))

    return groupedag_dir, slicedar_dir


class UniRunnerDslTest(unittest.TestCase):
    def test_groupedag_example_semantics_and_runtime_json(self):
        workflow = build_groupedag_example()

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
        workflow = build_slicedar_example()

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


def main():
    parser = argparse.ArgumentParser(description="Test or generate uniRunner DSL examples.")
    parser.add_argument(
        "--generate-examples",
        action="store_true",
        help="Generate groupedag and slicedar example runtime cache files instead of running tests.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "output", "unirunner"),
        help="Directory where generated example files should be written.",
    )
    args, unittest_args = parser.parse_known_args()

    if args.generate_examples:
        groupedag_dir, slicedar_dir = write_examples(args.output_dir)
        print(f"groupedag runtime cache: {groupedag_dir}")
        print(f"slicedar runtime cache: {slicedar_dir}")
        return

    unittest.main(argv=[sys.argv[0]] + unittest_args)


if __name__ == "__main__":
    main()
