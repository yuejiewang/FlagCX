from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:
    from .semantics import CollectiveSemantics, SemanticCheckResult, check_dag_semantics, make_collective_semantics
    from .utils import Collective, RedOp
except ImportError:  # Keep compatibility with direct execution from flagcx/tools.
    from semantics import CollectiveSemantics, SemanticCheckResult, check_dag_semantics, make_collective_semantics
    from utils import Collective, RedOp


FORMAT_VERSION = 1
DEFAULT_NUM_SLICES = 1
DEFAULT_NUM_RED_SLICES = 0
DEFAULT_RED_SLICE_SIZE = 65536
DEFAULT_NTHREADS = 32
SIZE_T_MASK = (1 << 64) - 1
HASH_CONSTANT = 0x9E3779B97F4A7C15


class DataType(Enum):
    int8 = 0
    uint8 = 1
    int32 = 2
    uint32 = 3
    int64 = 4
    uint64 = 5
    float16 = 6
    float32 = 7
    float = 7
    float64 = 8
    bfloat16 = 9


DATA_TYPE_SIZES = {
    DataType.int8: 1,
    DataType.uint8: 1,
    DataType.int32: 4,
    DataType.uint32: 4,
    DataType.int64: 8,
    DataType.uint64: 8,
    DataType.float16: 2,
    DataType.float32: 4,
    DataType.float64: 8,
    DataType.bfloat16: 2,
}


class BufferKind(str, Enum):
    none = "none"
    input = "input"
    output = "output"
    scratch = "scratch"


class P2pType(str, Enum):
    send = "send"
    recv = "recv"
    term = "term"
    wait = "wait"
    put = "put"
    signal = "signal"
    barrier_signal = "barrier_signal"
    wait_signal = "wait_signal"
    put_value = "put_value"
    put_signal = "put_signal"
    get = "get"


class AlgoType(str, Enum):
    dummy = "dummy"
    loc_red = "loc_red"
    grouped_ag = "grouped_ag"
    ring_ag = "ring_ag"
    ring_ar = "ring_ar"
    sliced_ar = "sliced_ar"
    ring_rs = "ring_rs"
    tree_red = "tree_red"


ALGO_TO_INT = {
    "dummy": 0,
    "loc_red": 1,
    "grouped_ag": 2,
    "ring_ag": 3,
    "ring_ar": 4,
    "sliced_ar": 5,
    "ring_rs": 6,
    "tree_red": 7,
}

COMM_OP_TO_INT = {
    "send": 0,
    "recv": 1,
    "broadcast": 2,
    "gather": 3,
    "scatter": 4,
    "reduce": 5,
    "all_reduce": 6,
    "all_gather": 7,
    "reduce_scatter": 8,
    "all_to_all": 9,
    "all_to_allv": 10,
    "noop": 11,
}

COLLECTIVE_TO_COMM_OP = {
    Collective.P2P: "send",
    Collective.Custom: "noop",
    Collective.Broadcast: "broadcast",
    Collective.Gather: "gather",
    Collective.Scatter: "scatter",
    Collective.Reduce: "reduce",
    Collective.AllReduce: "all_reduce",
    Collective.AllGather: "all_gather",
    Collective.ReduceScatter: "reduce_scatter",
    Collective.AlltoAll: "all_to_all",
    Collective.AlltoAllv: "all_to_allv",
    Collective.Nop: "noop",
}


def _enum_value(value):
    return value.value if isinstance(value, Enum) else value


def _datatype_value(datatype: Union[DataType, int]) -> int:
    return int(_enum_value(datatype))


def _datatype_size(datatype: Union[DataType, int]) -> int:
    if isinstance(datatype, DataType):
        return DATA_TYPE_SIZES[datatype]
    for candidate, size in DATA_TYPE_SIZES.items():
        if candidate.value == datatype:
            return size
    raise ValueError(f"Unsupported datatype value {datatype}")


def _red_op_value(red_op: Union[RedOp, int]) -> int:
    return int(_enum_value(red_op))


def _red_op_from_value(red_op: Union[RedOp, int]) -> RedOp:
    if isinstance(red_op, RedOp):
        return red_op
    return RedOp(red_op)


def _comm_op_name(collective: Collective, comm_op: Optional[str]) -> str:
    if comm_op is not None:
        return comm_op
    return COLLECTIVE_TO_COMM_OP[collective]


def _hash_combine(seed: int, value: int) -> int:
    return (
        seed
        ^ ((value & SIZE_T_MASK) + HASH_CONSTANT + ((seed << 6) & SIZE_T_MASK) + (seed >> 2))
    ) & SIZE_T_MASK


def _write_json_file(path: str, payload: Mapping):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


@dataclass(frozen=True)
class BufferRef:
    buffer: str
    offset_bytes: int = 0

    @staticmethod
    def none() -> "BufferRef":
        return BufferRef(BufferKind.none.value, 0)

    def to_json(self) -> Dict[str, Union[str, int]]:
        return {"buffer": self.buffer, "offset_bytes": self.offset_bytes}


@dataclass
class P2pOp:
    type: str
    peer_rank: int
    buffer: BufferRef
    count: int
    datatype: int

    def to_json(self) -> Dict:
        return {
            "type": self.type,
            "peer_rank": self.peer_rank,
            "count": self.count,
            "datatype": self.datatype,
            "buffer": self.buffer.to_json(),
        }


@dataclass
class RedOpDesc:
    input1: BufferRef
    input2: BufferRef
    output: BufferRef
    count: int
    nthreads: int
    datatype: int
    red_op: RedOp

    def to_json(self) -> Dict:
        return {
            "input1": self.input1.to_json(),
            "input2": self.input2.to_json(),
            "output": self.output.to_json(),
            "count": self.count,
            "nthreads": self.nthreads,
            "datatype": self.datatype,
            "red_op": self.red_op.value,
        }


@dataclass
class CpyOpDesc:
    src: BufferRef
    dst: BufferRef
    count: int
    datatype: int

    def to_json(self) -> Dict:
        return {
            "src": self.src.to_json(),
            "dst": self.dst.to_json(),
            "count": self.count,
            "datatype": self.datatype,
        }


@dataclass
class UniRunnerDagNode:
    node_idx: int
    node_type: str
    parents: List[int] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    p2p_ops: List[P2pOp] = field(default_factory=list)
    red: Optional[RedOpDesc] = None
    cpy: Optional[CpyOpDesc] = None
    name: str = ""

    def to_json(self) -> Dict:
        payload = {
            "node_idx": self.node_idx,
            "node_type": self.node_type,
            "parents": list(self.parents),
            "children": list(self.children),
        }
        if self.node_type == "p2p":
            payload["p2p_ops"] = [op.to_json() for op in self.p2p_ops]
        elif self.node_type == "red":
            payload["red"] = self.red.to_json()
        elif self.node_type == "cpy":
            payload["cpy"] = self.cpy.to_json()
        else:
            raise ValueError(f"Unsupported node type {self.node_type}")
        return payload


@dataclass
class UniRunnerDagCacheKey:
    algo: str
    comm_op: str
    count: int
    datatype: int
    red_op: int
    rank: int
    nranks: int
    root: int = -1
    group_size: int = 0
    num_slices: int = DEFAULT_NUM_SLICES
    num_red_slices: int = DEFAULT_NUM_RED_SLICES
    red_slice_size: int = DEFAULT_RED_SLICE_SIZE
    nthreads: int = DEFAULT_NTHREADS
    input_output_aliased: int = 0
    input_scratch_aliased: int = 0
    output_scratch_aliased: int = 0
    format_version: int = FORMAT_VERSION

    def hash_value(self) -> int:
        hash_value = 0
        values = [
            self.format_version,
            ALGO_TO_INT[self.algo],
            COMM_OP_TO_INT[self.comm_op],
            self.count,
            self.datatype,
            self.red_op,
            self.rank,
            self.nranks,
            self.root + 1,
            self.group_size + 1,
            self.num_slices,
            self.num_red_slices,
            self.red_slice_size,
            self.nthreads,
            self.input_output_aliased,
            self.input_scratch_aliased,
            self.output_scratch_aliased,
        ]
        for value in values:
            hash_value = _hash_combine(hash_value, int(value))
        return hash_value

    def file_name(self) -> str:
        return f"dag_hash_{self.hash_value()}_rank_{self.rank}.json"

    def to_json(self) -> Dict[str, Union[str, int]]:
        return {
            "format_version": self.format_version,
            "algo": self.algo,
            "comm_op": self.comm_op,
            "count": self.count,
            "datatype": self.datatype,
            "red_op": self.red_op,
            "rank": self.rank,
            "nranks": self.nranks,
            "root": self.root,
            "group_size": self.group_size,
            "num_slices": self.num_slices,
            "num_red_slices": self.num_red_slices,
            "red_slice_size": self.red_slice_size,
            "nthreads": self.nthreads,
            "input_output_aliased": self.input_output_aliased,
            "input_scratch_aliased": self.input_scratch_aliased,
            "output_scratch_aliased": self.output_scratch_aliased,
        }


@dataclass
class UniRunnerRankDag:
    rank: int
    nodes: List[UniRunnerDagNode] = field(default_factory=list)

    def add_node(self, node: UniRunnerDagNode):
        if node.node_idx != len(self.nodes):
            raise ValueError("node_idx must match append order")
        self.nodes.append(node)

    def finalize_dependencies(self):
        for node in self.nodes:
            node.parents = list(dict.fromkeys(node.parents))
            node.children = []
        for node in self.nodes:
            for parent in node.parents:
                if parent < 0 or parent >= len(self.nodes):
                    raise ValueError(f"rank {self.rank} node {node.node_idx} has invalid parent {parent}")
                self.nodes[parent].children.append(node.node_idx)
        for node in self.nodes:
            node.children = list(dict.fromkeys(node.children))

    def to_json_nodes(self) -> List[Dict]:
        return [node.to_json() for node in self.nodes]


class UniRunnerWorkflow:
    """Context-managed DSL for building per-rank uniRunner DAG templates."""

    def __init__(
        self,
        name: str,
        collective: Collective = Collective.Custom,
        world_size: int = 1,
        count: int = 0,
        datatype: Union[DataType, int] = DataType.float32,
        red_op: Union[RedOp, int] = RedOp.sum,
        root: int = -1,
        algo: Union[AlgoType, str] = AlgoType.dummy,
        comm_op: Optional[str] = None,
        group_size: int = 0,
        num_slices: int = DEFAULT_NUM_SLICES,
        num_red_slices: int = DEFAULT_NUM_RED_SLICES,
        red_slice_size: int = DEFAULT_RED_SLICE_SIZE,
        nthreads: int = DEFAULT_NTHREADS,
        input_count: Optional[int] = None,
        output_count: Optional[int] = None,
        scratch_count: int = 0,
        semantic: Optional[CollectiveSemantics] = None,
    ):
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        self.name = name
        self.collective = collective
        self.world_size = world_size
        self.count = count
        self.datatype = _datatype_value(datatype)
        self.type_size = _datatype_size(datatype)
        self.red_op = _red_op_from_value(red_op)
        self.root = root
        self.algo = _enum_value(algo)
        self.comm_op = _comm_op_name(collective, comm_op)
        self.group_size = group_size
        self.num_slices = num_slices
        self.num_red_slices = num_red_slices
        self.red_slice_size = red_slice_size
        self.nthreads = nthreads
        self.input_count = count if input_count is None else input_count
        self.output_count = count if output_count is None else output_count
        self.scratch_count = scratch_count
        self.semantic = semantic
        self._ranks = {rank: UniRunnerRankDag(rank) for rank in range(world_size)}
        self._active_rank: Optional[int] = None
        self._finalized = False

    def __enter__(self) -> "UniRunnerWorkflow":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.finalize()
        return False

    def rank(self, rank: int) -> "_RankBuilder":
        self._check_rank(rank)
        return _RankBuilder(self, rank)

    def ranks(self) -> range:
        return range(self.world_size)

    def input(self, offset: int = 0) -> BufferRef:
        return self.buffer(BufferKind.input, offset)

    def output(self, offset: int = 0) -> BufferRef:
        return self.buffer(BufferKind.output, offset)

    def scratch(self, offset: int = 0) -> BufferRef:
        return self.buffer(BufferKind.scratch, offset)

    def input_bytes(self, offset_bytes: int = 0) -> BufferRef:
        return BufferRef(BufferKind.input.value, offset_bytes)

    def output_bytes(self, offset_bytes: int = 0) -> BufferRef:
        return BufferRef(BufferKind.output.value, offset_bytes)

    def scratch_bytes(self, offset_bytes: int = 0) -> BufferRef:
        return BufferRef(BufferKind.scratch.value, offset_bytes)

    def buffer(self, kind: Union[BufferKind, str], offset: int = 0) -> BufferRef:
        kind_value = _enum_value(kind)
        return BufferRef(kind_value, offset * self.type_size)

    def send(self, peer_rank: int, buffer: BufferRef, count: int, datatype: Optional[int] = None) -> P2pOp:
        return P2pOp(P2pType.send.value, peer_rank, buffer, count, self.datatype if datatype is None else datatype)

    def recv(self, peer_rank: int, buffer: BufferRef, count: int, datatype: Optional[int] = None) -> P2pOp:
        return P2pOp(P2pType.recv.value, peer_rank, buffer, count, self.datatype if datatype is None else datatype)

    def p2p(
        self,
        rank: Optional[int] = None,
        ops: Optional[Sequence[P2pOp]] = None,
        parents: Optional[Sequence[int]] = None,
        name: str = "",
    ) -> int:
        rank = self._resolve_rank(rank)
        node = self._new_node(rank, "p2p", parents, name)
        node.p2p_ops = list(ops or [])
        self._ranks[rank].add_node(node)
        self._finalized = False
        return node.node_idx

    def red(
        self,
        rank: Optional[int] = None,
        input1: Optional[BufferRef] = None,
        input2: Optional[BufferRef] = None,
        output: Optional[BufferRef] = None,
        count: int = 0,
        red_op: Optional[Union[RedOp, int]] = None,
        parents: Optional[Sequence[int]] = None,
        name: str = "",
    ) -> int:
        rank = self._resolve_rank(rank)
        node = self._new_node(rank, "red", parents, name)
        node.red = RedOpDesc(
            input1=input1 or BufferRef.none(),
            input2=input2 or BufferRef.none(),
            output=output or BufferRef.none(),
            count=count,
            nthreads=self.nthreads,
            datatype=self.datatype,
            red_op=self.red_op if red_op is None else _red_op_from_value(red_op),
        )
        self._ranks[rank].add_node(node)
        self._finalized = False
        return node.node_idx

    def cpy(
        self,
        rank: Optional[int] = None,
        src: Optional[BufferRef] = None,
        dst: Optional[BufferRef] = None,
        count: int = 0,
        parents: Optional[Sequence[int]] = None,
        name: str = "",
    ) -> int:
        rank = self._resolve_rank(rank)
        node = self._new_node(rank, "cpy", parents, name)
        node.cpy = CpyOpDesc(src or BufferRef.none(), dst or BufferRef.none(), count, self.datatype)
        self._ranks[rank].add_node(node)
        self._finalized = False
        return node.node_idx

    def dep(self, rank: int, parent: int, child: int):
        self._check_rank(rank)
        nodes = self._ranks[rank].nodes
        if child < 0 or child >= len(nodes):
            raise ValueError(f"rank {rank} child node {child} does not exist")
        nodes[child].parents.append(parent)
        self._finalized = False

    def rank_templates(self) -> Mapping[int, UniRunnerRankDag]:
        self.finalize()
        return self._ranks

    def finalize(self):
        if self._finalized:
            return
        for rank_dag in self._ranks.values():
            rank_dag.finalize_dependencies()
        self._finalized = True

    def validate(self) -> SemanticCheckResult:
        return self.semantic_check()

    def semantic_check(self, semantic: Optional[CollectiveSemantics] = None) -> SemanticCheckResult:
        if semantic is None:
            semantic = self.semantic
        if semantic is None and self.collective != Collective.Custom:
            semantic = make_collective_semantics(
                self.collective,
                self.world_size,
                self.count,
                self.red_op,
                self.root if self.root >= 0 else 0,
            )
        if semantic is None:
            return SemanticCheckResult(False, ["custom workflows need an explicit semantic specification"])
        return check_dag_semantics(self, semantic)

    def cache_key(self, rank: int) -> UniRunnerDagCacheKey:
        self._check_rank(rank)
        return UniRunnerDagCacheKey(
            algo=self.algo,
            comm_op=self.comm_op,
            count=self.count,
            datatype=self.datatype,
            red_op=_red_op_value(self.red_op),
            rank=rank,
            nranks=self.world_size,
            root=self.root,
            group_size=self.group_size,
            num_slices=self.num_slices,
            num_red_slices=self.num_red_slices,
            red_slice_size=self.red_slice_size,
            nthreads=self.nthreads,
        )

    def runtime_entry(self, rank: int) -> Dict:
        self.finalize()
        key = self.cache_key(rank)
        nodes = self._ranks[rank].to_json_nodes()
        return {
            "hash": str(key.hash_value()),
            "key": key.to_json(),
            "dag": {"num_nodes": len(nodes), "nodes": nodes},
        }

    def runtime_cache_json(self, ranks: Optional[Iterable[int]] = None) -> Dict:
        selected_ranks = list(self.ranks() if ranks is None else ranks)
        return {
            "format_version": FORMAT_VERSION,
            "address_model": "buffer_kind+offset_bytes",
            "buffer_kinds": ["input", "output", "scratch"],
            "entries": [self.runtime_entry(rank) for rank in selected_ranks],
        }

    def dag_json(self) -> Dict:
        self.finalize()
        return {
            "name": self.name,
            "collective": self.collective.name,
            "world_size": self.world_size,
            "count": self.count,
            "datatype": self.datatype,
            "type_size": self.type_size,
            "algo": self.algo,
            "comm_op": self.comm_op,
            "ranks": {
                str(rank): {"nodes": rank_dag.to_json_nodes()}
                for rank, rank_dag in self._ranks.items()
            },
        }

    def write_rank_files(self, output_dir: str) -> List[str]:
        self.finalize()
        paths = []
        os.makedirs(output_dir, exist_ok=True)
        for rank in self.ranks():
            key = self.cache_key(rank)
            path = os.path.join(output_dir, key.file_name())
            _write_json_file(path, self.runtime_entry(rank))
            paths.append(path)
        return paths

    def write_runtime_cache(self, path: str, ranks: Optional[Iterable[int]] = None):
        _write_json_file(path, self.runtime_cache_json(ranks))

    def write_dag_json(self, path: str):
        _write_json_file(path, self.dag_json())

    def _new_node(
        self,
        rank: int,
        node_type: str,
        parents: Optional[Sequence[int]],
        name: str,
    ) -> UniRunnerDagNode:
        self._check_rank(rank)
        return UniRunnerDagNode(
            node_idx=len(self._ranks[rank].nodes),
            node_type=node_type,
            parents=list(parents or []),
            name=name,
        )

    def _resolve_rank(self, rank: Optional[int]) -> int:
        if rank is None:
            if self._active_rank is None:
                raise ValueError("rank must be supplied outside a workflow.rank(...) block")
            rank = self._active_rank
        self._check_rank(rank)
        return rank

    def _check_rank(self, rank: int):
        if rank < 0 or rank >= self.world_size:
            raise ValueError(f"rank {rank} is outside [0, {self.world_size})")


class _RankBuilder:
    def __init__(self, workflow: UniRunnerWorkflow, rank: int):
        self.workflow = workflow
        self.rank = rank

    def __enter__(self) -> "_RankBuilder":
        if self.workflow._active_rank is not None:
            raise RuntimeError("nested rank builders are not supported")
        self.workflow._active_rank = self.rank
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.workflow._active_rank = None
        return False

    def p2p(self, ops: Sequence[P2pOp], parents: Optional[Sequence[int]] = None, name: str = "") -> int:
        return self.workflow.p2p(self.rank, ops, parents, name)

    def red(
        self,
        input1: BufferRef,
        input2: BufferRef,
        output: BufferRef,
        count: int,
        red_op: Optional[Union[RedOp, int]] = None,
        parents: Optional[Sequence[int]] = None,
        name: str = "",
    ) -> int:
        return self.workflow.red(self.rank, input1, input2, output, count, red_op, parents, name)

    def cpy(self, src: BufferRef, dst: BufferRef, count: int, parents: Optional[Sequence[int]] = None, name: str = "") -> int:
        return self.workflow.cpy(self.rank, src, dst, count, parents, name)

    def dep(self, parent: int, child: int):
        self.workflow.dep(self.rank, parent, child)


def _rank_chunk(count: int, nranks: int, chunk: int) -> Tuple[int, int]:
    base = count // nranks
    remainder = count % nranks
    chunk_count = base + (1 if chunk < remainder else 0)
    offset = chunk * base + min(chunk, remainder)
    return offset, chunk_count


def _slice_chunk(count: int, num_slices: int, slice_idx: int) -> Tuple[int, int]:
    base = count // num_slices
    remainder = count % num_slices
    slice_count = base + (1 if slice_idx < remainder else 0)
    offset = slice_idx * base + min(slice_idx, remainder)
    return offset, slice_count


def _effective_red_slices(
    count: int,
    nranks: int,
    num_slices: int,
    num_red_slices: int,
    red_slice_size: int,
) -> int:
    if num_red_slices != 0:
        return num_red_slices
    if count == 0 or red_slice_size == 0 or num_slices == 0 or nranks <= 0:
        return 1
    divisor = nranks * num_slices * red_slice_size
    return max(1, (count + divisor - 1) // divisor)


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
                                parent_idx = (num_slices - 1) * nodes_per_slice + (step - 1) * (1 + effective_red_slices)
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
                                parent_idx = (num_slices - 1) * nodes_per_slice + (world_size - 2) * (1 + effective_red_slices)
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


def _rank_slice(count: int, nranks: int, num_slices: int, chunk: int, slice_idx: int) -> Tuple[int, int]:
    chunk_offset, chunk_count = _rank_chunk(count, nranks, chunk)
    slice_offset, slice_count = _slice_chunk(chunk_count, num_slices, slice_idx)
    return chunk_offset + slice_offset, slice_count


unirunnerworkflow = UniRunnerWorkflow


__all__ = [
    "AlgoType",
    "BufferKind",
    "BufferRef",
    "DataType",
    "P2pOp",
    "P2pType",
    "UniRunnerDagCacheKey",
    "UniRunnerDagNode",
    "UniRunnerWorkflow",
    "build_groupedag",
    "build_slicedar",
    "unirunnerworkflow",
]
