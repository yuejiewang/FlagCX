# uniRunner DSL Design and Usage

This document describes the Python DSL in `flagcx/tools/uni_runner.py` for
building uniRunner algorithm DAGs. The DSL follows the same high-level idea as
MSCCL-style workflow builders: an algorithm is described in Python inside a
`with UniRunnerWorkflow(...)` block, and the block adds typed operation nodes
plus dependencies. The generated JSON can be written in the cache format that
the existing uniRunner runtime loads from `FLAGCX_UNIRUNNER_DAG_CACHE_PATH`.

## Goals

- Build one DAG template per rank.
- Support the uniRunner node kinds already implemented in C++: `p2p`, `red`,
  and `cpy`.
- Address buffers by runtime buffer kind plus byte offset: `input`, `output`,
  and `scratch`.
- Generate JSON matching `uni_runner_helper.h`.
- Reproduce the C++ DAG cache key hash so files can be named
  `dag_hash_<hash>_rank_<rank>.json`.
- Provide semantic validation for common collective operations and custom
  communication semantics.

## Runtime JSON Shape

The runtime accepts either a single DAG template or a cache collection. The DSL
uses the single-template file form for rank files:

```json
{
  "hash": "123",
  "key": {
    "format_version": 1,
    "algo": "sliced_ar",
    "comm_op": "all_reduce",
    "count": 17,
    "datatype": 7,
    "red_op": 0,
    "rank": 0,
    "nranks": 4,
    "root": -1,
    "group_size": 0,
    "num_slices": 2,
    "num_red_slices": 2,
    "red_slice_size": 65536,
    "nthreads": 32,
    "input_output_aliased": 0,
    "input_scratch_aliased": 0,
    "output_scratch_aliased": 0
  },
  "dag": {
    "num_nodes": 1,
    "nodes": []
  }
}
```

The key must match the C++ initialization path that will be called at runtime.
For example, `initUniRunnerStateSlicedAR` looks for `algo = "sliced_ar"` and
`comm_op = "all_reduce"`. The runtime builds the same hash from the key and
then tries to load:

```bash
FLAGCX_UNIRUNNER_DAG_CACHE_PATH=/path/to/cache
/path/to/your/flagcx/test
```

Each rank file in that directory must be named:

```text
dag_hash_<computed_hash>_rank_<rank>.json
```

Use `workflow.write_rank_files(cache_dir)` to write those files.

## Basic DSL

```python
from uni_runner import UniRunnerWorkflow
from utils import Collective, DataType, RedOp

with UniRunnerWorkflow(
    "my_allreduce",
    collective=Collective.AllReduce,
    world_size=4,
    count=1024,
    datatype=DataType.float32,
    red_op=RedOp.sum,
    algo="sliced_ar",
    num_slices=2,
    num_red_slices=2,
    output_count=1024,
) as workflow:
    for rank in workflow.ranks():
        next_rank = (rank + 1) % workflow.world_size
        prev_rank = (rank - 1 + workflow.world_size) % workflow.world_size

        with workflow.rank(rank) as rb:
            p2p = rb.p2p(
                [
                    workflow.send(next_rank, workflow.input(0), 256),
                    workflow.recv(prev_rank, workflow.output(0), 256),
                ],
                name="exchange_chunk",
            )
            rb.red(
                workflow.output(0),
                workflow.input(0),
                workflow.output(0),
                256,
                red_op=RedOp.sum,
                parents=[p2p],
                name="accumulate_chunk",
            )

workflow.semantic_check().raise_for_error()
workflow.write_rank_files("/tmp/unirunner-cache")
workflow.write_dag_json("/tmp/my_allreduce_dag.json")
```

`workflow.input(i)`, `workflow.output(i)`, and `workflow.scratch(i)` use element
offsets. The emitted JSON stores byte offsets. Use `input_bytes`,
`output_bytes`, or `scratch_bytes` only when you already have byte offsets.

## Nodes

### `p2p`

A `p2p` node groups one or more device primitives. The common primitives are
`send` and `recv`.

```python
rb.p2p(
    [
        workflow.send(peer_rank, workflow.output(offset), count),
        workflow.recv(peer_rank, workflow.output(offset), count),
    ],
    parents=[parent_node_idx],
)
```

### `red`

A `red` node performs elementwise local reduction:

```python
rb.red(
    workflow.output(offset),
    workflow.input(offset),
    workflow.output(offset),
    count,
    red_op=RedOp.sum,
    parents=[p2p_node_idx],
)
```

### `cpy`

A `cpy` node performs a device-to-device copy:

```python
rb.cpy(workflow.input(0), workflow.output(rank * count), count)
```

Dependencies are per-rank node indices. A node may depend on a node that is
created later; `workflow.finalize()` computes the reverse `children` edges after
all nodes are present.

## Built-In Examples

`flagcx/tools/test_uni_runner.py` defines two complete example algorithms and
also contains the Python-only tests.

```bash
python flagcx/tools/test_uni_runner.py --generate-examples --output-dir /tmp/unirunner-dsl
```

This writes:

- `/tmp/unirunner-dsl/groupedag`: runtime cache files for grouped AllGather.
- `/tmp/unirunner-dsl/slicedar`: runtime cache files for sliced AllReduce.
- `groupedag_dag.json` and `slicedar_dag.json`: readable DAG dumps.

The example builders are also importable:

```python
from test_uni_runner import build_groupedag_example, build_slicedar_example

groupedag = build_groupedag_example()
slicedar = build_slicedar_example()
```

## Semantic Checking

`semantics.py` includes symbolic correctness rules for:

- `allreduce`
- `allgather`
- `reducescatter`
- `broadcast`
- `reduce`
- `gather`
- `scatter`
- `alltoall`

The checker validates:

- all ranks are present;
- node indices are contiguous;
- parent and child edges are symmetric;
- dependencies are acyclic;
- P2P sends and receives match during symbolic execution;
- final output buffers match the selected collective semantics.

Use:

```python
workflow.semantic_check().raise_for_error()
```

For custom communication semantics, provide an explicit `CollectiveSemantics`
object:

```python
from semantics import CollectiveSemantics
from utils import Collective, RedOp

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
```

You can also register a procedural checker with
`register_custom_semantic(name, checker)` when expected output is easier to
compute from the simulated context than to enumerate.

## Current Runtime Integration Notes

The C++ runtime does not yet expose a generic `custom` uniRunner entry point.
It loads cache files by matching the cache key generated by the active static
initializer, such as `grouped_ag` for AllGather or `sliced_ar` for AllReduce.
Therefore, a custom DAG intended to run today should use the key of the runtime
path it is replacing. The Python DSL can still build arbitrary DAGs and custom
semantic checks, but runtime loading requires an existing key path.

## Test

Run the Python-only test suite:

```bash
python flagcx/tools/test_uni_runner.py
```

The test does not compile C/CUDA code. It checks JSON generation, runtime cache
file names, and symbolic correctness of the grouped AllGather and sliced
AllReduce examples.
