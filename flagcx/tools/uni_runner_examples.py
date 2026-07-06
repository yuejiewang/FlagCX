from __future__ import annotations

import argparse
import os
from typing import Tuple

try:
    from .uni_runner import DataType, build_groupedag, build_slicedar
    from .utils import RedOp
except ImportError:  # Keep compatibility with direct execution from flagcx/tools.
    from uni_runner import DataType, build_groupedag, build_slicedar
    from utils import RedOp


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


def main():
    parser = argparse.ArgumentParser(description="Generate uniRunner DSL example DAG JSON files.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "output", "unirunner"),
        help="Directory where example runtime cache files should be written.",
    )
    args = parser.parse_args()
    groupedag_dir, slicedar_dir = write_examples(args.output_dir)
    print(f"groupedag runtime cache: {groupedag_dir}")
    print(f"slicedar runtime cache: {slicedar_dir}")


if __name__ == "__main__":
    main()
