#!/usr/bin/env python3

"""Plot bus bandwidth curves from FlagCX perf result files."""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent


INPUT_FILES = [
    # 8 rank allreduce
    # SCRIPT_DIR / "data/ar8_buff_cache_08.txt",
    # SCRIPT_DIR / "data/ar8_zcpy_cache_08.txt",
    # 4 rank allreduce
    # SCRIPT_DIR / "data/ar4_buff_05.txt",
    # SCRIPT_DIR / "data/ar4_zcpy_05.txt",
    # SCRIPT_DIR / "data/ar4_buff_cache_05.txt",
    # SCRIPT_DIR / "data/ar4_zcpy_cache_05.txt",
    # 4 rank allgather
    # SCRIPT_DIR / "data/ag4_buff_cache_05.txt",
    # SCRIPT_DIR / "data/ag4_zcpy_cache_05.txt",
    SCRIPT_DIR / "data/ar4_multi-fifo_1.txt",
    SCRIPT_DIR / "data/ar4_multi-fifo_8.txt",
]

# OUTPUT_FILE = SCRIPT_DIR / "figs/ar8_cache.svg"
OUTPUT_FILE = SCRIPT_DIR / "figs/ar4_multi-fifo.svg"
NORMALIZED_OUTPUT_FILE = OUTPUT_FILE.parent / f"{OUTPUT_FILE.stem}_normalized{OUTPUT_FILE.suffix}"

LINE_PATTERN = re.compile(
    r"Comm size:\s*(?P<comm_size>\d+)\s*bytes;\s*"
    r"Elapsed time:\s*(?P<elapsed>[0-9.eE+-]+)\s*sec;\s*"
    r"Algo bandwidth:\s*(?P<algo>[0-9.eE+-]+)\s*GB/s;\s*"
    r"Bus bandwidth:\s*(?P<bus>[0-9.eE+-]+)\s*GB/s"
)


def format_message_size(size_in_bytes: int) -> str:
    units = [
        (1024**3, "G"),
        (1024**2, "M"),
        (1024, "K"),
    ]
    for unit_size, suffix in units:
        if size_in_bytes >= unit_size and size_in_bytes % unit_size == 0:
            return f"{size_in_bytes // unit_size}{suffix}"
    return f"{size_in_bytes}B"


def parse_message_size(size_value: str | int) -> int:
    if isinstance(size_value, int):
        return size_value

    normalized = size_value.strip().upper()
    match = re.fullmatch(r"(?P<value>\d+)\s*(?P<unit>[KMG]?)(?P<byte>B)?", normalized)
    if match is None:
        raise ValueError(f"Unsupported message size format: {size_value}")

    value = int(match.group("value"))
    unit = match.group("unit")
    multiplier = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
    }[unit]
    return value * multiplier


def build_power_of_two_range(start_size: str | int, end_size: str | int) -> list[int]:
    start = parse_message_size(start_size)
    end = parse_message_size(end_size)
    if start > end:
        raise ValueError("start_size must be less than or equal to end_size")

    start_power = math.ceil(math.log2(start))
    end_power = math.floor(math.log2(end))
    return [2**power for power in range(start_power, end_power + 1)]


def normalize_selected_x_values(values: list[str | int]) -> list[int]:
    return sorted({parse_message_size(value) for value in values})


# Leave empty to plot all points.
# Example 1:
# SELECTED_X_VALUES = ["64K", "128K", "256K", "512K", "1M", "2M", "4M", "8M", "16M"]
# Example 2:
# SELECTED_X_VALUES = build_power_of_two_range("16K", "16M")
SELECTED_X_VALUES: list[str | int] = []


def build_x_ticks(min_size: int, max_size: int) -> list[int]:
    min_power = max(14, math.floor(math.log2(min_size)))
    max_power = math.ceil(math.log2(max_size))
    return [2**power for power in range(min_power, max_power + 1)]


def load_points(file_path: str | Path) -> tuple[list[int], list[float]]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    points: list[tuple[int, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            match = LINE_PATTERN.fullmatch(line)
            if match is None:
                raise ValueError(
                    f"Failed to parse {path}:{line_no}: {raw_line.rstrip()}"
                )

            comm_size = int(match.group("comm_size"))
            bus_bandwidth = float(match.group("bus"))
            points.append((comm_size, bus_bandwidth))

    if not points:
        raise ValueError(f"No valid data points found in {path}")

    points.sort(key=lambda item: item[0])
    comm_sizes = [item[0] for item in points]
    bus_bandwidths = [item[1] for item in points]
    return comm_sizes, bus_bandwidths


def filter_points(
    comm_sizes: list[int],
    bus_bandwidths: list[float],
    selected_x_value_set: set[int],
    file_path: str | Path,
) -> tuple[list[int], list[float]]:
    if not selected_x_value_set:
        return comm_sizes, bus_bandwidths

    filtered_points = [
        (comm_size, bus_bandwidth)
        for comm_size, bus_bandwidth in zip(comm_sizes, bus_bandwidths)
        if comm_size in selected_x_value_set
    ]
    if not filtered_points:
        raise ValueError(
            f"No data points left in {file_path} after applying SELECTED_X_VALUES"
        )

    filtered_comm_sizes = [point[0] for point in filtered_points]
    filtered_bus_bandwidths = [point[1] for point in filtered_points]
    return filtered_comm_sizes, filtered_bus_bandwidths


def load_plot_series(
    input_files: list[str | Path],
    selected_x_values: list[str | int] | None = None,
) -> tuple[list[tuple[Path, list[int], list[float]]], list[int]]:
    if selected_x_values is None:
        selected_x_values = []
    selected_x_values = normalize_selected_x_values(selected_x_values)
    selected_x_value_set = set(selected_x_values)
    all_comm_sizes: list[int] = []
    series: list[tuple[Path, list[int], list[float]]] = []

    for file_path in input_files:
        comm_sizes, bus_bandwidths = load_points(file_path)
        comm_sizes, bus_bandwidths = filter_points(
            comm_sizes, bus_bandwidths, selected_x_value_set, file_path
        )
        all_comm_sizes.extend(comm_sizes)
        series.append((Path(file_path), comm_sizes, bus_bandwidths))

    if not series:
        raise ValueError("No input series available for plotting")

    x_ticks = (
        selected_x_values
        if selected_x_values
        else build_x_ticks(min(all_comm_sizes), max(all_comm_sizes))
    )
    return series, x_ticks


def configure_x_axis(ax: plt.Axes, x_ticks: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([format_message_size(tick) for tick in x_ticks])
    ax.set_xlim(x_ticks[0], x_ticks[-1])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def save_figure(fig: plt.Figure, output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()

    savefig_kwargs = {}
    if output_path.suffix:
        savefig_kwargs["format"] = output_path.suffix.lstrip(".")

    fig.savefig(output_path, **savefig_kwargs)
    plt.close(fig)
    print(f"Saved figure to {output_path.resolve()}")


def plot_bw(
    input_files: list[str | Path],
    output_file: str | Path,
    selected_x_values: list[str | int] | None = None,
) -> None:
    series, x_ticks = load_plot_series(input_files, selected_x_values)

    fig, ax = plt.subplots(figsize=(10, 6))
    color_map = plt.get_cmap("tab10")

    for index, (file_path, comm_sizes, bus_bandwidths) in enumerate(series):
        ax.plot(
            comm_sizes,
            bus_bandwidths,
            marker="o",
            linewidth=2,
            markersize=5,
            color=color_map(index % color_map.N),
            label=file_path.stem,
        )

    configure_x_axis(ax, x_ticks)
    ax.set_xlabel("Comm size (bytes)")
    ax.set_ylabel("Bus bandwidth (GB/s)")
    ax.set_title("Bus bandwidth vs comm size")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    save_figure(fig, output_file)


def plot_bw_norm(
    input_files: list[str | Path],
    output_file: str | Path,
    selected_x_values: list[str | int] | None = None,
) -> None:
    series, x_ticks = load_plot_series(input_files, selected_x_values)
    baseline_path, baseline_comm_sizes, baseline_bus_bandwidths = series[0]
    baseline_bandwidth_map = dict(zip(baseline_comm_sizes, baseline_bus_bandwidths))

    normalized_x_ticks = [tick for tick in x_ticks if tick in baseline_bandwidth_map]
    if not normalized_x_ticks:
        raise ValueError(
            f"No baseline x values available in {baseline_path} for normalized plot"
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    color_map = plt.get_cmap("tab10")
    average_bandwidths: list[tuple[str, float]] = []

    for index, (file_path, comm_sizes, bus_bandwidths) in enumerate(series):
        normalized_points: list[tuple[int, float]] = []
        for comm_size, bus_bandwidth in zip(comm_sizes, bus_bandwidths):
            if comm_size not in baseline_bandwidth_map:
                continue

            baseline_bandwidth = baseline_bandwidth_map[comm_size]
            if baseline_bandwidth == 0:
                raise ValueError(
                    f"Baseline bandwidth is zero at {format_message_size(comm_size)} "
                    f"in {baseline_path}"
                )

            normalized_points.append((comm_size, bus_bandwidth / baseline_bandwidth))

        if not normalized_points:
            raise ValueError(
                f"No comparable data points left in {file_path} after normalization"
            )

        normalized_comm_sizes = [point[0] for point in normalized_points]
        normalized_bandwidths = [point[1] for point in normalized_points]
        average_bandwidths.append(
            (file_path.stem, sum(normalized_bandwidths) / len(normalized_bandwidths))
        )
        ax.plot(
            normalized_comm_sizes,
            normalized_bandwidths,
            marker="o",
            linewidth=2,
            markersize=5,
            color=color_map(index % color_map.N),
            label=file_path.stem,
        )

    configure_x_axis(ax, normalized_x_ticks)
    ax.set_xlabel("Comm size (bytes)")
    ax.set_ylabel("Normalized bus bandwidth")
    ax.set_title("Normalized bus bandwidth vs comm size")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    save_figure(fig, output_file)
    print("Average speed-up:")
    for label, average in average_bandwidths:
        print(f"  {label}: {average:.6f}")


def main() -> None:
    if not INPUT_FILES:
        return

    plot_bw(INPUT_FILES, OUTPUT_FILE, SELECTED_X_VALUES)
    plot_bw_norm(INPUT_FILES, NORMALIZED_OUTPUT_FILE, SELECTED_X_VALUES)


if __name__ == "__main__":
    main()
