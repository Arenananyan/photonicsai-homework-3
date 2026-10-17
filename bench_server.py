"""The Homework 2 optical bench retained as an MCP tool, using binary macropixels."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vendor/homework2_bench"))
import numpy as np
from fastmcp import FastMCP
from simulator import BenchConfig, SimulatedBench

mcp = FastMCP("Homework 2 simulated optical bench")
bench = SimulatedBench(BenchConfig(seed=42))


@mcp.tool
def measure_binary_pattern(on_indices: list[int]) -> dict:
    """Set on/off macropixels (indices 0..255) and measure simulated camera enhancement."""
    if len(set(on_indices)) != len(on_indices) or any(
        type(i) is not int or not 0 <= i < 256 for i in on_indices
    ):
        raise ValueError("Unique integer macropixel indices 0..255 required")
    pattern = np.zeros(256)
    pattern[on_indices] = 1
    measured = bench.measure(pattern.reshape(16, 16))
    return dict(
        simulated=True,
        on_count=len(on_indices),
        enhancement=measured.enhancement,
        target_mean=measured.target_mean,
        background_mean=measured.background_mean,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
