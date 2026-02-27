"""Analytical roofline model for stencil kernels.

Counts FLOPs from AST (additions, multiplications, divisions),
counts unique array loads, and classifies the kernel as
memory-bound or compute-bound.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from cutile_stencil.dsl.types import HardwareSpec, StencilSpec, RooflineResult


class _FlopCounter(ast.NodeVisitor):
    """Count arithmetic operations in a function body."""

    def __init__(self):
        self.adds = 0
        self.muls = 0
        self.divs = 0

    def visit_BinOp(self, node: ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            self.adds += 1
        elif isinstance(node.op, ast.Mult):
            self.muls += 1
        elif isinstance(node.op, (ast.Div, ast.FloorDiv)):
            self.divs += 1
        self.generic_visit(node)

    @property
    def total(self) -> int:
        return self.adds + self.muls + self.divs


def _count_unique_loads(spec: StencilSpec) -> int:
    """Number of distinct array element loads (accesses)."""
    if not spec.accesses:
        return max(1, len(spec.inputs))
    seen = set()
    for acc in spec.accesses:
        seen.add((acc.array_name, acc.offsets))
    return len(seen)


def roofline_analysis(
    spec: StencilSpec,
    hw: HardwareSpec | None = None,
    peak_gflops: float = 10000.0,
) -> RooflineResult:
    """Compute roofline metrics for a stencil."""
    if hw is None:
        hw = HardwareSpec()

    src = textwrap.dedent(inspect.getsource(spec.update_fn))
    tree = ast.parse(src)
    counter = _FlopCounter()
    counter.visit(tree)
    flops = max(counter.total, 1)

    loads = _count_unique_loads(spec)
    # One store per output point
    bytes_per_point = (loads + 1) * hw.dtype_bytes

    ai = flops / bytes_per_point  # FLOPs / byte

    # Memory-bound ceiling: BW * AI
    mem_ceiling_gflops = hw.peak_bandwidth_gbs * ai
    bound = "memory" if mem_ceiling_gflops < peak_gflops else "compute"

    effective_gflops = min(mem_ceiling_gflops, peak_gflops)
    peak_gpoints = effective_gflops / flops  # G-points/s

    return RooflineResult(
        flops_per_point=flops,
        bytes_per_point=bytes_per_point,
        arithmetic_intensity=ai,
        bound=bound,
        peak_gpoints_s=peak_gpoints,
    )
