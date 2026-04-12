"""
Tiling pass: select optimal tile sizes for shared-memory stencil execution.

For each ``stencil.ApplyOp`` that has ``halo_widths`` (set by
``AnalysisPass``), this pass evaluates candidate tile sizes and picks the
configuration with minimum halo overhead that fits within the shared-memory
budget.  It attaches ``tile_sizes`` (``ArrayAttr[IntAttr]``) on the op.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from xdsl.context import Context
from xdsl.dialects.builtin import ArrayAttr, IntAttr, ModuleOp, StringAttr
from xdsl.dialects.stencil import ApplyOp
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class TilingPass(ModulePass):
    """Select tile sizes that minimise halo overhead within a shared-memory budget.

    Parameters
    ----------
    shared_mem_bytes : int
        Per-SM shared memory budget in bytes.
    dtype_bytes : int
        Size of a single element in bytes (8 for float64, 4 for float32).
    candidate_sizes : tuple[int, ...]
        Power-of-2 candidate tile widths to evaluate (same for all dims).
    """

    name: ClassVar[str] = "stencil-tiling"

    shared_mem_bytes: int = 49152
    dtype_bytes: int = 8
    candidate_sizes: tuple[int, ...] = (32, 64, 128, 256)

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for child in op.walk():
            if isinstance(child, ApplyOp):
                self._tile_apply(child)

    # ------------------------------------------------------------------

    def _tile_apply(self, apply_op: ApplyOp) -> None:
        # Must have halo_widths (set by AnalysisPass)
        halo_attr = apply_op.attributes.get("halo_widths")
        if halo_attr is None:
            return

        halo: list[int] = [a.data for a in halo_attr]
        ndim: int = len(halo)

        # Number of arrays accessed (input + output)
        num_arrays_attr = apply_op.attributes.get("num_arrays")
        num_arrays: int = num_arrays_attr.data if num_arrays_attr is not None else 1

        best_tile: tuple[int, ...] | None = None
        best_overhead: float = float("inf")

        for ts in self.candidate_sizes:
            tile = (ts,) * ndim

            # Shared memory: input tile (tile + 2*halo per dim) + output tile
            expanded = tuple(t + 2 * h for t, h in zip(tile, halo))
            prod_expanded = math.prod(expanded)
            prod_tile = math.prod(tile)

            smem_usage = (prod_expanded + prod_tile) * self.dtype_bytes * num_arrays

            if smem_usage > self.shared_mem_bytes:
                continue

            # Halo overhead: fraction of expanded tile that is halo
            overhead = 1.0 - prod_tile / prod_expanded
            if overhead < best_overhead:
                best_overhead = overhead
                best_tile = tile

        # Fallback: (32,) * ndim
        if best_tile is None:
            best_tile = (32,) * ndim

        apply_op.attributes["tile_sizes"] = ArrayAttr(
            [IntAttr(s) for s in best_tile]
        )
