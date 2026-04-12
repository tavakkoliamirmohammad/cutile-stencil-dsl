"""
Temporal blocking pass: fuse multiple time-steps into a single GPU kernel.

For each ``stencil.ApplyOp`` that already has ``tile_sizes`` and
``halo_widths``, this pass determines the maximum number of time steps
that can be fused while keeping the expanded tile (with compounding
halos) within the shared-memory budget.

Attached attributes:
    ``temporal_steps``        -- ``IntAttr``: number of fused time steps.
    ``expanded_halo``         -- ``ArrayAttr[IntAttr]``: T * halo per dim.
    ``temporal_buffer_count`` -- ``IntAttr``: T + 1 intermediate buffers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from xdsl.context import Context
from xdsl.dialects.builtin import ArrayAttr, IntAttr, ModuleOp
from xdsl.dialects.stencil import ApplyOp
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class TemporalBlockingPass(ModulePass):
    """Determine temporal-blocking depth for tiled stencil kernels.

    Parameters
    ----------
    max_steps : int
        Maximum temporal blocking depth to consider.
    shared_mem_bytes : int
        Per-SM shared memory budget in bytes.
    dtype_bytes : int
        Size of a single element in bytes.
    """

    name: ClassVar[str] = "temporal-blocking"

    max_steps: int = 8
    shared_mem_bytes: int = 49152
    dtype_bytes: int = 8

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for child in op.walk():
            if isinstance(child, ApplyOp):
                self._process_apply(child)

    # ------------------------------------------------------------------

    def _process_apply(self, apply_op: ApplyOp) -> None:
        tile_attr = apply_op.attributes.get("tile_sizes")
        halo_attr = apply_op.attributes.get("halo_widths")

        if tile_attr is None or halo_attr is None:
            return

        tile: list[int] = [a.data for a in tile_attr]
        halo: list[int] = [a.data for a in halo_attr]

        best_t: int = 1

        for t in range(self.max_steps, 1, -1):
            # Expanded tile: tile[d] + 2 * T * halo[d] per dimension
            expanded = [tile[d] + 2 * t * halo[d] for d in range(len(tile))]
            # Double-buffer shared memory requirement
            smem = math.prod(expanded) * self.dtype_bytes * 2

            if smem <= self.shared_mem_bytes:
                best_t = t
                break

        # Attach attributes
        apply_op.attributes["temporal_steps"] = IntAttr(best_t)
        apply_op.attributes["expanded_halo"] = ArrayAttr(
            [IntAttr(best_t * h) for h in halo]
        )
        apply_op.attributes["temporal_buffer_count"] = IntAttr(best_t + 1)
