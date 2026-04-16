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
from xdsl.dialects.builtin import ArrayAttr, FileLineColLoc, IntAttr, ModuleOp, StringAttr
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
    candidate_sizes: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256)

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for child in op.walk():
            if isinstance(child, ApplyOp):
                self._tile_apply(child)

    # ------------------------------------------------------------------

    def _tile_apply(self, apply_op: ApplyOp) -> None:
        from itertools import product as iterproduct
        from xdsl.dialects.stencil import AccessOp

        # Must have halo_widths (set by AnalysisPass)
        halo_attr = apply_op.attributes.get("halo_widths")
        if halo_attr is None:
            return

        halo: list[int] = [a.data for a in halo_attr]
        ndim: int = len(halo)

        # Count actual number of unique loads (stencil accesses) + 1 store
        num_loads = 0
        for child_op in apply_op.region.block.ops:
            if isinstance(child_op, AccessOp):
                num_loads += 1
        num_loads = max(num_loads, 1)
        num_tiles = num_loads + 1  # loads + 1 store

        # Generate candidates: all combinations of candidate sizes per dim
        # For 3D, also include non-square tiles (e.g., 4x4x32, 8x8x16)
        candidates: list[tuple[int, ...]] = []
        for combo in iterproduct(self.candidate_sizes, repeat=ndim):
            candidates.append(combo)

        best_tile: tuple[int, ...] | None = None
        best_overhead: float = float("inf")

        for tile in candidates:
            # Each ct.load brings a (tile,) shaped tile into shared memory
            # (halo is realised by shifting source slices, not by expanding
            # the smem tile). Same for the store. So smem = num_tiles *
            # prod(tile). The halo expansion only matters for bandwidth
            # accounting (redundant loads at CTA boundaries).
            prod_tile = math.prod(tile)
            smem_usage = num_tiles * prod_tile * self.dtype_bytes

            if smem_usage > self.shared_mem_bytes:
                continue

            # Halo overhead: fraction of the loaded data that is redundant
            # halo (lower = better bandwidth utilisation).
            expanded = tuple(t + 2 * h for t, h in zip(tile, halo))
            overhead = 1.0 - prod_tile / math.prod(expanded)
            if overhead < best_overhead:
                best_overhead = overhead
                best_tile = tile

        # Fallback per ndim
        if best_tile is None:
            fallback = {1: (32,), 2: (16, 16), 3: (8, 8, 8)}
            best_tile = fallback.get(ndim, (8,) * ndim)
            loc = getattr(apply_op, "location", None)
            loc_prefix = ""
            if isinstance(loc, FileLineColLoc):
                loc_prefix = f"{loc.filename.data}:{loc.line.data} -- "
            import warnings
            warnings.warn(
                f"{loc_prefix}stencil-tiling: no candidate tile size fits in "
                f"shared memory ({self.shared_mem_bytes} B); "
                f"falling back to {best_tile}",
                stacklevel=2,
            )

        apply_op.attributes["tile_sizes"] = ArrayAttr(
            [IntAttr(s) for s in best_tile]
        )
