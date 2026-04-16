"""
Tiling pass: select tile sizes for shared-memory stencil execution.

Thin wrapper around :class:`cutile.passes.analysis.tile_selection.OverheadHeuristic`.
The heuristic + smem model + fallback are unified so the empirical
autotuner and the static pass cannot drift apart.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import ClassVar

from xdsl.context import Context
from xdsl.dialects.builtin import ArrayAttr, FileLineColLoc, IntAttr, ModuleOp
from xdsl.dialects.stencil import AccessOp, ApplyOp
from xdsl.passes import ModulePass

from cutile.passes.analysis.tile_selection import (
    OverheadHeuristic,
    fallback_tile,
)


@dataclass(frozen=True)
class TilingPass(ModulePass):
    """Attach ``tile_sizes`` to each ``stencil.ApplyOp``."""

    name: ClassVar[str] = "stencil-tiling"

    shared_mem_bytes: int = 49152
    dtype_bytes: int = 8
    candidate_sizes: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256)

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        selector = OverheadHeuristic(
            shared_mem_bytes=self.shared_mem_bytes,
            dtype_bytes=self.dtype_bytes,
            candidate_widths=self.candidate_sizes,
        )
        for child in op.walk():
            if isinstance(child, ApplyOp):
                self._tile_apply(child, selector)

    def _tile_apply(self, apply_op: ApplyOp,
                    selector: OverheadHeuristic) -> None:
        halo_attr = apply_op.attributes.get("halo_widths")
        if halo_attr is None:
            return

        halo: tuple[int, ...] = tuple(a.data for a in halo_attr)
        ndim = len(halo)

        num_loads = max(
            sum(1 for c in apply_op.region.block.ops
                if isinstance(c, AccessOp)),
            1,
        )

        chosen = selector.select(ndim, halo, num_loads)
        if chosen is None:
            chosen = fallback_tile(ndim)
            loc = getattr(apply_op, "location", None)
            loc_prefix = ""
            if isinstance(loc, FileLineColLoc):
                loc_prefix = f"{loc.filename.data}:{loc.line.data} -- "
            warnings.warn(
                f"{loc_prefix}stencil-tiling: no candidate tile size fits in "
                f"shared memory ({selector.shared_mem_bytes} B); "
                f"falling back to {chosen}",
                stacklevel=2,
            )

        apply_op.attributes["tile_sizes"] = ArrayAttr(
            [IntAttr(s) for s in chosen]
        )
