"""
Domain decomposition pass: annotate ``stencil.ApplyOp`` for multi-GPU execution.

For each ``stencil.ApplyOp`` with ``halo_widths``, this pass determines how
the domain should be split across GPUs and attaches the decomposition metadata.

Attached attributes (on each ``stencil.ApplyOp``):
    ``num_gpus``         -- ``IntAttr``: number of GPUs.
    ``split_axis``       -- ``IntAttr``: axis along which the domain is split.
    ``topology``         -- ``ArrayAttr[IntAttr]``: GPU grid topology.
    ``overlap_enabled``  -- ``IntAttr``: 1 if compute-communication overlap.
    ``interior_size``    -- ``IntAttr``: interior tile count (halo-independent).
    ``boundary_size``    -- ``IntAttr``: boundary tile count (halo-dependent).
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
class DecompositionPass(ModulePass):
    """Annotate stencil ops for multi-GPU domain decomposition.

    Parameters
    ----------
    num_gpus : int
        Total number of GPUs to decompose across.
    topology : tuple[int, ...] | None
        GPU grid topology (e.g. ``(2, 2)`` for a 2x2 Cartesian grid).
        ``None`` means 1-D split along the longest axis (axis 0 default).
    overlap : bool
        Whether to enable compute-communication overlap.
    """

    name: ClassVar[str] = "domain-decomposition"

    num_gpus: int = 1
    topology: tuple[int, ...] | None = None
    overlap: bool = True

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        if self.num_gpus <= 1:
            return

        for child in op.walk():
            if isinstance(child, ApplyOp):
                self._decompose(child)

    # ------------------------------------------------------------------

    def _decompose(self, apply_op: ApplyOp) -> None:
        # Determine ndim from halo_widths or ndim attribute
        halo_attr = apply_op.attributes.get("halo_widths")
        ndim_attr = apply_op.attributes.get("ndim")

        if halo_attr is not None:
            ndim = len(halo_attr)
        elif ndim_attr is not None:
            ndim = ndim_attr.data
        else:
            ndim = 2  # safe default

        # --- Determine topology and split axis ---
        if self.topology is not None:
            topo = self.topology
            # Primary split axis is the first axis with extent > 1
            split_axis = 0
            for d, extent in enumerate(topo):
                if extent > 1:
                    split_axis = d
                    break
        else:
            # 1-D split: default to axis 0 (longest axis in row-major)
            topo = (self.num_gpus,) + (1,) * max(0, ndim - 1)
            split_axis = 0

        # --- Interior / boundary sizing ---
        # Halo width on the split axis determines the boundary region.
        halo_on_split = 0
        if halo_attr is not None and split_axis < len(halo_attr):
            halo_on_split = halo_attr.data[split_axis].data

        # Tile size on the split axis (if tiling has been done)
        tile_attr = apply_op.attributes.get("tile_sizes")
        tile_on_split = 0
        if tile_attr is not None and split_axis < len(tile_attr):
            tile_on_split = tile_attr.data[split_axis].data

        # boundary_size = number of tile rows that overlap with halo
        # interior_size = remaining tile rows
        if tile_on_split > 0 and halo_on_split > 0:
            boundary_tiles = math.ceil(halo_on_split / tile_on_split) * 2
            interior_tiles = max(0, tile_on_split - boundary_tiles)
        else:
            boundary_tiles = halo_on_split * 2  # one halo layer per side
            interior_tiles = max(0, 1 - boundary_tiles)

        # --- Attach attributes ---
        apply_op.attributes["num_gpus"] = IntAttr(self.num_gpus)
        apply_op.attributes["split_axis"] = IntAttr(split_axis)
        apply_op.attributes["topology"] = ArrayAttr(
            [IntAttr(t) for t in topo]
        )
        apply_op.attributes["overlap_enabled"] = IntAttr(
            1 if self.overlap else 0
        )
        apply_op.attributes["interior_size"] = IntAttr(interior_tiles)
        apply_op.attributes["boundary_size"] = IntAttr(boundary_tiles)
