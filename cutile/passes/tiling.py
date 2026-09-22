"""
Tiling pass: select tile shapes for cuTile stencil execution.

cuTile executes a tile with a fixed 128-thread block, so a tile's element
count is simply how many elements each thread owns.  Two measured facts on
Blackwell drive the choice (see README, "Performance notes"):

* The innermost tile dimension must span the row.  Near-cubic tiles such as
  ``(4, 8, 8)`` load 64-byte row fragments and run ~3x slower than a row
  tile of the same size.
* tileiras keeps every loaded tile live until it is consumed, so a stencil
  with many accesses needs one element per thread (a 27-point stencil at two
  elements per thread doubles its registers and runs 1.6x slower), while a
  narrow, memory-bound stencil gains ~8% from two elements per thread.

For each ``stencil.ApplyOp`` that has ``halo_widths`` (set by
``AnalysisPass``), this pass attaches ``tile_sizes`` (``ArrayAttr[IntAttr]``)
and, for very wide stencils, ``occupancy`` (``IntAttr``):

* narrow (at most ``wide_stencil_loads`` accesses): two elements per thread
  in a single row, the shape that streams best when memory-bound;
* wide: one element per thread, as two rows of ``wide_inner`` (64) elements.
  A y-shifted access re-fetches ``rows + 2r`` rows per tile, so two rows
  halve that traffic; a 64-wide row is still four full 128-byte lines.
  Measured: 27-point equal, 25-point 4% and a 49-point in-plane box 20%
  faster than one row of 128;
* very wide (more than ``very_wide_stencil_loads`` accesses): two elements
  per thread plus a ``@ct.kernel(occupancy=4)`` hint (a 128-register
  budget).  One element per thread would need ~2 registers per access (254
  for 125 accesses, one block per SM); with the hint tileiras schedules the
  two-element tile in 62 registers (125-point box: 9.4 ms -> 6.5 ms).

The access count only ranks the configurations; the register count of the
compiled kernel decides.  :meth:`TilingPass.candidates` returns the tier's
``(tile, hints)`` pairs in order of preference, the first of which this pass
attaches, and :func:`cutile.runtime.compile` keeps the first whose compiled
kernel neither spills nor exceeds the register budget of the occupancy
target (see :mod:`cutile.runtime.regcheck`).  Wide stencils fall over to the
hinted two-element tile when one element per thread compiles to too many
registers: a 56-access stencil made of two-field products takes 210
registers unhinted (two blocks per SM, 5.2 ms) and 64 with the hint
(3.4 ms), while the 27-point box fits in 84 registers and is fastest
unhinted.  Very wide stencils fall back to one element per thread without
a hint when the hinted kernel spills.  Also:

* the tile has ``threads_per_block * elements_per_thread`` elements, laid
  along the innermost dimension up to ``max_inner`` and then along the
  second-innermost;
* 1D stencils use a 4x larger tile since the whole array is one row.

The array shape is unknown at compile time, so the generated launcher clamps
each tile dimension to the next power of two of the interior extent
(``_fit_tiles``).  ``shared_mem_bytes`` and ``dtype_bytes`` are accepted for
compatibility with older pipelines but no longer influence the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from xdsl.context import Context
from xdsl.dialects.builtin import ArrayAttr, IntAttr, ModuleOp
from xdsl.dialects.stencil import AccessOp, ApplyOp
from xdsl.passes import ModulePass


def unique_access_count(apply_op: ApplyOp) -> int:
    """Number of distinct (field, offset) values a ``stencil.apply`` reads.

    The emitter loads each distinct access once, so this is the number of
    tile loads the kernel will issue; repeated uses of the same value in the
    expression do not add loads.
    """
    seen = set()
    for child in apply_op.region.walk():
        if isinstance(child, AccessOp):
            seen.add((id(child.temp), tuple(child.offset)))
    return len(seen)


@dataclass(frozen=True)
class TilingPass(ModulePass):
    """Select row-shaped tile sizes from the stencil's access count.

    Parameters
    ----------
    shared_mem_bytes, dtype_bytes : int
        Unused; kept so existing pipelines keep constructing the pass.
    threads_per_block : int
        Threads cuTile assigns to one tile (128).
    wide_stencil_loads : int
        Stencils with more accesses than this are "wide".
    very_wide_stencil_loads : int
        Stencils with more accesses than this are "very wide".
    narrow_elements_per_thread, wide_elements_per_thread : int
        Elements each thread owns for narrow / wide stencils.
    very_wide_occupancy : int
        ``occupancy`` hint attached to very wide stencils; also the top of
        the hint ladder :meth:`candidates` ranks (down to ``min_occupancy``).
    min_occupancy : int
        Lowest occupancy hint the ladder tries before giving up on hints.
    wide_inner : int
        Row width (innermost tile dimension) for the wide tier.
    max_inner : int
        Cap on the innermost tile dimension (elements).
    """

    name: ClassVar[str] = "stencil-tiling"

    shared_mem_bytes: int = 49152
    dtype_bytes: int = 8
    threads_per_block: int = 128
    wide_stencil_loads: int = 16
    very_wide_stencil_loads: int = 64
    narrow_elements_per_thread: int = 2
    wide_elements_per_thread: int = 1
    very_wide_occupancy: int = 4
    min_occupancy: int = 2
    wide_inner: int = 64
    max_inner: int = 1024

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for child in op.walk():
            if isinstance(child, ApplyOp):
                self._tile_apply(child)

    # ------------------------------------------------------------------

    def _tile_apply(self, apply_op: ApplyOp) -> None:
        halo_attr = apply_op.attributes.get("halo_widths")
        if halo_attr is None:
            return

        ndim = len(halo_attr)
        num_loads = unique_access_count(apply_op)
        tile, hints = self.candidates(ndim, num_loads)[0]
        apply_op.attributes["tile_sizes"] = ArrayAttr([IntAttr(s) for s in tile])
        if "occupancy" in hints:
            apply_op.attributes["occupancy"] = IntAttr(hints["occupancy"])

    # ------------------------------------------------------------------

    def tier(self, num_loads: int) -> str:
        """``"narrow"``, ``"wide"`` or ``"very_wide"`` by access count."""
        if num_loads <= self.wide_stencil_loads:
            return "narrow"
        if num_loads > self.very_wide_stencil_loads:
            return "very_wide"
        return "wide"

    def elements_per_thread(self, num_loads: int) -> int:
        if self.tier(num_loads) == "wide":
            return self.wide_elements_per_thread
        return self.narrow_elements_per_thread  # very wide: paired with the occupancy hint

    def is_wide(self, num_loads: int) -> bool:
        return self.tier(num_loads) == "wide"

    def candidates(self, ndim: int, num_loads: int) -> list[tuple[tuple[int, ...], dict]]:
        """``(tile_sizes, kernel_hints)`` configurations, best guess first.

        Narrow stencils have a single configuration.  Wide and very wide
        stencils share a list in opposite orders: the one-element tile
        without a hint, and a ladder of hinted configurations, the
        two-element tile at the occupancy target followed by the one-element
        tile at the target and at each lower occupancy down to
        ``min_occupancy`` (a 144-load fused kernel spills at occupancy 4 but
        fits 164 registers at occupancy 3).  The runtime keeps the first
        whose compiled kernel fits its budget (see module doc).
        """
        one_row = self._row_tile(ndim, self.wide_elements_per_thread, self.wide_inner)
        two_rows = self._row_tile(ndim, self.narrow_elements_per_thread, self.max_inner)
        tier = self.tier(num_loads)
        if tier == "narrow":
            return [(two_rows, {})]
        ladder = [(two_rows, {"occupancy": self.very_wide_occupancy})] + [
            (one_row, {"occupancy": occ})
            for occ in range(self.very_wide_occupancy, self.min_occupancy - 1, -1)
        ]
        unhinted = (one_row, {})
        if tier == "wide":
            return [unhinted] + ladder
        return ladder + [unhinted]

    def tile_for(self, ndim: int, num_loads: int) -> list[int]:
        """Row tile for a stencil with *num_loads* accesses (first candidate)."""
        return list(self.candidates(ndim, num_loads)[0][0])

    def _row_tile(self, ndim: int, elements_per_thread: int, inner_cap: int) -> tuple[int, ...]:
        elements = self.threads_per_block * elements_per_thread
        if ndim == 1:
            return (elements * 4,)
        tile = [1] * ndim
        inner = min(elements, inner_cap)
        tile[-1] = inner
        tile[-2] = max(1, elements // inner)
        return tuple(tile)
