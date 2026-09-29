"""Fused stencil lowering: several Dialect 1 modules to one cuTile kernel.

Stencils over the same domain (CNS hypterm: five outputs from eight fields;
Gray-Scott u/v) are compiled into a single ``@ct.kernel`` that loads each
distinct (field, offset) once and stores every output.  The kernel goes
through the same Dialect 3 builders and emitter as a single stencil, so it
gets the same row tiles, inline loads, exact constants and kernel hints;
see :func:`cutile.lowering.stencil_to_target.extract_fused_meta` for how
inputs are merged by parameter name.

Public API
----------
lower_fused_stencils_to_python(modules, tile_sizes, halo_widths, ...)
    -> str  (complete Python source with fused kernel + launcher)
"""

from __future__ import annotations

from typing import Sequence

from xdsl.dialects.builtin import ModuleOp

from cutile.lowering.stencil_to_target import extract_fused_meta, lower_fused_to_target_ir
from cutile.lowering.target_to_python import emit_python


def lower_fused_stencils_to_python(
    modules: Sequence[ModuleOp],
    tile_sizes: tuple[int, ...] | None = None,
    halo_widths: tuple[int, ...] | None = None,
    temporal_steps: int = 1,
    kernel_hints: dict | None = None,
    spatial_term_order: bool | None = None,
) -> str:
    """Lower several Dialect 1 modules into one fused kernel and launcher.

    ``tile_sizes`` and ``halo_widths`` default to the tiling pass's choice
    for the merged access count and to the widest member's footprint.
    Temporal blocking is not available for fused kernels.
    """
    if temporal_steps != 1:
        raise NotImplementedError(
            "temporal blocking of fused kernels is not supported (temporal_steps must be 1)"
        )
    if tile_sizes is None or halo_widths is None:
        from cutile.passes.tiling import TilingPass

        meta = extract_fused_meta(modules)
        if halo_widths is None:
            halo_widths = (max(1, meta.order // 2),) * meta.ndim
        if tile_sizes is None:
            tile_sizes = tuple(TilingPass().tile_for(meta.ndim, len(meta.accesses)))
    target_ir = lower_fused_to_target_ir(
        modules, tuple(tile_sizes), tuple(halo_widths), kernel_hints, spatial_term_order
    )
    return emit_python(target_ir)
