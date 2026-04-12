"""Fused stencil lowering: multiple Dialect 1 IR modules to a single cuTile kernel.

Multi-field stencils (e.g. Gray-Scott u/v updates) that read overlapping
input data are compiled into a single ``@ct.kernel`` that loads shared data
once and computes all outputs, eliminating redundant global memory traffic.

Public API
----------
lower_fused_stencils_to_python(modules, tile_sizes, halo_widths, temporal_steps)
    -> str  (complete Python source with fused kernel + launcher)
"""

from __future__ import annotations

from typing import Sequence

from xdsl.dialects.builtin import ModuleOp

from cutile.dialects.cutile_stencil.dialect import FuncOp
from cutile.lowering.emitter import CodeEmitter
from cutile.lowering.stencil_to_cutile import (
    _AccessInfo,
    _StencilMeta,
    _extract_meta,
    _format_offset,
    _offset_expr,
)

# -------------------------------------------------------------------- #
# Internal helpers
# -------------------------------------------------------------------- #


def _find_func_and_block(module: ModuleOp):
    """Extract the FuncOp and its first block from a Dialect 1 module."""
    for op in module.body.ops:
        if isinstance(op, FuncOp):
            block = list(op.body.blocks)[0]
            return op, block
    raise ValueError("No cutile_stencil.FuncOp found in module")


def _merge_accesses(
    metas: list[_StencilMeta],
    global_input_names: list[str],
    name_remap: list[dict[str, str]],
) -> list[_AccessInfo]:
    """Merge accesses from all stencils, deduplicating by (global_name, offsets).

    Parameters
    ----------
    metas : list[_StencilMeta]
        Per-stencil metadata (each with its own local input_names).
    global_input_names : list[str]
        Global unique names for all input arrays across all stencils.
    name_remap : list[dict[str, str]]
        Per-stencil mapping from local input name to global input name.

    Returns
    -------
    list[_AccessInfo]
        Deduplicated accesses with view_name based on global input names.
    """
    seen: dict[tuple[str, tuple[int, ...]], _AccessInfo] = {}
    merged: list[_AccessInfo] = []

    for stencil_idx, meta in enumerate(metas):
        remap = name_remap[stencil_idx]
        for acc in meta.accesses:
            local_name = meta.input_names[acc.array_index]
            global_name = remap[local_name]
            # Find the index of global_name in global_input_names
            global_idx = global_input_names.index(global_name)
            key = (global_name, acc.offsets)
            if key not in seen:
                off_parts = "_".join(_format_offset(o) for o in acc.offsets)
                view_name = f"{global_name}_{off_parts}"
                info = _AccessInfo(global_idx, acc.offsets, view_name)
                seen[key] = info
                merged.append(info)

    return merged


def _remap_expression(
    meta: _StencilMeta,
    name_remap: dict[str, str],
) -> str:
    """Rewrite a stencil expression to use global input names in tile references.

    The expression uses tile variable names like ``t_u_0_0`` where ``u`` is a
    local name.  We replace with the global name equivalent.
    """
    expr = meta.expression
    for local_name, global_name in name_remap.items():
        if local_name != global_name:
            # Replace t_localname_ with t_globalname_
            expr = expr.replace(f"t_{local_name}_", f"t_{global_name}_")
    return expr


# -------------------------------------------------------------------- #
# Fused code emission
# -------------------------------------------------------------------- #


def _emit_fused_header(
    e: CodeEmitter,
    fused_name: str,
    all_constants: dict[str, float],
) -> None:
    """Emit module header for fused kernel."""
    e.line(f'"""cuTile fused kernel for {fused_name} (auto-generated)."""')
    e.blank()
    e.line("import cuda.tile as ct")
    e.line("import cupy as cp")

    if all_constants:
        e.blank()
        e.line("# Captured constants from stencil definition scope")
        for name, value in sorted(all_constants.items()):
            e.line(f"{name} = {value!r}")


def _emit_fused_kernel(
    e: CodeEmitter,
    fused_name: str,
    ndim: int,
    global_input_names: list[str],
    output_names: list[str],
    merged_accesses: list[_AccessInfo],
    stencil_expressions: list[tuple[str, str]],
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
) -> None:
    """Emit a single ``@ct.kernel`` that computes all stencil outputs."""
    bid_vars = ["bx", "by", "bz"][:ndim]
    tile_vars = ["TX", "TY", "TZ"][:ndim]
    halo_vars = ["HX", "HY", "HZ"][:ndim]
    n_vars = ["nx", "ny", "nz"][:ndim]

    tile_const_params = ", ".join(f"{tv}: ConstInt" for tv in tile_vars)
    halo_const_params = ", ".join(f"{hv}: ConstInt" for hv in halo_vars)

    input_params = ", ".join(global_input_names)
    output_params = ", ".join(output_names)

    e.line("@ct.kernel")
    e.line(
        f"def {fused_name}_fused_kernel({input_params}, {output_params}, "
        f"{tile_const_params}, {halo_const_params}):"
    )
    with e.indent():
        # Block indices
        for i, bv in enumerate(bid_vars):
            e.line(f"{bv} = ct.bid({i})")

        # Interior sizes (use first input array)
        first_arr = global_input_names[0]
        for d in range(ndim):
            e.line(f"{n_vars[d]} = {first_arr}.shape[{d}] - 2 * {halo_vars[d]}")

        e.blank()
        e.line("# --- Sliced input views (deduplicated across stencils) ---")
        for info in merged_accesses:
            arr_name = global_input_names[info.array_index]
            chain = arr_name
            for d in range(ndim):
                off = info.offsets[d]
                start = _offset_expr(halo_vars[d], off)
                stop = _offset_expr(halo_vars[d], off, add_n=True, n_var=n_vars[d])
                chain = f"{chain}.slice(axis={d}, start={start}, stop={stop})"
            e.line(f"{info.view_name} = {chain}")

        # Output views (zero offsets, one per output)
        e.blank()
        e.line("# --- Output views ---")
        for out_name in output_names:
            out_chain = out_name
            for d in range(ndim):
                out_chain = (
                    f"{out_chain}.slice(axis={d}, start={halo_vars[d]}, "
                    f"stop={halo_vars[d]} + {n_vars[d]})"
                )
            e.line(f"out_{out_name} = {out_chain}")

        # Load tiles
        e.blank()
        e.line("# --- Load input tiles (shared across stencils) ---")
        idx_tuple = ", ".join(bid_vars)
        shape_tuple = ", ".join(tile_vars)
        if ndim == 1:
            idx_arg = f"({idx_tuple},)"
            shape_arg = f"({shape_tuple},)"
        else:
            idx_arg = f"({idx_tuple})"
            shape_arg = f"({shape_tuple})"
        for info in merged_accesses:
            e.line(
                f"t_{info.view_name} = ct.load({info.view_name}, "
                f"index={idx_arg}, shape={shape_arg})"
            )

        # Compute each stencil expression
        e.blank()
        e.line("# --- Compute stencil expressions ---")
        for result_name, expr in stencil_expressions:
            e.line(f"{result_name} = {expr}")

        # Store all outputs
        e.blank()
        e.line("# --- Store outputs ---")
        for (result_name, _), out_name in zip(stencil_expressions, output_names):
            e.line(f"ct.store(out_{out_name}, index={idx_arg}, tile={result_name})")


def _emit_fused_launcher(
    e: CodeEmitter,
    fused_name: str,
    ndim: int,
    global_input_names: list[str],
    output_names: list[str],
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
) -> None:
    """Emit a launcher function for the fused kernel."""
    t_vars = ["TX", "TY", "TZ"][:ndim]
    h_vars = ["HX", "HY", "HZ"][:ndim]
    n_vars = ["Nx", "Ny", "Nz"][:ndim]

    input_params = ", ".join(global_input_names)
    output_params = ", ".join(output_names)

    e.blank()
    e.blank()
    e.line(f"def launch_{fused_name}_fused({input_params}, {output_params}, stream=None):")
    with e.indent():
        e.line(f"{', '.join(t_vars)} = {', '.join(str(t) for t in tile_sizes)}")
        e.line(f"{', '.join(h_vars)} = {', '.join(str(h) for h in halo_widths)}")

        first_input = global_input_names[0]
        trailing = "," if ndim == 1 else ""
        e.line(f"{', '.join(n_vars)}{trailing} = {first_input}.shape")

        e.line("if stream is None:")
        with e.indent():
            e.line("stream = cp.cuda.get_current_stream()")

        grid_parts = ", ".join(
            f"ct.cdiv({n_vars[d]} - 2 * {h_vars[d]}, {t_vars[d]})"
            for d in range(ndim)
        )
        grid_trailing = "," if ndim == 1 else ""
        e.line(f"grid = ({grid_parts}{grid_trailing})")

        t_args = ", ".join(t_vars)
        h_args = ", ".join(h_vars)
        e.line(
            f"ct.launch(stream, grid, {fused_name}_fused_kernel, "
            f"({input_params}, {output_params}, {t_args}, {h_args}))"
        )


def _emit_fused_temporal_launcher(
    e: CodeEmitter,
    fused_name: str,
    ndim: int,
    global_input_names: list[str],
    output_names: list[str],
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
    temporal_steps: int,
) -> None:
    """Emit a temporal-blocking launcher for the fused kernel."""
    t_vars = ["TX", "TY", "TZ"][:ndim]
    h_vars = ["HX", "HY", "HZ"][:ndim]
    T = temporal_steps

    input_params = ", ".join(global_input_names)
    output_params = ", ".join(output_names)

    e.blank()
    e.blank()
    e.line(f"def launch_{fused_name}_fused({input_params}, {output_params}, stream=None):")
    with e.indent():
        e.line(f"{', '.join(t_vars)} = {', '.join(str(t) for t in tile_sizes)}")
        e.line(f"{', '.join(h_vars)} = {', '.join(str(h) for h in halo_widths)}")

        first_input = global_input_names[0]

        e.line("if stream is None:")
        with e.indent():
            e.line("stream = cp.cuda.get_current_stream()")

        grid_parts = []
        for d in range(ndim):
            h = halo_widths[d]
            grid_parts.append(
                f"ct.cdiv({first_input}.shape[{d}] - {2 * h}, {t_vars[d]})"
            )
        trailing = "," if ndim == 1 else ""
        e.line(f"grid = ({', '.join(grid_parts)}{trailing})")

        e.line(f"# Temporal blocking: {T} steps with buffer swapping")
        # For fused kernels with multiple inputs/outputs, we swap each pair
        for i, (inp, out) in enumerate(zip(global_input_names, output_names)):
            e.line(f"bufs_{inp} = [{inp}]")
            e.line(f"for _ in range({T - 1}):")
            with e.indent():
                e.line(f"bufs_{inp}.append(cp.zeros_like({inp}))")
            e.line(f"bufs_{inp}.append({out})")

        e.line(f"for _step in range({T}):")
        with e.indent():
            buf_inputs = ", ".join(f"bufs_{inp}[_step]" for inp in global_input_names)
            buf_outputs = ", ".join(f"bufs_{inp}[_step + 1]" for inp in global_input_names)
            t_args = ", ".join(t_vars)
            h_args = ", ".join(h_vars)
            e.line(
                f"ct.launch(stream, grid, {fused_name}_fused_kernel, "
                f"({buf_inputs}, {buf_outputs}, {t_args}, {h_args}))"
            )


# -------------------------------------------------------------------- #
# Public API
# -------------------------------------------------------------------- #


def lower_fused_stencils_to_python(
    modules: list,
    tile_sizes: tuple[int, ...] | None = None,
    halo_widths: tuple[int, ...] | None = None,
    temporal_steps: int = 1,
) -> str:
    """Lower multiple Dialect 1 stencil IR modules into a single fused kernel.

    Each module is expected to contain a single ``cutile_stencil.FuncOp``.
    The function extracts metadata from each, merges (deduplicates) their
    input accesses, and emits a single ``@ct.kernel`` with one launcher.

    Parameters
    ----------
    modules : list[ModuleOp]
        List of xDSL ``ModuleOp`` objects, one per stencil in the fusion group.
    tile_sizes : tuple[int, ...] | None
        Tile sizes per dimension. Defaults chosen based on ndim.
    halo_widths : tuple[int, ...] | None
        Halo widths per dimension. Defaults derived from stencil order.
    temporal_steps : int
        Number of temporal blocking steps. 1 means no temporal blocking.

    Returns
    -------
    str
        Complete Python source string with a fused ``@ct.kernel`` and launcher.
    """
    if not modules:
        raise ValueError("At least one module is required for fusion")

    # ---------------------------------------------------------------- #
    # 1. Extract metadata from each module
    # ---------------------------------------------------------------- #
    metas: list[_StencilMeta] = []
    for module in modules:
        func_op, block = _find_func_and_block(module)
        meta = _extract_meta(func_op, block)
        metas.append(meta)

    # Validate: all stencils must have the same ndim
    ndim = metas[0].ndim
    for i, meta in enumerate(metas):
        if meta.ndim != ndim:
            raise ValueError(
                f"All stencils must have the same ndim for fusion. "
                f"Stencil 0 has ndim={ndim}, stencil {i} has ndim={meta.ndim}"
            )

    # ---------------------------------------------------------------- #
    # 2. Build global input name mapping
    # ---------------------------------------------------------------- #
    # Each stencil has its own input_names (e.g. ["u"] or ["u", "v"]).
    # We need a global unique set. The convention is to use the stencil
    # names as a hint: if stencil "gs_u" has input "u" and stencil "gs_v"
    # has input "v", the globals are ["u", "v"].
    #
    # When two stencils share an input name (same position), they share
    # the global name. When names differ, both appear.

    global_input_names_ordered: list[str] = []
    global_input_set: set[str] = set()
    name_remap: list[dict[str, str]] = []

    for meta in metas:
        remap: dict[str, str] = {}
        for local_name in meta.input_names:
            global_name = local_name
            # If this name is already used by a different stencil with a
            # different meaning, make it unique by prefixing with stencil name.
            # For the common case (same field name = same array), just reuse.
            if global_name not in global_input_set:
                global_input_names_ordered.append(global_name)
                global_input_set.add(global_name)
            remap[local_name] = global_name
        name_remap.append(remap)

    global_input_names = global_input_names_ordered

    # ---------------------------------------------------------------- #
    # 3. Build output names (one per stencil)
    # ---------------------------------------------------------------- #
    output_names: list[str] = []
    for meta in metas:
        output_names.append(f"{meta.name}_out")

    # ---------------------------------------------------------------- #
    # 4. Apply defaults for tile_sizes and halo_widths
    # ---------------------------------------------------------------- #
    _default_tiles = {1: (1024,), 2: (64, 64), 3: (32, 32, 32)}
    if tile_sizes is None:
        tile_sizes = _default_tiles.get(ndim, (64,) * ndim)
    if halo_widths is None:
        max_order = max(m.order for m in metas)
        hw = max_order // 2 if max_order else 1
        halo_widths = (hw,) * ndim

    # ---------------------------------------------------------------- #
    # 5. Merge accesses (deduplicate across stencils)
    # ---------------------------------------------------------------- #
    merged_accesses = _merge_accesses(metas, global_input_names, name_remap)

    # ---------------------------------------------------------------- #
    # 6. Build stencil expressions (remapped to global names)
    # ---------------------------------------------------------------- #
    stencil_expressions: list[tuple[str, str]] = []
    for i, meta in enumerate(metas):
        result_name = f"result_{meta.name}"
        expr = _remap_expression(meta, name_remap[i])
        stencil_expressions.append((result_name, expr))

    # ---------------------------------------------------------------- #
    # 7. Build a fused name
    # ---------------------------------------------------------------- #
    # Derive from the individual stencil names
    stencil_names = [m.name for m in metas]
    if len(stencil_names) <= 3:
        fused_name = "_".join(stencil_names)
    else:
        fused_name = f"{stencil_names[0]}_and_{len(stencil_names) - 1}_others"

    # ---------------------------------------------------------------- #
    # 8. Merge constants from all stencils
    # ---------------------------------------------------------------- #
    all_constants: dict[str, float] = {}
    for meta in metas:
        all_constants.update(meta.constants)

    # ---------------------------------------------------------------- #
    # 9. Emit Python source
    # ---------------------------------------------------------------- #
    e = CodeEmitter()
    _emit_fused_header(e, fused_name, all_constants)
    e.blank()
    e.line("ConstInt = ct.Constant[int]")
    e.blank()

    _emit_fused_kernel(
        e, fused_name, ndim,
        global_input_names, output_names,
        merged_accesses, stencil_expressions,
        tile_sizes, halo_widths,
    )

    if temporal_steps > 1:
        _emit_fused_temporal_launcher(
            e, fused_name, ndim,
            global_input_names, output_names,
            tile_sizes, halo_widths, temporal_steps,
        )
    else:
        _emit_fused_launcher(
            e, fused_name, ndim,
            global_input_names, output_names,
            tile_sizes, halo_widths,
        )

    return e.render()
