"""Fused stencil lowering: multiple Dialect 1 IR modules to a single cuTile kernel.

Multi-field stencils (e.g. Gray-Scott u/v updates) that read overlapping
input data are compiled into a single ``@ct.kernel`` that loads shared data
once and computes all outputs, eliminating redundant global memory traffic.

The fused kernel and launcher are generated through Dialect 3 (cutile_target)
IR: a ``KernelOp`` with merged accesses and multiple stores is built, then
emitted via ``emit_python``.

Public API
----------
lower_fused_stencils_to_python(modules, tile_sizes, halo_widths, temporal_steps)
    -> str  (complete Python source with fused kernel + launcher)
"""

from __future__ import annotations

from typing import Sequence

from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
    ModuleOp,
    StringAttr,
)
from xdsl.ir import Block, Region

from cutile.dialects.cutile_stencil.dialect import FuncOp
from cutile.lowering.stencil_to_cutile import (
    _AccessInfo,
    _StencilMeta,
    _extract_meta,
    _format_offset,
    _offset_expr,
)
from cutile.dialects.cutile_target.dialect import (
    BidOp,
    ForLoopOp,
    HostProgramOp,
    KernelOp,
    LaunchOp,
    LoadOp,
    ReturnOp,
    SliceOp,
    StoreOp,
)
from cutile.lowering.target_to_python import emit_python
from xdsl.dialects.builtin import IndexType

# -------------------------------------------------------------------- #
# Internal helpers
# -------------------------------------------------------------------- #

_IDX = IndexType()


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
# Dialect 3 IR builders for fused kernels
# -------------------------------------------------------------------- #


def _build_fused_kernel_body(
    fused_name: str,
    ndim: int,
    global_input_names: list[str],
    output_names: list[str],
    merged_accesses: list[_AccessInfo],
    stencil_expressions: list[tuple[str, str]],
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
) -> Region:
    """Build the body region for a fused ``KernelOp``.

    The region contains BidOps, SliceOp chains for each merged access,
    SliceOp chains for each output, LoadOps, StoreOps, and ReturnOp.
    """
    halo_vars = ["HX", "HY", "HZ"][:ndim]
    n_vars = ["nx", "ny", "nz"][:ndim]

    # Block args: one per input + one per output
    num_block_args = len(global_input_names) + len(output_names)
    block = Block(arg_types=[_IDX] * num_block_args)

    # BidOps
    for d in range(ndim):
        bid = BidOp.build(properties={"axis": IntAttr(d)}, result_types=[_IDX])
        block.add_op(bid)

    # SliceOp chains for each merged access
    for info in merged_accesses:
        arr_arg = block.args[info.array_index]
        prev_result = arr_arg
        for d in range(ndim):
            off = info.offsets[d]
            start_expr = _offset_expr(halo_vars[d], off)
            stop_expr = _offset_expr(halo_vars[d], off, add_n=True, n_var=n_vars[d])
            is_last = (d == ndim - 1)
            props = {
                "axis": IntAttr(d),
                "start": StringAttr(start_expr),
                "stop": StringAttr(stop_expr),
            }
            if is_last:
                props["var_name"] = StringAttr(info.view_name)
            s = SliceOp.build(
                properties=props,
                operands=[prev_result],
                result_types=[_IDX],
            )
            block.add_op(s)
            prev_result = s.result

        # LoadOp
        load = LoadOp.build(
            properties={"view_name": StringAttr(info.view_name)},
            operands=[prev_result],
            result_types=[_IDX],
        )
        block.add_op(load)

    # Output view slice chains
    for out_idx, out_name in enumerate(output_names):
        out_arg = block.args[len(global_input_names) + out_idx]
        prev_result = out_arg
        for d in range(ndim):
            is_last = (d == ndim - 1)
            props = {
                "axis": IntAttr(d),
                "start": StringAttr(halo_vars[d]),
                "stop": StringAttr(f"{halo_vars[d]} + {n_vars[d]}"),
            }
            if is_last:
                props["var_name"] = StringAttr(f"out_{out_name}")
            s = SliceOp.build(
                properties=props,
                operands=[prev_result],
                result_types=[_IDX],
            )
            block.add_op(s)
            prev_result = s.result

        # StoreOp (one per output)
        store = StoreOp.build(operands=[prev_result, prev_result])
        block.add_op(store)

    # ReturnOp
    ret = ReturnOp.build()
    block.add_op(ret)

    return Region([block])


def _build_fused_kernel_op(
    fused_name: str,
    ndim: int,
    global_input_names: list[str],
    output_names: list[str],
    merged_accesses: list[_AccessInfo],
    stencil_expressions: list[tuple[str, str]],
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
    all_constants: dict[str, float],
) -> KernelOp:
    """Build a fused ``KernelOp`` from merged metadata."""
    body = _build_fused_kernel_body(
        fused_name, ndim,
        global_input_names, output_names,
        merged_accesses, stencil_expressions,
        tile_sizes, halo_widths,
    )

    # Build constants array
    constants_items: list = []
    if all_constants:
        for k, v in sorted(all_constants.items()):
            constants_items.append(StringAttr(k))
            constants_items.append(StringAttr(repr(v)))

    # For the fused kernel, input_names includes both inputs and outputs
    all_names = global_input_names + output_names

    # Build a multi-expression string: "result_name1=expr1;result_name2=expr2"
    expr_parts = [f"{rn}={ex}" for rn, ex in stencil_expressions]
    fused_expression = ";".join(expr_parts)

    props: dict = {
        "kernel_name": StringAttr(f"{fused_name}_fused"),
        "tile_shape": ArrayAttr([IntAttr(t) for t in tile_sizes]),
        "halo": ArrayAttr([IntAttr(h) for h in halo_widths]),
        "ndim": IntAttr(ndim),
        "input_names": ArrayAttr([StringAttr(n) for n in all_names]),
        "expression": StringAttr(fused_expression),
    }
    if constants_items:
        props["constants"] = ArrayAttr(constants_items)

    return KernelOp.build(
        properties=props,
        operands=[[]],
        regions=[body],
    )


def _build_fused_launcher_preamble(
    fused_name: str,
    ndim: int,
    global_input_names: list[str],
    output_names: list[str],
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
) -> str:
    """Build the preamble string for the fused launcher."""
    t_vars = ["TX", "TY", "TZ"][:ndim]
    h_vars = ["HX", "HY", "HZ"][:ndim]
    n_vars = ["Nx", "Ny", "Nz"][:ndim]

    lines = []
    lines.append(f"{', '.join(t_vars)} = {', '.join(str(t) for t in tile_sizes)}")
    lines.append(f"{', '.join(h_vars)} = {', '.join(str(h) for h in halo_widths)}")

    first_input = global_input_names[0]
    trailing = "," if ndim == 1 else ""
    lines.append(f"{', '.join(n_vars)}{trailing} = {first_input}.shape")

    lines.append("if stream is None:")
    lines.append("    stream = cp.cuda.get_current_stream()")

    return "\n".join(lines)


def _build_fused_standard_host(
    fused_name: str,
    ndim: int,
    global_input_names: list[str],
    output_names: list[str],
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
) -> HostProgramOp:
    """Build a ``HostProgramOp`` for the fused standard launcher."""
    t_vars = ["TX", "TY", "TZ"][:ndim]
    h_vars = ["HX", "HY", "HZ"][:ndim]
    n_vars = ["Nx", "Ny", "Nz"][:ndim]

    # Grid expression
    grid_parts = ", ".join(
        f"ct.cdiv({n_vars[d]} - 2 * {h_vars[d]}, {t_vars[d]})"
        for d in range(ndim)
    )
    grid_trailing = "," if ndim == 1 else ""
    grid_expr = f"({grid_parts}{grid_trailing})"

    # Args expression
    input_args = ", ".join(global_input_names)
    output_args = ", ".join(output_names)
    t_args = ", ".join(t_vars)
    h_args = ", ".join(h_vars)
    args_expr = f"({input_args}, {output_args}, {t_args}, {h_args})"

    preamble = _build_fused_launcher_preamble(
        fused_name, ndim, global_input_names, output_names,
        tile_sizes, halo_widths,
    )

    host_block = Block()
    launch = LaunchOp.build(
        properties={
            "kernel_name": StringAttr(f"{fused_name}_fused_kernel"),
            "grid_expr": StringAttr(grid_expr),
            "args_expr": StringAttr(args_expr),
        },
        operands=[[]],
    )
    host_block.add_op(launch)
    host_region = Region([host_block])

    input_params = ", ".join(global_input_names)
    output_params = ", ".join(output_names)
    prog_name = f"launch_{fused_name}_fused({input_params}, {output_params}, stream=None)"

    return HostProgramOp.build(
        properties={
            "program_name": StringAttr(prog_name),
            "preamble": StringAttr(preamble),
        },
        regions=[host_region],
    )


def _build_fused_temporal_host(
    fused_name: str,
    ndim: int,
    global_input_names: list[str],
    output_names: list[str],
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
    temporal_steps: int,
) -> HostProgramOp:
    """Build a ``HostProgramOp`` for the fused temporal-blocking launcher."""
    t_vars = ["TX", "TY", "TZ"][:ndim]
    h_vars = ["HX", "HY", "HZ"][:ndim]
    T = temporal_steps

    first_input = global_input_names[0]

    lines = []
    lines.append(f"{', '.join(t_vars)} = {', '.join(str(t) for t in tile_sizes)}")
    lines.append(f"{', '.join(h_vars)} = {', '.join(str(h) for h in halo_widths)}")

    lines.append("if stream is None:")
    lines.append("    stream = cp.cuda.get_current_stream()")

    # Grid
    grid_parts = []
    for d in range(ndim):
        h = halo_widths[d]
        grid_parts.append(
            f"ct.cdiv({first_input}.shape[{d}] - {2 * h}, {t_vars[d]})"
        )
    trailing = "," if ndim == 1 else ""
    lines.append(f"grid = ({', '.join(grid_parts)}{trailing})")

    # Buffer chains
    lines.append(f"# Temporal blocking: {T} steps with buffer swapping")
    for inp in global_input_names:
        lines.append(f"bufs_{inp} = [{inp}]")
        lines.append(f"for _ in range({T - 1}):")
        lines.append(f"    bufs_{inp}.append(cp.zeros_like({inp}))")
    for inp, out in zip(global_input_names, output_names):
        lines.append(f"bufs_{inp}.append({out})")

    preamble = "\n".join(lines)

    # Build loop body with LaunchOp
    buf_inputs = ", ".join(f"bufs_{inp}[_step]" for inp in global_input_names)
    buf_outputs = ", ".join(f"bufs_{inp}[_step + 1]" for inp in global_input_names)
    t_args = ", ".join(t_vars)
    h_args = ", ".join(h_vars)
    args_expr = f"({buf_inputs}, {buf_outputs}, {t_args}, {h_args})"

    loop_block = Block()
    launch = LaunchOp.build(
        properties={
            "kernel_name": StringAttr(f"{fused_name}_fused_kernel"),
            "grid_expr": StringAttr("grid"),
            "args_expr": StringAttr(args_expr),
        },
        operands=[[]],
    )
    loop_block.add_op(launch)
    loop_region = Region([loop_block])

    loop = ForLoopOp.build(
        properties={"count": IntAttr(T)},
        regions=[loop_region],
    )

    host_block = Block()
    host_block.add_op(loop)
    host_region = Region([host_block])

    input_params = ", ".join(global_input_names)
    output_params = ", ".join(output_names)
    prog_name = f"launch_{fused_name}_fused({input_params}, {output_params}, stream=None)"

    return HostProgramOp.build(
        properties={
            "program_name": StringAttr(prog_name),
            "preamble": StringAttr(preamble),
        },
        regions=[host_region],
    )


# -------------------------------------------------------------------- #
# Fused kernel emitter (extends target_to_python for fused patterns)
# -------------------------------------------------------------------- #


def _emit_fused_kernel(
    e,
    kernel: KernelOp,
    output_names: list[str],
) -> None:
    """Emit the fused ``@ct.kernel`` function from a ``KernelOp``.

    For fused kernels, the kernel has multiple outputs and the expression
    is a semicolon-separated list of "result_name=expr" pairs.
    """
    from cutile.lowering.emitter import CodeEmitter

    ndim = kernel.ndim.data
    all_names = [a.data for a in kernel.input_names.data]
    fused_expression = kernel.expression.data

    # Split all_names into inputs and outputs
    input_names = [n for n in all_names if n not in output_names]

    bid_vars = ["bx", "by", "bz"][:ndim]
    tile_vars = ["TX", "TY", "TZ"][:ndim]
    halo_vars = ["HX", "HY", "HZ"][:ndim]
    n_vars = ["nx", "ny", "nz"][:ndim]

    tile_const_params = ", ".join(f"{tv}: ConstInt" for tv in tile_vars)
    halo_const_params = ", ".join(f"{hv}: ConstInt" for hv in halo_vars)

    input_params = ", ".join(input_names)
    output_params = ", ".join(output_names)

    kernel_name = kernel.kernel_name.data

    e.line("@ct.kernel")
    e.line(
        f"def {kernel_name}_kernel({input_params}, {output_params}, "
        f"{tile_const_params}, {halo_const_params}):"
    )

    with e.indent():
        # Block indices
        for i, bv in enumerate(bid_vars):
            e.line(f"{bv} = ct.bid({i})")

        # Interior sizes
        first_arr = input_names[0]
        for d in range(ndim):
            e.line(f"{n_vars[d]} = {first_arr}.shape[{d}] - 2 * {halo_vars[d]}")

        # Walk kernel body for slice chains and loads
        body_block = list(kernel.body.blocks)[0]
        val_to_expr: dict = {}

        # Map block args
        for i, name in enumerate(input_names):
            val_to_expr[body_block.args[i]] = name
        for i, name in enumerate(output_names):
            val_to_expr[body_block.args[len(input_names) + i]] = name

        # Collect slice chains and loads
        e.blank()
        e.line("# --- Sliced input views (deduplicated across stencils) ---")

        load_ops: list = []
        store_count = 0
        for op in body_block.ops:
            if isinstance(op, SliceOp):
                parent_expr = val_to_expr.get(op.input, "?")
                axis = op.axis.data
                start = op.start.data
                stop = op.stop.data
                chain_expr = (
                    f"{parent_expr}.slice(axis={axis}, "
                    f"start={start}, stop={stop})"
                )
                val_to_expr[op.result] = chain_expr

                if op.var_name is not None:
                    var_name = op.var_name.data
                    if var_name.startswith("out_"):
                        pass  # output views emitted separately
                    else:
                        e.line(f"{var_name} = {chain_expr}")

            elif isinstance(op, LoadOp):
                load_ops.append(op)

        # Output views
        e.blank()
        e.line("# --- Output views ---")
        for op in body_block.ops:
            if isinstance(op, SliceOp) and op.var_name is not None:
                var_name = op.var_name.data
                if var_name.startswith("out_"):
                    chain_expr = val_to_expr.get(op.result, "?")
                    e.line(f"{var_name} = {chain_expr}")

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

        for load in load_ops:
            view_name = (
                load.view_name.data if load.view_name is not None else "view"
            )
            e.line(
                f"t_{view_name} = ct.load({view_name}, "
                f"index={idx_arg}, shape={shape_arg})"
            )

        # Parse fused expression
        expr_pairs = fused_expression.split(";")

        # Compute each stencil expression
        e.blank()
        e.line("# --- Compute stencil expressions ---")
        for pair in expr_pairs:
            result_name, expr = pair.split("=", 1)
            e.line(f"{result_name} = {expr}")

        # Store all outputs
        e.blank()
        e.line("# --- Store outputs ---")
        for pair, out_name in zip(expr_pairs, output_names):
            result_name = pair.split("=", 1)[0]
            e.line(f"ct.store(out_{out_name}, index={idx_arg}, tile={result_name})")


def _emit_fused_python(module: ModuleOp, output_names: list[str]) -> str:
    """Walk a fused cutile_target IR module and emit Python source.

    This is a variant of ``emit_python`` that handles the fused kernel
    pattern (multiple outputs, multi-expression).
    """
    from cutile.lowering.emitter import CodeEmitter
    from cutile.lowering.target_to_python import _emit_header, _emit_host

    e = CodeEmitter()

    kernel_op: KernelOp | None = None
    host_op: HostProgramOp | None = None

    for op in module.body.ops:
        if isinstance(op, KernelOp):
            kernel_op = op
        elif isinstance(op, HostProgramOp):
            host_op = op

    if kernel_op is None:
        raise ValueError("No KernelOp found in target IR module")
    if host_op is None:
        raise ValueError("No HostProgramOp found in target IR module")

    # Header
    _emit_header(e, kernel_op)
    e.blank()
    e.line("ConstInt = ct.Constant[int]")
    e.blank()

    # Fused kernel (custom emitter for multi-output)
    _emit_fused_kernel(e, kernel_op, output_names)

    # Host launcher
    _emit_host(e, host_op)

    return e.render()


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

    The fused kernel is built as Dialect 3 (cutile_target) IR and then
    emitted to Python via the target emitter.

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
    global_input_names_ordered: list[str] = []
    global_input_set: set[str] = set()
    name_remap: list[dict[str, str]] = []

    for meta in metas:
        remap: dict[str, str] = {}
        for local_name in meta.input_names:
            global_name = local_name
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
    # 9. Build Dialect 3 IR (KernelOp + HostProgramOp)
    # ---------------------------------------------------------------- #
    kernel_op = _build_fused_kernel_op(
        fused_name, ndim,
        global_input_names, output_names,
        merged_accesses, stencil_expressions,
        tile_sizes, halo_widths,
        all_constants,
    )

    if temporal_steps > 1:
        host_op = _build_fused_temporal_host(
            fused_name, ndim,
            global_input_names, output_names,
            tile_sizes, halo_widths, temporal_steps,
        )
    else:
        host_op = _build_fused_standard_host(
            fused_name, ndim,
            global_input_names, output_names,
            tile_sizes, halo_widths,
        )

    # Assemble target module
    mod_block = Block()
    mod_block.add_op(kernel_op)
    mod_block.add_op(host_op)
    mod_region = Region([mod_block])
    target_module = ModuleOp(mod_region)

    # ---------------------------------------------------------------- #
    # 10. Emit Python source from Dialect 3 IR
    # ---------------------------------------------------------------- #
    return _emit_fused_python(target_module, output_names)
