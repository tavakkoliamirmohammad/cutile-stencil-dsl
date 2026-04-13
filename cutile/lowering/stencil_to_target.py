"""Stencil-to-Target lowering: Dialect 1 IR to Dialect 3 (cutile_target) IR.

Given an xDSL ``ModuleOp`` produced by the frontend parser (Dialect 1),
this module walks the IR, extracts stencil metadata and arithmetic
structure, and builds a ``cutile_target`` IR module containing:

* ``KernelOp`` with ``BidOp``, ``SliceOp`` chains, ``LoadOp``, ``StoreOp``
* ``HostProgramOp`` with ``LaunchOp`` (and ``ForLoopOp`` for temporal)

This is a *structural* lowering -- the stencil expression is carried as
a ``StringAttr`` on ``KernelOp`` rather than being decomposed into
individual arith ops in the target IR.  The point is that the *skeleton*
(kernel shape, slice chains, load/store pattern, launcher structure) is
expressed as first-class IR ops.
"""

from __future__ import annotations

from typing import Sequence

from xdsl.dialects import arith
from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IndexType,
    IntAttr,
    ModuleOp,
    StringAttr,
)
from xdsl.ir import Block, Region

from cutile.dialects.cutile_stencil.dialect import AccessOp, FuncOp, YieldOp
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


# -------------------------------------------------------------------- #
# Internal data structures (reused from stencil_to_cutile)
# -------------------------------------------------------------------- #


class _AccessInfo:
    """Describes one unique (array_index, offsets) stencil access."""

    __slots__ = ("array_index", "offsets", "view_name")

    def __init__(self, array_index: int, offsets: tuple[int, ...], view_name: str):
        self.array_index = array_index
        self.offsets = offsets
        self.view_name = view_name


class _StencilMeta:
    """All metadata extracted from a Dialect 1 ``FuncOp``."""

    __slots__ = (
        "name",
        "ndim",
        "order",
        "dtype",
        "num_inputs",
        "input_names",
        "accesses",
        "expression",
        "constants",
        "boundary",
    )

    def __init__(self) -> None:
        self.name: str = ""
        self.ndim: int = 2
        self.order: int = 2
        self.dtype: str = "float64"
        self.num_inputs: int = 1
        self.input_names: list[str] = []
        self.accesses: list[_AccessInfo] = []
        self.expression: str = ""
        self.constants: dict[str, float] = {}
        self.boundary: dict | None = None


# -------------------------------------------------------------------- #
# IR walking helpers (identical to stencil_to_cutile)
# -------------------------------------------------------------------- #


def _format_offset(off: int) -> str:
    """0 -> '0', +k -> 'pk', -k -> 'mk'."""
    if off == 0:
        return "0"
    elif off > 0:
        return f"p{off}"
    else:
        return f"m{abs(off)}"


def _offset_expr(halo_var: str, off: int, *, add_n: bool = False, n_var: str = "n") -> str:
    """Build start/stop expression for ``.slice()``."""
    if add_n:
        if off == 0:
            return f"{halo_var} + {n_var}"
        elif off > 0:
            return f"{halo_var} + {off} + {n_var}"
        else:
            return f"{halo_var} - {abs(off)} + {n_var}"
    else:
        if off == 0:
            return halo_var
        elif off > 0:
            return f"{halo_var} + {off}"
        else:
            return f"{halo_var} - {abs(off)}"


def _extract_meta(func_op: FuncOp, block: Block) -> _StencilMeta:
    """Extract :class:`_StencilMeta` from a Dialect 1 ``FuncOp``."""
    meta = _StencilMeta()
    meta.name = func_op.func_name.data
    meta.ndim = func_op.ndim.data
    meta.order = func_op.order.data
    if func_op.dtype is not None:
        meta.dtype = func_op.dtype.data
    meta.num_inputs = len(block.args)

    _default_names = list("uvwxyz")
    meta.input_names = [
        _default_names[i] if i < len(_default_names) else f"arr{i}"
        for i in range(meta.num_inputs)
    ]

    if func_op.constants is not None:
        for key, val in func_op.constants.data.items():
            if isinstance(val, FloatAttr):
                meta.constants[key] = val.value.data

    if func_op.boundary is not None:
        meta.boundary = {
            "type": func_op.boundary.bc_type_str,
        }
        if func_op.boundary.has_value:
            meta.boundary["value"] = func_op.boundary.value.value.data

    seen: dict[tuple[int, tuple[int, ...]], _AccessInfo] = {}
    access_order: list[_AccessInfo] = []
    val_to_name: dict = {}

    for op in block.ops:
        if isinstance(op, AccessOp):
            offsets = tuple(item.data for item in op.offset.parameters[0].data)
            arr_idx: int | None = None
            for i, arg in enumerate(block.args):
                if op.field is arg:
                    arr_idx = i
                    break
            if arr_idx is None:
                arr_idx = 0

            key = (arr_idx, offsets)
            if key not in seen:
                off_parts = "_".join(_format_offset(o) for o in offsets)
                arr_name = meta.input_names[arr_idx]
                view_name = f"{arr_name}_{off_parts}"
                info = _AccessInfo(arr_idx, offsets, view_name)
                seen[key] = info
                access_order.append(info)

            val_to_name[op.res] = f"t_{seen[key].view_name}"

    meta.accesses = access_order
    meta.expression = _reconstruct_expr(block, val_to_name)
    return meta


def _reconstruct_expr(block: Block, val_to_name: dict) -> str:
    """Walk arith ops in *block* and reconstruct a Python expression string."""
    from xdsl.ir import SSAValue

    for op in block.ops:
        if isinstance(op, arith.ConstantOp):
            val_attr = op.properties.get("value", op.attributes.get("value", None))
            if isinstance(val_attr, FloatAttr):
                fval = val_attr.value.data
                if fval == int(fval) and abs(fval) < 1e15:
                    val_to_name[op.result] = repr(fval)
                else:
                    val_to_name[op.result] = repr(fval)
            else:
                val_to_name[op.result] = str(val_attr)

        elif isinstance(op, arith.AddfOp):
            left = val_to_name.get(op.lhs, "?")
            right = val_to_name.get(op.rhs, "?")
            val_to_name[op.result] = f"{left} + {right}"

        elif isinstance(op, arith.SubfOp):
            left = val_to_name.get(op.lhs, "?")
            right = val_to_name.get(op.rhs, "?")
            if any(c in right for c in ("+", "-")) and not right.startswith("("):
                right = f"({right})"
            val_to_name[op.result] = f"{left} - {right}"

        elif isinstance(op, arith.MulfOp):
            left = val_to_name.get(op.lhs, "?")
            right = val_to_name.get(op.rhs, "?")
            if any(c in left for c in ("+", "-")) and not left.startswith("("):
                left = f"({left})"
            if any(c in right for c in ("+", "-")) and not right.startswith("("):
                right = f"({right})"
            val_to_name[op.result] = f"{left} * {right}"

        elif isinstance(op, arith.DivfOp):
            left = val_to_name.get(op.lhs, "?")
            right = val_to_name.get(op.rhs, "?")
            if any(c in left for c in ("+", "-")) and not left.startswith("("):
                left = f"({left})"
            if any(c in right for c in ("+", "-", "*")) and not right.startswith("("):
                right = f"({right})"
            val_to_name[op.result] = f"{left} / {right}"

        elif isinstance(op, arith.NegfOp):
            operand_name = op.operands[0]
            expr = val_to_name.get(operand_name, "?")
            val_to_name[op.result] = f"-({expr})"

        elif isinstance(op, YieldOp):
            return val_to_name.get(op.value, "0")

    return "0"


# -------------------------------------------------------------------- #
# Target IR builders
# -------------------------------------------------------------------- #

_IDX = IndexType()


def _build_kernel_body(
    meta: _StencilMeta,
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
) -> Region:
    """Build the body region for a ``KernelOp``.

    The region contains:
    - Block arguments for each input array + output
    - BidOp per dimension
    - SliceOp chains for each access + output
    - LoadOp per access
    - StoreOp for the output
    - ReturnOp terminator
    """
    ndim = meta.ndim
    halo_vars = ["HX", "HY", "HZ"][:ndim]
    n_vars = ["nx", "ny", "nz"][:ndim]

    # Block args: one per input array + output
    num_block_args = meta.num_inputs + 1  # inputs + output
    block = Block(arg_types=[_IDX] * num_block_args)

    # BidOps
    for d in range(ndim):
        bid = BidOp.build(properties={"axis": IntAttr(d)}, result_types=[_IDX])
        block.add_op(bid)

    # SliceOp chains for each access
    for info in meta.accesses:
        arr_arg = block.args[info.array_index]
        prev_result = arr_arg
        for d in range(ndim):
            off = info.offsets[d]
            start_expr = _offset_expr(halo_vars[d], off)
            stop_expr = _offset_expr(halo_vars[d], off, add_n=True, n_var=n_vars[d])
            # Last slice in chain gets the var_name
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

    # Output view slice chain (zero offsets)
    out_arg = block.args[meta.num_inputs]
    prev_result = out_arg
    for d in range(ndim):
        is_last = (d == ndim - 1)
        props = {
            "axis": IntAttr(d),
            "start": StringAttr(halo_vars[d]),
            "stop": StringAttr(f"{halo_vars[d]} + {n_vars[d]}"),
        }
        if is_last:
            props["var_name"] = StringAttr("out")
        s = SliceOp.build(
            properties=props,
            operands=[prev_result],
            result_types=[_IDX],
        )
        block.add_op(s)
        prev_result = s.result

    # StoreOp
    # We use the last slice result as the tile operand and a dummy for value
    # (the expression is stored on KernelOp)
    store = StoreOp.build(operands=[prev_result, prev_result])
    block.add_op(store)

    # ReturnOp
    ret = ReturnOp.build()
    block.add_op(ret)

    return Region([block])


def _build_kernel_op(
    meta: _StencilMeta,
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
) -> KernelOp:
    """Build a ``KernelOp`` from stencil metadata."""
    ndim = meta.ndim

    body = _build_kernel_body(meta, tile_sizes, halo_widths)

    # Build constants array: [key1, val1, key2, val2, ...]
    constants_items: list = []
    if meta.constants:
        for k, v in sorted(meta.constants.items()):
            constants_items.append(StringAttr(k))
            constants_items.append(StringAttr(repr(v)))

    props: dict = {
        "kernel_name": StringAttr(meta.name),
        "tile_shape": ArrayAttr([IntAttr(t) for t in tile_sizes]),
        "halo": ArrayAttr([IntAttr(h) for h in halo_widths]),
        "ndim": IntAttr(ndim),
        "input_names": ArrayAttr([StringAttr(n) for n in meta.input_names]),
        "expression": StringAttr(meta.expression),
    }
    if constants_items:
        props["constants"] = ArrayAttr(constants_items)

    return KernelOp.build(
        properties=props,
        operands=[[]],
        regions=[body],
    )


def _build_launcher_preamble(
    meta: _StencilMeta,
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
) -> str:
    """Build the preamble string for the standard (non-temporal) launcher."""
    ndim = meta.ndim
    multi_input = meta.num_inputs > 1
    t_vars = ["TX", "TY", "TZ"][:ndim]
    h_vars = ["HX", "HY", "HZ"][:ndim]
    n_vars = ["Nx", "Ny", "Nz"][:ndim]

    lines = []
    lines.append(f"{', '.join(t_vars)} = {', '.join(str(t) for t in tile_sizes)}")
    lines.append(f"{', '.join(h_vars)} = {', '.join(str(h) for h in halo_widths)}")

    first_input = meta.input_names[0] if multi_input else "u_in"
    trailing = "," if ndim == 1 else ""
    lines.append(f"{', '.join(n_vars)}{trailing} = {first_input}.shape")

    lines.append("if stream is None:")
    lines.append("    stream = cp.cuda.get_current_stream()")

    return "\n".join(lines)


def _build_standard_host(
    meta: _StencilMeta,
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
) -> HostProgramOp:
    """Build a ``HostProgramOp`` for the standard (non-temporal) launcher."""
    ndim = meta.ndim
    multi_input = meta.num_inputs > 1
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
    if multi_input:
        args = ", ".join(meta.input_names)
    else:
        args = "u_in"
    t_args = ", ".join(t_vars)
    h_args = ", ".join(h_vars)
    args_expr = f"({args}, u_out, {t_args}, {h_args})"

    # Preamble
    preamble = _build_launcher_preamble(meta, tile_sizes, halo_widths)

    # Build host body with LaunchOp
    host_block = Block()
    launch = LaunchOp.build(
        properties={
            "kernel_name": StringAttr(f"{meta.name}_kernel"),
            "grid_expr": StringAttr(grid_expr),
            "args_expr": StringAttr(args_expr),
        },
        operands=[[]],
    )
    host_block.add_op(launch)
    host_region = Region([host_block])

    # Program name (function signature)
    if multi_input:
        input_params = ", ".join(meta.input_names)
        prog_name = f"launch_{meta.name}({input_params}, u_out, stream=None)"
    else:
        prog_name = f"launch_{meta.name}(u_in, u_out, stream=None)"

    return HostProgramOp.build(
        properties={
            "program_name": StringAttr(prog_name),
            "preamble": StringAttr(preamble),
        },
        regions=[host_region],
    )


def _build_temporal_launcher_preamble(
    meta: _StencilMeta,
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
    temporal_steps: int,
) -> str:
    """Build the preamble string for the temporal-blocking launcher."""
    ndim = meta.ndim
    multi_input = meta.num_inputs > 1
    t_vars = ["TX", "TY", "TZ"][:ndim]
    h_vars = ["HX", "HY", "HZ"][:ndim]
    T = temporal_steps

    lines = []
    lines.append(f"{', '.join(t_vars)} = {', '.join(str(t) for t in tile_sizes)}")
    lines.append(f"{', '.join(h_vars)} = {', '.join(str(h) for h in halo_widths)}")

    first_input = meta.input_names[0] if multi_input else "u_in"

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

    # Buffer chain
    lines.append(f"# Temporal blocking: {T} steps with buffer swapping")
    lines.append(f"bufs = [{first_input}]")
    lines.append(f"for _ in range({T - 1}):")
    lines.append(f"    bufs.append(cp.zeros_like({first_input}))")
    lines.append("bufs.append(u_out)")

    return "\n".join(lines)


def _build_temporal_host(
    meta: _StencilMeta,
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
    temporal_steps: int,
) -> HostProgramOp:
    """Build a ``HostProgramOp`` for temporal-blocking launcher."""
    ndim = meta.ndim
    multi_input = meta.num_inputs > 1
    t_vars = ["TX", "TY", "TZ"][:ndim]
    h_vars = ["HX", "HY", "HZ"][:ndim]
    T = temporal_steps

    # Args expression inside the loop
    t_args = ", ".join(t_vars)
    h_args = ", ".join(h_vars)
    args_expr = f"(bufs[_step], bufs[_step + 1], {t_args}, {h_args})"

    # Preamble
    preamble = _build_temporal_launcher_preamble(
        meta, tile_sizes, halo_widths, temporal_steps
    )

    # Build loop body with LaunchOp
    loop_block = Block()
    launch = LaunchOp.build(
        properties={
            "kernel_name": StringAttr(f"{meta.name}_kernel"),
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

    # Host body contains the loop
    host_block = Block()
    host_block.add_op(loop)
    host_region = Region([host_block])

    # Program name
    if multi_input:
        input_params = ", ".join(meta.input_names)
        prog_name = f"launch_{meta.name}({input_params}, u_out, stream=None)"
    else:
        prog_name = f"launch_{meta.name}(u_in, u_out, stream=None)"

    return HostProgramOp.build(
        properties={
            "program_name": StringAttr(prog_name),
            "preamble": StringAttr(preamble),
        },
        regions=[host_region],
    )


# -------------------------------------------------------------------- #
# Public API
# -------------------------------------------------------------------- #


def lower_to_target_ir(
    module: ModuleOp,
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
    temporal_steps: int = 1,
    boundary_spec: dict | None = None,
) -> ModuleOp:
    """Convert Dialect 1 stencil IR into Dialect 3 cutile_target IR.

    Parameters
    ----------
    module:
        xDSL ``ModuleOp`` containing a ``cutile_stencil.FuncOp``.
    tile_sizes:
        Tile sizes per dimension.
    halo_widths:
        Halo widths per dimension.
    temporal_steps:
        Number of temporal blocking steps (1 = no temporal blocking).
    boundary_spec:
        Optional boundary condition specification dict.

    Returns
    -------
    ModuleOp
        New module containing ``cutile_target`` dialect ops.
    """
    # Find the FuncOp
    func_op: FuncOp | None = None
    for op in module.body.ops:
        if isinstance(op, FuncOp):
            func_op = op
            break
    if func_op is None:
        raise ValueError("No cutile_stencil.FuncOp found in module")

    block = list(func_op.body.blocks)[0]

    # Extract metadata
    meta = _extract_meta(func_op, block)

    # Merge boundary info
    if boundary_spec is None and meta.boundary is not None:
        boundary_spec = meta.boundary

    # Build target IR ops
    kernel_op = _build_kernel_op(meta, tile_sizes, halo_widths)

    if temporal_steps > 1:
        host_op = _build_temporal_host(meta, tile_sizes, halo_widths, temporal_steps)
    else:
        host_op = _build_standard_host(meta, tile_sizes, halo_widths)

    # Assemble module
    mod_block = Block()
    mod_block.add_op(kernel_op)
    mod_block.add_op(host_op)
    mod_region = Region([mod_block])
    target_module = ModuleOp(mod_region)

    # Attach boundary spec as a dict attribute on the module if present
    # (the emitter will check for this)
    if boundary_spec is not None:
        bc_items = [StringAttr(boundary_spec.get("type", "dirichlet"))]
        if "value" in boundary_spec:
            bc_items.append(StringAttr(str(boundary_spec["value"])))
        target_module.attributes["boundary"] = ArrayAttr(bc_items)
        # Also store the stencil name and ndim/halo for boundary emission
        target_module.attributes["stencil_name"] = StringAttr(meta.name)
        target_module.attributes["stencil_ndim"] = IntAttr(meta.ndim)
        target_module.attributes["stencil_halo"] = ArrayAttr(
            [IntAttr(h) for h in halo_widths]
        )

    return target_module
