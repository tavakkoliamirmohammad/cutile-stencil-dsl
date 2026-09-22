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

import math
from typing import Sequence

import numpy as np

from xdsl.dialects import arith
from xdsl.dialects import math as xmath
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
        "kernel_constants",
        "expressions",
        "output_names",
        "stencil_inputs",
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
        # float64 constants that must reach the kernel exactly (see
        # ``_reconstruct_expr``); referenced as ``_c0``, ``_c1``, ... in
        # ``expression`` and loaded from a device array at run time.
        self.kernel_constants: list[float] = []
        # One expression per output; a single stencil has one, a fused kernel
        # one per member stencil (``output_names`` are the members' names and
        # ``stencil_inputs`` each member's array parameter names).
        self.expressions: list[str] = []
        self.output_names: list[str] = []
        self.stencil_inputs: list[list[str]] = []

    @property
    def num_outputs(self) -> int:
        return max(1, len(self.expressions))

    def output_views(self) -> list[str]:
        """Kernel-side view variable per output (``out`` or ``out_<k>``)."""
        if self.num_outputs == 1:
            return ["out"]
        return [f"out_{k}" for k in range(self.num_outputs)]

    def output_params(self) -> list[str]:
        """Launcher parameter per output (``u_out`` or ``<stencil>_out``)."""
        if self.output_names:
            return [f"{n}_out" for n in self.output_names]
        return ["u_out"]


def _default_input_names(n: int) -> list[str]:
    defaults = list("uvwxyz")
    return [defaults[i] if i < len(defaults) else f"arr{i}" for i in range(n)]


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


def _extract_meta(
    func_op: FuncOp,
    block: Block,
    spatial_term_order: bool | None = None,
    *,
    input_names: Sequence[str] | None = None,
    kernel_constants: list[float] | None = None,
) -> _StencilMeta:
    """Extract :class:`_StencilMeta` from a Dialect 1 ``FuncOp``.

    ``spatial_term_order`` re-associates a top-level sum so its terms follow
    the (x, y, z) order of the accesses they read; ``None`` enables it for
    very wide stencils only (see :class:`~cutile.passes.tiling.TilingPass`).
    ``input_names`` overrides the generated array names (``u``, ``v``, ...)
    so view names like ``t_mx_p1_0_0`` are shared between fused stencils;
    ``kernel_constants`` lets several stencils share one constants table.
    """
    meta = _StencilMeta()
    meta.name = func_op.func_name.data
    meta.ndim = func_op.ndim.data
    meta.order = func_op.order.data
    if func_op.dtype is not None:
        meta.dtype = func_op.dtype.data
    meta.num_inputs = len(block.args)

    if input_names is not None:
        if len(input_names) != meta.num_inputs:
            raise ValueError(
                f"{meta.name}: {len(input_names)} input names for {meta.num_inputs} arrays"
            )
        meta.input_names = list(input_names)
    else:
        meta.input_names = _default_input_names(meta.num_inputs)
    if kernel_constants is not None:
        meta.kernel_constants = kernel_constants

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
    access_offsets: dict = {}

    for op in block.ops:
        if isinstance(op, AccessOp):
            offsets = tuple(item.data for item in op.offset.parameters[0].data)
            access_offsets[op.res] = offsets
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
    if spatial_term_order is None:
        from cutile.passes.tiling import TilingPass

        spatial_term_order = len(access_order) > TilingPass.very_wide_stencil_loads
    meta.expression = _reconstruct_expr(
        block,
        val_to_name,
        meta.kernel_constants,
        inline_all=meta.dtype in ("float32", "fp32"),
        access_offsets=access_offsets if spatial_term_order else None,
    )
    meta.expressions = [meta.expression]
    return meta


def _f32_exact(value: float) -> bool:
    """True when *value* survives a round trip through float32 unchanged."""
    return math.isfinite(value) and float(np.float32(value)) == value


def _fold(op, a: float, b: float) -> float | None:
    """Evaluate a binary arith op on two Python floats (float64)."""
    try:
        if isinstance(op, arith.AddfOp):
            return a + b
        if isinstance(op, arith.SubfOp):
            return a - b
        if isinstance(op, arith.MulfOp):
            return a * b
        if isinstance(op, arith.DivfOp):
            return a / b
    except ZeroDivisionError:
        return None
    return None


def _needs_parens(expr: str, ops: tuple[str, ...]) -> bool:
    return any(c in expr for c in ops) and not expr.startswith("(")


# arith.cmpf predicate (integer encoding, ordered comparisons) -> Python operator
_CMPF_SYMBOLS = {1: "==", 2: ">", 3: ">=", 4: "<", 5: "<=", 6: "!=", 13: "!="}


def _flatten_sum(value, const_vals: dict, sign: int = 1) -> list:
    """Flatten a chain of ``addf``/``subf`` into signed leaf terms."""
    owner = getattr(value, "owner", None)
    if value not in const_vals and isinstance(owner, arith.AddfOp):
        return _flatten_sum(owner.lhs, const_vals, sign) + _flatten_sum(owner.rhs, const_vals, sign)
    if value not in const_vals and isinstance(owner, arith.SubfOp):
        return _flatten_sum(owner.lhs, const_vals, sign) + _flatten_sum(owner.rhs, const_vals, -sign)
    return [(sign, value)]


def _term_offset(value, access_offsets: dict):
    """Smallest (x, y, z) access offset a term reads; constants sort first."""
    found = []
    stack = [value]
    seen = set()
    while stack:
        v = stack.pop()
        if id(v) in seen:
            continue
        seen.add(id(v))
        if v in access_offsets:
            found.append(access_offsets[v])
            continue
        owner = getattr(v, "owner", None)
        if owner is not None and hasattr(owner, "operands"):
            stack.extend(owner.operands)
    return (1, min(found)) if found else (0, ())


def _reconstruct_expr(
    block: Block,
    val_to_name: dict,
    kernel_constants: list[float] | None = None,
    *,
    inline_all: bool = False,
    access_offsets: dict | None = None,
) -> str:
    """Walk arith ops in *block* and reconstruct a Python expression string.

    cuTile evaluates Python float scalars in float32, even inside float64
    kernels, so two things happen here that a plain decompiler would not do:

    * constant sub-expressions (``c * (-1 / 12)``) are folded in Python
      float64 instead of being emitted as scalar arithmetic;
    * a constant that is not exactly representable in float32 is emitted as
      ``_c<k>`` and appended to *kernel_constants*; the emitter loads those
      from a float64 device array.  Constants such as ``0.25`` or ``6.0``
      stay inline.  ``inline_all`` (float32 stencils) disables this.

    When *access_offsets* (access SSA value -> offsets) is given, a top-level
    sum is re-associated so its terms follow the spatial order of the
    accesses they read: tileiras issues loads in program order, and for very
    wide stencils that order decides whether a low-register schedule is found
    (125-point box: 6.5 ms sorted vs 7.3-7.7 ms in source order).
    """
    if kernel_constants is None:
        kernel_constants = []
    const_vals: dict = {}

    def const_str(value: float) -> str:
        if inline_all or _f32_exact(value):
            return repr(float(value))
        if value not in kernel_constants:
            kernel_constants.append(value)
        return f"_c{kernel_constants.index(value)}"

    def name(v) -> str:
        if v in const_vals:
            return const_str(const_vals[v])
        return val_to_name.get(v, "?")

    for op in block.ops:
        if isinstance(op, arith.ConstantOp):
            val_attr = op.properties.get("value", op.attributes.get("value", None))
            if isinstance(val_attr, FloatAttr):
                const_vals[op.result] = float(val_attr.value.data)
            else:
                val_to_name[op.result] = str(val_attr)

        elif isinstance(op, (arith.AddfOp, arith.SubfOp, arith.MulfOp, arith.DivfOp)):
            if op.lhs in const_vals and op.rhs in const_vals:
                folded = _fold(op, const_vals[op.lhs], const_vals[op.rhs])
                if folded is not None and math.isfinite(folded):
                    const_vals[op.result] = folded
                    continue
            left, right = name(op.lhs), name(op.rhs)
            if isinstance(op, arith.AddfOp):
                val_to_name[op.result] = f"{left} + {right}"
            elif isinstance(op, arith.SubfOp):
                if _needs_parens(right, ("+", "-")):
                    right = f"({right})"
                val_to_name[op.result] = f"{left} - {right}"
            elif isinstance(op, arith.MulfOp):
                if _needs_parens(left, ("+", "-")):
                    left = f"({left})"
                if _needs_parens(right, ("+", "-", "/")):
                    right = f"({right})"
                val_to_name[op.result] = f"{left} * {right}"
            else:  # DivfOp
                if _needs_parens(left, ("+", "-")):
                    left = f"({left})"
                if _needs_parens(right, ("+", "-", "*", "/")):
                    right = f"({right})"
                val_to_name[op.result] = f"{left} / {right}"

        elif isinstance(op, arith.NegfOp):
            operand = op.operands[0]
            if operand in const_vals:
                const_vals[op.result] = -const_vals[operand]
                continue
            val_to_name[op.result] = f"-({name(operand)})"

        elif isinstance(op, (arith.MaximumfOp, arith.MinimumfOp)):
            fn = max if isinstance(op, arith.MaximumfOp) else min
            if op.lhs in const_vals and op.rhs in const_vals:
                const_vals[op.result] = fn(const_vals[op.lhs], const_vals[op.rhs])
                continue
            ct_fn = "ct.maximum" if isinstance(op, arith.MaximumfOp) else "ct.minimum"
            val_to_name[op.result] = f"{ct_fn}({name(op.lhs)}, {name(op.rhs)})"

        elif isinstance(op, xmath.AbsFOp):
            operand = op.operands[0]
            if operand in const_vals:
                const_vals[op.result] = abs(const_vals[operand])
                continue
            val_to_name[op.result] = f"ct.abs({name(operand)})"

        elif isinstance(op, arith.CmpfOp):
            sym = _CMPF_SYMBOLS.get(op.predicate.value.data, None)
            if sym is None:
                raise ValueError(f"Unsupported cmpf predicate {op.predicate}")
            val_to_name[op.result] = f"({name(op.lhs)} {sym} {name(op.rhs)})"

        elif isinstance(op, arith.SelectOp):
            cond, a, b = op.operands
            val_to_name[op.result] = f"ct.where({name(cond)}, {name(a)}, {name(b)})"

        elif isinstance(op, YieldOp):
            if access_offsets:
                terms = _flatten_sum(op.value, const_vals)
                if len(terms) > 1:
                    terms.sort(key=lambda t: _term_offset(t[1], access_offsets))
                    return _join_sum(terms, name)
            return name(op.value)

    return "0"


def _join_sum(terms: list, name) -> str:
    """Rebuild ``terms`` (sign, value) as a left-to-right sum string."""
    parts = []
    for i, (sign, value) in enumerate(terms):
        expr = name(value)
        if i == 0:
            parts.append(f"-({expr})" if sign < 0 else expr)
            continue
        if _needs_parens(expr, ("+", "-")):
            expr = f"({expr})"
        parts.append(f"{'-' if sign < 0 else '+'} {expr}")
    return " ".join(parts)


# -------------------------------------------------------------------- #
# Target IR builders
# -------------------------------------------------------------------- #

_IDX = IndexType()


def grid_order(ndim: int) -> tuple[int, ...]:
    """Array dimensions in block-index order.

    ``bid(0)`` varies fastest across consecutive thread blocks, so it should
    walk the *second-innermost* dimension (rows of one plane): consecutive
    blocks then stream through contiguous memory instead of hopping between
    planes.  Remaining outer dimensions follow, and the innermost dimension
    (which a row tile usually covers with a single block) comes last.

    1D -> ``(0,)``, 2D -> ``(0, 1)`` (already rows-fastest), 3D -> ``(1, 0, 2)``.
    """
    if ndim <= 2:
        return tuple(range(ndim))
    return tuple(range(ndim - 2, -1, -1)) + (ndim - 1,)


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

    # Block args: one per input array + one per output
    num_block_args = meta.num_inputs + meta.num_outputs
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

    # Output view slice chain (zero offsets) and StoreOp, one per output.
    # The store uses the last slice result as the tile operand and a dummy
    # for the value (the expressions are stored on KernelOp).
    for k, view in enumerate(meta.output_views()):
        prev_result = block.args[meta.num_inputs + k]
        for d in range(ndim):
            is_last = (d == ndim - 1)
            props = {
                "axis": IntAttr(d),
                "start": StringAttr(halo_vars[d]),
                "stop": StringAttr(f"{halo_vars[d]} + {n_vars[d]}"),
            }
            if is_last:
                props["var_name"] = StringAttr(view)
            s = SliceOp.build(
                properties=props,
                operands=[prev_result],
                result_types=[_IDX],
            )
            block.add_op(s)
            prev_result = s.result
        block.add_op(StoreOp.build(operands=[prev_result, prev_result]))

    # ReturnOp
    ret = ReturnOp.build()
    block.add_op(ret)

    return Region([block])


def _build_kernel_op(
    meta: _StencilMeta,
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
    kernel_hints: dict | None = None,
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
        "grid_order": ArrayAttr([IntAttr(d) for d in grid_order(ndim)]),
    }
    if meta.num_outputs > 1:
        props["expressions"] = ArrayAttr([StringAttr(x) for x in meta.expressions])
        props["output_names"] = ArrayAttr([StringAttr(n) for n in meta.output_names])
    if meta.kernel_constants:
        props["kernel_constants"] = ArrayAttr(
            [StringAttr(repr(v)) for v in meta.kernel_constants]
        )
    if kernel_hints:
        items: list = []
        for key in sorted(kernel_hints):
            items.append(StringAttr(str(key)))
            items.append(StringAttr(repr(kernel_hints[key])))
        props["kernel_hints"] = ArrayAttr(items)
    if constants_items:
        props["constants"] = ArrayAttr(constants_items)

    return KernelOp.build(
        properties=props,
        operands=[[]],
        regions=[body],
    )


def _fit_tiles_line(ndim: int, first_input: str) -> str:
    """Preamble line clamping compile-time tiles to the runtime domain.

    Compile time does not know the array shape, so a wide row tile is
    shrunk to the next power of two of each interior extent at launch
    (``_fit_tiles`` is emitted into every generated module).
    """
    t_vars = ["TX", "TY", "TZ"][:ndim]
    h_vars = ["HX", "HY", "HZ"][:ndim]
    tr = "," if ndim == 1 else ""
    t_tuple = f"({', '.join(t_vars)}{tr})"
    h_tuple = f"({', '.join(h_vars)}{tr})"
    return (
        f"{', '.join(t_vars)}{tr} = _fit_tiles({t_tuple}, "
        f"{first_input}.shape, {h_tuple})"
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
    lines.append(_fit_tiles_line(ndim, first_input))

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

    # Grid expression (same dimension order as the kernel's bid mapping)
    grid_parts = ", ".join(
        f"ct.cdiv({n_vars[d]} - 2 * {h_vars[d]}, {t_vars[d]})"
        for d in grid_order(ndim)
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
    consts = "_kernel_constants(), " if meta.kernel_constants else ""
    outs = ", ".join(meta.output_params())
    args_expr = f"({args}, {outs}, {consts}{t_args}, {h_args})"

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
        prog_name = f"launch_{meta.name}({input_params}, {outs}, stream=None)"
    else:
        prog_name = f"launch_{meta.name}(u_in, {outs}, stream=None)"

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
    lines.append(_fit_tiles_line(ndim, first_input))

    lines.append("if stream is None:")
    lines.append("    stream = cp.cuda.get_current_stream()")

    # Grid (same dimension order as the kernel's bid mapping)
    grid_parts = []
    for d in grid_order(ndim):
        h = halo_widths[d]
        grid_parts.append(
            f"ct.cdiv({first_input}.shape[{d}] - {2 * h}, {t_vars[d]})"
        )
    trailing = "," if ndim == 1 else ""
    lines.append(f"grid = ({', '.join(grid_parts)}{trailing})")

    # Buffer chain: scratch buffers are cached per (shape, dtype, device)
    # so a launch does not pay for allocation + zero-fill on every call.
    lines.append(f"# Temporal blocking: {T} steps through reusable scratch buffers")
    lines.append(
        f"bufs = [{first_input}] + _temporal_buffers({first_input}, {T - 1}) + [u_out]"
    )

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
    consts = "_kernel_constants(), " if meta.kernel_constants else ""
    args_expr = f"(bufs[_step], bufs[_step + 1], {consts}{t_args}, {h_args})"

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
    kernel_hints: dict | None = None,
    spatial_term_order: bool | None = None,
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
    kernel_hints:
        Optional ``@ct.kernel`` keyword arguments (e.g. ``{"occupancy": 8}``).
    spatial_term_order:
        Re-associate the sum in (x, y, z) access order; ``None`` = only for
        very wide stencils.

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
    meta = _extract_meta(func_op, block, spatial_term_order)

    # Merge boundary info
    if boundary_spec is None and meta.boundary is not None:
        boundary_spec = meta.boundary

    # Build target IR ops
    kernel_op = _build_kernel_op(meta, tile_sizes, halo_widths, kernel_hints)

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


# -------------------------------------------------------------------- #
# Fused (multi-output) kernels
# -------------------------------------------------------------------- #


def _func_and_block(module: ModuleOp) -> tuple[FuncOp, Block]:
    for op in module.body.ops:
        if isinstance(op, FuncOp):
            return op, list(op.body.blocks)[0]
    raise ValueError("No cutile_stencil.FuncOp found in module")


def _arg_names(func_op: FuncOp, block: Block) -> list[str]:
    """Array parameter names recorded by the frontend (positional fallback)."""
    if func_op.arg_names is not None:
        return [a.data for a in func_op.arg_names.data]
    return _default_input_names(len(block.args))


def _access_keys(block: Block, names: Sequence[str]) -> set:
    """Distinct ``(array name, offsets)`` a stencil body reads."""
    keys = set()
    for op in block.ops:
        if isinstance(op, AccessOp):
            idx = next((i for i, arg in enumerate(block.args) if op.field is arg), 0)
            keys.add((names[idx], tuple(item.data for item in op.offset.parameters[0].data)))
    return keys


def fused_name(names: Sequence[str]) -> str:
    return "_".join(names) if len(names) <= 3 else f"{names[0]}_and_{len(names) - 1}_others"


def extract_fused_meta(
    modules: Sequence[ModuleOp], spatial_term_order: bool | None = None
) -> _StencilMeta:
    """Merge the stencils in *modules* into the metadata of one kernel.

    Inputs are matched by parameter name (``FuncOp.arg_names``), so member
    stencils may read different subsets of the fields; each distinct
    ``(field, offset)`` becomes one load shared by every expression that
    uses it, and all members share one table of exact float64 constants.
    ``spatial_term_order`` defaults to the very-wide rule on the *merged*
    access count.
    """
    if not modules:
        raise ValueError("At least one module is required for fusion")
    funcs = [_func_and_block(m) for m in modules]
    stencil_inputs = [_arg_names(f, b) for f, b in funcs]
    inputs: list[str] = []
    for names in stencil_inputs:
        for n in names:
            if n not in inputs:
                inputs.append(n)
    if spatial_term_order is None:
        from cutile.passes.tiling import TilingPass

        merged = set().union(*(_access_keys(b, names) for (_, b), names in zip(funcs, stencil_inputs)))
        spatial_term_order = len(merged) > TilingPass.very_wide_stencil_loads

    shared_constants: list[float] = []
    metas = [
        _extract_meta(f, b, spatial_term_order, input_names=names, kernel_constants=shared_constants)
        for (f, b), names in zip(funcs, stencil_inputs)
    ]
    names = [m.name for m in metas]
    if len(set(names)) != len(names):
        raise ValueError(f"fused stencils must have distinct names, got {names}")
    for m in metas[1:]:
        if m.ndim != metas[0].ndim:
            raise ValueError(f"cannot fuse {m.ndim}D {m.name} with {metas[0].ndim}D {metas[0].name}")
        if m.dtype != metas[0].dtype:
            raise ValueError(f"cannot fuse {m.dtype} {m.name} with {metas[0].dtype} {metas[0].name}")

    fused = _StencilMeta()
    fused.name = fused_name(names)
    fused.ndim = metas[0].ndim
    fused.order = max(m.order for m in metas)
    fused.dtype = metas[0].dtype
    fused.num_inputs = len(inputs)
    fused.input_names = inputs
    seen: dict[str, _AccessInfo] = {}
    for m in metas:
        for acc in m.accesses:
            if acc.view_name not in seen:
                seen[acc.view_name] = _AccessInfo(
                    inputs.index(m.input_names[acc.array_index]), acc.offsets, acc.view_name
                )
    fused.accesses = list(seen.values())
    fused.expressions = [m.expression for m in metas]
    fused.expression = fused.expressions[0]
    fused.output_names = names
    fused.stencil_inputs = stencil_inputs
    for m in metas:
        fused.constants.update(m.constants)
    fused.kernel_constants = shared_constants
    return fused


def lower_fused_to_target_ir(
    modules: Sequence[ModuleOp],
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
    kernel_hints: dict | None = None,
    spatial_term_order: bool | None = None,
) -> ModuleOp:
    """Lower several Dialect 1 modules into one multi-output ``KernelOp``
    plus its launcher (see :func:`extract_fused_meta`)."""
    meta = extract_fused_meta(modules, spatial_term_order)
    kernel_op = _build_kernel_op(meta, tile_sizes, halo_widths, kernel_hints)
    host_op = _build_standard_host(meta, tile_sizes, halo_widths)
    mod_block = Block()
    mod_block.add_op(kernel_op)
    mod_block.add_op(host_op)
    return ModuleOp(Region([mod_block]))
