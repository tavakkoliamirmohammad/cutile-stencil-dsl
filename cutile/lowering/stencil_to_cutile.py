"""Stencil-to-cuTile lowering: Dialect 1 IR to cuTile Python source.

Given an xDSL ``ModuleOp`` produced by the frontend parser (Dialect 1),
this module walks the IR, extracts stencil metadata and arithmetic
structure, and emits a complete Python file that uses the ``cuda.tile``
(cuTile) API:

* ``@ct.kernel`` function with ``.slice()`` / ``ct.load`` / ``ct.store``
* ``launch_<name>`` host function with grid calculation and ``ct.launch``
* Optional temporal-blocking wrapper (buffer-swap loop)

This is the *pragmatic* lowering path -- it goes straight from Dialect 1
to Python source without constructing an intermediate Dialect 3 IR,
keeping the implementation compact while matching the output format of
the existing ``StencilCodeGenerator`` exactly.
"""

from __future__ import annotations

from typing import Sequence

from xdsl.dialects import arith
from xdsl.dialects.builtin import FloatAttr, ModuleOp
from xdsl.ir import Block, Operation, SSAValue

from cutile.dialects.cutile_stencil.dialect import AccessOp, FuncOp, YieldOp
from cutile.lowering.emitter import CodeEmitter


# -------------------------------------------------------------------- #
# Internal data structures
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
# IR walking helpers
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

    # Determine input parameter names.
    # Convention: single input -> "u", multiple -> "u", "v", "w", ...
    _default_names = list("uvwxyz")
    meta.input_names = [
        _default_names[i] if i < len(_default_names) else f"arr{i}"
        for i in range(meta.num_inputs)
    ]

    # Extract constants from the FuncOp
    if func_op.constants is not None:
        for key, val in func_op.constants.data.items():
            if isinstance(val, FloatAttr):
                meta.constants[key] = val.value.data

    # Extract boundary info
    if func_op.boundary is not None:
        meta.boundary = {
            "type": func_op.boundary.bc_type_str,
        }
        if func_op.boundary.has_value:
            meta.boundary["value"] = func_op.boundary.value.value.data

    # Collect unique accesses (preserving order of first occurrence).
    seen: dict[tuple[int, tuple[int, ...]], _AccessInfo] = {}
    access_order: list[_AccessInfo] = []

    # Map from SSAValue -> variable name used in expression reconstruction.
    val_to_name: dict[SSAValue, str] = {}

    for op in block.ops:
        if isinstance(op, AccessOp):
            offsets = tuple(item.data for item in op.offset.parameters[0].data)
            # Determine which block arg (= which input array) this accesses.
            arr_idx: int | None = None
            for i, arg in enumerate(block.args):
                if op.field is arg:
                    arr_idx = i
                    break
            if arr_idx is None:
                arr_idx = 0  # fallback

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

    # Reconstruct the stencil expression from arith ops.
    meta.expression = _reconstruct_expr(block, val_to_name)

    return meta


def _reconstruct_expr(block: Block, val_to_name: dict[SSAValue, str]) -> str:
    """Walk arith ops in *block* and reconstruct a Python expression string.

    The approach mirrors a simple stack-based decompiler: each op's result
    gets an expression string, and we return the expression string yielded
    by the final ``YieldOp``.
    """
    for op in block.ops:
        if isinstance(op, arith.ConstantOp):
            val_attr = op.properties.get("value", op.attributes.get("value", None))
            if isinstance(val_attr, FloatAttr):
                fval = val_attr.value.data
                # Emit clean float literals
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
            # Wrap RHS in parens if it contains + or -
            if any(c in right for c in ("+", "-")) and not right.startswith("("):
                right = f"({right})"
            val_to_name[op.result] = f"{left} - {right}"

        elif isinstance(op, arith.MulfOp):
            left = val_to_name.get(op.lhs, "?")
            right = val_to_name.get(op.rhs, "?")
            # Wrap operands in parens if they contain + or -
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

    return "0"  # fallback


# -------------------------------------------------------------------- #
# Code emission
# -------------------------------------------------------------------- #


def _emit_header(e: CodeEmitter, meta: _StencilMeta) -> None:
    """Emit module header: docstring, imports, constants."""
    e.line(f'"""cuTile kernel for {meta.name} stencil (auto-generated)."""')
    e.blank()
    e.line("import cuda.tile as ct")
    e.line("import cupy as cp")

    if meta.constants:
        e.blank()
        e.line("# Captured constants from stencil definition scope")
        for name, value in sorted(meta.constants.items()):
            e.line(f"{name} = {value!r}")


def _emit_kernel(
    e: CodeEmitter,
    meta: _StencilMeta,
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
) -> None:
    """Emit the ``@ct.kernel`` function."""
    ndim = meta.ndim
    bid_vars = ["bx", "by", "bz"][:ndim]
    tile_vars = ["TX", "TY", "TZ"][:ndim]
    halo_vars = ["HX", "HY", "HZ"][:ndim]
    n_vars = ["nx", "ny", "nz"][:ndim]

    tile_const_params = ", ".join(f"{tv}: ConstInt" for tv in tile_vars)
    halo_const_params = ", ".join(f"{hv}: ConstInt" for hv in halo_vars)

    multi_input = meta.num_inputs > 1
    if multi_input:
        input_params = ", ".join(meta.input_names)
    else:
        input_params = meta.input_names[0]

    e.line("@ct.kernel")
    e.line(
        f"def {meta.name}_kernel({input_params}, output, "
        f"{tile_const_params}, {halo_const_params}):"
    )
    with e.indent():
        # Block indices
        for i, bv in enumerate(bid_vars):
            e.line(f"{bv} = ct.bid({i})")

        # Interior sizes
        first_arr = meta.input_names[0]
        for d in range(ndim):
            e.line(f"{n_vars[d]} = {first_arr}.shape[{d}] - 2 * {halo_vars[d]}")

        # Sliced views for each unique access
        for info in meta.accesses:
            arr_name = meta.input_names[info.array_index]
            chain = arr_name
            for d in range(ndim):
                off = info.offsets[d]
                start = _offset_expr(halo_vars[d], off)
                stop = _offset_expr(halo_vars[d], off, add_n=True, n_var=n_vars[d])
                chain = f"{chain}.slice(axis={d}, start={start}, stop={stop})"
            e.line(f"{info.view_name} = {chain}")

        # Output view (zero offsets)
        out_chain = "output"
        for d in range(ndim):
            out_chain = (
                f"{out_chain}.slice(axis={d}, start={halo_vars[d]}, "
                f"stop={halo_vars[d]} + {n_vars[d]})"
            )
        e.line(f"out = {out_chain}")

        # Load tiles
        idx_tuple = ", ".join(bid_vars)
        shape_tuple = ", ".join(tile_vars)
        for info in meta.accesses:
            e.line(
                f"t_{info.view_name} = ct.load({info.view_name}, "
                f"index=({idx_tuple}), shape=({shape_tuple}))"
            )

        e.blank()
        # Stencil expression
        e.line(f"result = {meta.expression}")
        e.blank()

        # Store
        e.line(f"ct.store(out, index=({idx_tuple}), tile=result)")


def _emit_launcher(
    e: CodeEmitter,
    meta: _StencilMeta,
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
) -> None:
    """Emit the ``launch_<name>`` host function."""
    ndim = meta.ndim
    multi_input = meta.num_inputs > 1
    t_vars = ["TX", "TY", "TZ"][:ndim]
    h_vars = ["HX", "HY", "HZ"][:ndim]
    n_vars = ["Nx", "Ny", "Nz"][:ndim]

    e.blank()
    e.blank()
    if multi_input:
        input_params = ", ".join(meta.input_names)
        e.line(f"def launch_{meta.name}({input_params}, u_out, stream=None):")
    else:
        e.line(f"def launch_{meta.name}(u_in, u_out, stream=None):")
    with e.indent():
        e.line(f"{', '.join(t_vars)} = {', '.join(str(t) for t in tile_sizes)}")
        e.line(f"{', '.join(h_vars)} = {', '.join(str(h) for h in halo_widths)}")

        first_input = meta.input_names[0] if multi_input else "u_in"
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

        if multi_input:
            args = ", ".join(meta.input_names)
        else:
            args = "u_in"
        t_args = ", ".join(t_vars)
        h_args = ", ".join(h_vars)
        e.line(
            f"ct.launch(stream, grid, {meta.name}_kernel, "
            f"({args}, u_out, {t_args}, {h_args}))"
        )


def _emit_temporal_launcher(
    e: CodeEmitter,
    meta: _StencilMeta,
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
    temporal_steps: int,
) -> None:
    """Emit a temporal-blocking launcher with buffer swapping."""
    ndim = meta.ndim
    multi_input = meta.num_inputs > 1
    t_vars = ["TX", "TY", "TZ"][:ndim]
    h_vars = ["HX", "HY", "HZ"][:ndim]
    T = temporal_steps

    e.blank()
    e.blank()
    if multi_input:
        input_params = ", ".join(meta.input_names)
        e.line(f"def launch_{meta.name}({input_params}, u_out, stream=None):")
    else:
        e.line(f"def launch_{meta.name}(u_in, u_out, stream=None):")

    with e.indent():
        e.line(f"{', '.join(t_vars)} = {', '.join(str(t) for t in tile_sizes)}")
        e.line(f"{', '.join(h_vars)} = {', '.join(str(h) for h in halo_widths)}")

        first_input = meta.input_names[0] if multi_input else "u_in"

        e.line("if stream is None:")
        with e.indent():
            e.line("stream = cp.cuda.get_current_stream()")

        # Grid based on shape - 2*halo per dim
        grid_parts = []
        for d in range(ndim):
            h = halo_widths[d]
            grid_parts.append(
                f"ct.cdiv({first_input}.shape[{d}] - {2 * h}, {t_vars[d]})"
            )
        trailing = "," if ndim == 1 else ""
        e.line(f"grid = ({', '.join(grid_parts)}{trailing})")

        # Buffer chain
        e.line(f"# Temporal blocking: {T} steps with buffer swapping")
        e.line(f"bufs = [{first_input}]")
        e.line(f"for _ in range({T - 1}):")
        with e.indent():
            e.line(f"bufs.append(cp.zeros_like({first_input}))")
        e.line("bufs.append(u_out)")

        # Launch loop
        e.line(f"for _step in range({T}):")
        with e.indent():
            e.line(
                f"ct.launch(stream, grid, {meta.name}_kernel, "
                f"(bufs[_step], bufs[_step + 1], "
                f"{', '.join(t_vars)}, {', '.join(h_vars)}))"
            )


def _emit_boundary(
    e: CodeEmitter,
    meta: _StencilMeta,
    halo_widths: tuple[int, ...],
    boundary_spec: dict,
) -> None:
    """Emit a boundary application function."""
    ndim = meta.ndim
    bc_type = boundary_spec.get("type", "dirichlet")
    bc_value = boundary_spec.get("value", 0.0)

    e.blank()
    e.blank()
    e.line(f"def apply_boundary_{meta.name}(u, stream=None):")
    with e.indent():
        e.line('"""Apply boundary conditions to halo cells."""')
        e.line("if stream is None:")
        with e.indent():
            e.line("stream = cp.cuda.get_current_stream()")

        for d in range(ndim):
            hw = halo_widths[d]
            if bc_type == "periodic":
                e.line(f"# Dim {d}: periodic BC")
                if ndim == 1:
                    e.line(f"u[:{hw}] = u[-2*{hw}:-{hw}]")
                    e.line(f"u[-{hw}:] = u[{hw}:2*{hw}]")
                else:
                    low_dst = ", ".join(
                        ":" if i != d else f":{hw}" for i in range(ndim)
                    )
                    low_src = ", ".join(
                        ":" if i != d else f"-2*{hw}:-{hw}" for i in range(ndim)
                    )
                    high_dst = ", ".join(
                        ":" if i != d else f"-{hw}:" for i in range(ndim)
                    )
                    high_src = ", ".join(
                        ":" if i != d else f"{hw}:2*{hw}" for i in range(ndim)
                    )
                    e.line(f"u[{low_dst}] = u[{low_src}]")
                    e.line(f"u[{high_dst}] = u[{high_src}]")

            elif bc_type == "neumann":
                e.line(f"# Dim {d}: Neumann BC (zero gradient)")
                if ndim == 1:
                    e.line(f"u[:{hw}] = u[{hw}:{hw}+1]")
                    e.line(f"u[-{hw}:] = u[-{hw}-1:-{hw}]")
                else:
                    low_dst = ", ".join(
                        ":" if i != d else f":{hw}" for i in range(ndim)
                    )
                    low_src = ", ".join(
                        ":" if i != d else f"{hw}:{hw}+1" for i in range(ndim)
                    )
                    high_dst = ", ".join(
                        ":" if i != d else f"-{hw}:" for i in range(ndim)
                    )
                    high_src = ", ".join(
                        ":" if i != d else f"-{hw}-1:-{hw}" for i in range(ndim)
                    )
                    e.line(f"u[{low_dst}] = u[{low_src}]")
                    e.line(f"u[{high_dst}] = u[{high_src}]")

            elif bc_type == "dirichlet":
                e.line(f"# Dim {d}: Dirichlet BC (value={bc_value})")
                if ndim == 1:
                    e.line(f"u[:{hw}] = {bc_value}")
                    e.line(f"u[-{hw}:] = {bc_value}")
                else:
                    low_dst = ", ".join(
                        ":" if i != d else f":{hw}" for i in range(ndim)
                    )
                    high_dst = ", ".join(
                        ":" if i != d else f"-{hw}:" for i in range(ndim)
                    )
                    e.line(f"u[{low_dst}] = {bc_value}")
                    e.line(f"u[{high_dst}] = {bc_value}")

            elif bc_type == "reflecting":
                e.line(f"# Dim {d}: reflecting BC")
                if ndim == 1:
                    e.line(f"u[:{hw}] = u[2*{hw}-1:{hw}-1:-1]")
                    e.line(f"u[-{hw}:] = u[-{hw}-1:-2*{hw}-1:-1]")
                else:
                    low_dst = ", ".join(
                        ":" if i != d else f":{hw}" for i in range(ndim)
                    )
                    low_src = ", ".join(
                        ":" if i != d else f"2*{hw}-1:{hw}-1:-1"
                        for i in range(ndim)
                    )
                    high_dst = ", ".join(
                        ":" if i != d else f"-{hw}:" for i in range(ndim)
                    )
                    high_src = ", ".join(
                        ":" if i != d else f"-{hw}-1:-2*{hw}-1:-1"
                        for i in range(ndim)
                    )
                    e.line(f"u[{low_dst}] = u[{low_src}]")
                    e.line(f"u[{high_dst}] = u[{high_src}]")


# -------------------------------------------------------------------- #
# Public API
# -------------------------------------------------------------------- #


def lower_stencil_to_python(
    module: ModuleOp,
    domain: tuple[int, ...] | None = None,
    tile_sizes: tuple[int, ...] | None = None,
    halo_widths: tuple[int, ...] | None = None,
    temporal_steps: int = 1,
    boundary_spec: dict | None = None,
) -> str:
    """Lower a Dialect 1 stencil IR module to cuTile Python source code.

    Parameters
    ----------
    module:
        xDSL ``ModuleOp`` containing a ``cutile_stencil.FuncOp``.
    domain:
        Optional domain shape (not used for code emission, reserved for
        future analysis).
    tile_sizes:
        Tile sizes per dimension (e.g. ``(64, 64)``).  If ``None``,
        defaults are chosen based on ndim.
    halo_widths:
        Halo widths per dimension (e.g. ``(1, 1)``).  If ``None``,
        derived from ``order // 2``.
    temporal_steps:
        Number of temporal blocking steps.  ``1`` means no temporal
        blocking.
    boundary_spec:
        Optional dict ``{"type": "...", "value": ...}`` for boundary
        conditions.

    Returns
    -------
    str
        Complete Python source string with ``@ct.kernel`` and launcher.
    """
    # ---------------------------------------------------------------- #
    # 1. Find the FuncOp inside the module
    # ---------------------------------------------------------------- #
    func_op: FuncOp | None = None
    for op in module.body.ops:
        if isinstance(op, FuncOp):
            func_op = op
            break
    if func_op is None:
        raise ValueError("No cutile_stencil.FuncOp found in module")

    block = list(func_op.body.blocks)[0]

    # ---------------------------------------------------------------- #
    # 2. Extract metadata from the IR
    # ---------------------------------------------------------------- #
    meta = _extract_meta(func_op, block)
    ndim = meta.ndim

    # Apply defaults for tile_sizes and halo_widths
    _default_tiles = {1: (1024,), 2: (64, 64), 3: (32, 32, 32)}
    if tile_sizes is None:
        tile_sizes = _default_tiles.get(ndim, (64,) * ndim)
    if halo_widths is None:
        hw = meta.order // 2 if meta.order else 1
        halo_widths = (hw,) * ndim

    # Merge boundary info: IR boundary attr takes precedence, then
    # the explicit boundary_spec parameter.
    if boundary_spec is None and meta.boundary is not None:
        boundary_spec = meta.boundary

    # ---------------------------------------------------------------- #
    # 3. Emit Python source
    # ---------------------------------------------------------------- #
    e = CodeEmitter()
    _emit_header(e, meta)
    e.blank()
    e.line("ConstInt = ct.Constant[int]")
    e.blank()

    _emit_kernel(e, meta, tile_sizes, halo_widths)

    if boundary_spec is not None:
        _emit_boundary(e, meta, halo_widths, boundary_spec)

    if temporal_steps > 1:
        _emit_temporal_launcher(e, meta, tile_sizes, halo_widths, temporal_steps)
    else:
        _emit_launcher(e, meta, tile_sizes, halo_widths)

    return e.render()
