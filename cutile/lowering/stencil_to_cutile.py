"""Stencil-to-cuTile lowering: Dialect 1 IR to cuTile Python source.

This module provides the public ``lower_stencil_to_python()`` API that
takes a Dialect 1 stencil IR module and produces cuTile Python source.

The lowering now goes through the three-dialect stack:

    Dialect 1 (cutile_stencil) -> Dialect 3 (cutile_target) -> Python

Internally it delegates to:

* ``stencil_to_target.lower_to_target_ir()`` -- builds Dialect 3 IR
  (``KernelOp``, ``SliceOp``, ``LoadOp``, ``StoreOp``, ``LaunchOp``, etc.)
* ``target_to_python.emit_python()`` -- walks Dialect 3 IR and emits
  Python source

The generated Python file contains:

* ``@ct.kernel`` function with ``.slice()`` / ``ct.load`` / ``ct.store``
* ``launch_<name>`` host function with grid calculation and ``ct.launch``
* Optional temporal-blocking wrapper (buffer-swap loop)
* Optional boundary condition function
"""

from __future__ import annotations

from typing import Sequence

from xdsl.dialects import arith
from xdsl.dialects.builtin import FloatAttr, ModuleOp
from xdsl.ir import Block, SSAValue

from cutile.dialects.cutile_stencil.dialect import AccessOp, FuncOp, YieldOp
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

    The lowering now goes through the three-dialect stack:

        Dialect 1 (cutile_stencil) -> Dialect 3 (cutile_target) -> Python

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
    from cutile.lowering.stencil_to_target import lower_to_target_ir
    from cutile.lowering.target_to_python import emit_python

    # ---------------------------------------------------------------- #
    # 1. Resolve defaults (need to peek at FuncOp for ndim/order)
    # ---------------------------------------------------------------- #
    func_op: FuncOp | None = None
    for op in module.body.ops:
        if isinstance(op, FuncOp):
            func_op = op
            break
    if func_op is None:
        raise ValueError("No cutile_stencil.FuncOp found in module")

    ndim = func_op.ndim.data
    order = func_op.order.data

    _default_tiles = {1: (1024,), 2: (64, 64), 3: (32, 32, 32)}
    if tile_sizes is None:
        tile_sizes = _default_tiles.get(ndim, (64,) * ndim)
    if halo_widths is None:
        hw = order // 2 if order else 1
        halo_widths = (hw,) * ndim

    # ---------------------------------------------------------------- #
    # 2. Lower to Dialect 3 (cutile_target) IR
    # ---------------------------------------------------------------- #
    target_ir = lower_to_target_ir(
        module,
        tile_sizes=tile_sizes,
        halo_widths=halo_widths,
        temporal_steps=temporal_steps,
        boundary_spec=boundary_spec,
    )

    # ---------------------------------------------------------------- #
    # 3. Emit Python source from Dialect 3 IR
    # ---------------------------------------------------------------- #
    return emit_python(target_ir)
