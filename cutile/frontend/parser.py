"""
Python AST to cuTile Stencil Dialect IR parser.

Converts a Python function decorated with ``@stencil`` into an xDSL
``ModuleOp`` containing a ``cutile_stencil.FuncOp``.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

from xdsl.dialects import arith
from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    FileLineColLoc,
    FloatAttr,
    IntAttr,
    ModuleOp,
    StringAttr,
    f32,
    f64,
)
from xdsl.dialects.stencil import IndexAttr
from xdsl.ir import Block, Region, SSAValue

from cutile.dialects.cutile_stencil.dialect import (
    AccessOp,
    BoundaryAttr,
    FuncOp,
    YieldOp,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FLOAT_TYPE_MAP = {
    "float64": f64,
    "float32": f32,
}


def _make_index_attr(offsets: list[int]) -> IndexAttr:
    """Build an IndexAttr without triggering deprecation warnings."""
    return IndexAttr(ArrayAttr([IntAttr(o) for o in offsets]))


class _ParseError(Exception):
    """Raised when the stencil function cannot be parsed."""


# ---------------------------------------------------------------------------
# AST analysis
# ---------------------------------------------------------------------------


def _resolve_offset(node: ast.expr, index_vars: list[str]) -> tuple[str, list[int]]:
    """Resolve a subscript slice to (array_name_unused, offset_per_dim).

    For a 1-D access like ``u[i+1]``, returns ``[1]``.
    For a 2-D access like ``u[i-1, j+2]``, returns ``[-1, 2]``.
    """
    # The slice may be a single expression (1-D) or a Tuple (N-D).
    if isinstance(node, ast.Tuple):
        elts = node.elts
    else:
        elts = [node]

    offsets: list[int] = []
    for elt in elts:
        offsets.append(_resolve_single_offset(elt, index_vars))
    return offsets


def _resolve_single_offset(node: ast.expr, index_vars: list[str]) -> int:
    """Resolve a single dimension subscript expression to an integer offset.

    Handles patterns:
        i       -> 0
        i + 1   -> +1
        i - 1   -> -1
        i + N   -> +N  (for any integer N)
    """
    if isinstance(node, ast.Name):
        if node.id in index_vars:
            return 0
        raise _ParseError(
            f"Unknown index variable '{node.id}', expected one of {index_vars}"
        )

    if isinstance(node, ast.BinOp):
        # Pattern: index_var +/- constant
        if isinstance(node.left, ast.Name) and node.left.id in index_vars:
            const = _eval_constant_expr(node.right)
            if isinstance(node.op, ast.Add):
                return const
            elif isinstance(node.op, ast.Sub):
                return -const
            else:
                raise _ParseError(
                    f"Unsupported offset operation: {type(node.op).__name__}"
                )
        # Pattern: constant + index_var  (commutative)
        if isinstance(node.right, ast.Name) and node.right.id in index_vars:
            const = _eval_constant_expr(node.left)
            if isinstance(node.op, ast.Add):
                return const
            else:
                raise _ParseError(
                    f"Unsupported offset operation: {type(node.op).__name__}"
                )

    # Try evaluating as a constant offset expression (e.g., bare integer)
    raise _ParseError(
        f"Cannot resolve subscript expression to an offset: {ast.dump(node)}"
    )


def _eval_constant_expr(node: ast.expr) -> int:
    """Evaluate an AST node that should be a compile-time integer constant."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_constant_expr(node.operand)
    raise _ParseError(
        f"Expected compile-time integer constant, got: {ast.dump(node)}"
    )


def _collect_accesses(
    node: ast.expr,
    array_params: list[str],
    index_vars: list[str],
) -> list[tuple[str, list[int], ast.expr | None]]:
    """Recursively collect all array accesses (array_name, offsets, ast_node) from an expression."""
    results: list[tuple[str, list[int], ast.expr | None]] = []

    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in array_params:
            offsets = _resolve_offset(node.slice, index_vars)
            results.append((node.value.id, offsets, node))
            return results

    # Recurse into child nodes
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr):
            results.extend(_collect_accesses(child, array_params, index_vars))

    return results


def _infer_ndim_order(
    accesses: list[tuple[str, list[int], ...] | tuple[str, list[int]]],
) -> tuple[int, int]:
    """Infer ndim and order from collected accesses."""
    if not accesses:
        raise _ParseError("No array accesses found; cannot infer ndim/order")

    ndim = len(accesses[0][1])
    max_offset = 0
    for item in accesses:
        offsets = item[1]
        if len(offsets) != ndim:
            raise _ParseError(
                f"Inconsistent dimensionality: expected {ndim}, "
                f"got {len(offsets)}"
            )
        for o in offsets:
            max_offset = max(max_offset, abs(o))

    order = 2 * max_offset
    return ndim, order


# ---------------------------------------------------------------------------
# Expression codegen: AST expression -> SSAValue (arith ops in block)
# ---------------------------------------------------------------------------


class _ExprBuilder:
    """Translates a Python AST expression into arith SSA ops.

    Emits ops into *block* and returns the SSAValue of the result.
    """

    def __init__(
        self,
        block: Block,
        float_type,
        access_map: dict[tuple[str, tuple[int, ...]], SSAValue],
        constant_map: dict[str, float],
        source_file: str | None = None,
        source_line_offset: int = 0,
    ):
        self.block = block
        self.float_type = float_type
        self.access_map = access_map
        self.constant_map = constant_map
        self._const_cache: dict[float, SSAValue] = {}
        self._source_file = source_file
        self._source_line_offset = source_line_offset

    def _make_loc(self, node: ast.expr) -> FileLineColLoc | None:
        """Create a FileLineColLoc from an AST node if source info is available."""
        if self._source_file is None:
            return None
        line = getattr(node, "lineno", 0) + self._source_line_offset
        col = getattr(node, "col_offset", 0)
        return FileLineColLoc(
            StringAttr(self._source_file),
            IntAttr(line),
            IntAttr(col),
        )

    def _attach_loc(self, op, node: ast.expr) -> None:
        """Attach source location to an IR op if available."""
        loc = self._make_loc(node)
        if loc is not None:
            op.location = loc

    def _emit_const(self, value: float, node: ast.expr | None = None) -> SSAValue:
        if value in self._const_cache:
            return self._const_cache[value]
        op = arith.ConstantOp(FloatAttr(value, self.float_type))
        if node is not None:
            self._attach_loc(op, node)
        self.block.add_op(op)
        self._const_cache[value] = op.result
        return op.result

    def build(self, node: ast.expr) -> SSAValue:
        """Recursively lower an AST expression node."""
        if isinstance(node, ast.BinOp):
            return self._build_binop(node)
        if isinstance(node, ast.UnaryOp):
            return self._build_unaryop(node)
        if isinstance(node, ast.Constant):
            return self._build_constant(node)
        if isinstance(node, ast.Subscript):
            return self._build_access(node)
        if isinstance(node, ast.Name):
            return self._build_name(node)
        raise _ParseError(
            f"Unsupported expression node: {type(node).__name__} "
            f"({ast.dump(node)})"
        )

    def _build_binop(self, node: ast.BinOp) -> SSAValue:
        left = self.build(node.left)
        right = self.build(node.right)

        op_map = {
            ast.Add: arith.AddfOp,
            ast.Sub: arith.SubfOp,
            ast.Mult: arith.MulfOp,
            ast.Div: arith.DivfOp,
        }
        op_cls = op_map.get(type(node.op))
        if op_cls is None:
            raise _ParseError(
                f"Unsupported binary operator: {type(node.op).__name__}"
            )
        op = op_cls(left, right)
        self._attach_loc(op, node)
        self.block.add_op(op)
        return op.result

    def _build_unaryop(self, node: ast.UnaryOp) -> SSAValue:
        if isinstance(node.op, ast.USub):
            operand = self.build(node.operand)
            op = arith.NegfOp(operand)
            self._attach_loc(op, node)
            self.block.add_op(op)
            return op.result
        raise _ParseError(
            f"Unsupported unary operator: {type(node.op).__name__}"
        )

    def _build_constant(self, node: ast.Constant) -> SSAValue:
        if isinstance(node.value, (int, float)):
            return self._emit_const(float(node.value), node)
        raise _ParseError(f"Unsupported constant type: {type(node.value)}")

    def _build_access(self, node: ast.Subscript) -> SSAValue:
        if not isinstance(node.value, ast.Name):
            raise _ParseError("Subscript target must be a simple name")
        array_name = node.value.id
        # We don't re-parse offsets here; we look up from access_map
        # which was pre-populated during the collection phase.
        # Re-derive offsets from the slice to form the key.
        from cutile.frontend.parser import _resolve_offset

        # index_vars stored on self would be needed; store them in __init__
        offsets = _resolve_offset(node.slice, self._index_vars)
        key = (array_name, tuple(offsets))
        if key not in self.access_map:
            raise _ParseError(
                f"Access {array_name}{list(offsets)} not found in access map"
            )
        return self.access_map[key]

    def _build_name(self, node: ast.Name) -> SSAValue:
        """Handle a bare name reference -- must be a closure constant."""
        name = node.id
        if name in self.constant_map:
            return self._emit_const(self.constant_map[name], node)
        raise _ParseError(
            f"Unknown variable '{name}'. Not an array parameter or "
            f"known constant. If it is a closure variable, pass it via "
            f"the 'constants' argument."
        )


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def _extract_constants(fn) -> dict[str, float]:
    """Extract closure and global constants from a function."""
    constants: dict[str, float] = {}

    # Closure variables
    if fn.__code__.co_freevars and fn.__closure__:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            try:
                val = cell.cell_contents
                if isinstance(val, (int, float)):
                    constants[name] = float(val)
            except ValueError:
                pass

    return constants


def parse_stencil(
    fn,
    ndim: int | None,
    order: int | None,
    dtype: str = "float64",
    boundary=None,
    constants: dict[str, float] | None = None,
) -> ModuleOp:
    """Parse a ``@stencil``-decorated Python function into Dialect 1 IR.

    Parameters
    ----------
    fn:
        The Python function (callable).
    ndim:
        Number of dimensions.  ``None`` to auto-infer.
    order:
        Stencil order.  ``None`` to auto-infer.
    dtype:
        Data type string (``"float64"`` or ``"float32"``).
    boundary:
        Optional boundary specification (string or dict).
    constants:
        Optional dict mapping names to float values for closure variables.

    Returns
    -------
    ModuleOp
        xDSL ``ModuleOp`` containing a ``cutile_stencil.FuncOp``.
    """
    # ------------------------------------------------------------------
    # 1. Parse the source into an AST
    # ------------------------------------------------------------------
    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)

    # Extract source file and starting line number for diagnostics.
    # The AST line numbers are relative to the parsed source snippet;
    # source_line_offset adjusts them to absolute file positions.
    source_file: str | None = None
    source_line_offset: int = 0
    try:
        source_file = inspect.getfile(fn)
        _, start_lineno = inspect.getsourcelines(fn)
        # AST lineno is 1-based within the snippet; the snippet starts at
        # start_lineno in the file.  offset = start_lineno - 1 so that
        # AST line 1 maps to start_lineno.
        source_line_offset = start_lineno - 1
    except (OSError, TypeError):
        pass  # source info unavailable (e.g., python -c)

    # Find the function definition (skip decorators)
    func_def: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_def = node
            break
    if func_def is None:
        raise _ParseError("No function definition found in source")

    func_name = func_def.name

    # ------------------------------------------------------------------
    # 2. Extract parameters
    # ------------------------------------------------------------------
    params = [arg.arg for arg in func_def.args.args]
    if len(params) < 2:
        raise _ParseError(
            f"Stencil function '{func_name}' must have at least 2 parameters "
            f"(array + index variables), got {len(params)}: {params}"
        )

    # Heuristic: index variables are the last N params where N = ndim (if known)
    # or we determine them from subscript analysis.
    # Strategy: collect all subscript accesses first, then partition params.

    # ------------------------------------------------------------------
    # 3. Find the return statement
    # ------------------------------------------------------------------
    return_node: ast.Return | None = None
    for node in ast.walk(func_def):
        if isinstance(node, ast.Return) and node.value is not None:
            return_node = node
            break
    if return_node is None:
        raise _ParseError(
            f"Stencil function '{func_name}' must have a return statement"
        )

    return_expr = return_node.value

    # ------------------------------------------------------------------
    # 4. Identify array params vs index vars from subscript usage
    # ------------------------------------------------------------------
    # Scan for all Subscript nodes to discover which params are arrays
    # and which are used as index variables.
    subscript_arrays: set[str] = set()
    subscript_indices: set[str] = set()

    for node in ast.walk(return_expr):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in params:
                subscript_arrays.add(node.value.id)
                # Collect names used in the slice
                for sub_node in ast.walk(node.slice):
                    if isinstance(sub_node, ast.Name) and sub_node.id in params:
                        subscript_indices.add(sub_node.id)

    # Also scan assignment targets if the body has intermediate assignments
    for stmt in func_def.body:
        if isinstance(stmt, ast.Assign):
            for node in ast.walk(stmt.value):
                if isinstance(node, ast.Subscript) and isinstance(
                    node.value, ast.Name
                ):
                    if node.value.id in params:
                        subscript_arrays.add(node.value.id)
                        for sub_node in ast.walk(node.slice):
                            if (
                                isinstance(sub_node, ast.Name)
                                and sub_node.id in params
                            ):
                                subscript_indices.add(sub_node.id)

    # array_params: params that appear as subscript targets
    # index_vars: params that appear as subscript indices
    array_params = [p for p in params if p in subscript_arrays]
    index_vars = [p for p in params if p in subscript_indices]

    if not array_params:
        # Fallback: first param is the array, rest are index vars
        array_params = [params[0]]
        index_vars = params[1:]

    if not index_vars:
        # Fallback: everything after array params
        array_param_set = set(array_params)
        index_vars = [p for p in params if p not in array_param_set]

    # ------------------------------------------------------------------
    # 5. Collect all accesses and infer ndim/order if needed
    # ------------------------------------------------------------------
    all_accesses = _collect_accesses(return_expr, array_params, index_vars)

    # Also collect from assignment statements in the body
    for stmt in func_def.body:
        if isinstance(stmt, ast.Assign):
            all_accesses.extend(
                _collect_accesses(stmt.value, array_params, index_vars)
            )

    if ndim is None or order is None:
        inferred_ndim, inferred_order = _infer_ndim_order(all_accesses)
        if ndim is None:
            ndim = inferred_ndim
        if order is None:
            order = inferred_order

    # ------------------------------------------------------------------
    # 6. Resolve float type
    # ------------------------------------------------------------------
    float_type = _FLOAT_TYPE_MAP.get(dtype)
    if float_type is None:
        raise _ParseError(
            f"Unsupported dtype '{dtype}'. Use 'float32' or 'float64'."
        )

    # ------------------------------------------------------------------
    # 7. Merge constants
    # ------------------------------------------------------------------
    all_constants = _extract_constants(fn)
    if constants:
        all_constants.update(constants)

    # Also pull from fn.__globals__ for module-level variables referenced
    # in the expression.  We only grab numeric ones that appear as bare
    # Name nodes in the return expression.
    _referenced_names: set[str] = set()
    for node in ast.walk(return_expr):
        if isinstance(node, ast.Name):
            _referenced_names.add(node.id)

    for name in _referenced_names:
        if name not in all_constants and name not in set(array_params):
            if hasattr(fn, "__globals__") and name in fn.__globals__:
                val = fn.__globals__[name]
                if isinstance(val, (int, float)):
                    all_constants[name] = float(val)

    # ------------------------------------------------------------------
    # 8. Build the IR
    # ------------------------------------------------------------------
    # FuncOp body block with one block arg per array parameter
    func_block = Block()
    field_args: dict[str, SSAValue] = {}
    for idx, ap in enumerate(array_params):
        arg = func_block.insert_arg(float_type, idx)
        field_args[ap] = arg

    # De-duplicate accesses: (array_name, (offset_tuple)) -> AccessOp result
    unique_accesses: dict[tuple[str, tuple[int, ...]], SSAValue] = {}
    for array_name, offsets, ast_node in all_accesses:
        key = (array_name, tuple(offsets))
        if key in unique_accesses:
            continue
        field_val = field_args[array_name]
        acc_op = AccessOp(
            field_val,
            _make_index_attr(offsets),
            float_type,
            index_names=index_vars,
        )
        # Attach source location to AccessOp if available
        if source_file is not None and ast_node is not None:
            line = getattr(ast_node, "lineno", 0) + source_line_offset
            col = getattr(ast_node, "col_offset", 0)
            acc_op.location = FileLineColLoc(
                StringAttr(source_file),
                IntAttr(line),
                IntAttr(col),
            )
        func_block.add_op(acc_op)
        unique_accesses[key] = acc_op.res

    # Build arithmetic expression from the return value
    builder = _ExprBuilder(
        func_block, float_type, unique_accesses, all_constants,
        source_file=source_file,
        source_line_offset=source_line_offset,
    )
    builder._index_vars = index_vars  # needed by _build_access
    result_val = builder.build(return_expr)

    # Yield
    yield_op = YieldOp(result_val)
    func_block.add_op(yield_op)

    # Boundary attr
    boundary_attr = None
    if boundary is not None:
        if isinstance(boundary, str):
            boundary_attr = BoundaryAttr(boundary)
        elif isinstance(boundary, dict):
            bc_type = boundary.get("type", boundary.get("bc_type", "dirichlet"))
            bc_val = boundary.get("value", None)
            boundary_attr = BoundaryAttr(bc_type, bc_val)

    # Constants attr
    constants_attr = None
    if all_constants:
        constants_attr = DictionaryAttr(
            {k: FloatAttr(v, float_type) for k, v in all_constants.items()}
        )

    # FuncOp
    func_op = FuncOp(
        func_name=func_name,
        ndim=ndim,
        order=order,
        body=Region(func_block),
        dtype=dtype,
        boundary=boundary_attr,
        constants=constants_attr if all_constants else None,
    )

    # Attach source location to FuncOp
    if source_file is not None:
        func_line = getattr(func_def, "lineno", 0) + source_line_offset
        func_col = getattr(func_def, "col_offset", 0)
        func_op.location = FileLineColLoc(
            StringAttr(source_file),
            IntAttr(func_line),
            IntAttr(func_col),
        )

    # Wrap in ModuleOp
    module = ModuleOp([func_op])
    return module
