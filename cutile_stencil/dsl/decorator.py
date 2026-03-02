"""The @stencil decorator for capturing stencil definitions."""

from __future__ import annotations

import ast
import functools
import inspect
import textwrap
from typing import Optional

from cutile_stencil.dsl.types import StencilSpec
from cutile_stencil.dsl.registry import register


def _resolve_offset(node: ast.expr, index_names: set) -> int | None:
    """Resolve an AST index expression to a constant integer offset."""
    if isinstance(node, ast.Name) and node.id in index_names:
        return 0
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _resolve_offset(node.left, index_names)
        right = _resolve_offset(node.right, index_names)
        if left is not None and right is not None:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _resolve_offset(node.operand, index_names)
        if val is not None:
            return -val
    return None


def _infer_stencil_params(func):
    """Infer ndim and order from the stencil function body via AST analysis.

    Returns (ndim, order) where:
    - ndim = number of subscript dimensions in array accesses
    - order = 2 * max absolute offset across all subscript accesses
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    param_set = set(params)

    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)

    # Find all subscript accesses where the base is a function parameter
    subscript_bases = set()
    subscript_dims = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in param_set:
                subscript_bases.add(node.value.id)
                slc = node.slice
                if isinstance(slc, ast.Tuple):
                    subscript_dims.add(len(slc.elts))
                else:
                    subscript_dims.add(1)

    if not subscript_dims:
        raise ValueError(
            f"Cannot infer ndim for '{func.__name__}': "
            f"no array subscript accesses found"
        )

    if len(subscript_dims) > 1:
        raise ValueError(
            f"Inconsistent subscript dimensions in '{func.__name__}': "
            f"{sorted(subscript_dims)}"
        )

    ndim = subscript_dims.pop()

    # Index params = params not used as subscript bases
    index_names = set(p for p in params if p not in subscript_bases)

    # Compute max absolute offset across all subscript accesses
    max_abs_offset = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in subscript_bases:
                slc = node.slice
                elts = slc.elts if isinstance(slc, ast.Tuple) else [slc]
                for elt in elts:
                    off = _resolve_offset(elt, index_names)
                    if off is not None:
                        max_abs_offset = max(max_abs_offset, abs(off))

    order = 2 * max_abs_offset
    return ndim, order


def stencil(
    fn=None,
    *,
    ndim: int | None = None,
    order: int | None = None,
    dtype: str = "float64",
    output: str = "result",
    boundary=None,
):
    """Decorator that captures a stencil function and builds a StencilSpec.

    Parameters ``ndim`` and ``order`` are inferred from the function body
    when not provided explicitly:

    - ``ndim`` is determined from the number of subscript dimensions.
    - ``order`` is computed as ``2 * max_absolute_offset``.

    Usage::

        @stencil  # ndim and order inferred
        def heat_1d(u, i):
            return 0.25 * (u[i-1] + 2*u[i] + u[i+1])

        @stencil(ndim=2, order=2, boundary=BoundarySpec.periodic(2))
        def wave_2d(u, i, j):
            ...
    """
    def decorator(func):
        _ndim = ndim
        _order = order

        # Infer ndim and order if not provided
        if _ndim is None or _order is None:
            inferred_ndim, inferred_order = _infer_stencil_params(func)
            if _ndim is None:
                _ndim = inferred_ndim
            if _order is None:
                _order = inferred_order

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Validation
        if len(params) < _ndim + 1:
            raise ValueError(
                f"Stencil '{func.__name__}' has {len(params)} parameters, "
                f"but ndim={_ndim} requires at least {_ndim + 1} "
                f"(at least 1 array + {_ndim} index parameters)"
            )

        # Check for return statement
        src = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(src)
        has_return = any(isinstance(n, ast.Return) and n.value is not None
                         for n in ast.walk(tree))
        if not has_return:
            raise ValueError(
                f"Stencil '{func.__name__}' must contain a return statement"
            )

        # Extract closure constants (captured scope variables like dt, eps, dx)
        from cutile_stencil.codegen.ast_transform import extract_closure_constants
        closure_consts = extract_closure_constants(func)

        # First param(s) are inputs (arrays), last ndim params are indices
        inputs = tuple(params[:-_ndim])
        spec = StencilSpec(
            name=func.__name__,
            ndim=_ndim,
            order=_order,
            inputs=inputs,
            output=output,
            update_fn=func,
            dtype=dtype,
            boundary=boundary,
            closure_constants=closure_consts,
        )
        func._stencil_spec = spec
        register(spec)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._stencil_spec = spec
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
