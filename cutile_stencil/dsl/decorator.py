"""The @stencil decorator for capturing stencil definitions."""

from __future__ import annotations

import functools
import inspect
from typing import Optional

from cutile_stencil.dsl.types import StencilSpec
from cutile_stencil.dsl.registry import register


def stencil(
    fn=None,
    *,
    ndim: int = 1,
    order: int = 2,
    dtype: str = "float64",
    output: str = "result",
    boundary=None,
):
    """Decorator that captures a stencil function and builds a StencilSpec.

    Usage::

        @stencil(ndim=1, order=2)
        def heat_1d(u, i):
            return 0.25 * (u[i-1] + 2*u[i] + u[i+1])

        @stencil(ndim=2, order=2, boundary=BoundarySpec.periodic(2))
        def wave_2d(u, i, j):
            ...
    """
    def decorator(func):
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Validation
        if len(params) < ndim + 1:
            raise ValueError(
                f"Stencil '{func.__name__}' has {len(params)} parameters, "
                f"but ndim={ndim} requires at least {ndim + 1} "
                f"(at least 1 array + {ndim} index parameters)"
            )

        # Check for return statement
        import ast
        import textwrap
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
        inputs = tuple(params[:-ndim])
        spec = StencilSpec(
            name=func.__name__,
            ndim=ndim,
            order=order,
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
