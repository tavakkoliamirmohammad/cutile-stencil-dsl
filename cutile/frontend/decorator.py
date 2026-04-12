"""
The ``@stencil`` decorator for the cuTile Stencil DSL.

Converts a Python stencil function into Dialect 1 IR and attaches metadata
to the wrapped callable.
"""

from __future__ import annotations

import functools
from typing import Any

from cutile.frontend.parser import parse_stencil


def stencil(fn=None, *, ndim=None, order=None, dtype="float64", boundary=None):
    """Decorator that converts a Python function into stencil IR.

    Usage::

        @stencil(ndim=2, order=2)
        def heat(u, i, j):
            return 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])

    The decorated function gains:

    - ``heat._ir``:  xDSL ``ModuleOp`` (Dialect 1 IR)
    - ``heat._fn``:  the original Python function (for CPU reference)
    - ``heat._ndim``, ``heat._order``, ``heat._dtype``: metadata
    - ``heat._boundary``: boundary specification (or ``None``)

    Supports both ``@stencil`` (bare) and ``@stencil(ndim=2, ...)`` syntax.
    """

    def _wrap(func):
        ir = parse_stencil(
            func,
            ndim=ndim,
            order=order,
            dtype=dtype,
            boundary=boundary,
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._ir = ir
        wrapper._fn = func
        wrapper._ndim = ndim
        wrapper._order = order
        wrapper._dtype = dtype
        wrapper._boundary = boundary
        return wrapper

    # Support both @stencil and @stencil(...) syntax
    if fn is not None:
        # Called as @stencil without arguments
        return _wrap(fn)
    # Called as @stencil(...) with keyword arguments
    return _wrap
