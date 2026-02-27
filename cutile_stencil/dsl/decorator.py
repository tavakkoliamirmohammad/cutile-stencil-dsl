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
):
    """Decorator that captures a stencil function and builds a StencilSpec.

    Usage::

        @stencil(ndim=1, order=2)
        def heat_1d(u, i):
            return 0.25 * (u[i-1] + 2*u[i] + u[i+1])
    """
    def decorator(func):
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
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
