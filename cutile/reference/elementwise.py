"""Array-aware element-wise helpers usable both in stencil bodies and on the host.

Inside a ``@stencil`` body the frontend lowers ``max``/``min``/``abs`` and
``where`` to Tile IR.  When the same Python function is executed by the NumPy
reference (:func:`cutile.reference.stencil_ref.apply_stencil`) it runs on
arrays, where the builtins ``max``/``min`` would fail; these helpers dispatch
to NumPy or CuPy depending on the operands.
"""

from __future__ import annotations

import builtins
import importlib
import types


def _xp(*args):
    for a in args:
        mod = type(a).__module__.split(".")[0]
        if mod in ("numpy", "cupy"):
            return importlib.import_module(mod)
    return None


def maximum(a, b):
    xp = _xp(a, b)
    return builtins.max(a, b) if xp is None else xp.maximum(a, b)


def minimum(a, b):
    xp = _xp(a, b)
    return builtins.min(a, b) if xp is None else xp.minimum(a, b)


def where(cond, a, b):
    xp = _xp(cond, a, b)
    return (a if cond else b) if xp is None else xp.where(cond, a, b)


def array_aware(fn):
    """Return *fn* with ``max``/``min`` builtins replaced by array-aware ones.

    The stencil function is executed unchanged on array slices by the
    reference executor; only the two builtins that reject arrays are swapped.
    """
    g = dict(fn.__globals__)
    b = dict(vars(builtins))
    b["max"] = maximum
    b["min"] = minimum
    g["__builtins__"] = b
    g.setdefault("where", where)
    return types.FunctionType(fn.__code__, g, fn.__name__, fn.__defaults__, fn.__closure__)
