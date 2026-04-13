"""NumPy stencil executor -- runs without GPU.

Uses an ``_ArrayProxy`` to intercept ``u[offset]`` subscripts and map them
to numpy slices over the interior domain.  This version is decoupled from
the DSL types: it takes the raw Python function, ``ndim``, and
``halo_widths`` directly instead of a ``StencilSpec``.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Array proxy
# ---------------------------------------------------------------------------

class _ArrayProxy:
    """Proxy that translates integer-offset subscripts to numpy slices.

    For a 1-D array of size *N* with halo *h*, ``proxy[k]`` returns the
    slice ``u[h+k : N-h+k]``, i.e. the interior shifted by *k*.

    For n-D arrays, subscripts are tuples of offsets.
    """

    def __init__(self, data: np.ndarray, halo: Tuple[int, ...]):
        self._data = data
        self._halo = halo
        self._ndim = data.ndim

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        slices: list[slice] = []
        for dim, off in enumerate(key):
            h = self._halo[dim]
            n = self._data.shape[dim]
            start = h + off
            stop = n - h + off
            slices.append(slice(start, stop))
        return self._data[tuple(slices)]


# ---------------------------------------------------------------------------
# Single time-step application
# ---------------------------------------------------------------------------

def apply_stencil(
    u: np.ndarray,
    fn: Callable,
    ndim: int,
    halo_widths: Tuple[int, ...],
) -> np.ndarray:
    """Apply a stencil function once and return the updated array.

    Parameters
    ----------
    u : np.ndarray
        Full array including boundary / halo cells.
    fn : callable
        The stencil update function, e.g. ``def heat1d(u, i): ...``.
    ndim : int
        Number of spatial dimensions.
    halo_widths : tuple of int
        Halo width per dimension (e.g. ``(1,)`` for a 3-point 1-D stencil).

    Returns
    -------
    np.ndarray
        Copy of *u* with only the interior cells updated.
    """
    proxy = _ArrayProxy(u, halo_widths)
    idx_args = [0] * ndim
    result = fn(proxy, *idx_args)

    out = u.copy()
    interior = tuple(
        slice(h, s - h) for h, s in zip(halo_widths, u.shape)
    )
    out[interior] = result
    return out


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

class BoundaryKind(Enum):
    """Supported boundary condition types."""
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    PERIODIC = "periodic"
    REFLECTING = "reflecting"


def apply_boundary(
    u: np.ndarray,
    boundary_spec: dict,
    halo_widths: Tuple[int, ...],
) -> None:
    """Apply boundary conditions in-place.

    Parameters
    ----------
    u : np.ndarray
        Full array including halo cells (modified in-place).
    boundary_spec : dict
        Mapping from dimension index to a ``(low, high)`` pair, where each
        element is either a :class:`BoundaryKind` string
        (``"dirichlet"``, ``"neumann"``, ``"periodic"``, ``"reflecting"``)
        or a ``(kind_str, value)`` tuple for Dirichlet with a non-zero value.
    halo_widths : tuple of int
        Halo width per dimension.
    """
    ndim = u.ndim
    for dim in range(ndim):
        if dim not in boundary_spec:
            continue
        low_spec, high_spec = boundary_spec[dim]
        h = halo_widths[dim]
        n = u.shape[dim]

        low_kind, low_val = _parse_bc(low_spec)
        high_kind, high_val = _parse_bc(high_spec)

        _apply_bc_side(u, ndim, dim, h, n, low_kind, low_val, side="low")
        _apply_bc_side(u, ndim, dim, h, n, high_kind, high_val, side="high")


def _parse_bc(spec) -> Tuple[str, float]:
    """Normalise a boundary spec to ``(kind_str, value)``."""
    if isinstance(spec, str):
        return spec, 0.0
    if isinstance(spec, tuple):
        return spec[0], spec[1]
    # Allow BoundaryKind enum values directly
    if isinstance(spec, BoundaryKind):
        return spec.value, 0.0
    raise TypeError(f"Unsupported boundary spec: {spec!r}")


def _apply_bc_side(
    u: np.ndarray,
    ndim: int,
    dim: int,
    h: int,
    n: int,
    kind: str,
    value: float,
    side: str,
) -> None:
    """Apply a single boundary condition for one side of one dimension."""
    if kind == "dirichlet":
        sl = [slice(None)] * ndim
        if side == "low":
            sl[dim] = slice(0, h)
        else:
            sl[dim] = slice(n - h, n)
        u[tuple(sl)] = value

    elif kind == "neumann":
        for k in range(h):
            src = [slice(None)] * ndim
            dst = [slice(None)] * ndim
            if side == "low":
                src[dim] = h
                dst[dim] = h - 1 - k
            else:
                src[dim] = n - h - 1
                dst[dim] = n - h + k
            u[tuple(dst)] = u[tuple(src)]

    elif kind == "periodic":
        for k in range(h):
            src = [slice(None)] * ndim
            dst = [slice(None)] * ndim
            if side == "low":
                src[dim] = n - 2 * h + k
                dst[dim] = k
            else:
                src[dim] = h + k
                dst[dim] = n - h + k
            u[tuple(dst)] = u[tuple(src)]

    elif kind == "reflecting":
        for k in range(h):
            src = [slice(None)] * ndim
            dst = [slice(None)] * ndim
            if side == "low":
                src[dim] = 2 * h - 1 - k
                dst[dim] = k
            else:
                src[dim] = n - h - 1 - k
                dst[dim] = n - h + k
            u[tuple(dst)] = u[tuple(src)]


# ---------------------------------------------------------------------------
# Time marching
# ---------------------------------------------------------------------------

def time_march(
    u0: np.ndarray,
    fn: Callable,
    ndim: int,
    halo_widths: Tuple[int, ...],
    steps: int,
    boundary_spec: Optional[dict] = None,
) -> List[np.ndarray]:
    """Run multiple time steps of a stencil, returning snapshots.

    Parameters
    ----------
    u0 : np.ndarray
        Initial condition (full array with boundaries).
    fn : callable
        Stencil update function.
    ndim : int
        Number of spatial dimensions.
    halo_widths : tuple of int
        Halo width per dimension.
    steps : int
        Number of time steps.
    boundary_spec : dict, optional
        Boundary conditions (see :func:`apply_boundary`).

    Returns
    -------
    list of np.ndarray
        Snapshots ``[u0, u1, ..., u_steps]``.
    """
    history: List[np.ndarray] = [u0.copy()]
    u = u0.copy()
    for _ in range(steps):
        u = apply_stencil(u, fn, ndim, halo_widths)
        if boundary_spec is not None:
            apply_boundary(u, boundary_spec, halo_widths)
        history.append(u.copy())
    return history
