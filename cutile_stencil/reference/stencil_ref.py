"""NumPy stencil executor — runs without GPU.

Uses an _ArrayProxy to intercept ``u[offset]`` subscripts and map them
to numpy slices over the interior domain.
"""

from __future__ import annotations

from typing import List

import numpy as np

from cutile_stencil.dsl.types import StencilSpec


class _ArrayProxy:
    """Proxy that translates integer-offset subscripts to numpy slices.

    For a 1D array of size N with halo h, ``proxy[k]`` returns the slice
    ``u[h+k : N-h+k]``, i.e. the interior shifted by ``k``.

    For nD, subscripts are tuples of offsets.
    """

    def __init__(self, data: np.ndarray, halo: tuple[int, ...]):
        self._data = data
        self._halo = halo
        self._ndim = data.ndim

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        slices = []
        for dim, off in enumerate(key):
            h = self._halo[dim]
            n = self._data.shape[dim]
            start = h + off
            stop = n - h + off
            slices.append(slice(start, stop))
        return self._data[tuple(slices)]


def apply_stencil(u: np.ndarray, spec: StencilSpec) -> np.ndarray:
    """Apply a stencil once and return the updated interior.

    ``u`` should be the full array including boundary/halo cells.
    Returns an array the same shape as ``u`` with only the interior updated.
    """
    halo = spec.halo_widths
    if halo is None:
        halo = (spec.order // 2,) * spec.ndim

    proxies = {name: _ArrayProxy(u, halo) for name in spec.inputs}
    # Build index arguments (not used by update_fn, but needed for signature)
    # The update_fn expects (array_proxy, i, j, ...) where i,j are integer offsets.
    # We call it with offset=0 for all indices — the proxy handles the shifting.
    idx_args = [0] * spec.ndim
    arr_args = [proxies[name] for name in spec.inputs]

    result = spec.update_fn(*arr_args, *idx_args)

    out = u.copy()
    interior = tuple(slice(h, s - h) for h, s in zip(halo, u.shape))
    out[interior] = result
    return out


def time_march(
    u0: np.ndarray,
    spec: StencilSpec,
    steps: int,
    boundary_fn=None,
) -> List[np.ndarray]:
    """Run multiple time steps of a stencil, returning snapshots.

    Parameters
    ----------
    u0 : initial condition (full array with boundaries)
    spec : stencil specification
    steps : number of time steps
    boundary_fn : optional callable(u, step) to reset boundary conditions
    """
    history = [u0.copy()]
    u = u0.copy()
    for s in range(steps):
        u = apply_stencil(u, spec)
        if boundary_fn is not None:
            boundary_fn(u, s + 1)
        history.append(u.copy())
    return history
