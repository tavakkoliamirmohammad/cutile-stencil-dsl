"""Optional Numba JIT backend for stencil benchmarking.

Generates @njit functions from StencilSpec for CPU baseline comparison.
Gracefully returns None if numba is not installed.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from cutile_stencil.dsl.types import StencilSpec


def _has_numba() -> bool:
    try:
        import numba
        return True
    except ImportError:
        return False


def numba_apply_stencil(spec: StencilSpec) -> Optional[Callable]:
    """Generate a Numba JIT-compiled stencil application function.

    Returns None if numba is not installed or if the stencil is too complex
    for automatic Numba compilation.

    Currently supports single-input 1D stencils only.
    """
    if not _has_numba():
        return None

    if spec.ndim != 1 or len(spec.inputs) != 1:
        # Only 1D single-input stencils for now
        return None

    import numba

    halo = spec.halo_widths
    if halo is None:
        return None

    h = halo[0]

    # Build coefficient map from accesses
    # For simple stencils, extract coefficients by evaluating
    # the update function with unit vectors
    offsets = sorted({a.offsets[0] for a in spec.accesses})

    # Evaluate coefficients using probing
    class _Probe:
        def __init__(self, target_off):
            self._target = target_off
        def __getitem__(self, key):
            if key == self._target:
                return 1.0
            return 0.0

    coeffs = {}
    for off in offsets:
        probe = _Probe(off)
        try:
            val = spec.update_fn(probe, 0)
            coeffs[off] = float(val)
        except Exception:
            return None  # Too complex for probing

    # Generate the JIT function
    offset_list = sorted(coeffs.keys())
    coeff_list = [coeffs[o] for o in offset_list]

    @numba.njit
    def _apply(u, h, offsets_arr, coeffs_arr):
        N = u.shape[0]
        out = u.copy()
        for i in range(h, N - h):
            val = 0.0
            for k in range(len(offsets_arr)):
                val += coeffs_arr[k] * u[i + offsets_arr[k]]
            out[i] = val
        return out

    offsets_arr = np.array(offset_list, dtype=np.int64)
    coeffs_arr = np.array(coeff_list, dtype=np.float64)

    def apply_fn(u):
        return _apply(u, h, offsets_arr, coeffs_arr)

    return apply_fn
