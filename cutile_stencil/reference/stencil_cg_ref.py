"""NumPy reference for matrix-free CG using stencil operators.

Uses ``apply_stencil()`` as the operator instead of a DIA/BSR SpMV.
Works on full arrays (with halo cells) so that the stencil can read
neighbor values from the halo region.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from cutile_stencil.dsl.types import StencilSpec
from cutile_stencil.reference.stencil_ref import apply_stencil


def stencil_cg_solve(
    spec: StencilSpec,
    b_full: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, int, List[float]]:
    """Solve Ax = b with CG where A is a stencil operator.

    Parameters
    ----------
    spec : StencilSpec
        Stencil specification (with halo_widths set).
    b_full : np.ndarray
        Right-hand side, full array including halo cells.
    x0 : np.ndarray, optional
        Initial guess (full array with halo). Defaults to zeros.
    tol : float
        Convergence tolerance on ``||r||``.
    max_iter : int
        Maximum CG iterations.

    Returns
    -------
    x : np.ndarray
        Solution (full array with halo).
    iters : int
        Number of iterations performed.
    residuals : list of float
        ``||r||`` at each iteration.
    """
    halo = spec.halo_widths
    if halo is None:
        halo = (spec.order // 2,) * spec.ndim

    interior = tuple(slice(h, s - h) for h, s in zip(halo, b_full.shape))

    x = x0.copy() if x0 is not None else np.zeros_like(b_full)

    # r = b - A @ x  (only on interior)
    Ax = apply_stencil(x, spec)
    r = np.zeros_like(b_full)
    r[interior] = b_full[interior] - Ax[interior]

    p = r.copy()
    r_dot_r = float(np.dot(r[interior].ravel(), r[interior].ravel()))
    residuals: List[float] = [math.sqrt(r_dot_r)]

    for it in range(1, max_iter + 1):
        if residuals[-1] < tol:
            return x, it - 1, residuals

        # Ap = A @ p
        Ap = apply_stencil(p, spec)

        # p_dot_Ap on interior only
        p_dot_Ap = float(np.dot(p[interior].ravel(), Ap[interior].ravel()))
        alpha = r_dot_r / p_dot_Ap

        # Update on interior only
        x[interior] = x[interior] + alpha * p[interior]
        r[interior] = r[interior] - alpha * Ap[interior]

        new_r_dot_r = float(np.dot(r[interior].ravel(), r[interior].ravel()))
        residuals.append(math.sqrt(new_r_dot_r))

        beta = new_r_dot_r / r_dot_r
        p[interior] = r[interior] + beta * p[interior]
        r_dot_r = new_r_dot_r

    return x, max_iter, residuals
