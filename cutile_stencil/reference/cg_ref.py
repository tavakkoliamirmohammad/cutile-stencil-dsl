"""NumPy reference CG solver and mixed-precision CG."""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import numpy as np


def cg_solve(
    A_fn: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, int, List[float]]:
    """Standard Conjugate Gradient solver in NumPy.

    Parameters
    ----------
    A_fn : callable that computes A @ x
    b : right-hand side vector
    x0 : initial guess (zero if None)
    tol : convergence tolerance on ||r||
    max_iter : maximum iterations

    Returns
    -------
    x : solution vector
    iters : number of iterations performed
    residuals : list of ||r|| at each iteration
    """
    n = b.shape[0]
    x = x0.copy() if x0 is not None else np.zeros_like(b)
    r = b - A_fn(x)
    p = r.copy()
    r_dot_r = float(np.dot(r, r))
    residuals: List[float] = [math.sqrt(r_dot_r)]

    for it in range(1, max_iter + 1):
        if residuals[-1] < tol:
            return x, it - 1, residuals

        Ap = A_fn(p)
        p_dot_Ap = float(np.dot(p, Ap))
        alpha = r_dot_r / p_dot_Ap

        x = x + alpha * p
        r = r - alpha * Ap

        new_r_dot_r = float(np.dot(r, r))
        residuals.append(math.sqrt(new_r_dot_r))

        beta = new_r_dot_r / r_dot_r
        p = r + beta * p
        r_dot_r = new_r_dot_r

    return x, max_iter, residuals


def mixed_precision_cg(
    A_fn: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    inner_dtype=np.float32,
    tol: float = 1e-10,
    max_outer: int = 20,
    max_inner: int = 200,
    inner_tol: float = 1e-4,
) -> Tuple[np.ndarray, int, List[float]]:
    """Mixed-precision iterative refinement CG.

    Outer loop computes residual in FP64, inner CG solves in lower precision.

    Returns
    -------
    x : solution vector (FP64)
    outer_iters : number of outer refinement steps
    residuals : list of ||r|| at each outer step
    """
    n = b.shape[0]
    x = x0.copy() if x0 is not None else np.zeros_like(b)
    residuals: List[float] = []

    def A_fn_lp(v):
        return A_fn(v.astype(np.float64)).astype(inner_dtype)

    for outer in range(max_outer):
        # Residual in FP64
        r = b - A_fn(x)
        rnorm = float(np.linalg.norm(r))
        residuals.append(rnorm)

        if rnorm < tol:
            return x, outer, residuals

        # Inner solve in low precision
        r_lp = r.astype(inner_dtype)
        d_lp = np.zeros(n, dtype=inner_dtype)
        d_lp, _, _ = cg_solve(A_fn_lp, r_lp, d_lp, tol=inner_tol, max_iter=max_inner)

        # Update in FP64
        x = x + d_lp.astype(np.float64)

    r = b - A_fn(x)
    residuals.append(float(np.linalg.norm(r)))
    return x, max_outer, residuals
