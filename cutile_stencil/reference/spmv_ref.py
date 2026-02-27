"""NumPy reference implementations for DIA and BSR SpMV."""

from __future__ import annotations

import numpy as np

from cutile_stencil.solvers.formats import DIAMatrix, BSRMatrix


def dia_spmv(A: DIAMatrix, x: np.ndarray) -> np.ndarray:
    """Compute y = A @ x for a DIA matrix."""
    M, N = A.shape
    y = np.zeros(M, dtype=x.dtype)
    for k, off in enumerate(A.offsets):
        diag = A.data[k]
        if off >= 0:
            length = min(M, N - off)
            y[:length] += diag[:length] * x[off:off + length]
        else:
            aoff = -off
            length = min(M - aoff, N)
            y[aoff:aoff + length] += diag[aoff:aoff + length] * x[:length]
    return y


def bsr_spmv(A: BSRMatrix, x: np.ndarray) -> np.ndarray:
    """Compute y = A @ x for a BSR matrix."""
    M, N = A.shape
    br, bc = A.blocksize
    y = np.zeros(M, dtype=x.dtype)
    n_block_rows = len(A.indptr) - 1
    for bi in range(n_block_rows):
        for idx in range(A.indptr[bi], A.indptr[bi + 1]):
            bj = A.indices[idx]
            block = A.data[idx]  # (br, bc)
            y[bi * br:(bi + 1) * br] += block @ x[bj * bc:(bj + 1) * bc]
    return y
