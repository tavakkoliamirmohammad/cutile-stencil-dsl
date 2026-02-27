"""Sparse matrix formats: DIA and BSR, with Laplacian constructors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class DIAMatrix:
    """Diagonal (DIA) sparse matrix format.

    data[k, :] stores the k-th diagonal, offset given by offsets[k].
    Offset 0 = main diagonal, positive = superdiagonal, negative = subdiagonal.
    """
    data: np.ndarray       # shape (num_diags, N)
    offsets: np.ndarray    # shape (num_diags,) — integer offsets
    shape: Tuple[int, int]

    def to_dense(self) -> np.ndarray:
        M, N = self.shape
        A = np.zeros((M, N), dtype=self.data.dtype)
        for k, off in enumerate(self.offsets):
            diag = self.data[k]
            if off >= 0:
                length = min(M, N - off)
                for idx in range(length):
                    A[idx, idx + off] = diag[idx]
            else:
                length = min(M + off, N)
                for idx in range(length):
                    A[idx - off, idx] = diag[idx - off]
        return A


@dataclass
class BSRMatrix:
    """Block Sparse Row (BSR) matrix format."""
    data: np.ndarray       # shape (nnz_blocks, block_r, block_c)
    indices: np.ndarray    # column block indices
    indptr: np.ndarray     # row block pointers
    blocksize: Tuple[int, int]
    shape: Tuple[int, int]

    def to_dense(self) -> np.ndarray:
        M, N = self.shape
        br, bc = self.blocksize
        A = np.zeros((M, N), dtype=self.data.dtype)
        n_block_rows = len(self.indptr) - 1
        for bi in range(n_block_rows):
            for idx in range(self.indptr[bi], self.indptr[bi + 1]):
                bj = self.indices[idx]
                A[bi * br:(bi + 1) * br, bj * bc:(bj + 1) * bc] = self.data[idx]
        return A


# ── Laplacian constructors ──────────────────────────────────────────

def laplacian_1d_dia(N: int, dtype=np.float64) -> DIAMatrix:
    """1D Laplacian: tridiagonal [-1, 2, -1]."""
    data = np.zeros((3, N), dtype=dtype)
    data[0, :] = -1.0   # sub-diagonal
    data[1, :] = 2.0    # main diagonal
    data[2, :] = -1.0   # super-diagonal
    offsets = np.array([-1, 0, 1], dtype=np.int32)
    return DIAMatrix(data=data, offsets=offsets, shape=(N, N))


def laplacian_2d_dia(Nx: int, Ny: int, dtype=np.float64) -> DIAMatrix:
    """2D Laplacian on Nx×Ny grid: 5-point stencil [0,-1,-1,4,-1,-1,0]."""
    N = Nx * Ny
    data = np.zeros((5, N), dtype=dtype)
    offsets_list = [-Ny, -1, 0, 1, Ny]
    # Main diagonal
    data[2, :] = 4.0
    # ±1 diagonals (x-neighbors)
    data[1, :] = -1.0
    data[3, :] = -1.0
    # ±Ny diagonals (y-neighbors)
    data[0, :] = -1.0
    data[4, :] = -1.0
    offsets = np.array(offsets_list, dtype=np.int32)
    return DIAMatrix(data=data, offsets=offsets, shape=(N, N))


def laplacian_3d_dia(Nx: int, Ny: int, Nz: int, dtype=np.float64) -> DIAMatrix:
    """3D Laplacian on Nx×Ny×Nz grid: 7-point stencil."""
    N = Nx * Ny * Nz
    Nyz = Ny * Nz
    offsets_list = [-Nyz, -Nz, -1, 0, 1, Nz, Nyz]
    data = np.zeros((7, N), dtype=dtype)
    data[3, :] = 6.0    # main diagonal
    for k in [0, 1, 2, 4, 5, 6]:
        data[k, :] = -1.0
    offsets = np.array(offsets_list, dtype=np.int32)
    return DIAMatrix(data=data, offsets=offsets, shape=(N, N))
