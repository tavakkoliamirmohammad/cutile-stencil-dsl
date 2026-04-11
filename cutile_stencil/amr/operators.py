"""Inter-level transfer operators: prolongation (coarse->fine) and restriction (fine->coarse)."""

from __future__ import annotations

import numpy as np
from typing import Tuple

from cutile_stencil.codegen.emitter import CodeEmitter


def prolongation_weights_1d(ratio: int = 2) -> np.ndarray:
    """Linear interpolation weights for 1D prolongation.

    For ratio=2: each coarse cell maps to 2 fine cells.
    Fine cell at left: weight 0.75 from coarse, 0.25 from neighbor
    Fine cell at right: weight 0.25 from coarse, 0.75 from neighbor

    Returns array of shape (ratio,) with weights for the parent cell.
    """
    weights = np.zeros(ratio)
    for i in range(ratio):
        # Linear interpolation: center weight decreases as we move away
        weights[i] = 1.0 - (i + 0.5) / ratio
    return weights


def restriction_weights_1d(ratio: int = 2) -> np.ndarray:
    """Volume-weighted averaging for 1D restriction.

    For ratio=2: each fine pair averages to one coarse cell.
    Simple averaging: weight = 1/ratio for each fine cell.
    """
    return np.ones(ratio) / ratio


def generate_prolongation_kernel(
    ndim: int,
    ratio: int = 2,
    dtype: str = "float64",
    tile_size: int = 256,
) -> str:
    """Generate a cuTile kernel for coarse-to-fine prolongation.

    Uses bilinear (2D) or trilinear (3D) interpolation.
    For 1D: linear interpolation.
    """
    e = CodeEmitter()
    e.line('"""Prolongation kernel: coarse -> fine (auto-generated)."""')
    e.blank()
    e.line("import cuda.tile as ct")
    e.line("import cupy as cp")
    e.blank()
    e.line("ConstInt = ct.Constant[int]")
    e.blank()

    # For simplicity, generate a CuPy-based prolongation (not a ct.kernel)
    # since prolongation is a simple array operation
    e.line(f"def prolongate_{ndim}d(u_coarse, u_fine):")
    with e.indent():
        e.line(f'"""Prolongate coarse array to fine array (ratio={ratio})."""')
        if ndim == 1:
            e.line(f"ratio = {ratio}")
            e.line("N_coarse = u_coarse.shape[0]")
            e.line("for i in range(N_coarse - 1):")
            with e.indent():
                e.line("for r in range(ratio):")
                with e.indent():
                    e.line("alpha = (r + 0.5) / ratio")
                    e.line("u_fine[i * ratio + r] = (1 - alpha) * u_coarse[i] + alpha * u_coarse[i + 1]")
            # Last cell: copy
            e.line("u_fine[(N_coarse - 1) * ratio:] = u_coarse[-1]")
        elif ndim == 2:
            e.line(f"ratio = {ratio}")
            e.line("Nx, Ny = u_coarse.shape")
            e.line("for i in range(Nx - 1):")
            with e.indent():
                e.line("for j in range(Ny - 1):")
                with e.indent():
                    e.line("for ri in range(ratio):")
                    with e.indent():
                        e.line("for rj in range(ratio):")
                        with e.indent():
                            e.line("ai = (ri + 0.5) / ratio")
                            e.line("aj = (rj + 0.5) / ratio")
                            e.line("fi = i * ratio + ri")
                            e.line("fj = j * ratio + rj")
                            e.line("u_fine[fi, fj] = (")
                            with e.indent():
                                e.line("(1-ai)*(1-aj) * u_coarse[i, j] +")
                                e.line("ai*(1-aj) * u_coarse[i+1, j] +")
                                e.line("(1-ai)*aj * u_coarse[i, j+1] +")
                                e.line("ai*aj * u_coarse[i+1, j+1]")
                            e.line(")")
        else:
            e.line("raise NotImplementedError('3D prolongation not yet implemented')")

    return e.render()


def generate_restriction_kernel(
    ndim: int,
    ratio: int = 2,
    dtype: str = "float64",
) -> str:
    """Generate code for fine-to-coarse restriction (volume-weighted averaging)."""
    e = CodeEmitter()
    e.line('"""Restriction kernel: fine -> coarse (auto-generated)."""')
    e.blank()
    e.line("import cupy as cp")
    e.blank()

    e.line(f"def restrict_{ndim}d(u_fine, u_coarse):")
    with e.indent():
        e.line(f'"""Restrict fine array to coarse array (ratio={ratio})."""')
        if ndim == 1:
            e.line(f"ratio = {ratio}")
            e.line("N_coarse = u_coarse.shape[0]")
            e.line("for i in range(N_coarse):")
            with e.indent():
                e.line("u_coarse[i] = u_fine[i*ratio : (i+1)*ratio].mean()")
        elif ndim == 2:
            e.line(f"ratio = {ratio}")
            e.line("Nx, Ny = u_coarse.shape")
            e.line("for i in range(Nx):")
            with e.indent():
                e.line("for j in range(Ny):")
                with e.indent():
                    e.line("u_coarse[i, j] = u_fine[i*ratio:(i+1)*ratio, j*ratio:(j+1)*ratio].mean()")
        else:
            e.line("raise NotImplementedError('3D restriction not yet implemented')")

    return e.render()
