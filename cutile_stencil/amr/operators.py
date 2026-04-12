"""AMR prolongation and restriction operators.

Provides code generators for 1D, 2D, and 3D adaptive mesh refinement
interpolation (prolongation) and coarsening (restriction) kernels.
"""

from __future__ import annotations

from cutile_stencil.codegen.emitter import CodeEmitter


def generate_prolongation_kernel(ndim: int, refinement_factor: int = 2) -> str:
    """Generate a Python prolongation kernel for the given number of dimensions.

    Prolongation interpolates from a coarse grid to a fine grid.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions (1, 2, or 3).
    refinement_factor : int
        Refinement ratio between fine and coarse grids (default: 2).

    Returns
    -------
    str
        Python source code implementing the prolongation kernel.

    Raises
    ------
    NotImplementedError
        If ndim is not 1, 2, or 3.
    ValueError
        If ndim < 1 or refinement_factor < 1.
    """
    if ndim < 1:
        raise ValueError(f"ndim must be >= 1, got {ndim}")
    if refinement_factor < 1:
        raise ValueError(f"refinement_factor must be >= 1, got {refinement_factor}")

    e = CodeEmitter()
    r = refinement_factor

    if ndim == 1:
        _emit_prolongation_1d(e, r)
    elif ndim == 2:
        _emit_prolongation_2d(e, r)
    elif ndim == 3:
        _emit_prolongation_3d(e, r)
    else:
        raise NotImplementedError(
            f"Prolongation not implemented for ndim={ndim}. Supported: 1, 2, 3."
        )

    return e.render()


def generate_restriction_kernel(ndim: int, refinement_factor: int = 2) -> str:
    """Generate a Python restriction kernel for the given number of dimensions.

    Restriction coarsens from a fine grid to a coarse grid using volume-weighted
    averaging.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions (1, 2, or 3).
    refinement_factor : int
        Refinement ratio between fine and coarse grids (default: 2).

    Returns
    -------
    str
        Python source code implementing the restriction kernel.

    Raises
    ------
    NotImplementedError
        If ndim is not 1, 2, or 3.
    ValueError
        If ndim < 1 or refinement_factor < 1.
    """
    if ndim < 1:
        raise ValueError(f"ndim must be >= 1, got {ndim}")
    if refinement_factor < 1:
        raise ValueError(f"refinement_factor must be >= 1, got {refinement_factor}")

    e = CodeEmitter()
    r = refinement_factor

    if ndim == 1:
        _emit_restriction_1d(e, r)
    elif ndim == 2:
        _emit_restriction_2d(e, r)
    elif ndim == 3:
        _emit_restriction_3d(e, r)
    else:
        raise NotImplementedError(
            f"Restriction not implemented for ndim={ndim}. Supported: 1, 2, 3."
        )

    return e.render()


# ---------------------------------------------------------------------------
# 1D kernels
# ---------------------------------------------------------------------------

def _emit_prolongation_1d(e: CodeEmitter, r: int) -> None:
    """1D linear interpolation from coarse to fine."""
    e.line('"""1D prolongation: linear interpolation from coarse to fine grid."""')
    e.blank()
    e.line("import numpy as np")
    e.blank()
    e.blank()
    e.line("def prolongation_1d(u_coarse, u_fine):")
    with e.indent():
        e.line(f'"""Linear prolongation with refinement factor {r}."""')
        e.line("n_coarse = u_coarse.shape[0]")
        e.line("for i in range(n_coarse - 1):")
        with e.indent():
            e.line(f"for di in range({r}):")
            with e.indent():
                e.line(f"fi = i * {r} + di")
                e.line(f"ai = di / {r}")
                e.line("u_fine[fi] = (1 - ai) * u_coarse[i] + ai * u_coarse[i + 1]")
        e.line(f"# Fill last point")
        e.line(f"u_fine[(n_coarse - 1) * {r}] = u_coarse[n_coarse - 1]")


def _emit_restriction_1d(e: CodeEmitter, r: int) -> None:
    """1D volume-weighted restriction from fine to coarse."""
    e.line('"""1D restriction: volume-weighted averaging from fine to coarse grid."""')
    e.blank()
    e.line("import numpy as np")
    e.blank()
    e.blank()
    e.line("def restriction_1d(u_fine, u_coarse):")
    with e.indent():
        e.line(f'"""Volume-weighted restriction with refinement factor {r}."""')
        e.line("n_coarse = u_coarse.shape[0]")
        e.line("for i in range(n_coarse):")
        with e.indent():
            e.line(f"u_coarse[i] = u_fine[i * {r}:(i + 1) * {r}].mean()")


# ---------------------------------------------------------------------------
# 2D kernels
# ---------------------------------------------------------------------------

def _emit_prolongation_2d(e: CodeEmitter, r: int) -> None:
    """2D bilinear interpolation from coarse to fine."""
    e.line('"""2D prolongation: bilinear interpolation from coarse to fine grid."""')
    e.blank()
    e.line("import numpy as np")
    e.blank()
    e.blank()
    e.line("def prolongation_2d(u_coarse, u_fine):")
    with e.indent():
        e.line(f'"""Bilinear prolongation with refinement factor {r}."""')
        e.line("n0, n1 = u_coarse.shape")
        e.line("for i in range(n0 - 1):")
        with e.indent():
            e.line("for j in range(n1 - 1):")
            with e.indent():
                e.line(f"for di in range({r}):")
                with e.indent():
                    e.line(f"for dj in range({r}):")
                    with e.indent():
                        e.line(f"fi = i * {r} + di")
                        e.line(f"fj = j * {r} + dj")
                        e.line(f"ai = di / {r}")
                        e.line(f"aj = dj / {r}")
                        e.line("u_fine[fi, fj] = (")
                        with e.indent():
                            e.line("(1 - ai) * (1 - aj) * u_coarse[i, j] +")
                            e.line("ai * (1 - aj) * u_coarse[i + 1, j] +")
                            e.line("(1 - ai) * aj * u_coarse[i, j + 1] +")
                            e.line("ai * aj * u_coarse[i + 1, j + 1]")
                        e.line(")")


def _emit_restriction_2d(e: CodeEmitter, r: int) -> None:
    """2D volume-weighted restriction from fine to coarse."""
    e.line('"""2D restriction: volume-weighted averaging from fine to coarse grid."""')
    e.blank()
    e.line("import numpy as np")
    e.blank()
    e.blank()
    e.line("def restriction_2d(u_fine, u_coarse):")
    with e.indent():
        e.line(f'"""Volume-weighted restriction with refinement factor {r}."""')
        e.line("n0, n1 = u_coarse.shape")
        e.line("for i in range(n0):")
        with e.indent():
            e.line("for j in range(n1):")
            with e.indent():
                e.line(f"u_coarse[i, j] = u_fine[i * {r}:(i + 1) * {r}, j * {r}:(j + 1) * {r}].mean()")


# ---------------------------------------------------------------------------
# 3D kernels
# ---------------------------------------------------------------------------

def _emit_prolongation_3d(e: CodeEmitter, r: int) -> None:
    """3D trilinear interpolation from coarse to fine."""
    e.line('"""3D prolongation: trilinear interpolation from coarse to fine grid."""')
    e.blank()
    e.line("import numpy as np")
    e.blank()
    e.blank()
    e.line("def prolongation_3d(u_coarse, u_fine):")
    with e.indent():
        e.line(f'"""Trilinear prolongation with refinement factor {r}."""')
        e.line("n0, n1, n2 = u_coarse.shape")
        e.line("for i in range(n0 - 1):")
        with e.indent():
            e.line("for j in range(n1 - 1):")
            with e.indent():
                e.line("for k in range(n2 - 1):")
                with e.indent():
                    e.line(f"for di in range({r}):")
                    with e.indent():
                        e.line(f"for dj in range({r}):")
                        with e.indent():
                            e.line(f"for dk in range({r}):")
                            with e.indent():
                                e.line(f"fi = i * {r} + di")
                                e.line(f"fj = j * {r} + dj")
                                e.line(f"fk = k * {r} + dk")
                                e.line(f"ai = di / {r}")
                                e.line(f"aj = dj / {r}")
                                e.line(f"ak = dk / {r}")
                                e.line("u_fine[fi, fj, fk] = (")
                                with e.indent():
                                    e.line("(1 - ai) * (1 - aj) * (1 - ak) * u_coarse[i, j, k] +")
                                    e.line("ai * (1 - aj) * (1 - ak) * u_coarse[i + 1, j, k] +")
                                    e.line("(1 - ai) * aj * (1 - ak) * u_coarse[i, j + 1, k] +")
                                    e.line("(1 - ai) * (1 - aj) * ak * u_coarse[i, j, k + 1] +")
                                    e.line("ai * aj * (1 - ak) * u_coarse[i + 1, j + 1, k] +")
                                    e.line("ai * (1 - aj) * ak * u_coarse[i + 1, j, k + 1] +")
                                    e.line("(1 - ai) * aj * ak * u_coarse[i, j + 1, k + 1] +")
                                    e.line("ai * aj * ak * u_coarse[i + 1, j + 1, k + 1]")
                                e.line(")")


def _emit_restriction_3d(e: CodeEmitter, r: int) -> None:
    """3D volume-weighted restriction from fine to coarse."""
    e.line('"""3D restriction: volume-weighted averaging from fine to coarse grid."""')
    e.blank()
    e.line("import numpy as np")
    e.blank()
    e.blank()
    e.line("def restriction_3d(u_fine, u_coarse):")
    with e.indent():
        e.line(f'"""Volume-weighted restriction with refinement factor {r}."""')
        e.line("n0, n1, n2 = u_coarse.shape")
        e.line("for i in range(n0):")
        with e.indent():
            e.line("for j in range(n1):")
            with e.indent():
                e.line("for k in range(n2):")
                with e.indent():
                    e.line(
                        f"u_coarse[i, j, k] = u_fine["
                        f"i * {r}:(i + 1) * {r}, "
                        f"j * {r}:(j + 1) * {r}, "
                        f"k * {r}:(k + 1) * {r}].mean()"
                    )
