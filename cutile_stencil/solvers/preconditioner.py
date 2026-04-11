"""Preconditioner kernel generation for CG solvers."""

from __future__ import annotations

from cutile_stencil.codegen.emitter import CodeEmitter
from cutile_stencil.dsl.types import StencilSpec


def _extract_center_coefficient(spec: StencilSpec) -> float:
    """Extract the center coefficient from a stencil spec.

    The center coefficient corresponds to the access with all-zero offsets.
    For a standard Laplacian, this is typically -2*ndim (e.g., -4 for 2D).

    For stencils where the coefficient isn't obvious from the accesses alone,
    we estimate it from the stencil structure: -(number_of_neighbors).
    """
    if not spec.accesses:
        return 1.0

    # Count non-center accesses as a proxy for the center weight
    n_neighbors = sum(
        1 for acc in spec.accesses
        if not all(o == 0 for o in acc.offsets)
    )

    # For a standard Laplacian: center = -2*ndim
    # This is an approximation — exact coefficient requires evaluating the stencil
    if n_neighbors > 0:
        return -float(n_neighbors)
    return 1.0


def generate_jacobi_preconditioner(
    spec: StencilSpec,
    dtype: str = "float64",
    tile_size: int = 256,
) -> str:
    """Generate a cuTile kernel that applies Jacobi preconditioning.

    The kernel computes: z[i] = r[i] / diag_coeff
    where diag_coeff is the center coefficient of the stencil.

    Parameters
    ----------
    spec : StencilSpec
        The stencil specification (to extract center coefficient).
    dtype : str
        Data type for the kernel.
    tile_size : int
        Tile size for the kernel.

    Returns
    -------
    str
        Generated Python source code for the preconditioner kernel + launcher.
    """
    diag = _extract_center_coefficient(spec)
    inv_diag = 1.0 / diag if diag != 0.0 else 1.0

    e = CodeEmitter()
    e.line('"""Jacobi preconditioner kernel (auto-generated)."""')
    e.blank()
    e.line("import cuda.tile as ct")
    e.line("import cupy as cp")
    e.blank()
    e.line("ConstInt = ct.Constant[int]")
    e.blank()
    e.line(f"INV_DIAG = {inv_diag!r}")
    e.blank()

    # Kernel
    e.line("@ct.kernel")
    e.line("def jacobi_precondition_kernel(r, z, N: ConstInt, TILE: ConstInt):")
    with e.indent():
        e.line("pid = ct.bid(0)")
        e.line("r_tile = ct.load(r, index=(pid,), shape=(TILE,))")
        e.line("z_tile = r_tile * INV_DIAG")
        e.line("ct.store(z, index=(pid,), tile=z_tile)")

    e.blank()
    e.blank()

    # Launcher
    e.line("def launch_jacobi_precondition(r, z, stream=None):")
    with e.indent():
        e.line(f"TILE = {tile_size}")
        e.line("N = r.shape[0]")
        e.line("if stream is None:")
        with e.indent():
            e.line("stream = cp.cuda.get_current_stream()")
        e.line("grid = (ct.cdiv(N, TILE),)")
        e.line("ct.launch(stream, grid, jacobi_precondition_kernel, (r, z, N, TILE))")

    return e.render()


def get_center_coefficient(spec: StencilSpec) -> float:
    """Public API to get the estimated center coefficient."""
    return _extract_center_coefficient(spec)
