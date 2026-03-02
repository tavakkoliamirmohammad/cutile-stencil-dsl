"""Matrix-free Conjugate Gradient solver using stencil operators.

Generates a complete CG solver where the operator ``A @ p`` is a compiled
stencil kernel rather than a DIA SpMV.  This enables the full stencil
pipeline (analysis, tiling, temporal blocking) for the solver's dominant
kernel.
"""

from __future__ import annotations

from typing import Tuple

from cutile_stencil.codegen.emitter import CodeEmitter
from cutile_stencil.config import SolverConfig, DEFAULT_SOLVER
from cutile_stencil.dsl.types import StencilSpec, HardwareSpec
from cutile_stencil.solvers.blas_kernels import emit_dot_kernel, emit_axpy_kernel
from cutile_stencil.solvers.solver_base import emit_cg_loop


def generate_stencil_cg(
    spec: StencilSpec,
    domain: Tuple[int, ...],
    solver_config: SolverConfig | None = None,
    hw: HardwareSpec | None = None,
) -> str:
    """Generate a complete matrix-free CG solver with a stencil operator.

    The generated code contains:

    1. The compiled stencil kernel (from ``compile(spec, domain, hw)``)
    2. BLAS kernels (dot, axpy)
    3. A CG driver that calls ``launch_{spec.name}`` as the operator

    Parameters
    ----------
    spec : StencilSpec
        Stencil specification for the operator (e.g. Laplacian).
    domain : tuple of int
        Interior domain size per dimension.
    solver_config : SolverConfig, optional
        Solver configuration.
    hw : HardwareSpec, optional
        Hardware parameters for tiling analysis.

    Returns
    -------
    str
        Complete Python source for the matrix-free CG solver.
    """
    from cutile_stencil.dsl.pipeline import compile as stencil_compile

    cfg = solver_config or DEFAULT_SOLVER
    result = stencil_compile(spec, domain, hw)
    launcher_name = f"launch_{spec.name}"
    operator_call = f"{launcher_name}(p, Ap)"

    e = CodeEmitter()
    e.line('"""Matrix-free CG solver with stencil operator (auto-generated).')
    e.line("")
    e.line(f"Operator: {spec.name} (ndim={spec.ndim}, order={spec.order})")
    e.line(f"Domain: {domain}")
    e.line('"""')
    e.blank()
    e.line("import cuda.tile as ct")
    e.line("import cupy as cp")
    e.line("import math")
    e.blank()
    e.line("ConstInt = ct.Constant[int]")
    e.blank()

    # Embed the compiled stencil kernel (strip its import header)
    e.line("# ── Stencil operator kernel ───────────────────────────")
    stencil_code = result.code
    for line in stencil_code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        if stripped.startswith("ConstInt"):
            continue
        if stripped.startswith('"""') and "Auto-generated" in stripped:
            continue
        e.line(line)
    e.blank()

    # BLAS kernels
    e.line("# ── BLAS kernels ──────────────────────────────────────")
    e.blank()
    emit_dot_kernel(e, spec.dtype)
    e.blank()
    e.blank()
    emit_axpy_kernel(e, spec.dtype)
    e.blank()

    # CG driver using template
    e.blank()
    halo = spec.halo_widths

    e.line(f"def stencil_cg_solve(b, x0, tol={cfg.cg_tol}, max_iter={cfg.cg_max_iter}, tile_size={cfg.tile_size}):")
    with e.indent():
        e.line(f'"""Solve Ax = b with CG where A is the {spec.name} stencil.')
        e.line("")
        e.line("b and x0 are full arrays including halo/boundary cells.")
        e.line(f"Halo widths: {halo}")
        e.line('"""')
        e.line("N = b.shape[0]")
        e.line("x = x0.copy()")
        e.blank()

        # Compute initial residual
        e.line("# r = b - A @ x")
        e.line("Ax = cp.zeros_like(x)")
        e.line(f"{launcher_name}(x, Ax)")
        e.line("r = b - Ax")
        e.line("p = r.copy()")
        e.line("Ap = cp.zeros_like(x)")
        e.line("buf = cp.zeros(1, dtype=b.dtype)")
        e.blank()

        e.line("stream = cp.cuda.get_current_stream()")
        e.line("grid = (ct.cdiv(N, tile_size),)")
        e.blank()

        e.line("# r_dot_r = r^T r")
        e.line("buf.fill(0)")
        e.line("ct.launch(stream, grid, dot_kernel, (r, r, buf, tile_size))")
        e.line("r_dot_r = buf.item()")
        e.blank()

        emit_cg_loop(e, operator_call, cfg, spec.dtype)

    return e.render()


def compile_stencil_cg(
    spec: StencilSpec,
    domain: Tuple[int, ...],
    solver_config: SolverConfig | None = None,
    hw: HardwareSpec | None = None,
) -> "StencilCGCompileResult":
    """Compile a matrix-free stencil CG solver, returning a validatable result."""
    import ast
    from cutile_stencil.solvers.solver_compile import StencilCGCompileResult

    cfg = solver_config or DEFAULT_SOLVER
    code = generate_stencil_cg(spec, domain, cfg, hw)
    ast.parse(code)

    return StencilCGCompileResult(
        solver_type="StencilCG",
        code=code,
        config=cfg,
        dtype=spec.dtype,
        spec=spec,
        domain=domain,
    )
