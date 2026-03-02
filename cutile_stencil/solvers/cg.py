"""Standard multi-launch Conjugate Gradient driver (cuTile code).

Generates a complete CG solver that launches separate kernels per
operation (SpMV, axpy, dot, norm) — 7 launches per iteration.
"""

from __future__ import annotations

from cutile_stencil.codegen.emitter import CodeEmitter
from cutile_stencil.config import SolverConfig, DEFAULT_SOLVER
from cutile_stencil.solvers.blas_kernels import emit_dot_kernel, emit_axpy_kernel
from cutile_stencil.solvers.solver_base import emit_cg_function


def generate_cg_driver(solver_config: SolverConfig | None = None, dtype: str = "float64") -> str:
    """Generate cuTile CG driver Python file."""
    cfg = solver_config or DEFAULT_SOLVER
    e = CodeEmitter()
    e.line('"""Standard CG solver using cuTile kernels (auto-generated).')
    e.line("")
    e.line("Launches separate kernels per operation: SpMV, axpy, dot, norm.")
    e.line('"""')
    e.blank()
    e.line("import cuda.tile as ct")
    e.line("import cupy as cp")
    e.line("import math")
    e.blank()
    e.line("ConstInt = ct.Constant[int]")
    e.blank()

    # Inline kernels
    e.line("# ── Inline solver kernels ─────────────────────────────")
    e.blank()
    _emit_inline_kernels(e, dtype)

    # CG driver using template
    e.blank()
    emit_cg_function(
        e,
        func_name="cg_solve",
        operator_call="spmv_fn(p, Ap)",
        cfg=cfg,
        dtype=dtype,
        extra_params="spmv_fn, ",
    )
    return e.render()


def compile_cg(
    solver_config: SolverConfig | None = None,
    dtype: str = "float64",
) -> "CGCompileResult":
    """Compile a standard CG solver, returning a validatable result.

    Generates the CG driver code and a DIA SpMV auxiliary module.
    """
    import ast
    from cutile_stencil.solvers.solver_compile import CGCompileResult
    from cutile_stencil.solvers.kernels import generate_dia_spmv

    cfg = solver_config or DEFAULT_SOLVER
    code = generate_cg_driver(cfg, dtype)
    ast.parse(code)

    dia_code = generate_dia_spmv(dtype)
    ast.parse(dia_code)

    return CGCompileResult(
        solver_type="CG",
        code=code,
        auxiliary_code={"dia_spmv": dia_code},
        config=cfg,
        dtype=dtype,
    )


def _emit_inline_kernels(e: CodeEmitter, dtype: str = "float64"):
    emit_dot_kernel(e, dtype)
    e.blank()
    e.blank()
    emit_axpy_kernel(e, dtype)
