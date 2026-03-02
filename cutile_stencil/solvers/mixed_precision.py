"""Mixed-precision iterative refinement (cuTile code generation).

Outer loop in FP64 for residual computation, inner CG loop in FP32/FP16
using ct.astype casts. This reduces memory bandwidth for the inner solve
while maintaining FP64 accuracy in the outer residual.
"""

from __future__ import annotations

from cutile_stencil.codegen.emitter import CodeEmitter
from cutile_stencil.config import SolverConfig, DEFAULT_SOLVER
from cutile_stencil.solvers.blas_kernels import (
    emit_cast_kernel,
    emit_dot_kernel,
    emit_axpy_kernel,
)
from cutile_stencil.solvers.solver_base import (
    emit_inner_cg_function,
    emit_mixed_precision_wrapper,
)


def generate_mixed_precision_cg(
    solver_config: SolverConfig | None = None,
    outer_dtype: str = "float64",
    inner_dtype: str = "float32",
) -> str:
    """Generate mixed-precision iterative refinement CG driver."""
    cfg = solver_config or DEFAULT_SOLVER
    e = CodeEmitter()
    e.line('"""Mixed-precision CG with iterative refinement (auto-generated).')
    e.line("")
    e.line("Outer FP64 residual loop + inner FP32 CG solve.")
    e.line('"""')
    e.blank()
    e.line("import cuda.tile as ct")
    e.line("import cupy as cp")
    e.line("import math")
    e.blank()
    e.line("ConstInt = ct.Constant[int]")
    e.blank()

    # Cast kernel
    emit_cast_kernel(e)
    e.blank()

    # Inline dot & axpy (inner precision)
    emit_dot_kernel(e, inner_dtype)
    e.blank()

    emit_axpy_kernel(e, inner_dtype)
    e.blank()

    # Inner CG in low precision
    e.blank()
    emit_inner_cg_function(e, "spmv_fn(p, Ap)", cfg, inner_dtype)

    # Outer loop
    e.blank()
    e.blank()
    emit_mixed_precision_wrapper(
        e,
        inner_cg_name="inner_cg",
        outer_operator_call="spmv_fn_hp(p, Ap)",
        cfg=cfg,
        outer_dtype=outer_dtype,
        inner_dtype=inner_dtype,
    )
    return e.render()


def compile_mixed_precision_cg(
    solver_config: SolverConfig | None = None,
    outer_dtype: str = "float64",
    inner_dtype: str = "float32",
) -> "MixedPrecisionCGCompileResult":
    """Compile a mixed-precision CG solver, returning a validatable result."""
    import ast
    from cutile_stencil.solvers.solver_compile import MixedPrecisionCGCompileResult
    from cutile_stencil.solvers.kernels import generate_dia_spmv

    cfg = solver_config or DEFAULT_SOLVER
    code = generate_mixed_precision_cg(cfg, outer_dtype, inner_dtype)
    ast.parse(code)

    dia_code_hp = generate_dia_spmv(outer_dtype)
    ast.parse(dia_code_hp)
    dia_code_lp = generate_dia_spmv(inner_dtype)
    ast.parse(dia_code_lp)

    return MixedPrecisionCGCompileResult(
        solver_type="MixedPrecisionCG",
        code=code,
        auxiliary_code={"dia_spmv_hp": dia_code_hp, "dia_spmv_lp": dia_code_lp},
        config=cfg,
        dtype=outer_dtype,
        outer_dtype=outer_dtype,
        inner_dtype=inner_dtype,
    )
