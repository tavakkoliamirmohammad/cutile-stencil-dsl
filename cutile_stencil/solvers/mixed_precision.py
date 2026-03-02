"""Mixed-precision iterative refinement (cuTile code generation).

Outer loop in FP64 for residual computation, inner CG loop in FP32/FP16
using ct.astype casts. This reduces memory bandwidth for the inner solve
while maintaining FP64 accuracy in the outer residual.
"""

from __future__ import annotations

from cutile_stencil.codegen.emitter import CodeEmitter
from cutile_stencil.config import SolverConfig, DEFAULT_SOLVER


def generate_mixed_precision_cg(solver_config: SolverConfig | None = None) -> str:
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
    e.line("@ct.kernel")
    e.line("def cast_kernel(src, dst, TILE: ConstInt):")
    with e.indent():
        e.line("pid = ct.bid(0)")
        e.line("t = ct.load(src, index=(pid,), shape=(TILE,))")
        e.line("ct.store(dst, index=(pid,), tile=ct.astype(t, dst.dtype))")
    e.blank()

    # Inline dot & axpy
    e.line("@ct.kernel")
    e.line("def dot_kernel(a, b, result, TILE: ConstInt):")
    with e.indent():
        e.line("pid = ct.bid(0)")
        e.line("ta = ct.load(a, index=(pid,), shape=(TILE,))")
        e.line("tb = ct.load(b, index=(pid,), shape=(TILE,))")
        e.line("ct.atomic_add(result, (0,), ct.sum(ta * tb))")
    e.blank()

    e.line("@ct.kernel")
    e.line("def axpy_kernel(x, y, alpha, TILE: ConstInt):")
    with e.indent():
        e.line("pid = ct.bid(0)")
        e.line("tx = ct.load(x, index=(pid,), shape=(TILE,))")
        e.line("ty = ct.load(y, index=(pid,), shape=(TILE,))")
        e.line("ct.store(y, index=(pid,), tile=alpha * tx + ty)")
    e.blank()

    # Inner CG in low precision
    e.blank()
    e.line("def inner_cg(spmv_fn, rhs_lp, x_lp, tol, max_inner, tile_size=256):")
    with e.indent():
        e.line('"""Inner CG loop in low precision (FP32 or FP16)."""')
        e.line("N = rhs_lp.shape[0]")
        e.line("r = rhs_lp.clone()")
        e.line("Ax = cp.zeros_like(r)")
        e.line("spmv_fn(x_lp, Ax)")
        e.line("r = rhs_lp - Ax")
        e.line("p = r.clone()")
        e.line("Ap = cp.zeros_like(r)")
        e.line("buf = cp.zeros(1, dtype=r.dtype)")
        e.line("stream = cp.cuda.get_current_stream()")
        e.line("grid = (ct.cdiv(N, tile_size),)")
        e.blank()
        e.line("buf.zero_()")
        e.line("ct.launch(stream, grid, dot_kernel, (r, r, buf, tile_size))")
        e.line("r_dot_r = buf.item()")
        e.blank()
        e.line("for it in range(max_inner):")
        with e.indent():
            e.line("if math.sqrt(abs(r_dot_r)) < tol:")
            with e.indent():
                e.line("break")
            e.line("spmv_fn(p, Ap)")
            e.line("buf.zero_()")
            e.line("ct.launch(stream, grid, dot_kernel, (p, Ap, buf, tile_size))")
            e.line("p_dot_Ap = buf.item()")
            e.line("alpha = r_dot_r / p_dot_Ap")
            e.line("ct.launch(stream, grid, axpy_kernel, (p, x_lp, alpha, tile_size))")
            e.line("ct.launch(stream, grid, axpy_kernel, (Ap, r, -alpha, tile_size))")
            e.line("buf.zero_()")
            e.line("ct.launch(stream, grid, dot_kernel, (r, r, buf, tile_size))")
            e.line("new_rr = buf.item()")
            e.line("beta = new_rr / r_dot_r")
            e.line("p = r + beta * p")
            e.line("r_dot_r = new_rr")
        e.line("return x_lp")

    # Outer loop
    e.blank()
    e.blank()
    e.line("def mixed_precision_cg(spmv_fn_hp, spmv_fn_lp, b, x0,")
    e.line("                       inner_dtype=cp.float32,")
    e.line(f"                       tol={cfg.mixed_outer_tol}, max_outer={cfg.mixed_max_outer}, max_inner={cfg.mixed_max_inner},")
    e.line(f"                       tile_size={cfg.tile_size}):")
    with e.indent():
        e.line('"""Mixed-precision iterative refinement CG.')
        e.line("")
        e.line("spmv_fn_hp: A @ x in high precision (FP64)")
        e.line("spmv_fn_lp: A @ x in low precision (FP32/FP16)")
        e.line('"""')
        e.line("N = b.shape[0]")
        e.line("x = x0.clone()  # FP64")
        e.line("stream = cp.cuda.get_current_stream()")
        e.line("grid = (ct.cdiv(N, tile_size),)")
        e.blank()
        e.line("for outer in range(max_outer):")
        with e.indent():
            e.line("# Compute residual in FP64: r = b - A @ x")
            e.line("Ax = cp.zeros_like(b)")
            e.line("spmv_fn_hp(x, Ax)")
            e.line("r = b - Ax")
            e.blank()
            e.line("# Check convergence")
            e.line("buf = cp.zeros(1, dtype=b.dtype)")
            e.line("ct.launch(stream, grid, dot_kernel, (r, r, buf, tile_size))")
            e.line("rnorm = math.sqrt(buf.item())")
            e.line("if rnorm < tol:")
            with e.indent():
                e.line("return x, outer, rnorm")
            e.blank()
            e.line("# Cast residual to low precision")
            e.line("r_lp = r.astype(inner_dtype)")
            e.line("d_lp = cp.zeros(N, dtype=inner_dtype)")
            e.blank()
            e.line("# Inner CG solve in low precision: A @ d ≈ r")
            e.line(f"d_lp = inner_cg(spmv_fn_lp, r_lp, d_lp, tol={cfg.mixed_inner_tol}, max_inner=max_inner, tile_size=tile_size)")
            e.blank()
            e.line("# Update solution in FP64: x += d (cast back)")
            e.line("x = x + d_lp.astype(x.dtype)")
        e.blank()
        e.line("return x, max_outer, rnorm")
    return e.render()
