"""Standard multi-launch Conjugate Gradient driver (cuTile code).

Generates a complete CG solver that launches separate kernels per
operation (SpMV, axpy, dot, norm) — 7 launches per iteration.
"""

from __future__ import annotations

from cutile_stencil.codegen.emitter import CodeEmitter


def generate_cg_driver() -> str:
    """Generate cuTile CG driver Python file."""
    e = CodeEmitter()
    e.line('"""Standard CG solver using cuTile kernels (auto-generated).')
    e.line("")
    e.line("Launches separate kernels per operation: SpMV, axpy, dot, norm.")
    e.line('"""')
    e.blank()
    e.line("import cuda.tile as ct")
    e.line("import torch")
    e.line("import math")
    e.blank()
    e.line("ConstInt = ct.Constant[int]")
    e.blank()

    # Inline kernels
    e.line("# ── Inline solver kernels ─────────────────────────────")
    e.blank()
    _emit_inline_kernels(e)

    # CG driver
    e.blank()
    e.line("def cg_solve(spmv_fn, b, x0, tol=1e-10, max_iter=1000, tile_size=256):")
    with e.indent():
        e.line('"""Solve Ax = b with CG. spmv_fn(x, out) computes out = A @ x."""')
        e.line("N = b.shape[0]")
        e.line("x = x0.clone()")
        e.line("r = torch.zeros_like(b)")
        e.line("spmv_fn(x, r)       # r = A @ x")
        e.line("r = b - r            # r = b - A @ x")
        e.line("p = r.clone()")
        e.line("Ap = torch.zeros_like(b)")
        e.line("buf = torch.zeros(1, dtype=b.dtype, device=b.device)")
        e.blank()
        e.line("stream = torch.cuda.current_stream()")
        e.line("grid = (ct.cdiv(N, tile_size),)")
        e.blank()
        e.line("# r_dot_r = r^T r")
        e.line("buf.zero_()")
        e.line("ct.launch(stream, grid, dot_kernel, (r, r, buf, tile_size))")
        e.line("r_dot_r = buf.item()")
        e.blank()
        e.line("for it in range(max_iter):")
        with e.indent():
            e.line("if math.sqrt(r_dot_r) < tol:")
            with e.indent():
                e.line("return x, it, math.sqrt(r_dot_r)")
            e.blank()
            e.line("# Ap = A @ p")
            e.line("spmv_fn(p, Ap)")
            e.blank()
            e.line("# p_dot_Ap = p^T Ap")
            e.line("buf.zero_()")
            e.line("ct.launch(stream, grid, dot_kernel, (p, Ap, buf, tile_size))")
            e.line("p_dot_Ap = buf.item()")
            e.blank()
            e.line("alpha = r_dot_r / p_dot_Ap")
            e.blank()
            e.line("# x = x + alpha * p")
            e.line("ct.launch(stream, grid, axpy_kernel, (p, x, alpha, tile_size))")
            e.blank()
            e.line("# r = r - alpha * Ap")
            e.line("ct.launch(stream, grid, axpy_kernel, (Ap, r, -alpha, tile_size))")
            e.blank()
            e.line("# new_r_dot_r = r^T r")
            e.line("buf.zero_()")
            e.line("ct.launch(stream, grid, dot_kernel, (r, r, buf, tile_size))")
            e.line("new_r_dot_r = buf.item()")
            e.blank()
            e.line("beta = new_r_dot_r / r_dot_r")
            e.blank()
            e.line("# p = r + beta * p")
            e.line("p = r + beta * p")
            e.blank()
            e.line("r_dot_r = new_r_dot_r")
        e.blank()
        e.line("return x, max_iter, math.sqrt(r_dot_r)")
    return e.render()


def _emit_inline_kernels(e: CodeEmitter):
    e.line("@ct.kernel")
    e.line("def dot_kernel(a, b, result, TILE: ConstInt):")
    with e.indent():
        e.line("pid = ct.bid(0)")
        e.line("ta = ct.load(a, index=(pid,), shape=(TILE,))")
        e.line("tb = ct.load(b, index=(pid,), shape=(TILE,))")
        e.line("partial = ct.sum(ta * tb)")
        e.line("ct.atomic_add(result, (0,), partial)")
    e.blank()
    e.blank()
    e.line("@ct.kernel")
    e.line("def axpy_kernel(x, y, alpha, TILE: ConstInt):")
    with e.indent():
        e.line("pid = ct.bid(0)")
        e.line("tx = ct.load(x, index=(pid,), shape=(TILE,))")
        e.line("ty = ct.load(y, index=(pid,), shape=(TILE,))")
        e.line("result = alpha * tx + ty")
        e.line("ct.store(y, index=(pid,), tile=result)")
