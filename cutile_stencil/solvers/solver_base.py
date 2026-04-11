"""Reusable CG iteration templates for code generation.

Provides ``emit_cg_loop`` and ``emit_cg_function`` so that the CG
iteration body is written once and shared by ``cg.py``,
``mixed_precision.py``, and ``stencil_cg.py``.
"""

from __future__ import annotations

import re

from cutile_stencil.codegen.emitter import CodeEmitter
from cutile_stencil.config import SolverConfig, DEFAULT_SOLVER


def _replace_vars(s: str, replacements: dict) -> str:
    """Word-boundary-aware replacement for variable names in generated code."""
    for old, new in replacements.items():
        s = re.sub(rf'\b{re.escape(old)}\b', new, s)
    return s


def emit_cg_loop(
    e: CodeEmitter,
    operator_call: str,
    cfg: SolverConfig | None = None,
    dtype: str = "float64",
) -> None:
    """Emit the CG iteration body (the ``for it in range(...)`` block).

    Assumes the following names are already in scope in the generated code:
    ``x``, ``r``, ``p``, ``Ap``, ``buf``, ``stream``, ``grid``,
    ``dot_kernel``, ``axpy_kernel``, ``tile_size``, ``math``, ``ct``.

    Parameters
    ----------
    e : CodeEmitter
        The emitter to write into (at current indent level).
    operator_call : str
        Code string that computes ``Ap = A @ p``, e.g.
        ``"spmv_fn(p, Ap)"`` or ``"launch_laplacian_1d(p, Ap)"``.
    cfg : SolverConfig, optional
        Solver configuration for defaults.
    dtype : str
        Data type string.
    """
    cfg = cfg or DEFAULT_SOLVER

    e.line("for it in range(max_iter):")
    with e.indent():
        e.line("if math.sqrt(r_dot_r) < tol:")
        with e.indent():
            e.line("return x, it, math.sqrt(r_dot_r)")
        e.blank()
        e.line(f"# Ap = A @ p")
        e.line(operator_call)
        e.blank()
        e.line("# p_dot_Ap = p^T Ap")
        e.line("buf.fill(0)")
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
        e.line("buf.fill(0)")
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


def emit_cg_function(
    e: CodeEmitter,
    func_name: str,
    operator_call: str,
    cfg: SolverConfig | None = None,
    dtype: str = "float64",
    extra_params: str = "",
) -> None:
    """Emit a complete CG solver function.

    Emits the function signature, setup (initial residual, etc.), and the
    CG loop via ``emit_cg_loop``.

    Parameters
    ----------
    e : CodeEmitter
        Emitter to write into.
    func_name : str
        Name of the generated function.
    operator_call : str
        Code for the operator application (see ``emit_cg_loop``).
    cfg : SolverConfig, optional
        Solver configuration.
    dtype : str
        Data type string.
    extra_params : str
        Additional parameters to add to the function signature, e.g.
        ``"spmv_fn, "`` (note trailing comma+space).
    """
    cfg = cfg or DEFAULT_SOLVER

    e.line(f"def {func_name}({extra_params}b, x0, tol={cfg.cg_tol}, max_iter={cfg.cg_max_iter}, tile_size={cfg.tile_size}):")
    with e.indent():
        e.line(f'"""Solve Ax = b with CG."""')
        e.line("N = b.shape[0]")
        e.line("x = x0.copy()")
        e.line("r = cp.zeros_like(b)")
        e.line(_replace_vars(operator_call, {"Ap": "r", "p": "x"}))
        e.line("r = b - r")
        e.line("p = r.copy()")
        e.line("Ap = cp.zeros_like(b)")
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
        emit_cg_loop(e, operator_call, cfg, dtype)


def emit_preconditioned_cg_loop(
    e: CodeEmitter,
    operator_call: str,
    cfg: SolverConfig | None = None,
    dtype: str = "float64",
) -> None:
    """Emit the preconditioned CG iteration body.

    Implements preconditioned CG where z = M^{-1} r and beta is computed
    using dot(r, z) instead of dot(r, r).

    Assumes the following names are already in scope in the generated code:
    ``x``, ``r``, ``z``, ``p``, ``Ap``, ``buf``, ``stream``, ``grid``,
    ``dot_kernel``, ``axpy_kernel``, ``launch_jacobi_precondition``,
    ``tile_size``, ``math``, ``ct``.
    The variable ``r_dot_z`` must be initialised before calling this.

    Parameters
    ----------
    e : CodeEmitter
        The emitter to write into (at current indent level).
    operator_call : str
        Code string that computes ``Ap = A @ p``.
    cfg : SolverConfig, optional
        Solver configuration for defaults.
    dtype : str
        Data type string.
    """
    cfg = cfg or DEFAULT_SOLVER

    e.line("for it in range(max_iter):")
    with e.indent():
        e.line("if math.sqrt(abs(r_dot_z)) < tol:")
        with e.indent():
            e.line("return x, it, math.sqrt(abs(r_dot_z))")
        e.blank()
        e.line("# Ap = A @ p")
        e.line(operator_call)
        e.blank()
        e.line("# p_dot_Ap = p^T Ap")
        e.line("buf.fill(0)")
        e.line("ct.launch(stream, grid, dot_kernel, (p, Ap, buf, tile_size))")
        e.line("p_dot_Ap = buf.item()")
        e.blank()
        e.line("alpha = r_dot_z / p_dot_Ap")
        e.blank()
        e.line("# x = x + alpha * p")
        e.line("ct.launch(stream, grid, axpy_kernel, (p, x, alpha, tile_size))")
        e.blank()
        e.line("# r = r - alpha * Ap")
        e.line("ct.launch(stream, grid, axpy_kernel, (Ap, r, -alpha, tile_size))")
        e.blank()
        e.line("# z = M^{-1} r  (apply preconditioner to new residual)")
        e.line("launch_jacobi_precondition(r, z)")
        e.blank()
        e.line("# new_r_dot_z = r^T z")
        e.line("buf.fill(0)")
        e.line("ct.launch(stream, grid, dot_kernel, (r, z, buf, tile_size))")
        e.line("new_r_dot_z = buf.item()")
        e.blank()
        e.line("beta = new_r_dot_z / r_dot_z")
        e.blank()
        e.line("# p = z + beta * p")
        e.line("p = z + beta * p")
        e.blank()
        e.line("r_dot_z = new_r_dot_z")
    e.blank()
    e.line("return x, max_iter, math.sqrt(abs(r_dot_z))")


def emit_inner_cg_function(
    e: CodeEmitter,
    operator_call: str,
    cfg: SolverConfig | None = None,
    dtype: str = "float32",
) -> None:
    """Emit the inner CG function for mixed-precision refinement.

    This is a simplified CG that uses ``break`` instead of ``return``
    for early convergence, suitable for the inner solve.
    """
    cfg = cfg or DEFAULT_SOLVER

    e.line("def inner_cg(spmv_fn, rhs_lp, x_lp, tol, max_inner, tile_size=256):")
    with e.indent():
        e.line('"""Inner CG loop in low precision (FP32 or FP16)."""')
        e.line("N = rhs_lp.shape[0]")
        e.line("r = rhs_lp.copy()")
        e.line("Ax = cp.zeros_like(r)")
        e.line(_replace_vars(operator_call, {"Ap": "Ax", "p": "x_lp"}))
        e.line("r = rhs_lp - Ax")
        e.line("p = r.copy()")
        e.line("Ap = cp.zeros_like(r)")
        e.line("buf = cp.zeros(1, dtype=r.dtype)")
        e.line("stream = cp.cuda.get_current_stream()")
        e.line("grid = (ct.cdiv(N, tile_size),)")
        e.blank()
        e.line("buf.fill(0)")
        e.line("ct.launch(stream, grid, dot_kernel, (r, r, buf, tile_size))")
        e.line("r_dot_r = buf.item()")
        e.blank()
        e.line("for it in range(max_inner):")
        with e.indent():
            e.line("if math.sqrt(abs(r_dot_r)) < tol:")
            with e.indent():
                e.line("break")
            e.line(operator_call)
            e.line("buf.fill(0)")
            e.line("ct.launch(stream, grid, dot_kernel, (p, Ap, buf, tile_size))")
            e.line("p_dot_Ap = buf.item()")
            e.line("alpha = r_dot_r / p_dot_Ap")
            e.line("ct.launch(stream, grid, axpy_kernel, (p, x_lp, alpha, tile_size))")
            e.line("ct.launch(stream, grid, axpy_kernel, (Ap, r, -alpha, tile_size))")
            e.line("buf.fill(0)")
            e.line("ct.launch(stream, grid, dot_kernel, (r, r, buf, tile_size))")
            e.line("new_rr = buf.item()")
            e.line("beta = new_rr / r_dot_r")
            e.line("p = r + beta * p")
            e.line("r_dot_r = new_rr")
        e.line("return x_lp")


def emit_mixed_precision_wrapper(
    e: CodeEmitter,
    inner_cg_name: str,
    outer_operator_call: str,
    cfg: SolverConfig | None = None,
    outer_dtype: str = "float64",
    inner_dtype: str = "float32",
) -> None:
    """Emit the outer mixed-precision iterative refinement loop.

    Parameters
    ----------
    inner_cg_name : str
        Name of the inner CG function (already emitted).
    outer_operator_call : str
        Code for the high-precision operator application.
    """
    cfg = cfg or DEFAULT_SOLVER

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
        e.line("x = x0.copy()  # FP64")
        e.line("stream = cp.cuda.get_current_stream()")
        e.line("grid = (ct.cdiv(N, tile_size),)")
        e.blank()
        e.line("for outer in range(max_outer):")
        with e.indent():
            e.line("# Compute residual in FP64: r = b - A @ x")
            e.line("Ax = cp.zeros_like(b)")
            e.line(_replace_vars(outer_operator_call, {"Ap": "Ax", "p": "x"}))
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
            e.line(f"d_lp = {inner_cg_name}(spmv_fn_lp, r_lp, d_lp, tol={cfg.mixed_inner_tol}, max_inner=max_inner, tile_size=tile_size)")
            e.blank()
            e.line("# Update solution in FP64: x += d (cast back)")
            e.line("x = x + d_lp.astype(x.dtype)")
        e.blank()
        e.line("return x, max_outer, rnorm")
