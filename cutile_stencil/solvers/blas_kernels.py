"""Canonical BLAS kernel emit helpers for solver code generation.

Each ``emit_*`` function writes kernel or launch code into a supplied
CodeEmitter, eliminating duplication across cg.py, mixed_precision.py,
persistent_cg.py, and kernels.py.
"""

from __future__ import annotations

from cutile_stencil.codegen.emitter import CodeEmitter


def _ct_dtype(dtype: str) -> str:
    """Map a dtype string to the cuda.tile constant name."""
    mapping = {
        "float64": "ct.float64",
        "float32": "ct.float32",
        "float16": "ct.float16",
    }
    return mapping.get(dtype, "ct.float64")


# ── Dot product kernel ────────────────────────────────────────────────

def emit_dot_kernel(e: CodeEmitter, dtype: str = "float64") -> None:
    """Emit the atomic-reduction dot product kernel."""
    e.line("@ct.kernel")
    e.line("def dot_kernel(a, b, result, TILE: ConstInt):")
    with e.indent():
        e.line("pid = ct.bid(0)")
        e.line("ta = ct.load(a, index=(pid,), shape=(TILE,))")
        e.line("tb = ct.load(b, index=(pid,), shape=(TILE,))")
        e.line("partial = ct.sum(ta * tb)")
        e.line("ct.atomic_add(result, (0,), partial)")


def emit_dot_launch(e: CodeEmitter) -> None:
    """Emit the dot product launch helper."""
    e.line("def launch_dot(a, b, result, tile_size=256):")
    with e.indent():
        e.line("N = a.shape[0]")
        e.line("result.fill(0)")
        e.line("grid = (ct.cdiv(N, tile_size),)")
        e.line("stream = cp.cuda.get_current_stream()")
        e.line("ct.launch(stream, grid, dot_kernel, (a, b, result, tile_size))")


# ── AXPY kernel ───────────────────────────────────────────────────────

def emit_axpy_kernel(e: CodeEmitter, dtype: str = "float64") -> None:
    """Emit the y = alpha * x + y kernel."""
    e.line("@ct.kernel")
    e.line("def axpy_kernel(x, y, alpha, TILE: ConstInt):")
    with e.indent():
        e.line("pid = ct.bid(0)")
        e.line("tx = ct.load(x, index=(pid,), shape=(TILE,))")
        e.line("ty = ct.load(y, index=(pid,), shape=(TILE,))")
        e.line("result = alpha * tx + ty")
        e.line("ct.store(y, index=(pid,), tile=result)")


def emit_axpy_launch(e: CodeEmitter) -> None:
    """Emit the axpy launch helper."""
    e.line("def launch_axpy(x, y, alpha, tile_size=256):")
    with e.indent():
        e.line("N = x.shape[0]")
        e.line("grid = (ct.cdiv(N, tile_size),)")
        e.line("stream = cp.cuda.get_current_stream()")
        e.line("ct.launch(stream, grid, axpy_kernel, (x, y, alpha, tile_size))")


# ── Norm2 kernel ──────────────────────────────────────────────────────

def emit_norm2_kernel(e: CodeEmitter, dtype: str = "float64") -> None:
    """Emit the squared-norm kernel with atomic reduction."""
    e.line("@ct.kernel")
    e.line("def norm2_kernel(a, result, TILE: ConstInt):")
    with e.indent():
        e.line("pid = ct.bid(0)")
        e.line("ta = ct.load(a, index=(pid,), shape=(TILE,))")
        e.line("sq = ta * ta")
        e.line("partial = ct.sum(sq)")
        e.line("ct.atomic_add(result, (0,), partial)")


def emit_norm2_launch(e: CodeEmitter) -> None:
    """Emit the norm2 launch helper."""
    e.line("def launch_norm2(a, result, tile_size=256):")
    with e.indent():
        e.line("# result[0] = ||a||^2; caller takes sqrt if needed")
        e.line("N = a.shape[0]")
        e.line("result.fill(0)")
        e.line("grid = (ct.cdiv(N, tile_size),)")
        e.line("stream = cp.cuda.get_current_stream()")
        e.line("ct.launch(stream, grid, norm2_kernel, (a, result, tile_size))")


# ── Cast kernel ───────────────────────────────────────────────────────

def emit_cast_kernel(e: CodeEmitter) -> None:
    """Emit the precision-cast kernel (from mixed_precision.py)."""
    e.line("@ct.kernel")
    e.line("def cast_kernel(src, dst, TILE: ConstInt):")
    with e.indent():
        e.line("pid = ct.bid(0)")
        e.line("t = ct.load(src, index=(pid,), shape=(TILE,))")
        e.line("ct.store(dst, index=(pid,), tile=ct.astype(t, dst.dtype))")


# ── Arrive-wait barrier with sense reversal (for persistent CG) ──────

def emit_barrier(
    e: CodeEmitter,
    phase: int,
    *,
    last_block_lines: list[str] | None = None,
) -> None:
    """Emit an atomic-counter arrive-wait barrier with sense reversal.

    Uses ``counters[phase]`` for arrival counting and ``senses[phase]``
    for sense-flag notification.  The last arriving block performs any
    *last_block_lines* cleanup and flips the sense flag so all other
    blocks can proceed.

    The caller must have ``bid``, ``num_blocks``, ``iteration``,
    ``counters``, and ``senses`` in scope.
    """
    e.line(f"# ── Barrier {phase} ──")
    e.line("expected = iteration & 1")
    e.line("target = 1 - expected")
    e.line(f"cnt = ct.atomic_add(counters, ({phase},), 1, memory_order=ct.MemoryOrder.ACQ_REL)")
    e.line("if cnt == num_blocks - 1:")
    with e.indent():
        e.line(f"ct.atomic_xchg(counters, ({phase},), 0, memory_order=ct.MemoryOrder.RELAXED)")
        for ln in (last_block_lines or []):
            e.line(ln)
        e.line(f"ct.atomic_xchg(senses, ({phase},), target, memory_order=ct.MemoryOrder.RELEASE)")
    e.line(f"while ct.atomic_cas(senses, {phase}, target, target, memory_order=ct.MemoryOrder.ACQUIRE) != target: pass")
    e.blank()


# ── DIA SpMV phase (for persistent CG) ───────────────────────────────

def emit_dia_spmv_phase(e: CodeEmitter, dtype: str = "float64") -> None:
    """Emit the DIA SpMV loop body used inside persistent CG.

    Emits the tiled SpMV loop that computes ``Ap = A @ p`` for one tile,
    accumulating diagonal contributions via gather.  The caller must have
    ``tile_id``, ``diag_vals``, ``offsets_arr``, ``num_diags_val``, ``p``,
    ``Ap``, and ``TILE`` in scope.
    """
    e.line(f"acc = ct.full((TILE,), 0.0, dtype={_ct_dtype(dtype)})")
    e.line("for d in range(num_diags_val):")
    with e.indent():
        e.line("dv = ct.load(diag_vals, index=(d, tile_id), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)")
        e.line("dv = ct.reshape(dv, (TILE,))")
        e.line("base = tile_id * TILE + ct.arange(TILE, dtype=cp.int32)")
        e.line("off = ct.load(offsets_arr, index=d, shape=())")
        e.line("x_vals = ct.gather(p, base + off)")
        e.line("acc = acc + dv * x_vals")
    e.line("ct.store(Ap, index=(tile_id,), tile=acc)")
