"""cuTile kernel code generators for solver building blocks.

Each function returns a string of syntactically-valid cuTile Python code.
These are meant to be written to files and run on a GPU.
"""

from __future__ import annotations

from cutile_stencil.codegen.emitter import CodeEmitter


def _header(e: CodeEmitter):
    e.line('"""Auto-generated cuTile solver kernel."""')
    e.blank()
    e.line("import cuda.tile as ct")
    e.line("import cupy as cp")
    e.blank()
    e.line("ConstInt = ct.Constant[int]")


def generate_dia_spmv() -> str:
    """DIA SpMV kernel using shifted-view pattern (like stencil1d.py)."""
    e = CodeEmitter()
    _header(e)
    e.blank()
    e.line("@ct.kernel")
    e.line("def dia_spmv_kernel(diag_vals, offsets_arr, x, y, num_diags: ConstInt, TILE: ConstInt):")
    with e.indent():
        e.line("pid = ct.bid(0)")
        e.line("# Load output tile (accumulator)")
        e.line("acc = ct.full((TILE,), 0.0, dtype=ct.float64)")
        e.line("for d in range(num_diags):")
        with e.indent():
            e.line("# Load diagonal values for this tile")
            e.line("dv = ct.load(diag_vals, index=(d, pid), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)")
            e.line("dv = ct.reshape(dv, (TILE,))")
            e.line("# Compute shifted x indices via gather")
            e.line("base = pid * TILE + ct.arange(TILE, dtype=cp.int32)")
            e.line("off = ct.load(offsets_arr, index=d, shape=())")
            e.line("x_idx = base + off")
            e.line("x_vals = ct.gather(x, x_idx)")
            e.line("acc = acc + dv * x_vals")
        e.line("ct.store(y, index=(pid,), tile=acc)")
    e.blank()
    e.blank()
    e.line("def launch_dia_spmv(diag_vals, offsets_arr, x, y, num_diags, tile_size=256):")
    with e.indent():
        e.line("N = y.shape[0]")
        e.line("grid = (ct.cdiv(N, tile_size),)")
        e.line("stream = cp.cuda.get_current_stream()")
        e.line("ct.launch(stream, grid, dia_spmv_kernel, (diag_vals, offsets_arr, x, y, num_diags, tile_size))")
    return e.render()


def generate_bsr_spmv() -> str:
    """BSR SpMV kernel using gather + mma."""
    e = CodeEmitter()
    _header(e)
    e.blank()
    e.line("@ct.kernel")
    e.line("def bsr_spmv_kernel(data, indices, indptr, x, y, BR: ConstInt, BC: ConstInt):")
    with e.indent():
        e.line("bid = ct.bid(0)")
        e.line("# Each block handles one block-row")
        e.line("row_start = ct.load(indptr, index=bid, shape=())")
        e.line("row_end = ct.load(indptr, index=bid + 1, shape=())")
        e.line("acc = ct.full((BR, 1), 0.0, dtype=ct.float64)")
        e.line("for idx in range(row_start, row_end):")
        with e.indent():
            e.line("col_block = ct.load(indices, index=idx, shape=())")
            e.line("block = ct.load(data, index=(idx, 0, 0), shape=(1, BR, BC), padding_mode=ct.PaddingMode.ZERO)")
            e.line("block = ct.reshape(block, (BR, BC))")
            e.line("x_col = col_block * BC + ct.arange(BC, dtype=cp.int32)")
            e.line("x_tile = ct.gather(x, x_col)")
            e.line("x_tile = ct.reshape(x_tile, (BC, 1))")
            e.line("acc = ct.mma(block, x_tile, acc)")
        e.line("acc = ct.reshape(acc, (BR,))")
        e.line("ct.store(y, index=(bid,), tile=acc)")
    return e.render()


def generate_axpy() -> str:
    """y = alpha * x + y kernel."""
    e = CodeEmitter()
    _header(e)
    e.blank()
    e.line("@ct.kernel")
    e.line("def axpy_kernel(x, y, alpha, TILE: ConstInt):")
    with e.indent():
        e.line("pid = ct.bid(0)")
        e.line("tx = ct.load(x, index=(pid,), shape=(TILE,))")
        e.line("ty = ct.load(y, index=(pid,), shape=(TILE,))")
        e.line("result = alpha * tx + ty")
        e.line("ct.store(y, index=(pid,), tile=result)")
    e.blank()
    e.blank()
    e.line("def launch_axpy(x, y, alpha, tile_size=256):")
    with e.indent():
        e.line("N = x.shape[0]")
        e.line("grid = (ct.cdiv(N, tile_size),)")
        e.line("stream = cp.cuda.get_current_stream()")
        e.line("ct.launch(stream, grid, axpy_kernel, (x, y, alpha, tile_size))")
    return e.render()


def generate_dot() -> str:
    """Atomic-reduction dot product kernel (like dot_product.py)."""
    e = CodeEmitter()
    _header(e)
    e.blank()
    e.line("@ct.kernel")
    e.line("def dot_kernel(a, b, result, TILE: ConstInt):")
    with e.indent():
        e.line("pid = ct.bid(0)")
        e.line("ta = ct.load(a, index=(pid,), shape=(TILE,))")
        e.line("tb = ct.load(b, index=(pid,), shape=(TILE,))")
        e.line("prod = ta * tb")
        e.line("partial = ct.sum(prod)")
        e.line("ct.atomic_add(result, (0,), partial)")
    e.blank()
    e.blank()
    e.line("def launch_dot(a, b, result, tile_size=256):")
    with e.indent():
        e.line("N = a.shape[0]")
        e.line("result.zero_()")
        e.line("grid = (ct.cdiv(N, tile_size),)")
        e.line("stream = cp.cuda.get_current_stream()")
        e.line("ct.launch(stream, grid, dot_kernel, (a, b, result, tile_size))")
    return e.render()


def generate_norm2() -> str:
    """Squared-norm kernel with atomic reduction (like norm2.py)."""
    e = CodeEmitter()
    _header(e)
    e.blank()
    e.line("@ct.kernel")
    e.line("def norm2_kernel(a, result, TILE: ConstInt):")
    with e.indent():
        e.line("pid = ct.bid(0)")
        e.line("ta = ct.load(a, index=(pid,), shape=(TILE,))")
        e.line("sq = ta * ta")
        e.line("partial = ct.sum(sq)")
        e.line("ct.atomic_add(result, (0,), partial)")
    e.blank()
    e.blank()
    e.line("def launch_norm2(a, result, tile_size=256):")
    with e.indent():
        e.line("# result[0] = ||a||^2; caller takes sqrt if needed")
        e.line("N = a.shape[0]")
        e.line("result.zero_()")
        e.line("grid = (ct.cdiv(N, tile_size),)")
        e.line("stream = cp.cuda.get_current_stream()")
        e.line("ct.launch(stream, grid, norm2_kernel, (a, result, tile_size))")
    return e.render()
