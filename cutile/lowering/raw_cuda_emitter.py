"""Raw CUDA kernel emitter for memory-bound stencils.

When a stencil is classified as memory-bound by the roofline analysis,
the cuTile TMA-based approach adds overhead from the shared memory
detour. This module generates a simple CuPy RawKernel that uses
per-thread scalar loads — matching what JAX/XLA does internally.

The generated kernel achieves near-peak DRAM bandwidth because:
- No TMA descriptor overhead
- No shared memory detour (Global → L2 → Registers directly)
- L2 cache naturally handles overlapping stencil reads
- Simple index arithmetic, no .slice() chains
"""

from __future__ import annotations

from cutile.lowering.emitter import CodeEmitter
from cutile.dialects.cutile_stencil.dialect import AccessOp, FuncOp, YieldOp
from xdsl.dialects import arith
from xdsl.dialects.builtin import FloatAttr, ModuleOp
from xdsl.ir import SSAValue


def _find_func_op(module: ModuleOp) -> FuncOp:
    for op in module.body.ops:
        if isinstance(op, FuncOp):
            return op
    raise ValueError("No cutile_stencil.FuncOp found")


def lower_stencil_to_raw_cuda(
    module: ModuleOp,
    tile_sizes: tuple[int, ...] | None = None,
    halo_widths: tuple[int, ...] | None = None,
) -> str:
    """Generate a CuPy RawKernel for a stencil (bypasses cuTile TMA).

    Returns complete Python source with the raw CUDA kernel and launcher.
    """
    func_op = _find_func_op(module)
    block = func_op.body.block

    name = func_op.func_name.data
    ndim = func_op.ndim.data
    order = func_op.order.data

    if halo_widths is None:
        h = order // 2
        halo_widths = (h,) * ndim
    if tile_sizes is None:
        tile_sizes = (16, 16)[:ndim] if ndim >= 2 else (256,)

    # Collect accesses and build expression
    accesses: list[tuple[int, tuple[int, ...]]] = []  # (arr_idx, offsets)
    val_to_expr: dict[SSAValue, str] = {}

    # Map block args
    arr_names = list("uvwxyz")[:len(block.args)]
    for i, arg in enumerate(block.args):
        val_to_expr[arg] = arr_names[i]

    # Index variable names per dim
    idx_vars = ["i", "j", "k"][:ndim]
    # Stride variables
    stride_vars = [f"s{d}" for d in range(ndim)]

    for op in block.ops:
        if isinstance(op, AccessOp):
            offsets = tuple(item.data for item in op.offset.parameters[0].data)
            arr_idx = 0
            for i, arg in enumerate(block.args):
                if op.field is arg:
                    arr_idx = i
                    break
            accesses.append((arr_idx, offsets))

            # Build index expression: (i+off0)*s0 + (j+off1)*s1 + ...
            parts = []
            for d in range(ndim):
                idx = idx_vars[d]
                off = offsets[d]
                if off == 0:
                    parts.append(f"{idx} * {stride_vars[d]}")
                elif off > 0:
                    parts.append(f"({idx} + {off}) * {stride_vars[d]}")
                else:
                    parts.append(f"({idx} - {abs(off)}) * {stride_vars[d]}")
            idx_expr = " + ".join(parts)
            val_to_expr[op.res] = f"{arr_names[arr_idx]}[{idx_expr}]"

        elif isinstance(op, arith.ConstantOp):
            val_attr = op.properties.get("value", op.attributes.get("value", None))
            if isinstance(val_attr, FloatAttr):
                val_to_expr[op.result] = repr(val_attr.value.data)
            else:
                val_to_expr[op.result] = str(val_attr)

        elif isinstance(op, arith.AddfOp):
            l = val_to_expr.get(op.lhs, "?")
            r = val_to_expr.get(op.rhs, "?")
            val_to_expr[op.result] = f"({l} + {r})"

        elif isinstance(op, arith.SubfOp):
            l = val_to_expr.get(op.lhs, "?")
            r = val_to_expr.get(op.rhs, "?")
            val_to_expr[op.result] = f"({l} - {r})"

        elif isinstance(op, arith.MulfOp):
            l = val_to_expr.get(op.lhs, "?")
            r = val_to_expr.get(op.rhs, "?")
            val_to_expr[op.result] = f"({l} * {r})"

        elif isinstance(op, arith.DivfOp):
            l = val_to_expr.get(op.lhs, "?")
            r = val_to_expr.get(op.rhs, "?")
            val_to_expr[op.result] = f"({l} / {r})"

        elif isinstance(op, arith.NegfOp):
            operand = op.operands[0]
            val_to_expr[op.result] = f"(-{val_to_expr.get(operand, '?')})"

        elif isinstance(op, YieldOp):
            expr = val_to_expr.get(op.value, "0")

    num_inputs = len(block.args)

    # ================================================================
    # Generate CUDA C kernel
    # ================================================================
    e = CodeEmitter()
    e.line(f'"""Raw CUDA kernel for {name} stencil (auto-generated).')
    e.line("")
    e.line("Uses CuPy RawKernel for near-peak memory bandwidth.")
    e.line("Bypasses cuTile TMA — direct per-thread scalar loads.")
    e.line('"""')
    e.blank()
    e.line("import cupy as cp")
    e.blank()

    # Build CUDA C source
    e.line("_CUDA_SOURCE = r'''")

    # Kernel signature
    params = []
    for arr in arr_names[:num_inputs]:
        params.append(f"const double* __restrict__ {arr}")
    params.append("double* __restrict__ out")
    for d in range(ndim):
        params.append(f"int n{idx_vars[d]}")
    param_str = ", ".join(params)

    e.line(f"extern \"C\" __global__")
    e.line(f"void {name}_kernel({param_str}) {{")

    # Thread index computation
    if ndim == 1:
        h = halo_widths[0]
        e.line(f"    int i = blockIdx.x * blockDim.x + threadIdx.x + {h};")
        e.line(f"    if (i < ni + {h}) {{")
        # Strides for 1D: just 1
        # Replace stride vars in expression
        expr_cuda = expr.replace(f" * {stride_vars[0]}", "")
        for d in range(ndim):
            expr_cuda = expr_cuda.replace(f"* {stride_vars[d]}", "")
        # For 1D, index is just the offset from i
        # Rebuild manually
        out_idx = f"i"
        e.line(f"        out[{out_idx}] = {expr};")
        e.line("    }")

    elif ndim == 2:
        hx, hy = halo_widths
        e.line(f"    int i = blockIdx.x * blockDim.x + threadIdx.x + {hx};")
        e.line(f"    int j = blockIdx.y * blockDim.y + threadIdx.y + {hy};")
        e.line(f"    int stride = nj + {2*hy};")
        e.line(f"    if (i < ni + {hx} && j < nj + {hy}) {{")
        e.line(f"        int idx = i * stride + j;")
        # Build expression using idx arithmetic
        cuda_expr = _build_cuda_expr_2d(block, arr_names[:num_inputs], halo_widths)
        e.line(f"        out[idx] = {cuda_expr};")
        e.line("    }")

    elif ndim == 3:
        hx, hy, hz = halo_widths
        e.line(f"    int i = blockIdx.x * blockDim.x + threadIdx.x + {hx};")
        e.line(f"    int j = blockIdx.y * blockDim.y + threadIdx.y + {hy};")
        e.line(f"    int k = blockIdx.z * blockDim.z + threadIdx.z + {hz};")
        e.line(f"    int sy = nk + {2*hz};")
        e.line(f"    int sx = (nj + {2*hy}) * sy;")
        e.line(f"    if (i < ni + {hx} && j < nj + {hy} && k < nk + {hz}) {{")
        e.line(f"        int idx = i * sx + j * sy + k;")
        cuda_expr = _build_cuda_expr_3d(block, arr_names[:num_inputs], halo_widths)
        e.line(f"        out[idx] = {cuda_expr};")
        e.line("    }")

    e.line("}")
    e.line("'''")
    e.blank()

    # Compile kernel
    e.line(f"_kernel = cp.RawKernel(_CUDA_SOURCE, '{name}_kernel')")
    e.blank()

    # Launcher
    e.blank()
    if num_inputs == 1:
        e.line(f"def launch_{name}(u_in, u_out, stream=None):")
    else:
        inp_params = ", ".join(arr_names[:num_inputs])
        e.line(f"def launch_{name}({inp_params}, u_out, stream=None):")

    with e.indent():
        e.line(f'"""Launch raw CUDA {name} kernel."""')

        if ndim == 1:
            e.line(f"N, = u_in.shape")
            e.line(f"ni = N - {2*halo_widths[0]}")
            e.line(f"threads = {tile_sizes[0]}")
            e.line(f"blocks = (ni + threads - 1) // threads")
            args = "u_in, u_out, ni" if num_inputs == 1 else f"{inp_params}, u_out, ni"
            e.line(f"_kernel((blocks,), (threads,), ({args}))")

        elif ndim == 2:
            hx, hy = halo_widths
            tx, ty = tile_sizes
            e.line(f"Nx, Ny = u_in.shape")
            e.line(f"ni, nj = Nx - {2*hx}, Ny - {2*hy}")
            e.line(f"threads = ({tx}, {ty})")
            e.line(f"blocks = ((ni + {tx-1}) // {tx}, (nj + {ty-1}) // {ty})")
            args = "u_in, u_out, ni, nj" if num_inputs == 1 else f"{inp_params}, u_out, ni, nj"
            e.line(f"_kernel(blocks, threads, ({args}))")

        elif ndim == 3:
            hx, hy, hz = halo_widths
            tx, ty, tz = tile_sizes
            e.line(f"Nx, Ny, Nz = u_in.shape")
            e.line(f"ni, nj, nk = Nx - {2*hx}, Ny - {2*hy}, Nz - {2*hz}")
            e.line(f"threads = ({tx}, {ty}, {tz})")
            e.line(f"blocks = ((ni+{tx-1})//{tx}, (nj+{ty-1})//{ty}, (nk+{tz-1})//{tz})")
            args = "u_in, u_out, ni, nj, nk" if num_inputs == 1 else f"{inp_params}, u_out, ni, nj, nk"
            e.line(f"_kernel(blocks, threads, ({args}))")

    return e.render()


def _build_cuda_expr_2d(block, arr_names, halo_widths):
    """Build a CUDA C expression string for 2D stencils using idx+offset."""
    hx, hy = halo_widths
    stride = f"stride"
    val_to_expr = {}

    for i, arg in enumerate(block.args):
        val_to_expr[arg] = arr_names[i]

    for op in block.ops:
        if isinstance(op, AccessOp):
            offsets = tuple(item.data for item in op.offset.parameters[0].data)
            arr_idx = 0
            for i, arg in enumerate(block.args):
                if op.field is arg:
                    arr_idx = i
                    break
            arr = arr_names[arr_idx]
            di, dj = offsets
            offset_parts = []
            if di != 0:
                offset_parts.append(f"{di} * {stride}")
            if dj != 0:
                offset_parts.append(str(dj))
            if offset_parts:
                offset_str = " + ".join(offset_parts)
                val_to_expr[op.res] = f"{arr}[idx + {offset_str}]"
            else:
                val_to_expr[op.res] = f"{arr}[idx]"

        elif isinstance(op, arith.ConstantOp):
            val_attr = op.properties.get("value", op.attributes.get("value", None))
            if isinstance(val_attr, FloatAttr):
                val_to_expr[op.result] = repr(val_attr.value.data)
            else:
                val_to_expr[op.result] = str(val_attr)

        elif isinstance(op, arith.AddfOp):
            l = val_to_expr.get(op.lhs, "?")
            r = val_to_expr.get(op.rhs, "?")
            val_to_expr[op.result] = f"({l} + {r})"

        elif isinstance(op, arith.SubfOp):
            l = val_to_expr.get(op.lhs, "?")
            r = val_to_expr.get(op.rhs, "?")
            val_to_expr[op.result] = f"({l} - {r})"

        elif isinstance(op, arith.MulfOp):
            l = val_to_expr.get(op.lhs, "?")
            r = val_to_expr.get(op.rhs, "?")
            val_to_expr[op.result] = f"({l} * {r})"

        elif isinstance(op, arith.DivfOp):
            l = val_to_expr.get(op.lhs, "?")
            r = val_to_expr.get(op.rhs, "?")
            val_to_expr[op.result] = f"({l} / {r})"

        elif isinstance(op, arith.NegfOp):
            operand = op.operands[0]
            val_to_expr[op.result] = f"(-{val_to_expr.get(operand, '?')})"

        elif isinstance(op, YieldOp):
            return val_to_expr.get(op.value, "0")

    return "0"


def _build_cuda_expr_3d(block, arr_names, halo_widths):
    """Build a CUDA C expression for 3D stencils."""
    val_to_expr = {}

    for i, arg in enumerate(block.args):
        val_to_expr[arg] = arr_names[i]

    for op in block.ops:
        if isinstance(op, AccessOp):
            offsets = tuple(item.data for item in op.offset.parameters[0].data)
            arr_idx = 0
            for i, arg in enumerate(block.args):
                if op.field is arg:
                    arr_idx = i
                    break
            arr = arr_names[arr_idx]
            di, dj, dk = offsets
            parts = []
            if di != 0:
                parts.append(f"{di} * sx")
            if dj != 0:
                parts.append(f"{dj} * sy")
            if dk != 0:
                parts.append(str(dk))
            if parts:
                val_to_expr[op.res] = f"{arr}[idx + {' + '.join(parts)}]"
            else:
                val_to_expr[op.res] = f"{arr}[idx]"

        elif isinstance(op, arith.ConstantOp):
            val_attr = op.properties.get("value", op.attributes.get("value", None))
            if isinstance(val_attr, FloatAttr):
                val_to_expr[op.result] = repr(val_attr.value.data)
            else:
                val_to_expr[op.result] = str(val_attr)

        elif isinstance(op, arith.AddfOp):
            val_to_expr[op.result] = f"({val_to_expr.get(op.lhs, '?')} + {val_to_expr.get(op.rhs, '?')})"
        elif isinstance(op, arith.SubfOp):
            val_to_expr[op.result] = f"({val_to_expr.get(op.lhs, '?')} - {val_to_expr.get(op.rhs, '?')})"
        elif isinstance(op, arith.MulfOp):
            val_to_expr[op.result] = f"({val_to_expr.get(op.lhs, '?')} * {val_to_expr.get(op.rhs, '?')})"
        elif isinstance(op, arith.DivfOp):
            val_to_expr[op.result] = f"({val_to_expr.get(op.lhs, '?')} / {val_to_expr.get(op.rhs, '?')})"
        elif isinstance(op, arith.NegfOp):
            val_to_expr[op.result] = f"(-{val_to_expr.get(op.operands[0], '?')})"
        elif isinstance(op, YieldOp):
            return val_to_expr.get(op.value, "0")

    return "0"
