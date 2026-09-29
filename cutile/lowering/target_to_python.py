"""Target-to-Python emitter: Dialect 3 (cutile_target) IR to Python source.

Walks a ``ModuleOp`` containing ``cutile_target`` dialect ops and emits
the equivalent cuTile Python source code.  The emitter is purely
mechanical -- it reads properties from each op and formats them into
the Python API strings.
"""

from __future__ import annotations

import re

from xdsl.dialects.builtin import ArrayAttr, IntAttr, ModuleOp, StringAttr

from cutile.dialects.cutile_target.dialect import (
    BidOp,
    ForLoopOp,
    HostProgramOp,
    KernelOp,
    LaunchOp,
    LoadOp,
    ReturnOp,
    SliceOp,
    StoreOp,
)
from cutile.lowering.emitter import CodeEmitter


# -------------------------------------------------------------------- #
# Kernel emitter
# -------------------------------------------------------------------- #


def _emit_kernel(e: CodeEmitter, kernel: KernelOp) -> None:
    """Emit the ``@ct.kernel`` function from a ``KernelOp``.

    The old emitter produces a specific ordering:
    1. Block indices (bid)
    2. Interior sizes (nx, ny, ...)
    3. All slice-chain assignments (input views + output view)
    4. All ct.load statements
    5. Blank + result expression + blank
    6. ct.store

    We replicate that ordering by doing two passes over the body ops:
    first collecting slice chains, then collecting loads.
    """
    ndim = kernel.ndim.data
    name = kernel.kernel_name.data
    input_names = [a.data for a in kernel.input_names.data]
    expressions = (
        [a.data for a in kernel.expressions.data]
        if kernel.expressions is not None
        else [kernel.expression.data]
    )
    num_inputs = len(input_names)
    multi_input = num_inputs > 1
    body_block = list(kernel.body.blocks)[0]
    n_out = len(body_block.args) - num_inputs
    output_params = ["output"] if n_out == 1 else [f"output_{k}" for k in range(n_out)]
    result_names = ["result"] if n_out == 1 else [f"result_{k}" for k in range(n_out)]

    bid_vars = ["bx", "by", "bz"][:ndim]
    tile_vars = ["TX", "TY", "TZ"][:ndim]
    halo_vars = ["HX", "HY", "HZ"][:ndim]
    n_vars = ["nx", "ny", "nz"][:ndim]

    tile_const_params = ", ".join(f"{tv}: ConstInt" for tv in tile_vars)
    halo_const_params = ", ".join(f"{hv}: ConstInt" for hv in halo_vars)

    if multi_input:
        input_params = ", ".join(input_names)
    else:
        input_params = input_names[0]

    consts_param = "consts, " if kernel.kernel_constants is not None else ""

    if kernel.kernel_hints is not None and len(kernel.kernel_hints.data):
        items = [a.data for a in kernel.kernel_hints.data]
        kw = ", ".join(f"{k}={v}" for k, v in zip(items[0::2], items[1::2]))
        e.line(f"@ct.kernel({kw})")
    else:
        e.line("@ct.kernel")
    e.line(
        f"def {name}_kernel({input_params}, {', '.join(output_params)}, {consts_param}"
        f"{tile_const_params}, {halo_const_params}):"
    )

    with e.indent():
        # 1. Block indices: grid_order lists array dims in bid order, so
        #    bid(0) walks rows of a plane (see stencil_to_target.grid_order)
        order = (
            [a.data for a in kernel.grid_order.data]
            if kernel.grid_order is not None
            else list(range(ndim))
        )
        for pos, d in enumerate(order):
            e.line(f"{bid_vars[d]} = ct.bid({pos})")

        # 2. Interior sizes
        first_arr = input_names[0]
        for d in range(ndim):
            e.line(f"{n_vars[d]} = {first_arr}.shape[{d}] - 2 * {halo_vars[d]}")

        # Walk kernel body to build SSA -> expression mapping
        val_to_expr: dict = {}
        view_vars: dict = {}
        store_ops: list[StoreOp] = []

        # Map block args to input names + outputs
        for i in range(num_inputs):
            val_to_expr[body_block.args[i]] = input_names[i]
        for k, pname in enumerate(output_params):
            val_to_expr[body_block.args[num_inputs + k]] = pname

        # First pass: collect slice chains (emit assignments)
        # and remember load ops for second pass
        load_ops: list[LoadOp] = []
        for op in body_block.ops:
            if isinstance(op, SliceOp):
                parent_expr = val_to_expr.get(op.input, "?")
                axis = op.axis.data
                start = op.start.data
                stop = op.stop.data
                chain_expr = (
                    f"{parent_expr}.slice(axis={axis}, "
                    f"start={start}, stop={stop})"
                )
                val_to_expr[op.result] = chain_expr

                # Emit assignment only for the last slice in a chain
                if op.var_name is not None:
                    var_name = op.var_name.data
                    view_vars[op.result] = var_name
                    e.line(f"{var_name} = {chain_expr}")

            elif isinstance(op, LoadOp):
                load_ops.append(op)

            elif isinstance(op, StoreOp):
                store_ops.append(op)

        # 4. Emit all loads
        idx_tuple = ", ".join(bid_vars)
        shape_tuple = ", ".join(tile_vars)
        if ndim == 1:
            idx_arg = f"({idx_tuple},)"
            shape_arg = f"({shape_tuple},)"
        else:
            idx_arg = f"({idx_tuple})"
            shape_arg = f"({shape_tuple})"

        # A tile used once is loaded inline, at its use site, so loads
        # interleave with the arithmetic in evaluation order (tileiras keeps
        # every loaded tile live until consumed; hoisting all loads first
        # costs registers and occupancy on wide stencils).  Tiles used more
        # than once are loaded once, up front.
        for load in load_ops:
            view_name = (
                load.view_name.data if load.view_name is not None else "view"
            )
            load_expr = f"ct.load({view_name}, index={idx_arg}, shape={shape_arg})"
            pattern = rf"\bt_{re.escape(view_name)}\b"
            uses = sum(len(re.findall(pattern, x)) for x in expressions)
            if uses == 1:
                expressions = [re.sub(pattern, load_expr, x) for x in expressions]
            else:
                e.line(f"t_{view_name} = {load_expr}")

        # 4b. Exact float64 constants (cuTile rounds Python floats to f32)
        if kernel.kernel_constants is not None:
            values = [a.data for a in kernel.kernel_constants.data]
            n = _pow2_at_least(len(values))
            e.blank()
            e.line(f"_consts = ct.load(consts, index=(0,), shape=({n},))")
            for k, v in enumerate(values):
                sub = f"ct.extract(_consts, ({k},), shape=(1,))"
                if ndim > 1:
                    sub = f"ct.reshape({sub}, ({', '.join(['1'] * ndim)}))"
                e.line(f"_c{k} = {sub}  # {v}")

        # 5. Expressions (one per output)
        e.blank()
        for rname, x in zip(result_names, expressions):
            e.line(f"{rname} = {x}")
        e.blank()

        # 6. Stores
        for rname, store in zip(result_names, store_ops):
            view = view_vars.get(store.tile, "out")
            e.line(f"ct.store({view}, index={idx_arg}, tile={rname})")


# -------------------------------------------------------------------- #
# Boundary emitter
# -------------------------------------------------------------------- #


def _emit_boundary(
    e: CodeEmitter,
    name: str,
    ndim: int,
    halo_widths: tuple[int, ...],
    bc_type: str,
    bc_value: float = 0.0,
) -> None:
    """Emit a boundary application function."""
    e.blank()
    e.blank()
    e.line(f"def apply_boundary_{name}(u, stream=None):")
    with e.indent():
        e.line('"""Apply boundary conditions to halo cells."""')
        e.line("if stream is None:")
        with e.indent():
            e.line("stream = cp.cuda.get_current_stream()")

        for d in range(ndim):
            hw = halo_widths[d]
            if bc_type == "periodic":
                e.line(f"# Dim {d}: periodic BC")
                if ndim == 1:
                    e.line(f"u[:{hw}] = u[-2*{hw}:-{hw}]")
                    e.line(f"u[-{hw}:] = u[{hw}:2*{hw}]")
                else:
                    low_dst = ", ".join(
                        ":" if i != d else f":{hw}" for i in range(ndim)
                    )
                    low_src = ", ".join(
                        ":" if i != d else f"-2*{hw}:-{hw}" for i in range(ndim)
                    )
                    high_dst = ", ".join(
                        ":" if i != d else f"-{hw}:" for i in range(ndim)
                    )
                    high_src = ", ".join(
                        ":" if i != d else f"{hw}:2*{hw}" for i in range(ndim)
                    )
                    e.line(f"u[{low_dst}] = u[{low_src}]")
                    e.line(f"u[{high_dst}] = u[{high_src}]")

            elif bc_type == "neumann":
                e.line(f"# Dim {d}: Neumann BC (zero gradient)")
                if ndim == 1:
                    e.line(f"u[:{hw}] = u[{hw}:{hw}+1]")
                    e.line(f"u[-{hw}:] = u[-{hw}-1:-{hw}]")
                else:
                    low_dst = ", ".join(
                        ":" if i != d else f":{hw}" for i in range(ndim)
                    )
                    low_src = ", ".join(
                        ":" if i != d else f"{hw}:{hw}+1" for i in range(ndim)
                    )
                    high_dst = ", ".join(
                        ":" if i != d else f"-{hw}:" for i in range(ndim)
                    )
                    high_src = ", ".join(
                        ":" if i != d else f"-{hw}-1:-{hw}" for i in range(ndim)
                    )
                    e.line(f"u[{low_dst}] = u[{low_src}]")
                    e.line(f"u[{high_dst}] = u[{high_src}]")

            elif bc_type == "dirichlet":
                e.line(f"# Dim {d}: Dirichlet BC (value={bc_value})")
                if ndim == 1:
                    e.line(f"u[:{hw}] = {bc_value}")
                    e.line(f"u[-{hw}:] = {bc_value}")
                else:
                    low_dst = ", ".join(
                        ":" if i != d else f":{hw}" for i in range(ndim)
                    )
                    high_dst = ", ".join(
                        ":" if i != d else f"-{hw}:" for i in range(ndim)
                    )
                    e.line(f"u[{low_dst}] = {bc_value}")
                    e.line(f"u[{high_dst}] = {bc_value}")

            elif bc_type == "reflecting":
                e.line(f"# Dim {d}: reflecting BC")
                if ndim == 1:
                    e.line(f"u[:{hw}] = u[2*{hw}-1:{hw}-1:-1]")
                    e.line(f"u[-{hw}:] = u[-{hw}-1:-2*{hw}-1:-1]")
                else:
                    low_dst = ", ".join(
                        ":" if i != d else f":{hw}" for i in range(ndim)
                    )
                    low_src = ", ".join(
                        ":" if i != d else f"2*{hw}-1:{hw}-1:-1"
                        for i in range(ndim)
                    )
                    high_dst = ", ".join(
                        ":" if i != d else f"-{hw}:" for i in range(ndim)
                    )
                    high_src = ", ".join(
                        ":" if i != d else f"-{hw}-1:-2*{hw}-1:-1"
                        for i in range(ndim)
                    )
                    e.line(f"u[{low_dst}] = u[{low_src}]")
                    e.line(f"u[{high_dst}] = u[{high_src}]")


# -------------------------------------------------------------------- #
# Launch-time tile clamping
# -------------------------------------------------------------------- #


def _emit_fit_tiles_helper(e: CodeEmitter) -> None:
    """Emit ``_fit_tiles``: clamp tile dims to the runtime interior extents.

    The tiling pass picks wide row tiles without knowing the array shape; on
    a smaller domain the launcher shrinks each tile dimension to the next
    power of two of the interior so a tile never covers mostly padding.
    cuTile specialises the kernel per constant, so this costs nothing.
    """
    e.line("def _fit_tiles(tiles, shape, halos):")
    with e.indent():
        e.line('"""Clamp each tile dim to the next power of two >= interior."""')
        e.line("fitted = []")
        e.line("for t, n, h in zip(tiles, shape, halos):")
        with e.indent():
            e.line("interior = max(1, n - 2 * h)")
            e.line("fitted.append(min(t, 1 << max(0, (interior - 1).bit_length())))")
        e.line("return tuple(fitted)")


# -------------------------------------------------------------------- #
# Temporal scratch buffers
# -------------------------------------------------------------------- #


def _emit_temporal_buffers_helper(e: CodeEmitter) -> None:
    """Emit a module-level cache of scratch arrays for temporal launchers.

    The kernel only ever writes interior cells, so a scratch buffer's halo
    stays zero after allocation and the buffers can be reused across calls
    with identical semantics to allocating fresh zeros each time.
    """
    e.line("_TEMPORAL_BUFFERS = {}")
    e.blank()
    e.blank()
    e.line("def _temporal_buffers(ref, count):")
    with e.indent():
        e.line('"""Return `count` reusable scratch arrays shaped like `ref`."""')
        e.line("key = (tuple(ref.shape), str(ref.dtype), int(ref.device.id))")
        e.line("bufs = _TEMPORAL_BUFFERS.get(key)")
        e.line("if bufs is None or len(bufs) < count:")
        with e.indent():
            e.line("bufs = [cp.zeros_like(ref) for _ in range(count)]")
            e.line("_TEMPORAL_BUFFERS[key] = bufs")
        e.line("return bufs[:count]")


# -------------------------------------------------------------------- #
# Host program emitter
# -------------------------------------------------------------------- #


def _emit_host(e: CodeEmitter, host: HostProgramOp) -> None:
    """Emit the launcher function from a ``HostProgramOp``."""
    prog_name = host.program_name.data
    preamble = host.preamble.data if host.preamble is not None else ""

    e.blank()
    e.blank()
    e.line(f"def {prog_name}:")
    with e.indent():
        # Emit preamble lines
        for pline in preamble.split("\n"):
            e.line(pline)

        # Walk body ops
        body_block = list(host.body.blocks)[0]
        for op in body_block.ops:
            if isinstance(op, LaunchOp):
                kernel_name = op.kernel_name.data
                grid_expr = op.grid_expr.data
                args_expr = op.args_expr.data if op.args_expr is not None else "()"

                # For standard launcher, compute grid inline
                if grid_expr != "grid":
                    e.line(f"grid = {grid_expr}")

                e.line(
                    f"ct.launch(stream, grid, {kernel_name}, "
                    f"{args_expr})"
                )

            elif isinstance(op, ForLoopOp):
                count = op.count.data
                e.line(f"for _step in range({count}):")
                with e.indent():
                    # Walk loop body
                    loop_block = list(op.body.blocks)[0]
                    for loop_op in loop_block.ops:
                        if isinstance(loop_op, LaunchOp):
                            kernel_name = loop_op.kernel_name.data
                            args_expr = loop_op.args_expr.data if loop_op.args_expr is not None else "()"
                            e.line(
                                f"ct.launch(stream, grid, {kernel_name}, "
                                f"{args_expr})"
                            )


# -------------------------------------------------------------------- #
# Header emitter
# -------------------------------------------------------------------- #


def _emit_header(e: CodeEmitter, kernel: KernelOp) -> None:
    """Emit module header: docstring, imports, constants."""
    name = kernel.kernel_name.data
    e.line(f'"""cuTile kernel for {name} stencil (auto-generated)."""')
    e.blank()
    e.line("import cuda.tile as ct")
    e.line("import cupy as cp")

    # Emit constants if present
    if kernel.constants is not None:
        items = kernel.constants.data
        if items:
            e.blank()
            e.line("# Captured constants from stencil definition scope")
            # Constants are stored as [key1, val1, key2, val2, ...]
            i = 0
            while i < len(items):
                key = items[i].data
                val = items[i + 1].data
                e.line(f"{key} = {val}")
                i += 2

    if kernel.kernel_constants is not None:
        values = [a.data for a in kernel.kernel_constants.data]
        n = _pow2_at_least(len(values))
        e.blank()
        e.line("# float64 constants delivered through a device array: cuTile rounds")
        e.line("# Python float scalars to float32 even inside float64 kernels.")
        e.line(f"_KERNEL_CONSTANTS = [{', '.join(values)}]")
        e.line("_KERNEL_CONSTANTS_DEV = {}")
        e.blank()
        e.blank()
        e.line("def _kernel_constants():")
        with e.indent():
            e.line('"""Per-device float64 array holding the stencil constants."""')
            e.line("dev = cp.cuda.Device().id")
            e.line("arr = _KERNEL_CONSTANTS_DEV.get(dev)")
            e.line("if arr is None:")
            with e.indent():
                e.line(
                    f"padded = _KERNEL_CONSTANTS + [0.0] * ({n} - len(_KERNEL_CONSTANTS))"
                )
                e.line("arr = cp.asarray(padded, dtype=cp.float64)")
                e.line("_KERNEL_CONSTANTS_DEV[dev] = arr")
            e.line("return arr")


def _pow2_at_least(n: int) -> int:
    """Smallest power of two >= n (and >= 1)."""
    return 1 << max(0, (n - 1).bit_length())


# -------------------------------------------------------------------- #
# Public API
# -------------------------------------------------------------------- #


def emit_python(module: ModuleOp) -> str:
    """Walk cutile_target IR and emit Python source code.

    Parameters
    ----------
    module:
        ``ModuleOp`` containing ``cutile_target`` dialect ops
        (``KernelOp``, ``HostProgramOp``, etc.).

    Returns
    -------
    str
        Complete Python source string.
    """
    e = CodeEmitter()

    # Find KernelOp and HostProgramOp
    kernel_op: KernelOp | None = None
    host_op: HostProgramOp | None = None

    for op in module.body.ops:
        if isinstance(op, KernelOp):
            kernel_op = op
        elif isinstance(op, HostProgramOp):
            host_op = op

    if kernel_op is None:
        raise ValueError("No KernelOp found in target IR module")
    if host_op is None:
        raise ValueError("No HostProgramOp found in target IR module")

    # Header
    _emit_header(e, kernel_op)
    e.blank()
    e.line("ConstInt = ct.Constant[int]")
    e.blank()
    e.blank()
    _emit_fit_tiles_helper(e)
    e.blank()

    # Scratch-buffer cache for temporal launchers
    host_ops = list(list(host_op.body.blocks)[0].ops)
    if any(isinstance(op, ForLoopOp) for op in host_ops):
        _emit_temporal_buffers_helper(e)
        e.blank()

    # Kernel
    _emit_kernel(e, kernel_op)

    # Boundary (if present in module attributes)
    if "boundary" in module.attributes:
        bc_items = module.attributes["boundary"].data
        bc_type = bc_items[0].data
        bc_value = float(bc_items[1].data) if len(bc_items) > 1 else 0.0
        stencil_name = module.attributes["stencil_name"].data
        stencil_ndim = module.attributes["stencil_ndim"].data
        stencil_halo = tuple(a.data for a in module.attributes["stencil_halo"].data)
        _emit_boundary(e, stencil_name, stencil_ndim, stencil_halo, bc_type, bc_value)

    # Host launcher
    _emit_host(e, host_op)

    return e.render()
