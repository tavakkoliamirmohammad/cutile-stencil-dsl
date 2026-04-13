"""Target-to-Python emitter: Dialect 3 (cutile_target) IR to Python source.

Walks a ``ModuleOp`` containing ``cutile_target`` dialect ops and emits
the equivalent cuTile Python source code.  The emitter is purely
mechanical -- it reads properties from each op and formats them into
the Python API strings.
"""

from __future__ import annotations

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
    expression = kernel.expression.data
    num_inputs = len(input_names)
    multi_input = num_inputs > 1

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

    e.line("@ct.kernel")
    e.line(
        f"def {name}_kernel({input_params}, output, "
        f"{tile_const_params}, {halo_const_params}):"
    )

    with e.indent():
        # 1. Block indices
        for i, bv in enumerate(bid_vars):
            e.line(f"{bv} = ct.bid({i})")

        # 2. Interior sizes
        first_arr = input_names[0]
        for d in range(ndim):
            e.line(f"{n_vars[d]} = {first_arr}.shape[{d}] - 2 * {halo_vars[d]}")

        # Walk kernel body to build SSA -> expression mapping
        body_block = list(kernel.body.blocks)[0]
        val_to_expr: dict = {}

        # Map block args to input names + output
        for i in range(num_inputs):
            val_to_expr[body_block.args[i]] = input_names[i]
        val_to_expr[body_block.args[num_inputs]] = "output"

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
                    e.line(f"{var_name} = {chain_expr}")

            elif isinstance(op, LoadOp):
                load_ops.append(op)

        # 4. Emit all loads
        idx_tuple = ", ".join(bid_vars)
        shape_tuple = ", ".join(tile_vars)
        if ndim == 1:
            idx_arg = f"({idx_tuple},)"
            shape_arg = f"({shape_tuple},)"
        else:
            idx_arg = f"({idx_tuple})"
            shape_arg = f"({shape_tuple})"

        for load in load_ops:
            view_name = (
                load.view_name.data if load.view_name is not None else "view"
            )
            e.line(
                f"t_{view_name} = ct.load({view_name}, "
                f"index={idx_arg}, shape={shape_arg})"
            )

        # 5. Expression
        e.blank()
        e.line(f"result = {expression}")
        e.blank()

        # 6. Store
        e.line(f"ct.store(out, index={idx_arg}, tile=result)")


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
