"""cuTile Target Dialect (Dialect 3) — GPU kernel and host orchestration IR.

This dialect represents cuTile GPU operations (kernel, slice, load, store,
launch) and host-side orchestration (alloc, sync, loops, boundary).  It is
the last IR stage before Python source emission.
"""

from __future__ import annotations

from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    var_operand_def,
    result_def,
    region_def,
    prop_def,
    traits_def,
)
from xdsl.traits import IsTerminator, NoTerminator
from xdsl.dialects.builtin import StringAttr, IntAttr, ArrayAttr

# ---------------------------------------------------------------------------
# Device IR operations (inside a kernel body)
# ---------------------------------------------------------------------------


@irdl_op_definition
class KernelOp(IRDLOperation):
    """A ``@ct.kernel`` function.

    The *body* region contains device-side operations terminated by a
    ``ReturnOp``.  ``params`` captures the kernel's tensor parameters as
    SSA values produced by the enclosing host program.
    """

    name = "cutile.kernel"

    kernel_name = prop_def(StringAttr)
    tile_shape = prop_def(ArrayAttr)
    halo = prop_def(ArrayAttr)

    params = var_operand_def()
    body = region_def()


@irdl_op_definition
class SliceOp(IRDLOperation):
    """``tensor.slice(axis=N, start=S, stop=E)`` — extract a sub-range.

    *start* and *stop* are expression strings (e.g. ``"bid*T"``), kept as
    ``StringAttr`` so that the emitter can paste them verbatim into generated
    Python/CUDA source.
    """

    name = "cutile.slice"

    axis = prop_def(IntAttr)
    start = prop_def(StringAttr)
    stop = prop_def(StringAttr)

    input = operand_def()
    result = result_def()


@irdl_op_definition
class LoadOp(IRDLOperation):
    """``ct.load(tensor)`` — load a tile from global memory."""

    name = "cutile.load"

    tensor = operand_def()
    result = result_def()


@irdl_op_definition
class StoreOp(IRDLOperation):
    """``ct.store(tile, value)`` — store a value into a tile (side-effect)."""

    name = "cutile.store"

    tile = operand_def()
    value = operand_def()


@irdl_op_definition
class BidOp(IRDLOperation):
    """``ct.bid(axis)`` — read the block index for *axis*."""

    name = "cutile.bid"

    axis = prop_def(IntAttr)

    result = result_def()


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """Terminate a ``KernelOp`` body."""

    name = "cutile.return"

    traits = traits_def(IsTerminator())


# ---------------------------------------------------------------------------
# Host IR operations (launcher / orchestration)
# ---------------------------------------------------------------------------


@irdl_op_definition
class HostProgramOp(IRDLOperation):
    """Top-level host-side launcher function.

    The *body* region contains host operations (alloc, launch, sync, ...).
    """

    name = "cutile.host_program"

    program_name = prop_def(StringAttr)
    body = region_def()


@irdl_op_definition
class AllocOp(IRDLOperation):
    """Allocate a GPU buffer."""

    name = "cutile.alloc"

    var_name = prop_def(StringAttr)
    dtype = prop_def(StringAttr)

    result = result_def()


@irdl_op_definition
class LaunchOp(IRDLOperation):
    """Launch a compiled kernel."""

    name = "cutile.launch"

    kernel_name = prop_def(StringAttr)
    grid_expr = prop_def(StringAttr)

    args = var_operand_def()


@irdl_op_definition
class SyncOp(IRDLOperation):
    """Stream synchronisation barrier."""

    name = "cutile.sync"


@irdl_op_definition
class SwapBuffersOp(IRDLOperation):
    """Swap two double-buffer pointers."""

    name = "cutile.swap_buffers"

    buf_a = operand_def()
    buf_b = operand_def()


@irdl_op_definition
class ForLoopOp(IRDLOperation):
    """Host-side counted ``for`` loop (e.g. temporal stepping)."""

    name = "cutile.for_loop"

    count = prop_def(IntAttr)
    body = region_def()


@irdl_op_definition
class ForEachGpuOp(IRDLOperation):
    """Iterate over available GPUs."""

    name = "cutile.for_each_gpu"

    num_gpus = prop_def(IntAttr)
    body = region_def()


@irdl_op_definition
class AsyncOp(IRDLOperation):
    """Async execution block (computation / communication overlap)."""

    name = "cutile.async"

    body = region_def()


@irdl_op_definition
class BoundaryApplyOp(IRDLOperation):
    """Apply boundary conditions on the host side."""

    name = "cutile.boundary_apply"

    bc_type = prop_def(StringAttr)

    buffer = operand_def()


@irdl_op_definition
class LayoutCastOp(IRDLOperation):
    """Convert between data layouts (e.g. flat <-> bricked)."""

    name = "cutile.layout_cast"

    from_layout = prop_def(StringAttr)
    to_layout = prop_def(StringAttr)

    input = operand_def()
    result = result_def()


# ---------------------------------------------------------------------------
# Dialect registration
# ---------------------------------------------------------------------------

CutileTargetDialect = Dialect(
    "cutile",
    [
        # Device ops
        KernelOp,
        SliceOp,
        LoadOp,
        StoreOp,
        BidOp,
        ReturnOp,
        # Host ops
        HostProgramOp,
        AllocOp,
        LaunchOp,
        SyncOp,
        SwapBuffersOp,
        ForLoopOp,
        ForEachGpuOp,
        AsyncOp,
        BoundaryApplyOp,
        LayoutCastOp,
    ],
    [],
)
