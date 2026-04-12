"""Communication dialect for multi-GPU halo exchange operations.

Location-agnostic: ops can live in host or device IR.
"""

from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    var_operand_def,
    prop_def,
    AnyAttr,
)
from xdsl.dialects.builtin import IntAttr, StringAttr


@irdl_op_definition
class SendHaloOp(IRDLOperation):
    """Send halo data from a buffer to a destination GPU.

    The halo region is identified by axis, side, and width.
    """

    name = "comm.send_halo"

    dst_gpu = prop_def(IntAttr)
    axis = prop_def(IntAttr)
    side = prop_def(StringAttr)
    halo_width = prop_def(IntAttr)

    buffer = operand_def(AnyAttr())


@irdl_op_definition
class RecvHaloOp(IRDLOperation):
    """Receive halo data into a buffer from a source GPU.

    The halo region is identified by axis, side, and width.
    """

    name = "comm.recv_halo"

    src_gpu = prop_def(IntAttr)
    axis = prop_def(IntAttr)
    side = prop_def(StringAttr)
    halo_width = prop_def(IntAttr)

    buffer = operand_def(AnyAttr())


@irdl_op_definition
class ExchangeHalosOp(IRDLOperation):
    """Exchange halo regions across all partition buffers along an axis."""

    name = "comm.exchange_halos"

    halo_width = prop_def(IntAttr)
    axis = prop_def(IntAttr)

    partition_buffers = var_operand_def(AnyAttr())


@irdl_op_definition
class BarrierOp(IRDLOperation):
    """Global synchronization barrier across GPUs."""

    name = "comm.barrier"


CommDialect = Dialect(
    "comm", [SendHaloOp, RecvHaloOp, ExchangeHalosOp, BarrierOp], []
)
