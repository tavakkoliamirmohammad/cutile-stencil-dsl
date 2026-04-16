"""Communication dialect for multi-GPU stencil orchestration.

The dialect models four kinds of host-side actions that show up in
multi-GPU stencil step bodies:

1. **State retrieval**: :class:`GetHaloStateOp` returns the cached
   per-(num_gpus, halo_width) (streams, ev_right, ev_left) tuple.
2. **Cross-device synchronisation**: :class:`WaitNeighborEventOp`
   makes a per-GPU stream wait for the halo arrival event recorded
   by a neighbour during the previous step.
3. **Per-GPU compute launch**: :class:`LaunchPerGpuOp` launches a
   compiled stencil kernel on each GPU's partition.
4. **Halo exchange**: :class:`HaloExchangePairOp` (async P2P send +
   record events) and the original :class:`SendHaloOp` /
   :class:`RecvHaloOp` / :class:`ExchangeHalosOp` for synchronous
   variants.

The Python emitter walks ops of these kinds and turns each into a
short snippet of cuTile/CuPy code that calls into
``cutile.runtime.multigpu_helpers``. Generated code is therefore a
straightforward unrolling of the IR — no hand-written ``e.line`` text
spaghetti.
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


# ---------------------------------------------------------------------------
# Original (synchronous) ops — kept for compatibility.
# ---------------------------------------------------------------------------


@irdl_op_definition
class SendHaloOp(IRDLOperation):
    """Send halo data from a buffer to a destination GPU."""

    name = "comm.send_halo"

    dst_gpu = prop_def(IntAttr)
    axis = prop_def(IntAttr)
    side = prop_def(StringAttr)
    halo_width = prop_def(IntAttr)

    buffer = operand_def(AnyAttr())


@irdl_op_definition
class RecvHaloOp(IRDLOperation):
    """Receive halo data into a buffer from a source GPU."""

    name = "comm.recv_halo"

    src_gpu = prop_def(IntAttr)
    axis = prop_def(IntAttr)
    side = prop_def(StringAttr)
    halo_width = prop_def(IntAttr)

    buffer = operand_def(AnyAttr())


@irdl_op_definition
class ExchangeHalosOp(IRDLOperation):
    """Exchange halo regions across all partition buffers along an axis.

    Synchronous variant — emitted as a call to
    ``multigpu_helpers.exchange_halos``.
    """

    name = "comm.exchange_halos"

    halo_width = prop_def(IntAttr)
    axis = prop_def(IntAttr)

    partition_buffers = var_operand_def(AnyAttr())


@irdl_op_definition
class BarrierOp(IRDLOperation):
    """Global synchronisation barrier across GPUs."""

    name = "comm.barrier"


# ---------------------------------------------------------------------------
# Async / event-chained ops used by the multi-GPU step body.
# ---------------------------------------------------------------------------


@irdl_op_definition
class GetHaloStateOp(IRDLOperation):
    """Look up the lazy (streams, ev_right, ev_left) cache.

    Lowered to ``streams, ev_right, ev_left = get_halo_state(...)``.
    """

    name = "comm.get_halo_state"

    num_gpus = prop_def(IntAttr)
    halo_width = prop_def(IntAttr)


@irdl_op_definition
class WaitNeighborEventOp(IRDLOperation):
    """Wait on the cross-device event for a halo arriving INTO this GPU.

    ``side="right"`` waits for the event recorded by the GPU on the
    *left* (gpu_id - 1); ``side="left"`` waits for the GPU on the *right*
    (gpu_id + 1). Skipped at edges where no neighbour exists.
    """

    name = "comm.wait_neighbor_event"

    side = prop_def(StringAttr)


@irdl_op_definition
class LaunchPerGpuOp(IRDLOperation):
    """Per-GPU launch of a previously-emitted single-GPU kernel launcher.

    Lowered to ``launch_<kernel_name>(p_in[gid], p_out[gid])`` inside the
    enclosing ``cutile.for_each_gpu`` body.
    """

    name = "comm.launch_per_gpu"

    kernel_name = prop_def(StringAttr)


@irdl_op_definition
class HaloExchangePairOp(IRDLOperation):
    """Async P2P halo exchange between adjacent GPUs ``i`` and ``i+1``.

    Lowered to ``halo_send_pair(p_out, i, i+1, halo_width, axis,
    streams, ev_right, ev_left)``. Records the cross-device events
    that the next step's :class:`WaitNeighborEventOp`s consume.
    """

    name = "comm.halo_exchange_pair"

    halo_width = prop_def(IntAttr)
    axis = prop_def(IntAttr)


CommDialect = Dialect(
    "comm",
    [
        # Original ops
        SendHaloOp,
        RecvHaloOp,
        ExchangeHalosOp,
        BarrierOp,
        # Async / event-chained ops
        GetHaloStateOp,
        WaitNeighborEventOp,
        LaunchPerGpuOp,
        HaloExchangePairOp,
    ],
    [],
)
