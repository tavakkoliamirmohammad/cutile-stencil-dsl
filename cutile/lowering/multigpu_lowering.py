"""Multi-GPU lowering: stencil IR -> comm-dialect host programs -> Python.

The single-GPU pipeline (``cutile_stencil`` -> ``cutile_target`` ->
Python) handles the kernel and per-GPU launcher unchanged. This module
adds the host-side multi-GPU orchestration as a sequence of
``HostProgramOp`` ops whose bodies use the ``comm`` dialect:

* ``setup_multigpu_<name>`` — calls
  :func:`cutile.runtime.multigpu_helpers.decompose_domain` and allocates
  output buffers per GPU.
* ``step_multigpu_<name>`` — *the dynamic part*. Built from
  ``comm.GetHaloStateOp``, ``cutile_target.ForEachGpuOp`` containing
  ``comm.WaitNeighborEventOp`` + ``comm.LaunchPerGpuOp``, and
  ``cutile_target.ForLoopOp`` containing ``comm.HaloExchangePairOp``.
* ``gather_multigpu_<name>`` — calls
  :func:`cutile.runtime.multigpu_helpers.gather_results`.
* ``launch_multigpu_<name>`` — convenience wrapper.

The emitter walks each op type and produces a few lines of Python that
delegate to the runtime helpers. The result is structurally the same
code that the old text emitter produced, but every line traces back to
an op in the IR — no spaghetti ``e.line(...)`` text generation in the
multi-GPU path.
"""

from __future__ import annotations

from xdsl.dialects.builtin import IntAttr, ModuleOp, StringAttr
from xdsl.ir import Block, Region

from cutile.dialects.comm.dialect import (
    GetHaloStateOp,
    HaloExchangePairOp,
    LaunchPerGpuOp,
    WaitNeighborEventOp,
)
from cutile.dialects.cutile_target.dialect import (
    ForEachGpuOp,
    ForLoopOp,
    HostProgramOp,
)
from cutile.lowering.emitter import CodeEmitter


# ---------------------------------------------------------------------------
# IR construction
# ---------------------------------------------------------------------------


def _build_step_body(
    kernel_name: str,
    num_gpus: int,
    halo_width: int,
    axis: int,
) -> Region:
    """Build the body region of a ``step_multigpu_<name>`` HostProgramOp.

    Body:
        comm.get_halo_state {num_gpus, halo_width}
        cutile.for_each_gpu {num_gpus} {
            comm.wait_neighbor_event {side="right"}
            comm.wait_neighbor_event {side="left"}
            comm.launch_per_gpu {kernel_name}
        }
        cutile.for_loop {count=num_gpus-1} {
            comm.halo_exchange_pair {halo_width, axis}
        }
    """
    foreach_block = Block([
        WaitNeighborEventOp.create(properties={"side": StringAttr("right")}),
        WaitNeighborEventOp.create(properties={"side": StringAttr("left")}),
        LaunchPerGpuOp.create(
            properties={"kernel_name": StringAttr(kernel_name)},
        ),
    ])
    foreach_op = ForEachGpuOp.create(
        properties={"num_gpus": IntAttr(num_gpus)},
        regions=[Region(foreach_block)],
    )

    pair_block = Block([
        HaloExchangePairOp.create(
            properties={
                "halo_width": IntAttr(halo_width),
                "axis": IntAttr(axis),
            },
        ),
    ])
    pair_loop = ForLoopOp.create(
        properties={"count": IntAttr(max(num_gpus - 1, 0))},
        regions=[Region(pair_block)],
    )

    state_op = GetHaloStateOp.create(
        properties={
            "num_gpus": IntAttr(num_gpus),
            "halo_width": IntAttr(halo_width),
        },
    )

    return Region(Block([state_op, foreach_op, pair_loop]))


def build_multigpu_host_programs(
    kernel_name: str,
    num_gpus: int,
    halo_width: int,
    split_axis: int,
) -> list[HostProgramOp]:
    """Build the four multi-GPU HostProgramOps for *kernel_name*.

    Returns the ops in source order: setup, step, gather, launch. The
    emitter walks them in this order to produce Python.
    """
    # setup: pure helper call — no body ops needed; preamble carries
    # the function signature suffix and constants.
    setup = HostProgramOp.create(
        properties={
            "program_name": StringAttr(
                f"setup_multigpu_{kernel_name}(u_in, num_gpus={num_gpus})"
            ),
            "preamble": StringAttr(
                f'"""Decompose domain and allocate per-GPU buffers (call ONCE)."""\n'
                f"halo_width = {halo_width}\n"
                f"split_axis = {split_axis}\n"
                f"partitions_in = decompose_domain(u_in, num_gpus, split_axis, halo_width)\n"
                f"partitions_out = []\n"
                f"for gpu_id in range(num_gpus):\n"
                f"    with cp.cuda.Device(gpu_id):\n"
                f"        partitions_out.append(cp.zeros_like(partitions_in[gpu_id]))\n"
                f"return partitions_in, partitions_out"
            ),
        },
        regions=[Region(Block([]))],
    )

    step_body = _build_step_body(kernel_name, num_gpus, halo_width, split_axis)
    step = HostProgramOp.create(
        properties={
            "program_name": StringAttr(
                f"step_multigpu_{kernel_name}(partitions_in, partitions_out, "
                f"num_gpus={num_gpus})"
            ),
            "preamble": StringAttr(
                f'"""Execute one timestep on all GPUs + async halo exchange."""\n'
                f"halo_width = {halo_width}\n"
                f"split_axis = {split_axis}"
            ),
        },
        regions=[step_body],
    )

    gather = HostProgramOp.create(
        properties={
            "program_name": StringAttr(
                f"gather_multigpu_{kernel_name}(u_out, partitions, num_gpus={num_gpus})"
            ),
            "preamble": StringAttr(
                f'"""Gather per-GPU partition interiors into the full output array."""\n'
                f"halo_width = {halo_width}\n"
                f"split_axis = {split_axis}\n"
                f"gather_results(u_out, partitions, num_gpus, split_axis, halo_width)"
            ),
        },
        regions=[Region(Block([]))],
    )

    launch = HostProgramOp.create(
        properties={
            "program_name": StringAttr(
                f"launch_multigpu_{kernel_name}(u_in, u_out, num_gpus={num_gpus})"
            ),
            "preamble": StringAttr(
                f'"""Convenience: decompose + step + gather in one call."""\n'
                f"p_in, p_out = setup_multigpu_{kernel_name}(u_in, num_gpus)\n"
                f"step_multigpu_{kernel_name}(p_in, p_out, num_gpus)\n"
                f"gather_multigpu_{kernel_name}(u_out, p_out, num_gpus)"
            ),
        },
        regions=[Region(Block([]))],
    )

    return [setup, step, gather, launch]


# ---------------------------------------------------------------------------
# IR -> Python emitter
# ---------------------------------------------------------------------------


_HELPER_IMPORT = (
    "from cutile.runtime.multigpu_helpers import (\n"
    "    decompose_domain,\n"
    "    gather_results,\n"
    "    get_halo_state,\n"
    "    halo_send_pair,\n"
    "    exchange_halos,\n"
    ")"
)


def _emit_step_body(e: CodeEmitter, region: Region, kernel_name: str) -> None:
    """Walk a step-body region and emit Python for each comm/cutile op."""
    block = list(region.blocks)[0]
    for op in block.ops:
        if isinstance(op, GetHaloStateOp):
            e.line("streams, ev_right, ev_left = get_halo_state("
                   "num_gpus, halo_width)")
        elif isinstance(op, ForEachGpuOp):
            e.blank()
            e.line("for gpu_id in range(num_gpus):")
            with e.indent():
                e.line("with cp.cuda.Device(gpu_id):")
                with e.indent():
                    inner = list(op.body.blocks)[0]
                    for inner_op in inner.ops:
                        if isinstance(inner_op, WaitNeighborEventOp):
                            side = inner_op.side.data
                            if side == "right":
                                e.line("if gpu_id - 1 in ev_right:")
                                with e.indent():
                                    e.line("streams[gpu_id].wait_event("
                                           "ev_right[gpu_id - 1])")
                            elif side == "left":
                                e.line("if gpu_id + 1 in ev_left:")
                                with e.indent():
                                    e.line("streams[gpu_id].wait_event("
                                           "ev_left[gpu_id + 1])")
                        elif isinstance(inner_op, LaunchPerGpuOp):
                            kn = inner_op.kernel_name.data
                            e.line(f"launch_{kn}("
                                   "partitions_in[gpu_id], "
                                   "partitions_out[gpu_id])")
                        else:
                            raise ValueError(
                                f"Unsupported op in for_each_gpu body: {inner_op}"
                            )
        elif isinstance(op, ForLoopOp):
            e.blank()
            count = op.count.data
            e.line(f"for i in range({count}):")
            with e.indent():
                inner = list(op.body.blocks)[0]
                for inner_op in inner.ops:
                    if isinstance(inner_op, HaloExchangePairOp):
                        e.line("halo_send_pair(partitions_out, i, i + 1, "
                               "halo_width, split_axis, "
                               "streams, ev_right, ev_left)")
                    else:
                        raise ValueError(
                            f"Unsupported op in for_loop body: {inner_op}"
                        )
        else:
            raise ValueError(f"Unsupported op in step body: {op}")


def emit_multigpu_host_programs(host_programs: list[HostProgramOp]) -> str:
    """Walk the four multi-GPU HostProgramOps and emit Python source."""
    e = CodeEmitter()
    e.blank()
    e.blank()
    for line in _HELPER_IMPORT.split("\n"):
        e.line(line)

    for host in host_programs:
        e.blank()
        e.blank()
        sig = host.program_name.data
        e.line(f"def {sig}:")
        with e.indent():
            preamble = host.preamble.data if host.preamble is not None else ""
            for pline in preamble.split("\n"):
                e.line(pline)
            # The step program has IR ops in its body; walk them. Other
            # programs are pure-preamble (helper-call wrappers) with empty
            # body regions.
            block = list(host.body.blocks)[0]
            if list(block.ops):
                kernel_name = sig.split("step_multigpu_", 1)[1].split("(", 1)[0]
                _emit_step_body(e, host.body, kernel_name)

    return e.render()
