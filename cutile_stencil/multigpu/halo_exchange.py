"""Halo exchange utilities for multi-GPU stencil computations.

Provides P2P halo exchange with contiguous buffer packing for non-contiguous
2D/3D halo regions, and compute/communication overlap via CUDA streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class HaloExchangePlan:
    """Pre-computed plan for halo exchange between GPU partitions.

    Describes which slices to pack/unpack for each neighbor pair.
    """

    rank: int
    num_ranks: int
    split_axis: int
    halo_width: int  # halo along split axis
    local_shape: Tuple[int, ...]  # full local array shape (with halo)

    # Neighbor ranks (None = boundary)
    left_neighbor: Optional[int]
    right_neighbor: Optional[int]

    # Whether to wrap around (periodic BC)
    periodic: bool


def create_exchange_plan(
    decomposition,
    rank: int,
    local_shape: Tuple[int, ...],
    periodic: bool = False,
) -> HaloExchangePlan:
    """Create a halo exchange plan for a given rank."""
    part = decomposition.partitions[rank]
    split = decomposition.split_axis
    hw = part.halo_widths[split]

    left = part.neighbors.get("left")
    right = part.neighbors.get("right")

    # For periodic: wrap neighbors
    if periodic:
        if left is None:
            left = decomposition.num_gpus - 1
        if right is None:
            right = 0

    return HaloExchangePlan(
        rank=rank,
        num_ranks=decomposition.num_gpus,
        split_axis=split,
        halo_width=hw,
        local_shape=local_shape,
        left_neighbor=left,
        right_neighbor=right,
        periodic=periodic,
    )


def emit_halo_exchange_function(plan: HaloExchangePlan) -> str:
    """Generate Python code for P2P halo exchange with contiguous packing.

    For 1D splits, the halo is already contiguous (a slice along the split axis).
    For 2D+ splits, halo columns/planes are non-contiguous and must be packed
    into a contiguous buffer before P2P copy, then unpacked on the receiving side.

    Returns a string of Python code defining:
    - pack_halo_left/right(u) -> contiguous buffer
    - unpack_halo_left/right(u, buf)
    - exchange_halos(u_local, u_peers) -> performs full exchange
    """
    from cutile_stencil.codegen.emitter import CodeEmitter

    e = CodeEmitter()
    ndim = len(plan.local_shape)
    split = plan.split_axis
    hw = plan.halo_width

    e.line("import cupy as cp")
    e.blank()

    # Helper to build slice notation for arbitrary ndim
    def _slice_str(dim_slices):
        """Build Python slice string from per-dim slice strings."""
        return ", ".join(dim_slices)

    # Pack left boundary (rightmost interior strip -> send to left neighbor)
    # This is the `hw` cells at the low end of our interior
    e.line("def pack_halo_to_left(u):")
    with e.indent():
        e.line(
            '"""Pack interior cells near left boundary into contiguous buffer."""'
        )
        slices = [":" for _ in range(ndim)]
        slices[split] = f"{hw}:2*{hw}"
        e.line(f"return cp.ascontiguousarray(u[{_slice_str(slices)}])")

    e.blank()

    # Pack right boundary (leftmost interior strip near right edge -> send right)
    e.line("def pack_halo_to_right(u):")
    with e.indent():
        e.line(
            '"""Pack interior cells near right boundary into contiguous buffer."""'
        )
        slices = [":" for _ in range(ndim)]
        slices[split] = f"-2*{hw}:-{hw}"
        e.line(f"return cp.ascontiguousarray(u[{_slice_str(slices)}])")

    e.blank()

    # Unpack received halo into left ghost cells
    e.line("def unpack_halo_from_left(u, buf):")
    with e.indent():
        e.line('"""Unpack received buffer into left halo cells."""')
        slices = [":" for _ in range(ndim)]
        slices[split] = f":{hw}"
        e.line(f"u[{_slice_str(slices)}] = buf")

    e.blank()

    # Unpack received halo into right ghost cells
    e.line("def unpack_halo_from_right(u, buf):")
    with e.indent():
        e.line('"""Unpack received buffer into right halo cells."""')
        slices = [":" for _ in range(ndim)]
        slices[split] = f"-{hw}:"
        e.line(f"u[{_slice_str(slices)}] = buf")

    e.blank()

    # Main exchange function with P2P and optional overlap
    e.line("def exchange_halos(u_local, u_peers, rank, comm_stream=None):")
    with e.indent():
        e.line('"""Exchange halos with neighbors via P2P copy.')
        e.line("")
        e.line("Parameters")
        e.line("----------")
        e.line("u_local : cupy array on current device")
        e.line(
            "u_peers : list of cupy arrays (one per GPU, on their respective devices)"
        )
        e.line("rank : int, this GPU's rank")
        e.line("comm_stream : cupy.cuda.Stream, optional")
        e.line(
            "    If provided, exchange runs on this stream (for overlap with compute)"
        )
        e.line('"""')
        e.line("if comm_stream is None:")
        with e.indent():
            e.line("comm_stream = cp.cuda.get_current_stream()")
        e.blank()
        e.line("with comm_stream:")
        with e.indent():
            # Send to right, receive from left
            if plan.right_neighbor is not None:
                e.line(f"# Send right boundary to rank {plan.right_neighbor}")
                e.line("send_right = pack_halo_to_right(u_local)")
                e.line(
                    "# P2P copy: our right interior -> neighbor's left halo"
                )
                right_slices = [":" for _ in range(ndim)]
                right_slices[split] = f":{hw}"
                e.line(f"with cp.cuda.Device({plan.right_neighbor}):")
                with e.indent():
                    e.line(
                        f"u_peers[{plan.right_neighbor}]"
                        f"[{_slice_str(right_slices)}]"
                        " = cp.asarray(cp.asnumpy(send_right))"
                    )

            if plan.left_neighbor is not None:
                e.line(f"# Send left boundary to rank {plan.left_neighbor}")
                e.line("send_left = pack_halo_to_left(u_local)")
                e.line(
                    "# P2P copy: our left interior -> neighbor's right halo"
                )
                left_slices = [":" for _ in range(ndim)]
                left_slices[split] = f"-{hw}:"
                e.line(f"with cp.cuda.Device({plan.left_neighbor}):")
                with e.indent():
                    e.line(
                        f"u_peers[{plan.left_neighbor}]"
                        f"[{_slice_str(left_slices)}]"
                        " = cp.asarray(cp.asnumpy(send_left))"
                    )

    e.blank()

    # Overlapped version
    e.line(
        "def step_with_overlap(u_local, u_peers, rank, launch_fn, output):"
    )
    with e.indent():
        e.line(
            '"""Execute stencil + halo exchange with compute/communication overlap.'
        )
        e.line("")
        e.line("1. Launch halo exchange on comm_stream")
        e.line(
            "2. Launch stencil kernel on compute_stream (can start interior immediately)"
        )
        e.line("3. Sync both streams before next step")
        e.line('"""')
        e.line("compute_stream = cp.cuda.Stream(non_blocking=True)")
        e.line("comm_stream = cp.cuda.Stream(non_blocking=True)")
        e.blank()
        e.line("# Start halo exchange on comm stream")
        e.line(
            "exchange_halos(u_local, u_peers, rank, comm_stream=comm_stream)"
        )
        e.blank()
        e.line(
            "# Launch stencil on compute stream (interior doesn't need halos)"
        )
        e.line("with compute_stream:")
        with e.indent():
            e.line("launch_fn(u_local, output)")
        e.blank()
        e.line("# Wait for both to complete")
        e.line("compute_stream.synchronize()")
        e.line("comm_stream.synchronize()")

    return e.render()
