"""Runtime helpers for multi-GPU stencil execution.

These functions are stencil-agnostic and shared by every multi-GPU
launcher emitted by the cuTile pipeline. Moving them out of the text
emitter keeps generated code small (just the per-stencil step function)
and centralises the GPU-orchestration logic in a single, testable spot.

Two decomposition families:

**1D split** (the original path) — partition along a single axis,
neighbours form a chain, each pair exchanges one slab of halo per
direction.

  - :func:`decompose_domain`, :func:`gather_results`,
    :func:`get_halo_state`, :func:`halo_send_pair`,
    :func:`exchange_halos`

**Cartesian topology** — partition along two or more axes onto a
logical N-D process grid (e.g. 2x2 onto 4 GPUs). Per-GPU halo bandwidth
shrinks by ``sqrt(num_gpus)`` versus a 1D split because each GPU
exchanges with up to ``2*ndim_topo`` neighbours instead of two end-of-
chain neighbours.

  - :func:`cartesian_decompose`, :func:`cartesian_gather`,
    :func:`get_cartesian_halo_state`, :func:`cartesian_halo_send_axis`

All helpers enable peer access between every GPU pair, so subsequent
halo memcpys take the P2P path automatically.
"""

from __future__ import annotations

import cupy as cp


# Per-(num_gpus, halo_width) cache of streams + cross-device events.
# Keyed independently of the stencil so multiple compiled stencils with
# the same multi-GPU shape share streams and events.
_HALO_STATE_CACHE: dict = {}


def decompose_domain(u, num_gpus: int, split_axis: int, halo_width: int):
    """Split *u* into *num_gpus* halo-overlapped sub-domains across GPUs.

    Each partition lives on its own device. Peer access is enabled
    between every pair of GPUs so subsequent halo memcpys can use the
    P2P path. Returns a list of ``cupy.ndarray`` views (one per GPU).
    """
    shape = u.shape
    axis_len = shape[split_axis]
    interior_total = axis_len - 2 * halo_width
    base_chunk = interior_total // num_gpus
    remainder = interior_total % num_gpus

    for i in range(num_gpus):
        for j in range(num_gpus):
            if i != j:
                try:
                    cp.cuda.runtime.deviceEnablePeerAccess(j)
                except cp.cuda.runtime.CUDARuntimeError:
                    pass  # already enabled or not supported on this pair

    partitions = []
    offset = 0
    for gpu_id in range(num_gpus):
        chunk = base_chunk + (1 if gpu_id < remainder else 0)
        start = offset
        stop = offset + chunk + 2 * halo_width
        slices = [slice(None)] * len(shape)
        slices[split_axis] = slice(start, stop)
        sub = u[tuple(slices)]
        with cp.cuda.Device(gpu_id):
            partitions.append(sub.copy())
        offset += chunk
    return partitions


def gather_results(u_out, partitions, num_gpus: int, split_axis: int,
                   halo_width: int) -> None:
    """Gather per-GPU partition interiors back into the full ``u_out``."""
    shape = u_out.shape
    interior_total = shape[split_axis] - 2 * halo_width
    base_chunk = interior_total // num_gpus
    remainder = interior_total % num_gpus

    offset = 0
    for gpu_id in range(num_gpus):
        chunk = base_chunk + (1 if gpu_id < remainder else 0)
        src_sl = [slice(None)] * len(shape)
        src_sl[split_axis] = slice(halo_width, halo_width + chunk)
        dst_sl = [slice(None)] * len(shape)
        dst_sl[split_axis] = slice(halo_width + offset, halo_width + offset + chunk)
        u_out[tuple(dst_sl)] = cp.asarray(partitions[gpu_id][tuple(src_sl)])
        offset += chunk


def get_halo_state(num_gpus: int, halo_width: int):
    """Lazy per-shape cache of (streams, ev_right, ev_left).

    - ``streams[g]``: GPU *g*'s default (current) stream
    - ``ev_right[i]``: event recorded on *i* after *i*->(i+1) halo send
    - ``ev_left[j]``: event recorded on *j* after *j*->(j-1) halo send

    Cached so subsequent ``step`` calls on the same shape are
    allocation-free. Pre-prime via :func:`prime_halo_state` to inject
    custom streams (e.g. for CUDA Graph capture).
    """
    key = (num_gpus, halo_width)
    state = _HALO_STATE_CACHE.get(key)
    if state is not None:
        return state
    streams = []
    for gid in range(num_gpus):
        with cp.cuda.Device(gid):
            streams.append(cp.cuda.get_current_stream())
    ev_right: dict = {}
    ev_left: dict = {}
    for i in range(num_gpus - 1):
        with cp.cuda.Device(i):
            ev_right[i] = cp.cuda.Event(disable_timing=True)
        with cp.cuda.Device(i + 1):
            ev_left[i + 1] = cp.cuda.Event(disable_timing=True)
    state = (streams, ev_right, ev_left)
    _HALO_STATE_CACHE[key] = state
    return state


def prime_halo_state(num_gpus: int, halo_width: int,
                     streams: list, ev_right: dict, ev_left: dict) -> None:
    """Override the cached halo-state with externally-built streams.

    Used by :mod:`cutile.runtime.graph_helpers` to inject capture-aware
    streams into the multi-GPU step path. The next call to
    :func:`get_halo_state` for ``(num_gpus, halo_width)`` returns the
    primed values instead of constructing fresh defaults.
    """
    _HALO_STATE_CACHE[(num_gpus, halo_width)] = (streams, ev_right, ev_left)


def reset_halo_state(num_gpus: int | None = None,
                     halo_width: int | None = None) -> None:
    """Clear the halo-state cache (whole or one entry).

    Useful between capture experiments or when the lifetime of an
    underlying stream/event handle ends (e.g. captured streams are
    destroyed at end of a benchmark).
    """
    if num_gpus is None or halo_width is None:
        _HALO_STATE_CACHE.clear()
    else:
        _HALO_STATE_CACHE.pop((num_gpus, halo_width), None)


def halo_send_pair(parts, i: int, j: int, halo_width: int, axis: int,
                   streams, ev_right, ev_left) -> None:
    """Async P2P halo exchange between adjacent GPUs *i* and *j=i+1*.

    Records cross-device events on each sender so the next iteration's
    kernel on the receiving side can ``stream.wait_event`` on them
    instead of a host-side synchronize.
    """
    ndim = parts[0].ndim
    # i -> j (right halo of i)
    sl_src = [slice(None)] * ndim
    sl_src[axis] = slice(-2 * halo_width, -halo_width)
    sl_dst = [slice(None)] * ndim
    sl_dst[axis] = slice(0, halo_width)
    sv = parts[i][tuple(sl_src)]
    dv = parts[j][tuple(sl_dst)]
    with cp.cuda.Device(i):
        if sv.flags.c_contiguous and dv.flags.c_contiguous:
            cp.cuda.runtime.memcpyPeerAsync(
                dv.data.ptr, j, sv.data.ptr, i, sv.nbytes, streams[i].ptr,
            )
        else:
            buf = cp.ascontiguousarray(sv)
            with cp.cuda.Device(j):
                parts[j][tuple(sl_dst)] = buf.copy()
        ev_right[i].record(streams[i])
    # j -> i (left halo of j)
    sl_src2 = [slice(None)] * ndim
    sl_src2[axis] = slice(halo_width, 2 * halo_width)
    sl_dst2 = [slice(None)] * ndim
    sl_dst2[axis] = slice(-halo_width, None)
    sv2 = parts[j][tuple(sl_src2)]
    dv2 = parts[i][tuple(sl_dst2)]
    with cp.cuda.Device(j):
        if sv2.flags.c_contiguous and dv2.flags.c_contiguous:
            cp.cuda.runtime.memcpyPeerAsync(
                dv2.data.ptr, i, sv2.data.ptr, j, sv2.nbytes, streams[j].ptr,
            )
        else:
            buf2 = cp.ascontiguousarray(sv2)
            with cp.cuda.Device(i):
                parts[i][tuple(sl_dst2)] = buf2.copy()
        ev_left[j].record(streams[j])


## ----------------------------------------------------------------- ##
## Cartesian-topology decomposition (N-D process grid)                ##
## ----------------------------------------------------------------- ##


_CART_HALO_STATE_CACHE: dict = {}


def cartesian_decompose(u, topology: tuple[int, ...],
                        halo_widths: tuple[int, ...]) -> dict:
    """Split *u* over a Cartesian process grid.

    ``topology`` is the per-axis number of GPUs (e.g. ``(2, 2)`` =
    2x2 = 4 GPUs total). ``halo_widths`` is per-domain-axis halo
    thickness (must match ``u.ndim``).

    Returns a dict with:
      - ``parts``: list of ``cupy.ndarray``, one per GPU, indexed by
        the linearised rank ``rank = ravel_multi_index(coord, topology)``
      - ``coords``: list of per-GPU ``(c0, c1, ...)`` grid coordinates
      - ``starts``: list of per-GPU per-axis interior start offsets
      - ``chunks``: list of per-GPU per-axis interior chunk lengths
      - ``topology``: echoed for downstream helpers
    """
    import math
    ndim = u.ndim
    if len(topology) > ndim:
        raise ValueError(
            f"topology has {len(topology)} dims but array has {ndim}")
    # Pad topology with 1s on trailing axes so it matches u.ndim
    topo = tuple(topology) + (1,) * (ndim - len(topology))
    if len(halo_widths) != ndim:
        raise ValueError("halo_widths must match u.ndim")

    num_gpus = math.prod(topo)

    for i in range(num_gpus):
        for j in range(num_gpus):
            if i != j:
                try:
                    cp.cuda.runtime.deviceEnablePeerAccess(j)
                except cp.cuda.runtime.CUDARuntimeError:
                    pass

    # Per-axis chunking (equal split with remainders going to low ranks)
    interior_per_axis = [u.shape[d] - 2 * halo_widths[d] for d in range(ndim)]
    base = [interior_per_axis[d] // topo[d] for d in range(ndim)]
    rem = [interior_per_axis[d] % topo[d] for d in range(ndim)]

    parts: list = [None] * num_gpus
    coords: list = [None] * num_gpus
    starts: list = [None] * num_gpus
    chunks: list = [None] * num_gpus

    # Iterate every grid coordinate
    coord_iter = [(0,) * ndim]
    for d in range(ndim):
        new = []
        for c in coord_iter:
            for k in range(topo[d]):
                new.append(c[:d] + (k,) + c[d + 1:])
        coord_iter = new

    for coord in coord_iter:
        rank = 0
        for d in range(ndim):
            rank = rank * topo[d] + coord[d]
        gpu_id = rank
        per_axis_chunk = []
        per_axis_start = []
        slc = []
        for d in range(ndim):
            ch = base[d] + (1 if coord[d] < rem[d] else 0)
            st = sum(base[d] + (1 if k < rem[d] else 0)
                     for k in range(coord[d]))
            per_axis_chunk.append(ch)
            per_axis_start.append(st)
            # Slice on u: include halo on both sides
            slc.append(slice(st, st + ch + 2 * halo_widths[d]))
        with cp.cuda.Device(gpu_id):
            parts[rank] = u[tuple(slc)].copy()
        coords[rank] = coord
        starts[rank] = tuple(per_axis_start)
        chunks[rank] = tuple(per_axis_chunk)

    return {
        "parts": parts,
        "coords": coords,
        "starts": starts,
        "chunks": chunks,
        "topology": topo,
    }


def cartesian_gather(u_out, decomp: dict, halo_widths: tuple[int, ...]) -> None:
    """Gather Cartesian sub-domain interiors back into *u_out*.

    Per-rank slice may be non-contiguous along non-leading axes; CuPy
    cannot cross-device copy non-contiguous arrays directly, so we
    materialise a contiguous copy on the source device first.
    """
    ndim = u_out.ndim
    parts = decomp["parts"]
    starts = decomp["starts"]
    chunks = decomp["chunks"]
    for rank, sub in enumerate(parts):
        src_sl = []
        dst_sl = []
        for d in range(ndim):
            src_sl.append(slice(halo_widths[d], halo_widths[d] + chunks[rank][d]))
            dst_sl.append(slice(halo_widths[d] + starts[rank][d],
                                halo_widths[d] + starts[rank][d] + chunks[rank][d]))
        with cp.cuda.Device(rank):
            interior = cp.ascontiguousarray(sub[tuple(src_sl)])
        u_out[tuple(dst_sl)] = cp.asarray(interior)


def _cart_neighbour(coord: tuple[int, ...], topology: tuple[int, ...],
                    axis: int, delta: int) -> int | None:
    """Return the rank of the neighbour at +/-delta along ``axis``, or None."""
    new = list(coord)
    new[axis] += delta
    if not 0 <= new[axis] < topology[axis]:
        return None
    rank = 0
    for d in range(len(topology)):
        rank = rank * topology[d] + new[d]
    return rank


def get_cartesian_halo_state(topology: tuple[int, ...],
                             halo_widths: tuple[int, ...]):
    """Per-(topology, halo) cache of (streams, events_by_(rank, axis, side))."""
    import math
    key = (tuple(topology), tuple(halo_widths))
    state = _CART_HALO_STATE_CACHE.get(key)
    if state is not None:
        return state
    num_gpus = math.prod(topology)
    streams = []
    for gid in range(num_gpus):
        with cp.cuda.Device(gid):
            streams.append(cp.cuda.get_current_stream())
    # Events keyed by (sender_rank, axis, side): event recorded on sender
    # after sender->neighbour memcpy completes; neighbour waits on it.
    events: dict = {}
    # Build coords table
    coords = []
    for rank in range(num_gpus):
        c = []
        r = rank
        for d in range(len(topology) - 1, -1, -1):
            c.append(r % topology[d])
            r //= topology[d]
        coords.append(tuple(reversed(c)))
    for rank in range(num_gpus):
        with cp.cuda.Device(rank):
            for axis in range(len(topology)):
                if topology[axis] == 1:
                    continue
                for side in ("low", "high"):
                    events[(rank, axis, side)] = cp.cuda.Event(
                        disable_timing=True)
    state = (streams, events, coords)
    _CART_HALO_STATE_CACHE[key] = state
    return state


def cartesian_halo_send_axis(parts: list, coords: list, topology: tuple[int, ...],
                             halo_widths: tuple[int, ...], axis: int,
                             streams: list, events: dict) -> None:
    """Async halo exchange along ``axis`` for every adjacent pair.

    Each rank with a neighbour at +1 along ``axis`` sends its high-side
    halo to the neighbour's low-side halo region (and vice versa). The
    sends record cross-device events that the next iteration's kernels
    wait on.
    """
    import math
    ndim = parts[0].ndim
    h = halo_widths[axis]
    num_gpus = math.prod(topology)
    for rank in range(num_gpus):
        coord = coords[rank]
        right_rank = _cart_neighbour(coord, topology, axis, +1)
        if right_rank is None:
            continue
        # rank.high boundary -> right_rank.low halo
        sl_src = [slice(None)] * ndim
        sl_src[axis] = slice(-2 * h, -h)
        sl_dst = [slice(None)] * ndim
        sl_dst[axis] = slice(0, h)
        sv = parts[rank][tuple(sl_src)]
        dv = parts[right_rank][tuple(sl_dst)]
        with cp.cuda.Device(rank):
            if sv.flags.c_contiguous and dv.flags.c_contiguous:
                cp.cuda.runtime.memcpyPeerAsync(
                    dv.data.ptr, right_rank, sv.data.ptr, rank,
                    sv.nbytes, streams[rank].ptr,
                )
            else:
                buf = cp.ascontiguousarray(sv)
                with cp.cuda.Device(right_rank):
                    parts[right_rank][tuple(sl_dst)] = buf.copy()
            events[(rank, axis, "high")].record(streams[rank])
        # right_rank.low boundary -> rank.high halo
        sl_src2 = [slice(None)] * ndim
        sl_src2[axis] = slice(h, 2 * h)
        sl_dst2 = [slice(None)] * ndim
        sl_dst2[axis] = slice(-h, None)
        sv2 = parts[right_rank][tuple(sl_src2)]
        dv2 = parts[rank][tuple(sl_dst2)]
        with cp.cuda.Device(right_rank):
            if sv2.flags.c_contiguous and dv2.flags.c_contiguous:
                cp.cuda.runtime.memcpyPeerAsync(
                    dv2.data.ptr, rank, sv2.data.ptr, right_rank,
                    sv2.nbytes, streams[right_rank].ptr,
                )
            else:
                buf2 = cp.ascontiguousarray(sv2)
                with cp.cuda.Device(rank):
                    parts[rank][tuple(sl_dst2)] = buf2.copy()
            events[(right_rank, axis, "low")].record(streams[right_rank])


def reset_cartesian_halo_state(topology: tuple[int, ...] | None = None,
                               halo_widths: tuple[int, ...] | None = None) -> None:
    if topology is None or halo_widths is None:
        _CART_HALO_STATE_CACHE.clear()
    else:
        _CART_HALO_STATE_CACHE.pop((tuple(topology), tuple(halo_widths)), None)


def exchange_halos(partitions, halo_width: int, split_axis: int) -> None:
    """Synchronous halo exchange (kept as a non-async fallback)."""
    n = len(partitions)
    for i in range(n - 1):
        j = i + 1
        ndim = partitions[0].ndim
        sl_src = [slice(None)] * ndim
        sl_src[split_axis] = slice(-2 * halo_width, -halo_width)
        sl_dst = [slice(None)] * ndim
        sl_dst[split_axis] = slice(0, halo_width)
        sv = partitions[i][tuple(sl_src)]
        dv = partitions[j][tuple(sl_dst)]
        if sv.flags.c_contiguous and dv.flags.c_contiguous:
            cp.cuda.runtime.memcpyPeer(
                dv.data.ptr, j, sv.data.ptr, i, sv.nbytes,
            )
        else:
            with cp.cuda.Device(i):
                buf = cp.ascontiguousarray(sv)
            with cp.cuda.Device(j):
                partitions[j][tuple(sl_dst)] = buf.copy()
        sl_src2 = [slice(None)] * ndim
        sl_src2[split_axis] = slice(halo_width, 2 * halo_width)
        sl_dst2 = [slice(None)] * ndim
        sl_dst2[split_axis] = slice(-halo_width, None)
        sv2 = partitions[j][tuple(sl_src2)]
        dv2 = partitions[i][tuple(sl_dst2)]
        if sv2.flags.c_contiguous and dv2.flags.c_contiguous:
            cp.cuda.runtime.memcpyPeer(
                dv2.data.ptr, i, sv2.data.ptr, j, sv2.nbytes,
            )
        else:
            with cp.cuda.Device(j):
                buf2 = cp.ascontiguousarray(sv2)
            with cp.cuda.Device(i):
                partitions[i][tuple(sl_dst2)] = buf2.copy()
