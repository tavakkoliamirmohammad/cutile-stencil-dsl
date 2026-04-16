"""Runtime helpers for multi-GPU stencil execution.

These functions are stencil-agnostic and shared by every multi-GPU
launcher emitted by the cuTile pipeline. Moving them out of the text
emitter keeps generated code small (just the per-stencil step function)
and centralises the GPU-orchestration logic in a single, testable spot.

The pieces:

- :func:`decompose_domain` — split an input array along ``split_axis``
  into ``num_gpus`` overlapping sub-arrays, one per GPU, with halo
  padding on each side. Enables P2P access between every GPU pair.
- :func:`gather_results` — copy the interior of each per-GPU partition
  back into a contiguous host-visible output array.
- :func:`get_halo_state` — lazy per-(num_gpus, halo_width) cache of
  per-GPU streams + per-pair cross-device events. Created on first
  call; reused thereafter so each ``step`` is allocation-free.
- :func:`halo_send_pair` — async P2P exchange between two adjacent
  GPUs, recording cross-device events that the next iteration's
  kernels wait on (no host sync needed between kernel and halo).
- :func:`exchange_halos` — synchronous halo exchange kept for direct
  external callers and as a non-async fallback.
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
    allocation-free.
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
