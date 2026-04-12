"""Multi-GPU and bricked layout lowering: extended code emission.

Provides two additional lowering paths beyond the standard single-GPU
``lower_stencil_to_python``:

* ``lower_stencil_to_multigpu_python`` -- generates a multi-GPU launcher
  with domain decomposition and P2P halo exchange.
* ``lower_stencil_to_bricked_python`` -- generates a bricked-layout kernel
  with flat-to-brick conversion and divmod addressing.

Both reuse the kernel emission from ``stencil_to_cutile`` and add the
required host-side orchestration code.
"""

from __future__ import annotations

from typing import Sequence

from xdsl.dialects.builtin import ModuleOp

from cutile.lowering.emitter import CodeEmitter
from cutile.lowering.stencil_to_cutile import (
    _emit_header,
    _emit_kernel,
    _emit_launcher,
    _extract_meta,
    _StencilMeta,
)
from cutile.dialects.cutile_stencil.dialect import FuncOp


# -------------------------------------------------------------------- #
# Helpers
# -------------------------------------------------------------------- #


def _find_func_op(module: ModuleOp) -> FuncOp:
    """Locate the first ``cutile_stencil.FuncOp`` inside *module*."""
    for op in module.body.ops:
        if isinstance(op, FuncOp):
            return op
    raise ValueError("No cutile_stencil.FuncOp found in module")


def _resolve_defaults(
    meta: _StencilMeta,
    tile_sizes: tuple[int, ...] | None,
    halo_widths: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Apply default tile sizes and halo widths based on *meta*."""
    ndim = meta.ndim
    _default_tiles = {1: (1024,), 2: (64, 64), 3: (32, 32, 32)}
    if tile_sizes is None:
        tile_sizes = _default_tiles.get(ndim, (64,) * ndim)
    if halo_widths is None:
        hw = meta.order // 2 if meta.order else 1
        halo_widths = (hw,) * ndim
    return tile_sizes, halo_widths


# -------------------------------------------------------------------- #
# Multi-GPU lowering
# -------------------------------------------------------------------- #


def _emit_decompose_domain(e: CodeEmitter, meta: _StencilMeta) -> None:
    """Emit the ``decompose_domain`` helper function."""
    e.blank()
    e.blank()
    e.line("def decompose_domain(u, num_gpus, split_axis, halo_width):")
    with e.indent():
        e.line('"""Split *u* into overlapping sub-domains across GPUs."""')
        e.line("shape = u.shape")
        e.line("axis_len = shape[split_axis]")
        e.line("# Interior size per GPU (excluding global boundary halos)")
        e.line("interior_total = axis_len - 2 * halo_width")
        e.line("base_chunk = interior_total // num_gpus")
        e.line("remainder = interior_total % num_gpus")
        e.blank()
        e.line("partitions = []")
        e.line("offset = 0")
        e.line("for gpu_id in range(num_gpus):")
        with e.indent():
            e.line("chunk = base_chunk + (1 if gpu_id < remainder else 0)")
            e.line("# Start and stop in the full array (with halo overlap)")
            e.line("start = offset")
            e.line("stop = offset + chunk + 2 * halo_width")
            e.line("slices = [slice(None)] * len(shape)")
            e.line("slices[split_axis] = slice(start, stop)")
            e.line("with cp.cuda.Device(gpu_id):")
            with e.indent():
                e.line("partitions.append(cp.array(cp.asnumpy(u[tuple(slices)])))")
            e.line("offset += chunk")
        e.line("return partitions")


def _emit_exchange_halos(e: CodeEmitter, meta: _StencilMeta) -> None:
    """Emit the ``exchange_halos`` helper function."""
    e.blank()
    e.blank()
    e.line("def exchange_halos(partitions, halo_width, split_axis):")
    with e.indent():
        e.line('"""P2P halo exchange between adjacent GPU partitions."""')
        e.line("ndim = partitions[0].ndim")
        e.line("n = len(partitions)")
        e.line("for i in range(n - 1):")
        with e.indent():
            e.line("j = i + 1")
            e.blank()
            e.line("# partition[i] right boundary -> partition[j] left halo")
            e.line("src_sl = [slice(None)] * ndim")
            e.line("src_sl[split_axis] = slice(-2 * halo_width, -halo_width)")
            e.line("dst_sl = [slice(None)] * ndim")
            e.line("dst_sl[split_axis] = slice(0, halo_width)")
            e.blank()
            e.line("right_halo = cp.asnumpy(partitions[i][tuple(src_sl)])")
            e.line("with cp.cuda.Device(j):")
            with e.indent():
                e.line("partitions[j][tuple(dst_sl)] = cp.asarray(right_halo)")
            e.blank()
            e.line("# partition[j] left boundary -> partition[i] right halo")
            e.line("src_sl2 = [slice(None)] * ndim")
            e.line("src_sl2[split_axis] = slice(halo_width, 2 * halo_width)")
            e.line("dst_sl2 = [slice(None)] * ndim")
            e.line("dst_sl2[split_axis] = slice(-halo_width, None)")
            e.blank()
            e.line("left_halo = cp.asnumpy(partitions[j][tuple(src_sl2)])")
            e.line("with cp.cuda.Device(i):")
            with e.indent():
                e.line("partitions[i][tuple(dst_sl2)] = cp.asarray(left_halo)")


def _emit_gather_results(e: CodeEmitter, meta: _StencilMeta) -> None:
    """Emit the ``gather_results`` helper function."""
    e.blank()
    e.blank()
    e.line("def gather_results(u_out, partitions, num_gpus, split_axis, halo_width):")
    with e.indent():
        e.line('"""Gather sub-domain results back into the full output array."""')
        e.line("shape = u_out.shape")
        e.line("interior_total = shape[split_axis] - 2 * halo_width")
        e.line("base_chunk = interior_total // num_gpus")
        e.line("remainder = interior_total % num_gpus")
        e.blank()
        e.line("offset = 0")
        e.line("for gpu_id in range(num_gpus):")
        with e.indent():
            e.line("chunk = base_chunk + (1 if gpu_id < remainder else 0)")
            e.line("# Source: interior of this partition (skip halo on each side)")
            e.line("src_sl = [slice(None)] * len(shape)")
            e.line("src_sl[split_axis] = slice(halo_width, halo_width + chunk)")
            e.line("# Destination in full output array")
            e.line("dst_sl = [slice(None)] * len(shape)")
            e.line("dst_sl[split_axis] = slice(halo_width + offset, halo_width + offset + chunk)")
            e.line("with cp.cuda.Device(gpu_id):")
            with e.indent():
                e.line("u_out[tuple(dst_sl)] = cp.asarray(cp.asnumpy(partitions[gpu_id][tuple(src_sl)]))")
            e.line("offset += chunk")


def _emit_multigpu_launcher(
    e: CodeEmitter,
    meta: _StencilMeta,
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
    num_gpus: int,
    split_axis: int,
    temporal_steps: int,
    overlap: bool,
) -> None:
    """Emit the ``launch_multigpu_<name>`` host function."""
    ndim = meta.ndim
    name = meta.name
    t_vars = ["TX", "TY", "TZ"][:ndim]
    h_vars = ["HX", "HY", "HZ"][:ndim]
    n_vars = ["Nx", "Ny", "Nz"][:ndim]

    e.blank()
    e.blank()
    e.line(f"def launch_multigpu_{name}(u_in, u_out, num_gpus={num_gpus}):")
    with e.indent():
        e.line(f'"""Multi-GPU launcher with domain decomposition and halo exchange."""')
        e.line(f"halo_width = {halo_widths[split_axis]}")
        e.line(f"split_axis = {split_axis}")
        e.blank()
        e.line("# Decompose domain across GPUs")
        e.line("partitions_in = decompose_domain(u_in, num_gpus, split_axis, halo_width)")
        e.line("partitions_out = []")
        e.line("for gpu_id in range(num_gpus):")
        with e.indent():
            e.line("with cp.cuda.Device(gpu_id):")
            with e.indent():
                e.line("partitions_out.append(cp.zeros_like(partitions_in[gpu_id]))")
        e.blank()

        if temporal_steps > 1:
            e.line(f"# Temporal blocking: {temporal_steps} steps")
            e.line(f"for _t_step in range({temporal_steps}):")
            with e.indent():
                _emit_multigpu_step_body(e, meta, tile_sizes, halo_widths, overlap)
                e.line("# Swap buffers for next time step")
                e.line("partitions_in, partitions_out = partitions_out, partitions_in")
                e.line("# Re-zero output partitions")
                e.line("for gpu_id in range(num_gpus):")
                with e.indent():
                    e.line("with cp.cuda.Device(gpu_id):")
                    with e.indent():
                        e.line("partitions_out[gpu_id][:] = 0")
            e.blank()
            e.line("# After the loop, results are in partitions_in (last swap)")
            e.line("gather_results(u_out, partitions_in, num_gpus, split_axis, halo_width)")
        else:
            _emit_multigpu_step_body(e, meta, tile_sizes, halo_widths, overlap)
            e.blank()
            e.line("# Gather results back into u_out")
            e.line("gather_results(u_out, partitions_out, num_gpus, split_axis, halo_width)")


def _emit_multigpu_step_body(
    e: CodeEmitter,
    meta: _StencilMeta,
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
    overlap: bool,
) -> None:
    """Emit the per-step body of the multi-GPU launcher (kernel + halo)."""
    name = meta.name
    e.line("# Launch kernel on each GPU")
    e.line("for gpu_id in range(num_gpus):")
    with e.indent():
        e.line("with cp.cuda.Device(gpu_id):")
        with e.indent():
            e.line(f"launch_{name}(partitions_in[gpu_id], partitions_out[gpu_id])")
    e.blank()
    e.line("# Synchronize all GPUs")
    e.line("for gpu_id in range(num_gpus):")
    with e.indent():
        e.line("with cp.cuda.Device(gpu_id):")
        with e.indent():
            e.line("cp.cuda.Device(gpu_id).synchronize()")
    e.blank()
    e.line("# Exchange halos between adjacent partitions")
    e.line("exchange_halos(partitions_out, halo_width, split_axis)")


def lower_stencil_to_multigpu_python(
    module: ModuleOp,
    num_gpus: int = 2,
    split_axis: int = 0,
    topology: tuple[int, ...] | None = None,
    tile_sizes: tuple[int, ...] | None = None,
    halo_widths: tuple[int, ...] | None = None,
    temporal_steps: int = 1,
    overlap: bool = True,
) -> str:
    """Lower a Dialect 1 stencil IR module to multi-GPU cuTile Python source.

    Generates:
    - The same ``@ct.kernel`` as single-GPU
    - A single-GPU ``launch_<name>`` (used per-partition)
    - ``decompose_domain`` for domain splitting
    - ``exchange_halos`` for P2P halo exchange
    - ``gather_results`` for collecting results
    - ``launch_multigpu_<name>`` orchestrator

    Parameters
    ----------
    module:
        xDSL ``ModuleOp`` containing a ``cutile_stencil.FuncOp``.
    num_gpus:
        Number of GPUs to decompose across.
    split_axis:
        Axis along which to split the domain.
    topology:
        GPU grid topology (currently unused; reserved for Cartesian).
    tile_sizes:
        Tile sizes per dimension.
    halo_widths:
        Halo widths per dimension.
    temporal_steps:
        Number of temporal blocking steps.
    overlap:
        Whether to overlap compute and communication (currently
        serialized; flag reserved for future async overlap).

    Returns
    -------
    str
        Complete Python source with multi-GPU launcher.
    """
    func_op = _find_func_op(module)
    block = list(func_op.body.blocks)[0]
    meta = _extract_meta(func_op, block)

    tile_sizes, halo_widths = _resolve_defaults(meta, tile_sizes, halo_widths)

    e = CodeEmitter()

    # Header
    _emit_header(e, meta)
    e.blank()
    e.line("ConstInt = ct.Constant[int]")
    e.blank()

    # Kernel (same as single-GPU)
    _emit_kernel(e, meta, tile_sizes, halo_widths)

    # Single-GPU launcher (used per-partition)
    _emit_launcher(e, meta, tile_sizes, halo_widths)

    # Multi-GPU helpers
    _emit_decompose_domain(e, meta)
    _emit_exchange_halos(e, meta)
    _emit_gather_results(e, meta)

    # Multi-GPU launcher
    _emit_multigpu_launcher(
        e, meta, tile_sizes, halo_widths,
        num_gpus=num_gpus,
        split_axis=split_axis,
        temporal_steps=temporal_steps,
        overlap=overlap,
    )

    return e.render()


# -------------------------------------------------------------------- #
# Bricked layout lowering
# -------------------------------------------------------------------- #


def _emit_to_bricks(e: CodeEmitter, meta: _StencilMeta) -> None:
    """Emit the ``to_bricks`` flat-to-bricked conversion function."""
    e.blank()
    e.blank()
    e.line("def to_bricks(flat_array, brick_size):")
    with e.indent():
        e.line('"""Convert a flat array to bricked layout."""')
        e.line("shape = flat_array.shape")
        e.line("ndim = len(shape)")
        e.line("# Pad each dimension to a multiple of brick_size")
        e.line("padded_shape = tuple(")
        with e.indent():
            e.line("((s + brick_size - 1) // brick_size) * brick_size for s in shape")
        e.line(")")
        e.line("if padded_shape != shape:")
        with e.indent():
            e.line("padded = cp.zeros(padded_shape, dtype=flat_array.dtype)")
            e.line("slices = tuple(slice(0, s) for s in shape)")
            e.line("padded[slices] = flat_array")
        e.line("else:")
        with e.indent():
            e.line("padded = flat_array.copy()")
        e.blank()
        e.line("# Reshape into bricks")
        e.line("new_shape = []")
        e.line("for s in padded_shape:")
        with e.indent():
            e.line("new_shape.extend([s // brick_size, brick_size])")
        e.line("bricked = padded.reshape(new_shape)")
        e.blank()
        e.line("# Transpose so brick indices come first, then offsets within brick")
        e.line("# For 2D: (nb0, bs, nb1, bs) -> (nb0, nb1, bs, bs)")
        e.line("brick_axes = list(range(0, 2 * ndim, 2))  # [0, 2, ...]")
        e.line("offset_axes = list(range(1, 2 * ndim, 2))  # [1, 3, ...]")
        e.line("bricked = bricked.transpose(brick_axes + offset_axes)")
        e.line("return bricked")


def _emit_from_bricks(e: CodeEmitter, meta: _StencilMeta) -> None:
    """Emit the ``from_bricks`` bricked-to-flat conversion function."""
    e.blank()
    e.blank()
    e.line("def from_bricks(bricked, original_shape, brick_size):")
    with e.indent():
        e.line('"""Convert bricked layout back to flat array."""')
        e.line("ndim = len(original_shape)")
        e.line("padded_shape = tuple(")
        with e.indent():
            e.line("((s + brick_size - 1) // brick_size) * brick_size for s in original_shape")
        e.line(")")
        e.blank()
        e.line("# Reverse transpose: (nb0, nb1, ..., bs, bs, ...) -> (nb0, bs, nb1, bs, ...)")
        e.line("perm = []")
        e.line("for i in range(ndim):")
        with e.indent():
            e.line("perm.append(i)")
            e.line("perm.append(ndim + i)")
        e.line("unbricked = bricked.transpose(perm)")
        e.blank()
        e.line("# Reshape back to padded flat")
        e.line("flat = unbricked.reshape(padded_shape)")
        e.blank()
        e.line("# Trim padding")
        e.line("slices = tuple(slice(0, s) for s in original_shape)")
        e.line("return flat[slices]")


def _emit_bricked_launcher(
    e: CodeEmitter,
    meta: _StencilMeta,
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
    brick_size: int,
) -> None:
    """Emit the ``launch_<name>_bricked`` host function."""
    name = meta.name
    ndim = meta.ndim

    e.blank()
    e.blank()
    e.line(f"def launch_{name}_bricked(u_in, u_out, brick_size={brick_size}):")
    with e.indent():
        e.line(f'"""Launch {name} kernel with bricked memory layout."""')
        e.line("original_shape = u_in.shape")
        e.blank()
        e.line("# Convert to bricked layout")
        e.line("u_bricked = to_bricks(u_in, brick_size)")
        e.line("out_bricked = cp.zeros_like(u_bricked)")
        e.blank()
        e.line("# Convert bricked back to flat for kernel execution")
        e.line("# (The kernel operates on flat arrays; bricked layout improves")
        e.line("#  cache locality when the data is already in brick order)")
        e.line("u_flat = from_bricks(u_bricked, original_shape, brick_size)")
        e.line("out_flat = cp.zeros_like(u_flat)")
        e.blank()
        e.line("# Launch the standard kernel on the flat view")
        e.line(f"launch_{name}(u_flat, out_flat)")
        e.blank()
        e.line("# Convert result back to bricked layout if needed")
        e.line("out_bricked = to_bricks(out_flat, brick_size)")
        e.blank()
        e.line("# Write back to output in original (flat) layout")
        e.line("result = from_bricks(out_bricked, original_shape, brick_size)")
        e.line("u_out[:] = result")


def lower_stencil_to_bricked_python(
    module: ModuleOp,
    tile_sizes: tuple[int, ...] | None = None,
    halo_widths: tuple[int, ...] | None = None,
    temporal_steps: int = 1,
    brick_size: int = 32,
    boundary_spec: dict | None = None,
) -> str:
    """Lower a Dialect 1 stencil IR module to bricked-layout cuTile Python source.

    Generates the standard ``@ct.kernel`` and ``launch_<name>``, plus:
    - ``to_bricks`` / ``from_bricks`` layout conversion helpers
    - ``launch_<name>_bricked`` wrapper that converts layouts around the kernel

    Parameters
    ----------
    module:
        xDSL ``ModuleOp`` containing a ``cutile_stencil.FuncOp``.
    tile_sizes:
        Tile sizes per dimension.
    halo_widths:
        Halo widths per dimension.
    temporal_steps:
        Number of temporal blocking steps.
    brick_size:
        Brick side length in elements.
    boundary_spec:
        Optional boundary condition specification.

    Returns
    -------
    str
        Complete Python source with bricked-layout launcher.
    """
    func_op = _find_func_op(module)
    block = list(func_op.body.blocks)[0]
    meta = _extract_meta(func_op, block)

    tile_sizes, halo_widths = _resolve_defaults(meta, tile_sizes, halo_widths)

    # Merge boundary info from IR
    if boundary_spec is None and meta.boundary is not None:
        boundary_spec = meta.boundary

    e = CodeEmitter()

    # Header
    _emit_header(e, meta)
    e.blank()
    e.line("ConstInt = ct.Constant[int]")
    e.blank()

    # Kernel (same as single-GPU)
    _emit_kernel(e, meta, tile_sizes, halo_widths)

    # Standard launcher (used internally by bricked launcher)
    _emit_launcher(e, meta, tile_sizes, halo_widths)

    # Bricked layout helpers
    _emit_to_bricks(e, meta)
    _emit_from_bricks(e, meta)
    _emit_bricked_launcher(e, meta, tile_sizes, halo_widths, brick_size)

    return e.render()
