"""Multi-GPU and bricked layout lowering: extended code emission.

Provides two additional lowering paths beyond the standard single-GPU
``lower_stencil_to_python``:

* ``lower_stencil_to_multigpu_python`` -- generates a multi-GPU launcher
  with domain decomposition and P2P halo exchange.
* ``lower_stencil_to_bricked_python`` -- generates a bricked-layout kernel
  with flat-to-brick conversion and divmod addressing.

Both reuse the kernel/launcher emission from Dialect 3 (cutile_target) IR
via ``lower_to_target_ir`` + ``emit_python``, and add the required
host-side orchestration code as Python string appendages.
"""

from __future__ import annotations

from typing import Sequence

from xdsl.dialects.builtin import ModuleOp

from cutile.lowering.emitter import CodeEmitter
from cutile.lowering.stencil_to_cutile import (
    _extract_meta,
    _StencilMeta,
)
from cutile.lowering.stencil_to_target import lower_to_target_ir
from cutile.lowering.target_to_python import emit_python
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


def _get_kernel_and_launcher_source(
    module: ModuleOp,
    tile_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
    temporal_steps: int = 1,
    boundary_spec: dict | None = None,
) -> str:
    """Get kernel + launcher Python source via Dialect 3 IR pipeline."""
    target_ir = lower_to_target_ir(
        module,
        tile_sizes=tile_sizes,
        halo_widths=halo_widths,
        temporal_steps=temporal_steps,
        boundary_spec=boundary_spec,
    )
    return emit_python(target_ir)


# -------------------------------------------------------------------- #
# Multi-GPU lowering
# -------------------------------------------------------------------- #
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
    """Lower a stencil IR module to multi-GPU cuTile Python source.

    Two stages:

    1. Kernel + single-GPU launcher emitted via the standard
       ``cutile_stencil`` -> ``cutile_target`` -> Python pipeline.
    2. Host-side multi-GPU orchestration (setup / step / gather /
       launch) is built as ``cutile_target.HostProgramOp`` ops whose
       bodies use the ``comm`` dialect, then walked by
       :func:`cutile.lowering.multigpu_lowering.emit_multigpu_host_programs`.

    The runtime helpers live in ``cutile.runtime.multigpu_helpers``;
    generated code only imports them, so the per-stencil source stays
    small and the orchestration logic is centralised.
    """
    from cutile.lowering.multigpu_lowering import (
        build_multigpu_host_programs,
        emit_multigpu_host_programs,
    )

    func_op = _find_func_op(module)
    block = list(func_op.body.blocks)[0]
    meta = _extract_meta(func_op, block)

    tile_sizes, halo_widths = _resolve_defaults(meta, tile_sizes, halo_widths)

    # 1. Kernel + single-GPU launcher via the standard pipeline.
    kernel_source = _get_kernel_and_launcher_source(
        module, tile_sizes, halo_widths, temporal_steps=1,
    )

    # 2. Build comm-dialect host programs and walk them to emit Python.
    host_programs = build_multigpu_host_programs(
        kernel_name=meta.name,
        num_gpus=num_gpus,
        halo_width=halo_widths[split_axis],
        split_axis=split_axis,
    )
    multigpu_source = emit_multigpu_host_programs(host_programs)

    return kernel_source + multigpu_source


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

    The kernel and standard launcher are generated through Dialect 3
    (cutile_target) IR via ``lower_to_target_ir`` + ``emit_python``.
    Bricked layout helpers are appended as Python string code.

    Generates the standard ``@ct.kernel`` and ``launch_<name>`` (via Dialect 3),
    plus:
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

    # Generate kernel + standard launcher through Dialect 3 IR
    kernel_source = _get_kernel_and_launcher_source(
        module, tile_sizes, halo_widths,
        temporal_steps=temporal_steps,
        boundary_spec=boundary_spec,
    )

    # Append bricked layout helpers
    e = CodeEmitter()
    _emit_to_bricks(e, meta)
    _emit_from_bricks(e, meta)
    _emit_bricked_launcher(e, meta, tile_sizes, halo_widths, brick_size)

    return kernel_source + e.render()
