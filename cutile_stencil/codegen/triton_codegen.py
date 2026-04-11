"""Generate Triton kernel Python files from StencilSpec.

Alternative backend to the cuTile codegen. Produces @triton.jit kernels
with block-pointer arithmetic instead of cuTile's .slice() approach.
"""

from __future__ import annotations

from typing import Optional

from cutile_stencil.codegen.emitter import CodeEmitter
from cutile_stencil.codegen.ast_transform import transform_stencil_expr
from cutile_stencil.dsl.types import StencilSpec, TileConfig, TemporalConfig


class TritonCodeGenerator:
    """Generates a Triton kernel Python file from a stencil spec."""

    def __init__(
        self,
        spec: StencilSpec,
        tile_config: TileConfig,
        temporal_config: Optional[TemporalConfig] = None,
    ):
        self.spec = spec
        self.tile = tile_config
        self.temporal = temporal_config

    def emit(self) -> str:
        """Generate complete Triton kernel source."""
        e = CodeEmitter()
        spec = self.spec
        ndim = spec.ndim

        self._emit_header(e)
        e.blank()

        access_names = self._build_access_names(spec)
        self._emit_kernel(e, spec, access_names, ndim)
        e.blank()
        e.blank()
        self._emit_launcher(e, spec, ndim)
        return e.render()

    def _emit_header(self, e: CodeEmitter):
        e.line(f'"""Triton kernel for {self.spec.name} stencil (auto-generated)."""')
        e.blank()
        e.line("import triton")
        e.line("import triton.language as tl")
        e.line("import torch")
        # Emit closure constants
        consts = self.spec.closure_constants
        if consts:
            e.blank()
            e.line("# Captured constants from stencil definition scope")
            for name, value in sorted(consts.items()):
                e.line(f"{name} = {value!r}")

    @staticmethod
    def _build_access_names(spec: StencilSpec):
        """Build deduplicated access names, same logic as cuTile codegen."""
        seen = {}
        access_names = []
        for acc in spec.accesses:
            key = (acc.array_name, acc.offsets)
            if key not in seen:
                off_parts = "_".join(
                    ("0" if o == 0 else f"p{o}" if o > 0 else f"m{abs(o)}")
                    for o in acc.offsets
                )
                vname = f"{acc.array_name}_{off_parts}"
                seen[key] = vname
                access_names.append((vname, acc.offsets, acc.array_name))
        return access_names

    def _emit_kernel(self, e: CodeEmitter, spec, access_names, ndim):
        """Emit the @triton.jit kernel."""
        tile_sizes = self.tile.tile_sizes
        halo_widths = self.tile.halo_widths

        # Parameter names
        multi_input = len(spec.inputs) > 1
        input_ptrs = [f"{name}_ptr" for name in spec.inputs]
        all_ptr_params = input_ptrs + ["output_ptr"]

        # Stride parameters per input
        stride_params = []
        for name in spec.inputs:
            for d in range(ndim):
                stride_params.append(f"stride_{name}_{d}")
        for d in range(ndim):
            stride_params.append(f"stride_out_{d}")

        # Size and tile parameters
        size_params = [f"N{d}" for d in range(ndim)]
        tile_params = [f"BLOCK_{d}: tl.constexpr" for d in range(ndim)]

        all_params = all_ptr_params + stride_params + size_params + tile_params

        e.line("@triton.jit")
        e.line(f"def {spec.name}_kernel({', '.join(all_params)}):")
        with e.indent():
            # Program IDs
            for d in range(ndim):
                e.line(f"pid_{d} = tl.program_id({d})")

            e.blank()

            # Block offsets per dimension
            for d in range(ndim):
                e.line(f"offs_{d} = pid_{d} * BLOCK_{d} + tl.arange(0, BLOCK_{d})")

            e.blank()

            # Load each unique access
            for vname, offsets, arr_name in access_names:
                # Compute pointer offset with strides
                terms = []
                for d in range(ndim):
                    off = offsets[d]
                    if off == 0:
                        terms.append(f"offs_{d} * stride_{arr_name}_{d}")
                    elif off > 0:
                        terms.append(f"(offs_{d} + {off}) * stride_{arr_name}_{d}")
                    else:
                        terms.append(f"(offs_{d} - {abs(off)}) * stride_{arr_name}_{d}")

                ptr_expr = f"{arr_name}_ptr + " + " + ".join(terms)

                # Mask for bounds checking
                mask_parts = [f"offs_{d} < N{d}" for d in range(ndim)]
                mask = " & ".join(mask_parts)

                e.line(f"{vname} = tl.load({ptr_expr}, mask={mask}, other=0.0)")

            e.blank()

            # Stencil expression via AST transform
            offset_to_tile = {}
            for vname, offsets, arr_name in access_names:
                offset_to_tile[(arr_name, offsets)] = vname
            term_names = []
            for acc in spec.accesses:
                key = (acc.array_name, acc.offsets)
                term_names.append(offset_to_tile.get(key, "UNKNOWN"))

            try:
                expr_src = transform_stencil_expr(
                    spec.update_fn, spec.accesses, spec.ndim, term_names
                )
                e.line(f"result = {expr_src}")
            except Exception:
                e.line("# WARNING: AST transform failed, using fallback sum")
                e.line(f"result = {' + '.join(term_names)}")

            e.blank()

            # Store result
            out_terms = [f"offs_{d} * stride_out_{d}" for d in range(ndim)]
            out_ptr = "output_ptr + " + " + ".join(out_terms)
            mask_parts = [f"offs_{d} < N{d}" for d in range(ndim)]
            mask = " & ".join(mask_parts)
            e.line(f"tl.store({out_ptr}, result, mask={mask})")

    def _emit_launcher(self, e, spec, ndim):
        """Emit the Python launcher function."""
        tile_sizes = self.tile.tile_sizes
        halo_widths = self.tile.halo_widths

        multi_input = len(spec.inputs) > 1
        if multi_input:
            input_params = ", ".join(spec.inputs)
        else:
            input_params = "u_in"

        e.line(f"def launch_{spec.name}({input_params}, u_out):")
        with e.indent():
            # Get sizes (interior = total - 2*halo)
            first_input = spec.inputs[0] if multi_input else "u_in"
            for d in range(ndim):
                e.line(f"N{d} = {first_input}.shape[{d}] - 2 * {halo_widths[d]}")

            e.blank()

            # Block sizes
            for d in range(ndim):
                e.line(f"BLOCK_{d} = {tile_sizes[d]}")

            e.blank()

            # Grid
            grid_parts = [f"triton.cdiv(N{d}, BLOCK_{d})" for d in range(ndim)]
            e.line(f"grid = ({', '.join(grid_parts)},)")

            e.blank()

            # Build stride args
            stride_args = []
            for name in spec.inputs:
                arr = name if multi_input else "u_in"
                for d in range(ndim):
                    stride_args.append(f"{arr}.stride({d})")
            for d in range(ndim):
                stride_args.append(f"u_out.stride({d})")

            # Size args
            size_args = [f"N{d}" for d in range(ndim)]

            # Block args
            block_args = [f"BLOCK_{d}" for d in range(ndim)]

            # Pointer args
            ptr_args = []
            for name in spec.inputs:
                arr = name if multi_input else "u_in"
                ptr_args.append(arr)
            ptr_args.append("u_out")

            all_args = ptr_args + stride_args + size_args + block_args
            e.line(f"{spec.name}_kernel[grid]({', '.join(all_args)})")

    def emit_to_file(self, path) -> None:
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.emit())
