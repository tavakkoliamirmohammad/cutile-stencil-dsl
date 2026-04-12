"""Generate cuTile kernel Python files from StencilSpec.

Strategy: in-kernel .slice() approach for all dimensions.
Each unique stencil access creates a shifted view inside the kernel via
Array.slice(), so the launcher just passes original arrays + constants.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

from cutile_stencil.codegen.emitter import CodeEmitter
from cutile_stencil.codegen.ast_transform import transform_stencil_expr
from cutile_stencil.codegen.errors import CodegenWarning
from cutile_stencil.dsl.types import StencilSpec, TileConfig, TemporalConfig, BrickLayout
from cutile_stencil.config import BenchmarkConfig, DEFAULT_BENCHMARK


class StencilCodeGenerator:
    """Generates a complete cuTile kernel Python file from a stencil spec."""

    def __init__(
        self,
        spec: StencilSpec,
        tile_config: TileConfig,
        temporal_config: Optional[TemporalConfig] = None,
        benchmark_config: Optional[BenchmarkConfig] = None,
        layout: Optional[BrickLayout] = None,
    ):
        self.spec = spec
        self.tile = tile_config
        self.temporal = temporal_config
        self.bench = benchmark_config or DEFAULT_BENCHMARK
        self.layout = layout

    def emit(self) -> str:
        if self.layout is not None:
            return self._emit_bricked()
        e = CodeEmitter()
        self._emit_header(e)
        e.blank()
        e.line("ConstInt = ct.Constant[int]")
        e.blank()
        access_names = self._build_access_names(self.spec)
        ndim = self.spec.ndim
        self._emit_kernel_nd(e, self.spec, access_names, ndim,
                             self.tile.tile_sizes, self.tile.halo_widths)
        self._emit_launcher_nd(e, self.spec, ndim,
                               self.tile.tile_sizes, self.tile.halo_widths)
        self._emit_benchmark_nd(e, self.spec, ndim,
                                self.tile.tile_sizes, self.tile.halo_widths)
        return e.render()

    def emit_to_file(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.emit())

    # ------------------------------------------------------------------
    # Offset formatting helper
    # ------------------------------------------------------------------
    @staticmethod
    def _format_offset(off: int) -> str:
        """Format an integer offset for use in variable names.

        0 → "0", +k → "pk", -k → "mk"
        """
        if off == 0:
            return "0"
        elif off > 0:
            return f"p{off}"
        else:
            return f"m{abs(off)}"

    # ------------------------------------------------------------------
    # Build deduplicated access names (shared by 1D/2D/3D)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_access_names(spec: StencilSpec):
        """Build deduplicated list of (view_name, offsets_tuple, array_name).

        Naming: {array}_{off0}[_{off1}[_{off2}]]
        e.g. u_m1, E_0, u_m2_0, u_0_m1
        """
        seen = {}
        access_names = []
        for acc in spec.accesses:
            key = (acc.array_name, acc.offsets)
            if key not in seen:
                off_parts = "_".join(
                    StencilCodeGenerator._format_offset(o) for o in acc.offsets
                )
                vname = f"{acc.array_name}_{off_parts}"
                seen[key] = vname
                access_names.append((vname, acc.offsets, acc.array_name))
        return access_names

    # ------------------------------------------------------------------
    # Shared nD helpers: kernel, launcher, benchmark
    # ------------------------------------------------------------------
    def _emit_kernel_nd(self, e: CodeEmitter, spec, access_names, ndim, tile_sizes, halo_widths):
        """Emit an nD kernel using in-kernel .slice() for shifted views."""
        bid_vars = ["bx", "by", "bz"][:ndim]
        tile_vars = ["TX", "TY", "TZ"][:ndim]
        halo_vars = ["HX", "HY", "HZ"][:ndim]
        n_vars = ["nx", "ny", "nz"][:ndim]

        tile_const_params = ", ".join(f"{tv}: ConstInt" for tv in tile_vars)
        halo_const_params = ", ".join(f"{hv}: ConstInt" for hv in halo_vars)

        multi_input = len(spec.inputs) > 1
        input_params = ", ".join(spec.inputs) if multi_input else spec.inputs[0]

        e.line("@ct.kernel")
        e.line(f"def {spec.name}_kernel({input_params}, output, {tile_const_params}, {halo_const_params}):")
        with e.indent():
            for i, bv in enumerate(bid_vars):
                e.line(f"{bv} = ct.bid({i})")
            # Compute interior sizes
            first_arr = spec.inputs[0]
            for d in range(ndim):
                e.line(f"{n_vars[d]} = {first_arr}.shape[{d}] - 2 * {halo_vars[d]}")
            # Create sliced views for each unique access
            for vname, offsets, arr_name in access_names:
                slice_chain = arr_name
                for d in range(ndim):
                    off = offsets[d]
                    start = self._offset_expr(halo_vars[d], off)
                    stop = self._offset_expr(halo_vars[d], off, add_n=True, n_var=n_vars[d])
                    slice_chain = f"{slice_chain}.slice(axis={d}, start={start}, stop={stop})"
                e.line(f"{vname} = {slice_chain}")
            # Output view (all-zero offsets)
            out_chain = "output"
            for d in range(ndim):
                out_chain = f"{out_chain}.slice(axis={d}, start={halo_vars[d]}, stop={halo_vars[d]} + {n_vars[d]})"
            e.line(f"out = {out_chain}")
            # Load tiles
            shape_tuple = ", ".join(tile_vars)
            idx_tuple = ", ".join(bid_vars)
            for vname, _, _ in access_names:
                e.line(f"t_{vname} = ct.load({vname}, index=({idx_tuple}), shape=({shape_tuple}))")
            e.blank()
            # Stencil expression
            offset_to_tile = {(arr, offs): f"t_{vn}" for vn, offs, arr in access_names}
            term_names = [offset_to_tile[(acc.array_name, acc.offsets)] for acc in spec.accesses]
            self._emit_stencil_expr_nd(e, spec, term_names)
            e.blank()
            e.line(f"ct.store(out, index=({idx_tuple}), tile=result)")

    def _emit_launcher_nd(self, e, spec, ndim, tile_sizes, halo_widths):
        """Emit an nD launcher that just passes arrays + constants."""
        multi_input = len(spec.inputs) > 1
        t_vars = ["TX", "TY", "TZ"][:ndim]
        h_vars = ["HX", "HY", "HZ"][:ndim]
        n_vars = ["Nx", "Ny", "Nz"][:ndim]

        e.blank()
        e.blank()
        if multi_input:
            input_params = ", ".join(spec.inputs)
            e.line(f"def launch_{spec.name}({input_params}, u_out, stream=None):")
        else:
            e.line(f"def launch_{spec.name}(u_in, u_out, stream=None):")
        with e.indent():
            e.line(f"{', '.join(t_vars)} = {', '.join(str(t) for t in tile_sizes)}")
            e.line(f"{', '.join(h_vars)} = {', '.join(str(h) for h in halo_widths)}")
            first_input = spec.inputs[0] if multi_input else "u_in"
            # Trailing comma for 1D to unpack single-element tuple from .shape
            trailing = "," if ndim == 1 else ""
            e.line(f"{', '.join(n_vars)}{trailing} = {first_input}.shape")
            e.line("if stream is None:")
            with e.indent():
                e.line("stream = cp.cuda.get_current_stream()")
            grid_parts = ", ".join(
                f"ct.cdiv({n_vars[d]} - 2 * {h_vars[d]}, {t_vars[d]})" for d in range(ndim)
            )
            # Trailing comma for 1D so Python sees a tuple, not a parenthesized expr
            grid_trailing = "," if ndim == 1 else ""
            e.line(f"grid = ({grid_parts}{grid_trailing})")
            if multi_input:
                args = ", ".join(spec.inputs)
            else:
                args = "u_in"
            t_args = ", ".join(t_vars)
            h_args = ", ".join(h_vars)
            e.line(f"ct.launch(stream, grid, {spec.name}_kernel, ({args}, u_out, {t_args}, {h_args}))")

    def _emit_benchmark_nd(self, e, spec, ndim, tile_sizes, halo_widths):
        """Emit an nD benchmark __main__ block."""
        h_vars = ["hx", "hy", "hz"][:ndim]
        bench_grids = {1: (self.bench.grid_1d,), 2: self.bench.grid_2d, 3: self.bench.grid_3d}
        e.blank()
        e.blank()
        e.line("if __name__ == '__main__':")
        with e.indent():
            e.line(f"N = {bench_grids[ndim][0]}")
            h_assign = ", ".join(f"{h_vars[d]}" for d in range(ndim))
            h_vals = ", ".join(str(h) for h in halo_widths)
            e.line(f"{h_assign} = {h_vals}")
            shape = ", ".join(f"N + 2 * {h_vars[d]}" for d in range(ndim))
            e.line(f"u = cp.random.randn({shape}).astype(cp.float64)")
            e.line("out = cp.zeros_like(u)")
            e.line(f"launch_{spec.name}(u, out)")
            e.line("print('Kernel launched successfully')")

    # ------------------------------------------------------------------
    # Stencil expression emission
    # ------------------------------------------------------------------
    def _emit_stencil_expr_nd(self, e: CodeEmitter, spec: StencilSpec, term_names):
        """Emit the stencil expression via AST transform (nD)."""
        try:
            expr_src = transform_stencil_expr(
                spec.update_fn, spec.accesses, spec.ndim, term_names
            )
            e.line(f"result = {expr_src}")
        except Exception as exc:
            warnings.warn(
                f"AST transform failed for {spec.name}: {exc}. "
                f"Generated kernel uses fallback sum expression.",
                CodegenWarning,
                stacklevel=2,
            )
            e.line("# WARNING: AST transform failed, using fallback sum")
            e.line(f"result = {' + '.join(term_names)}")

    # ------------------------------------------------------------------
    # Bricked memory layout codegen
    # ------------------------------------------------------------------
    def _emit_bricked(self) -> str:
        """Top-level bricked codegen: header + kernel + launcher."""
        e = CodeEmitter()
        spec = self.spec
        ndim = spec.ndim
        tile_sizes = self.tile.tile_sizes
        halo_widths = self.tile.halo_widths
        brick_sizes = self.layout.brick_sizes

        self._emit_header(e)
        e.blank()
        e.line("ConstInt = ct.Constant[int]")
        e.blank()

        access_names = self._build_access_names(spec)
        self._emit_kernel_bricked_nd(
            e, spec, access_names, ndim, tile_sizes, brick_sizes, halo_widths
        )
        self._emit_launcher_bricked_nd(
            e, spec, ndim, tile_sizes, brick_sizes, halo_widths
        )
        return e.render()

    def _emit_kernel_bricked_nd(
        self, e, spec, access_names, ndim, tile_sizes, brick_sizes, halo_widths
    ):
        """Emit bricked kernel using divmod for brick/tile selection.

        Grid dim d = num_bricks_d * cdiv(B_d, T_d).
        Kernel computes brick_id = bid // tiles_per_brick, tile_id = bid % tiles_per_brick.
        Outer .slice() selects brick, inner .slice() applies stencil offset.
        """
        # Variable names
        tile_vars = ["TX", "TY", "TZ"][:ndim]
        brick_vars = ["BX", "BY", "BZ"][:ndim]
        halo_vars = ["HX", "HY", "HZ"][:ndim]
        tpb_vars = ["TPB_X", "TPB_Y", "TPB_Z"][:ndim]
        bid_vars = ["bid_x", "bid_y", "bid_z"][:ndim]
        bix_vars = ["bix", "biy", "biz"][:ndim]
        tid_vars = ["tid_x", "tid_y", "tid_z"][:ndim]

        tile_params = ", ".join(f"{tv}: ConstInt" for tv in tile_vars)
        brick_params = ", ".join(f"{bv}: ConstInt" for bv in brick_vars)
        halo_params = ", ".join(f"{hv}: ConstInt" for hv in halo_vars)
        tpb_params = ", ".join(f"{tp}: ConstInt" for tp in tpb_vars)

        multi_input = len(spec.inputs) > 1
        input_params = ", ".join(spec.inputs) if multi_input else spec.inputs[0]

        e.line("@ct.kernel")
        e.line(
            f"def {spec.name}_kernel("
            f"{input_params}, output, "
            f"{tile_params}, {brick_params}, {halo_params}, {tpb_params}):"
        )
        with e.indent():
            # Divmod to get brick index and tile-within-brick index
            for d in range(ndim):
                e.line(f"{bid_vars[d]} = ct.bid({d})")
                e.line(f"{bix_vars[d]} = {bid_vars[d]} // {tpb_vars[d]}")
                e.line(f"{tid_vars[d]} = {bid_vars[d]} % {tpb_vars[d]}")

            e.blank()
            # Slice outer dims to select brick, then inner dims for stencil offset
            for vname, offsets, arr_name in access_names:
                chain = arr_name
                # Outer dims: select brick
                for d in range(ndim):
                    chain = (
                        f"{chain}.slice(axis={d}, "
                        f"start={bix_vars[d]}, stop={bix_vars[d]} + 1)"
                    )
                # Inner dims: apply stencil offset within the padded brick
                for d in range(ndim):
                    axis = ndim + d
                    off = offsets[d]
                    start = self._offset_expr(halo_vars[d], off)
                    stop = self._offset_expr(
                        halo_vars[d], off, add_n=True, n_var=brick_vars[d]
                    )
                    chain = f"{chain}.slice(axis={axis}, start={start}, stop={stop})"
                e.line(f"{vname} = {chain}")

            # Output view
            out_chain = "output"
            for d in range(ndim):
                out_chain = (
                    f"{out_chain}.slice(axis={d}, "
                    f"start={bix_vars[d]}, stop={bix_vars[d]} + 1)"
                )
            for d in range(ndim):
                axis = ndim + d
                out_chain = (
                    f"{out_chain}.slice(axis={axis}, "
                    f"start={halo_vars[d]}, stop={halo_vars[d]} + {brick_vars[d]})"
                )
            e.line(f"out = {out_chain}")
            e.blank()

            # Load tiles using tile-within-brick index.
            # After .slice() the array is still 2*ndim-dimensional (outer
            # brick dims are size 1).  ct.load requires full-rank index/shape.
            outer_zeros = ", ".join(["0"] * ndim)
            outer_ones = ", ".join(["1"] * ndim)
            inner_idx = ", ".join(tid_vars)
            inner_shape = ", ".join(tile_vars)
            full_idx = f"{outer_zeros}, {inner_idx}"
            full_shape = f"{outer_ones}, {inner_shape}"
            for vname, _, _ in access_names:
                e.line(
                    f"t_{vname} = ct.load({vname}, "
                    f"index=({full_idx}), shape=({full_shape}))"
                )
            e.blank()

            # Stencil expression
            offset_to_tile = {
                (arr, offs): f"t_{vn}" for vn, offs, arr in access_names
            }
            term_names = [
                offset_to_tile[(acc.array_name, acc.offsets)]
                for acc in spec.accesses
            ]
            self._emit_stencil_expr_nd(e, spec, term_names)
            e.blank()
            e.line(f"ct.store(out, index=({full_idx}), tile=result)")

    def _emit_launcher_bricked_nd(
        self, e, spec, ndim, tile_sizes, brick_sizes, halo_widths
    ):
        """Emit bricked launcher: grid = num_bricks * tiles_per_brick per dim."""
        multi_input = len(spec.inputs) > 1
        t_vars = ["TX", "TY", "TZ"][:ndim]
        b_vars = ["BX", "BY", "BZ"][:ndim]
        h_vars = ["HX", "HY", "HZ"][:ndim]
        tpb_vars = ["TPB_X", "TPB_Y", "TPB_Z"][:ndim]
        nb_vars = ["NBX", "NBY", "NBZ"][:ndim]

        e.blank()
        e.blank()
        if multi_input:
            input_params = ", ".join(spec.inputs)
            e.line(f"def launch_{spec.name}({input_params}, u_out, stream=None):")
        else:
            e.line(f"def launch_{spec.name}(u_in, u_out, stream=None):")
        with e.indent():
            e.line(f"{', '.join(t_vars)} = {', '.join(str(t) for t in tile_sizes)}")
            e.line(f"{', '.join(b_vars)} = {', '.join(str(b) for b in brick_sizes)}")
            e.line(f"{', '.join(h_vars)} = {', '.join(str(h) for h in halo_widths)}")

            # Get num_bricks from array shape (outer dims)
            first_input = spec.inputs[0] if multi_input else "u_in"
            for d in range(ndim):
                e.line(f"{nb_vars[d]} = {first_input}.shape[{d}]")

            e.line("if stream is None:")
            with e.indent():
                e.line("stream = cp.cuda.get_current_stream()")

            # Tiles per brick
            for d in range(ndim):
                e.line(
                    f"{tpb_vars[d]} = ct.cdiv({b_vars[d]}, {t_vars[d]})"
                )

            grid_parts = ", ".join(
                f"{nb_vars[d]} * {tpb_vars[d]}" for d in range(ndim)
            )
            # Trailing comma for 1D so Python sees a tuple, not an int
            trailing = "," if ndim == 1 else ""
            e.line(f"grid = ({grid_parts}{trailing})")

            if multi_input:
                args = ", ".join(spec.inputs)
            else:
                args = "u_in"
            t_args = ", ".join(t_vars)
            b_args = ", ".join(b_vars)
            h_args = ", ".join(h_vars)
            tpb_args = ", ".join(tpb_vars)
            e.line(
                f"ct.launch(stream, grid, {spec.name}_kernel, "
                f"({args}, u_out, {t_args}, {b_args}, {h_args}, {tpb_args}))"
            )

    # ------------------------------------------------------------------
    # Offset expression helpers for .slice() calls
    # ------------------------------------------------------------------
    @staticmethod
    def _offset_expr(halo_var: str, off: int, add_n: bool = False, n_var: str = "n") -> str:
        """Build a start/stop expression for .slice().

        start: HALO + off
        stop:  HALO + off + n
        """
        if add_n:
            if off == 0:
                return f"{halo_var} + {n_var}"
            elif off > 0:
                return f"{halo_var} + {off} + {n_var}"
            else:
                return f"{halo_var} - {abs(off)} + {n_var}"
        else:
            if off == 0:
                return f"{halo_var}"
            elif off > 0:
                return f"{halo_var} + {off}"
            else:
                return f"{halo_var} - {abs(off)}"

    # ------------------------------------------------------------------
    # Header / closure constants
    # ------------------------------------------------------------------
    def _emit_header(self, e: CodeEmitter):
        e.line(f'"""cuTile kernel for {self.spec.name} stencil (auto-generated)."""')
        e.blank()
        e.line("import cuda.tile as ct")
        e.line("import cupy as cp")
        self._emit_closure_constants(e)

    def _emit_closure_constants(self, e: CodeEmitter):
        """Emit captured closure constants as module-level definitions."""
        consts = self.spec.closure_constants
        if not consts:
            return
        e.blank()
        e.line("# Captured constants from stencil definition scope")
        for name, value in sorted(consts.items()):
            e.line(f"{name} = {value!r}")
