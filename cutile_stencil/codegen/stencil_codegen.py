"""Generate cuTile kernel Python files from StencilSpec.

Strategy: shifted-view approach for all dimensions.
Each unique stencil access becomes a pre-shifted array view so that
ct.load always uses the base (power-of-two) tile size.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from cutile_stencil.codegen.emitter import CodeEmitter
from cutile_stencil.codegen.ast_transform import transform_stencil_expr
from cutile_stencil.dsl.types import StencilSpec, TileConfig, TemporalConfig
from cutile_stencil.config import BenchmarkConfig, DEFAULT_BENCHMARK


class StencilCodeGenerator:
    """Generates a complete cuTile kernel Python file from a stencil spec."""

    def __init__(
        self,
        spec: StencilSpec,
        tile_config: TileConfig,
        temporal_config: Optional[TemporalConfig] = None,
        benchmark_config: Optional[BenchmarkConfig] = None,
    ):
        self.spec = spec
        self.tile = tile_config
        self.temporal = temporal_config
        self.bench = benchmark_config or DEFAULT_BENCHMARK

    def emit(self) -> str:
        if self.spec.ndim == 1:
            return self._emit_1d()
        elif self.spec.ndim == 2:
            return self._emit_2d()
        else:
            return self._emit_3d()

    def emit_to_file(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.emit())

    # ------------------------------------------------------------------
    # 1D: shifted-view approach (like stencil1d.py)
    # ------------------------------------------------------------------
    def _emit_1d(self) -> str:
        e = CodeEmitter()
        spec = self.spec
        ts = self.tile.tile_sizes[0]
        halo = self.tile.halo_widths[0]

        self._emit_header(e)
        e.blank()

        # Constant
        e.line("ConstInt = ct.Constant[int]")
        e.blank()

        # Build the list of shifted views we need (per-array, per-offset)
        seen = set()
        view_names = []
        for acc in spec.accesses:
            key = (acc.array_name, acc.offsets[0])
            if key not in seen:
                seen.add(key)
                off = acc.offsets[0]
                vname = f"v_{acc.array_name}_off{off:+d}".replace("+", "p").replace("-", "m")
                view_names.append((vname, off, acc.array_name))

        # Kernel
        e.line("@ct.kernel")
        params = ", ".join(vn for vn, _, _ in view_names)
        e.line(f"def {spec.name}_kernel({params}, output, TILE: ConstInt):")
        with e.indent():
            e.line("pid = ct.bid(0)")
            for vn, _, _ in view_names:
                e.line(f"t_{vn} = ct.load({vn}, index=(pid,), shape=(TILE,))")
            # Build the stencil expression from the update_fn source
            self._emit_stencil_expr_1d(e, view_names)
            e.line("ct.store(output, index=(pid,), tile=result)")

        self._emit_launcher_1d(e, view_names, ts, halo)
        self._emit_benchmark_1d(e, view_names, ts, halo)
        return e.render()

    def _emit_stencil_expr_1d(self, e: CodeEmitter, view_names):
        """Build the arithmetic expression via AST transform."""
        spec = self.spec
        # Build term_names matching accesses order, mapping each access
        # to the corresponding tile variable
        offset_to_tile = {}
        for vn, off, arr_name in view_names:
            offset_to_tile[(arr_name, (off,))] = f"t_{vn}"

        term_names = []
        for acc in spec.accesses:
            key = (acc.array_name, acc.offsets)
            term_names.append(offset_to_tile.get(key, f"t_unknown"))

        try:
            expr_src = transform_stencil_expr(
                spec.update_fn, spec.accesses, spec.ndim, term_names
            )
            e.line(f"result = {expr_src}")
        except Exception:
            # Fallback: sum of tiles
            terms = [f"t_{vn}" for vn, _, _ in view_names]
            e.line(f"result = {' + '.join(terms)}")

    def _emit_launcher_1d(self, e: CodeEmitter, view_names, ts, halo):
        e.blank()
        e.blank()
        multi_input = len(self.spec.inputs) > 1
        if multi_input:
            input_params = ", ".join(self.spec.inputs)
            e.line(f"def launch_{self.spec.name}({input_params}, output, stream=None):")
        else:
            e.line(f"def launch_{self.spec.name}(u, output, stream=None):")
        with e.indent():
            e.line(f"TILE = {ts}")
            e.line(f"halo = {halo}")
            first_input = self.spec.inputs[0] if multi_input else "u"
            e.line(f"N = {first_input}.shape[0]")
            e.line("if stream is None:")
            with e.indent():
                e.line("stream = cp.cuda.get_current_stream()")
            # Create shifted views
            for vn, off, arr_name in view_names:
                src = arr_name if multi_input else "u"
                if off == 0:
                    e.line(f"{vn} = {src}[halo:N - halo]")
                elif off > 0:
                    e.line(f"{vn} = {src}[halo + {off}:N - halo + {off}]")
                else:
                    e.line(f"{vn} = {src}[halo - {abs(off)}:N - halo - {abs(off)}]")
            e.line("out_view = output[halo:N - halo]")
            e.line("n_interior = N - 2 * halo")
            e.line("grid = (ct.cdiv(n_interior, TILE),)")
            args = ", ".join(vn for vn, _, _ in view_names)
            e.line(f"ct.launch(stream, grid, {self.spec.name}_kernel, ({args}, out_view, TILE))")

    def _emit_benchmark_1d(self, e: CodeEmitter, view_names, ts, halo):
        e.blank()
        e.blank()
        e.line("if __name__ == '__main__':")
        with e.indent():
            e.line("import argparse")
            e.line("import triton")
            e.line("parser = argparse.ArgumentParser(add_help=False)")
            e.line('parser.add_argument("--precision", type=str, default="float64")')
            e.line("args, _ = parser.parse_known_args()")
            e.line("dtype_map = {'float64': cp.float64, 'float32': cp.float32, 'float16': cp.float16}")
            e.line("cp_dtype = dtype_map[args.precision]")
            e.blank()
            e.line(f"N = {self.bench.grid_1d} + 2 * {halo}")
            e.line("u = cp.random.randn(N).astype(cp_dtype)")
            e.line("out = cp.zeros_like(u)")
            e.line(f"launch_{self.spec.name}(u, out)")
            e.line("print('Kernel launched successfully')")

    # ------------------------------------------------------------------
    # 2D: shifted-view approach (same pattern as 1D)
    # ------------------------------------------------------------------
    def _emit_2d(self) -> str:
        e = CodeEmitter()
        spec = self.spec
        tx, ty = self.tile.tile_sizes
        hx, hy = self.tile.halo_widths

        self._emit_header(e)
        e.blank()
        e.line("ConstInt = ct.Constant[int]")
        e.blank()

        view_names = self._build_view_names_nd(spec)
        self._emit_kernel_nd(e, spec, view_names, 2, (tx, ty))
        self._emit_launcher_nd(e, spec, view_names, 2, (tx, ty), (hx, hy))
        self._emit_benchmark_nd(e, spec, 2, (tx, ty), (hx, hy))
        return e.render()

    def _emit_stencil_expr_nd(self, e: CodeEmitter, spec: StencilSpec, term_names):
        """Emit the stencil expression via AST transform (inlines intermediates)."""
        try:
            expr_src = transform_stencil_expr(
                spec.update_fn, spec.accesses, spec.ndim, term_names
            )
            e.line(f"result = {expr_src}")
        except Exception:
            # Fallback
            e.line(f"result = {' + '.join(term_names)}")

    # ------------------------------------------------------------------
    # 3D: shifted-view approach (same pattern as 1D/2D)
    # ------------------------------------------------------------------
    def _emit_3d(self) -> str:
        e = CodeEmitter()
        spec = self.spec
        tx, ty, tz = self.tile.tile_sizes
        hx, hy, hz = self.tile.halo_widths

        self._emit_header(e)
        e.blank()
        e.line("ConstInt = ct.Constant[int]")
        e.blank()

        view_names = self._build_view_names_nd(spec)
        self._emit_kernel_nd(e, spec, view_names, 3, (tx, ty, tz))
        self._emit_launcher_nd(e, spec, view_names, 3, (tx, ty, tz), (hx, hy, hz))
        self._emit_benchmark_nd(e, spec, 3, (tx, ty, tz), (hx, hy, hz))
        return e.render()

    # ------------------------------------------------------------------
    # Shared nD helpers (2D/3D shifted-view kernel, launcher, benchmark)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_view_names_nd(spec: StencilSpec):
        """Build deduplicated list of (view_name, offsets_tuple, array_name)."""
        seen = {}
        view_names = []
        for acc in spec.accesses:
            key = (acc.array_name, acc.offsets)
            if key not in seen:
                parts = "_".join(f"{o:+d}" for o in acc.offsets)
                vname = f"v_{acc.array_name}_{parts}".replace("+", "p").replace("-", "m")
                seen[key] = vname
                view_names.append((vname, acc.offsets, acc.array_name))
        return view_names

    def _emit_kernel_nd(self, e: CodeEmitter, spec, view_names, ndim, tile_sizes):
        """Emit an nD kernel using shifted-view parameters (power-of-two loads)."""
        bid_vars = ["bx", "by", "bz"][:ndim]
        tile_vars = ["TX", "TY", "TZ"][:ndim]
        tile_const_params = ", ".join(f"{tv}: ConstInt" for tv in tile_vars)

        e.line("@ct.kernel")
        params = ", ".join(vn for vn, _, _ in view_names)
        e.line(f"def {spec.name}_kernel({params}, output, {tile_const_params}):")
        with e.indent():
            for i, bv in enumerate(bid_vars):
                e.line(f"{bv} = ct.bid({i})")
            shape_tuple = ", ".join(tile_vars)
            idx_tuple = ", ".join(bid_vars)
            for vn, _, _ in view_names:
                e.line(f"t_{vn} = ct.load({vn}, index=({idx_tuple}), shape=({shape_tuple}))")
            e.blank()
            # Map accesses → tile variables
            offset_to_tile = {(vn_arr, offs): f"t_{vn}" for vn, offs, vn_arr in view_names}
            term_names = [offset_to_tile[(acc.array_name, acc.offsets)] for acc in spec.accesses]
            self._emit_stencil_expr_nd(e, spec, term_names)
            e.blank()
            e.line(f"ct.store(output, index=({idx_tuple}), tile=result)")

    @staticmethod
    def _slice_expr(h_var: str, off: int, n_var: str) -> str:
        """Format a Python slice string: ``h+off : N-h+off``."""
        if off == 0:
            return f"{h_var}:{n_var} - {h_var}"
        elif off > 0:
            return f"{h_var} + {off}:{n_var} - {h_var} + {off}"
        else:
            return f"{h_var} - {abs(off)}:{n_var} - {h_var} - {abs(off)}"

    def _emit_launcher_nd(self, e, spec, view_names, ndim, tile_sizes, halo_widths):
        """Emit an nD launcher that creates shifted views and launches the kernel."""
        multi_input = len(spec.inputs) > 1
        h_vars = ["hx", "hy", "hz"][:ndim]
        n_vars = ["Nx", "Ny", "Nz"][:ndim]
        t_vars = ["TX", "TY", "TZ"][:ndim]

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
            e.line(f"{', '.join(n_vars)} = {first_input}.shape")
            e.line("if stream is None:")
            with e.indent():
                e.line("stream = cp.cuda.get_current_stream()")
            # Create shifted views
            for vn, offsets, arr_name in view_names:
                src = arr_name if multi_input else "u_in"
                slices = ", ".join(
                    self._slice_expr(h_vars[d], offsets[d], n_vars[d])
                    for d in range(ndim)
                )
                e.line(f"{vn} = {src}[{slices}]")
            out_slices = ", ".join(f"{h_vars[d]}:{n_vars[d]} - {h_vars[d]}" for d in range(ndim))
            e.line(f"out_view = u_out[{out_slices}]")
            grid_parts = ", ".join(
                f"ct.cdiv({n_vars[d]} - 2 * {h_vars[d]}, {t_vars[d]})" for d in range(ndim)
            )
            e.line(f"grid = ({grid_parts})")
            args = ", ".join(vn for vn, _, _ in view_names)
            t_args = ", ".join(t_vars)
            e.line(f"ct.launch(stream, grid, {spec.name}_kernel, ({args}, out_view, {t_args}))")

    def _emit_benchmark_nd(self, e, spec, ndim, tile_sizes, halo_widths):
        """Emit an nD benchmark __main__ block."""
        h_vars = ["hx", "hy", "hz"][:ndim]
        bench_grids = {2: self.bench.grid_2d, 3: self.bench.grid_3d}
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
