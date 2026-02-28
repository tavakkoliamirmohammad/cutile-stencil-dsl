"""Generate cuTile kernel Python files from StencilSpec.

Strategy selection:
  - 1D + low-order → shifted-view approach (like stencil1d.py)
  - 2D/3D → offset-tile-load with padding_mode=ZERO
  - Temporal blocking → fused multi-step kernel
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
                e.line("stream = torch.cuda.current_stream()")
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
            e.line("dtype_map = {'float64': torch.float64, 'float32': torch.float32, 'float16': torch.float16}")
            e.line("torch_dtype = dtype_map[args.precision]")
            e.blank()
            e.line(f"N = {self.bench.grid_1d} + 2 * {halo}")
            e.line("u = torch.randn(N, dtype=torch_dtype, device='cuda')")
            e.line("out = torch.zeros_like(u)")
            e.line(f"launch_{self.spec.name}(u, out)")
            e.line("print('Kernel launched successfully')")

    # ------------------------------------------------------------------
    # 2D: offset-tile-load with padding_mode=ZERO
    # ------------------------------------------------------------------
    def _emit_2d(self) -> str:
        e = CodeEmitter()
        spec = self.spec
        tx, ty = self.tile.tile_sizes
        hx, hy = self.tile.halo_widths

        self._emit_header(e)
        e.blank()
        e.line("ConstInt = ct.Constant[int]")
        pad_mode = self._get_padding_mode()
        e.line(f"PAD_MODE = ct.PaddingMode.{pad_mode}")
        e.blank()

        if self.temporal and self.temporal.steps > 1:
            self._emit_2d_temporal(e, tx, ty, hx, hy)
        else:
            self._emit_2d_basic(e, tx, ty, hx, hy)

        self._emit_launcher_2d(e, tx, ty)
        self._emit_benchmark_2d(e, tx, ty)
        return e.render()

    def _emit_2d_basic(self, e: CodeEmitter, tx, ty, hx, hy):
        spec = self.spec
        multi_input = len(spec.inputs) > 1
        e.line("@ct.kernel")
        if multi_input:
            input_params = ", ".join(f"{inp}_in" for inp in spec.inputs)
            e.line(f"def {spec.name}_kernel({input_params}, u_out, TX: ConstInt, TY: ConstInt):")
        else:
            e.line(f"def {spec.name}_kernel(u_in, u_out, TX: ConstInt, TY: ConstInt):")
        with e.indent():
            e.line("bx = ct.bid(0)")
            e.line("by = ct.bid(1)")
            etx = tx + 2 * hx
            ety = ty + 2 * hy
            if multi_input:
                # Load separate expanded tiles per input array
                for inp in spec.inputs:
                    e.line(f"tile_{inp} = ct.load({inp}_in, index=(bx, by), shape=({etx}, {ety}), padding_mode=PAD_MODE)")
            else:
                e.line(f"# Load expanded tile ({etx} x {ety}) with halo")
                e.line(f"tile_in = ct.load(u_in, index=(bx, by), shape=({etx}, {ety}), padding_mode=PAD_MODE)")
            e.blank()
            e.line(f"# Extract stencil neighbors from loaded tile")
            terms = []
            for acc in spec.accesses:
                ox, oy = acc.offsets
                sx = hx + ox
                sy = hy + oy
                ex = sx + tx
                ey = sy + ty
                vname = f"nb_{acc.array_name}_{ox:+d}_{oy:+d}".replace("+", "p").replace("-", "m")
                tile_src = f"tile_{acc.array_name}" if multi_input else "tile_in"
                e.line(f"{vname} = {tile_src}[{sx}:{ex}, {sy}:{ey}]")
                terms.append(vname)

            e.blank()
            self._emit_stencil_expr_nd(e, spec, terms)
            e.blank()
            e.line("ct.store(u_out, index=(bx, by), tile=result)")

    def _emit_2d_temporal(self, e: CodeEmitter, tx, ty, hx, hy):
        spec = self.spec
        T = self.temporal.steps
        ehx, ehy = self.temporal.expanded_halo
        e.line("@ct.kernel")
        e.line(f"def {spec.name}_temporal_kernel(u_in, u_out, TX: ConstInt, TY: ConstInt):")
        with e.indent():
            e.line("bx = ct.bid(0)")
            e.line("by = ct.bid(1)")
            etx = tx + 2 * ehx
            ety = ty + 2 * ehy
            e.line(f"# Load expanded tile for {T} temporal steps")
            e.line(f"tile = ct.load(u_in, index=(bx, by), shape=({etx}, {ety}), padding_mode=PAD_MODE)")
            e.blank()
            e.line(f"# Fused {T}-step temporal blocking")
            e.line(f"for _t_step in range({T}):")
            with e.indent():
                # After each shrink, the remaining halo is always exactly (hx, hy)
                # per side, so use constant per-step halo offsets
                e.line(f"# Per-step halo is constant: ({hx}, {hy})")
                e.line(f"inner_x = tile.shape[0] - 2 * {hx}")
                e.line(f"inner_y = tile.shape[1] - 2 * {hy}")
                for i, acc in enumerate(spec.accesses):
                    ox, oy = acc.offsets
                    vname = f"nb{i}"
                    e.line(f"{vname} = tile[{hx} + {ox}:{hx} + {ox} + inner_x, {hy} + {oy}:{hy} + {oy} + inner_y]")
                # Stencil expression
                nb_terms = [f"nb{i}" for i in range(len(spec.accesses))]
                self._emit_stencil_expr_nd(e, spec, nb_terms)
                e.line(f"tile = result  # Shrinks by (2*{hx}, 2*{hy}) each step")
            e.blank()
            e.line("ct.store(u_out, index=(bx, by), tile=tile)")

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

    def _emit_launcher_2d(self, e: CodeEmitter, tx, ty):
        spec = self.spec
        multi_input = len(spec.inputs) > 1
        kname = spec.name + "_kernel"
        if self.temporal and self.temporal.steps > 1:
            kname = spec.name + "_temporal_kernel"
        e.blank()
        e.blank()
        if multi_input:
            input_params = ", ".join(spec.inputs)
            e.line(f"def launch_{spec.name}({input_params}, u_out, stream=None):")
        else:
            e.line(f"def launch_{spec.name}(u_in, u_out, stream=None):")
        with e.indent():
            e.line(f"TX, TY = {tx}, {ty}")
            first_input = spec.inputs[0] if multi_input else "u_in"
            e.line(f"Nx, Ny = {first_input}.shape")
            e.line("if stream is None:")
            with e.indent():
                e.line("stream = torch.cuda.current_stream()")
            e.line("grid = (ct.cdiv(Nx, TX), ct.cdiv(Ny, TY))")
            if multi_input:
                args = ", ".join(spec.inputs)
                e.line(f"ct.launch(stream, grid, {kname}, ({args}, u_out, TX, TY))")
            else:
                e.line(f"ct.launch(stream, grid, {kname}, (u_in, u_out, TX, TY))")

    def _emit_benchmark_2d(self, e: CodeEmitter, tx, ty):
        spec = self.spec
        e.blank()
        e.blank()
        e.line("if __name__ == '__main__':")
        with e.indent():
            e.line(f"N = {self.bench.grid_2d[0]}")
            e.line("u = torch.randn(N, N, dtype=torch.float64, device='cuda')")
            e.line("out = torch.zeros_like(u)")
            e.line(f"launch_{spec.name}(u, out)")
            e.line("print('Kernel launched successfully')")

    # ------------------------------------------------------------------
    # 3D: bid(0), bid(1), bid(2) grid pattern
    # ------------------------------------------------------------------
    def _emit_3d(self) -> str:
        e = CodeEmitter()
        spec = self.spec
        tx, ty, tz = self.tile.tile_sizes
        hx, hy, hz = self.tile.halo_widths

        self._emit_header(e)
        e.blank()
        e.line("ConstInt = ct.Constant[int]")
        pad_mode = self._get_padding_mode()
        e.line(f"PAD_MODE = ct.PaddingMode.{pad_mode}")
        e.blank()

        multi_input = len(spec.inputs) > 1
        e.line("@ct.kernel")
        if multi_input:
            input_params = ", ".join(f"{inp}_in" for inp in spec.inputs)
            e.line(f"def {spec.name}_kernel({input_params}, u_out, TX: ConstInt, TY: ConstInt, TZ: ConstInt):")
        else:
            e.line(f"def {spec.name}_kernel(u_in, u_out, TX: ConstInt, TY: ConstInt, TZ: ConstInt):")
        with e.indent():
            e.line("bx = ct.bid(0)")
            e.line("by = ct.bid(1)")
            e.line("bz = ct.bid(2)")
            etx = tx + 2 * hx
            ety = ty + 2 * hy
            etz = tz + 2 * hz
            if multi_input:
                for inp in spec.inputs:
                    e.line(f"tile_{inp} = ct.load({inp}_in, index=(bx, by, bz), shape=({etx}, {ety}, {etz}), padding_mode=PAD_MODE)")
            else:
                e.line(f"tile_in = ct.load(u_in, index=(bx, by, bz), shape=({etx}, {ety}, {etz}), padding_mode=PAD_MODE)")
            e.blank()
            terms = []
            for acc in spec.accesses:
                ox, oy, oz = acc.offsets
                sx, sy, sz = hx + ox, hy + oy, hz + oz
                ex, ey, ez = sx + tx, sy + ty, sz + tz
                vname = f"nb_{acc.array_name}_{ox:+d}_{oy:+d}_{oz:+d}".replace("+", "p").replace("-", "m")
                tile_src = f"tile_{acc.array_name}" if multi_input else "tile_in"
                e.line(f"{vname} = {tile_src}[{sx}:{ex}, {sy}:{ey}, {sz}:{ez}]")
                terms.append(vname)
            e.blank()
            self._emit_stencil_expr_nd(e, spec, terms)
            e.blank()
            e.line("ct.store(u_out, index=(bx, by, bz), tile=result)")

        # Launcher
        e.blank()
        e.blank()
        e.line(f"def launch_{spec.name}(u_in, u_out, stream=None):")
        with e.indent():
            e.line(f"TX, TY, TZ = {tx}, {ty}, {tz}")
            e.line("Nx, Ny, Nz = u_in.shape")
            e.line("if stream is None:")
            with e.indent():
                e.line("stream = torch.cuda.current_stream()")
            e.line("grid = (ct.cdiv(Nx, TX), ct.cdiv(Ny, TY), ct.cdiv(Nz, TZ))")
            e.line(f"ct.launch(stream, grid, {spec.name}_kernel, (u_in, u_out, TX, TY, TZ))")

        e.blank()
        e.blank()
        e.line("if __name__ == '__main__':")
        with e.indent():
            e.line(f"N = {self.bench.grid_3d[0]}")
            e.line("u = torch.randn(N, N, N, dtype=torch.float64, device='cuda')")
            e.line("out = torch.zeros_like(u)")
            e.line(f"launch_{spec.name}(u, out)")
            e.line("print('Kernel launched successfully')")

        return e.render()

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------
    def _get_padding_mode(self) -> str:
        """Get cuTile padding mode from boundary spec."""
        if self.spec.boundary is not None:
            return self.spec.boundary.cuTile_padding_mode(0)
        return "ZERO"

    def _emit_header(self, e: CodeEmitter):
        e.line(f'"""cuTile kernel for {self.spec.name} stencil (auto-generated)."""')
        e.blank()
        e.line("import cuda.tile as ct")
        e.line("import torch")
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
