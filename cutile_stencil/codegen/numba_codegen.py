"""Generate Numba CUDA kernel Python files from StencilSpec."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from cutile_stencil.codegen.emitter import CodeEmitter
from cutile_stencil.codegen.ast_transform import transform_stencil_expr
from cutile_stencil.dsl.types import StencilSpec, TileConfig, TemporalConfig


class NumbaCudaCodeGenerator:
    """Generate Numba CUDA (@cuda.jit) kernels from a StencilSpec.

    Each thread processes one interior grid point. No shared-memory tile
    abstraction is used — this is a direct port of the stencil computation
    to CUDA threads.

    Parameters
    ----------
    spec : StencilSpec
        The stencil specification (accesses, inputs, ndim, etc.).
    tile_config : TileConfig
        Tile sizes and halo widths for block/grid configuration.
    temporal_config : TemporalConfig, optional
        Temporal blocking config (unused in this backend, reserved).
    """

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
        """Emit the full Numba CUDA kernel source as a string."""
        e = CodeEmitter()
        self._emit_header(e)
        e.blank()
        self._emit_kernel(e)
        self._emit_launcher(e)
        return e.render()

    def _emit_header(self, e: CodeEmitter) -> None:
        e.line(f'"""Numba CUDA kernel for {self.spec.name} stencil (auto-generated)."""')
        e.blank()
        e.line("from numba import cuda")
        e.line("import numpy as np")
        consts = self.spec.closure_constants
        if consts:
            e.blank()
            for name, value in sorted(consts.items()):
                e.line(f"{name} = {value!r}")

    def _emit_kernel(self, e: CodeEmitter) -> None:
        spec = self.spec
        ndim = spec.ndim
        halo = self.tile.halo_widths

        multi_input = len(spec.inputs) > 1
        input_params = ", ".join(spec.inputs) if multi_input else spec.inputs[0]

        e.line("@cuda.jit")
        e.line(f"def {spec.name}_kernel({input_params}, output):")
        with e.indent():
            if ndim == 1:
                e.line("i = cuda.grid(1)")
                e.line(f"if i < output.shape[0] - 2 * {halo[0]}:")
                with e.indent():
                    e.line(f"gi = i + {halo[0]}")
                    self._emit_stencil_body_1d(e, spec, halo[0])
            elif ndim == 2:
                e.line("i, j = cuda.grid(2)")
                e.line(
                    f"if i < output.shape[0] - 2 * {halo[0]}"
                    f" and j < output.shape[1] - 2 * {halo[1]}:"
                )
                with e.indent():
                    e.line(f"gi = i + {halo[0]}")
                    e.line(f"gj = j + {halo[1]}")
                    self._emit_stencil_body_2d(e, spec, halo)
            elif ndim == 3:
                e.line("i, j, k = cuda.grid(3)")
                e.line(
                    f"if i < output.shape[0] - 2 * {halo[0]}"
                    f" and j < output.shape[1] - 2 * {halo[1]}"
                    f" and k < output.shape[2] - 2 * {halo[2]}:"
                )
                with e.indent():
                    e.line(f"gi = i + {halo[0]}")
                    e.line(f"gj = j + {halo[1]}")
                    e.line(f"gk = k + {halo[2]}")
                    self._emit_stencil_body_3d(e, spec, halo)
            else:
                e.line(f"pass  # ndim={ndim} not supported")

    def _emit_stencil_body_1d(
        self, e: CodeEmitter, spec: StencilSpec, halo: int
    ) -> None:
        """Emit direct array access stencil computation for 1D kernels."""
        terms = []
        for acc in spec.accesses:
            off = acc.offsets[0]
            if off == 0:
                terms.append(f"{acc.array_name}[gi]")
            elif off > 0:
                terms.append(f"{acc.array_name}[gi + {off}]")
            else:
                terms.append(f"{acc.array_name}[gi - {abs(off)}]")

        try:
            expr = transform_stencil_expr(spec.update_fn, spec.accesses, spec.ndim, terms)
            e.line(f"output[gi] = {expr}")
        except Exception:
            e.line(f"output[gi] = {' + '.join(terms)}")

    def _emit_stencil_body_2d(
        self, e: CodeEmitter, spec: StencilSpec, halo: tuple
    ) -> None:
        """Emit direct array access stencil computation for 2D kernels."""
        terms = []
        for acc in spec.accesses:
            parts = []
            for d, off in enumerate(acc.offsets):
                idx = ["gi", "gj"][d]
                if off == 0:
                    parts.append(idx)
                elif off > 0:
                    parts.append(f"{idx} + {off}")
                else:
                    parts.append(f"{idx} - {abs(off)}")
            terms.append(f"{acc.array_name}[{', '.join(parts)}]")

        try:
            expr = transform_stencil_expr(spec.update_fn, spec.accesses, spec.ndim, terms)
            e.line(f"output[gi, gj] = {expr}")
        except Exception:
            e.line(f"output[gi, gj] = {' + '.join(terms)}")

    def _emit_stencil_body_3d(
        self, e: CodeEmitter, spec: StencilSpec, halo: tuple
    ) -> None:
        """Emit direct array access stencil computation for 3D kernels."""
        terms = []
        for acc in spec.accesses:
            parts = []
            for d, off in enumerate(acc.offsets):
                idx = ["gi", "gj", "gk"][d]
                if off == 0:
                    parts.append(idx)
                elif off > 0:
                    parts.append(f"{idx} + {off}")
                else:
                    parts.append(f"{idx} - {abs(off)}")
            terms.append(f"{acc.array_name}[{', '.join(parts)}]")

        try:
            expr = transform_stencil_expr(spec.update_fn, spec.accesses, spec.ndim, terms)
            e.line(f"output[gi, gj, gk] = {expr}")
        except Exception:
            e.line(f"output[gi, gj, gk] = {' + '.join(terms)}")

    def _emit_launcher(self, e: CodeEmitter) -> None:
        spec = self.spec
        ndim = spec.ndim
        tile = self.tile.tile_sizes
        halo = self.tile.halo_widths
        multi_input = len(spec.inputs) > 1

        if multi_input:
            params = ", ".join(spec.inputs)
        else:
            params = "u_in"

        e.blank()
        e.blank()
        e.line(f"def launch_{spec.name}({params}, u_out):")
        with e.indent():
            first = spec.inputs[0] if multi_input else "u_in"
            if ndim == 1:
                e.line(f"n = {first}.shape[0] - 2 * {halo[0]}")
                e.line(f"threads = {tile[0]}")
                e.line("blocks = (n + threads - 1) // threads")
                args = f"{params}, u_out" if multi_input else "u_in, u_out"
                e.line(f"{spec.name}_kernel[blocks, threads]({args})")
            elif ndim == 2:
                e.line(f"nx = {first}.shape[0] - 2 * {halo[0]}")
                e.line(f"ny = {first}.shape[1] - 2 * {halo[1]}")
                e.line(f"threads = ({min(tile[0], 16)}, {min(tile[1], 16)})")
                e.line("blocks = ((nx + threads[0] - 1) // threads[0], (ny + threads[1] - 1) // threads[1])")
                args = f"{params}, u_out" if multi_input else "u_in, u_out"
                e.line(f"{spec.name}_kernel[blocks, threads]({args})")
            elif ndim == 3:
                e.line(f"nx = {first}.shape[0] - 2 * {halo[0]}")
                e.line(f"ny = {first}.shape[1] - 2 * {halo[1]}")
                e.line(f"nz = {first}.shape[2] - 2 * {halo[2]}")
                e.line(f"threads = ({min(tile[0], 8)}, {min(tile[1], 8)}, {min(tile[2], 8)})")
                e.line("blocks = ((nx + threads[0] - 1) // threads[0], (ny + threads[1] - 1) // threads[1], (nz + threads[2] - 1) // threads[2])")
                args = f"{params}, u_out" if multi_input else "u_in, u_out"
                e.line(f"{spec.name}_kernel[blocks, threads]({args})")

    def emit_to_file(self, path) -> None:
        """Write generated kernel source to a file.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path. Parent directories are created if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.emit())
