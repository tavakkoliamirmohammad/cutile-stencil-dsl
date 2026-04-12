"""Tests for the Numba CUDA backend code generator."""

import ast
import pytest
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import HardwareSpec
from cutile_stencil.dsl.registry import clear
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.analysis.tiling import compute_tile_config
from cutile_stencil.codegen.numba_codegen import NumbaCudaCodeGenerator


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


hw = HardwareSpec(shared_mem_bytes=49152, dtype_bytes=8)


def _make_gen(ndim):
    """Create a NumbaCudaCodeGenerator for a standard test stencil of given ndim."""
    if ndim == 1:
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
        spec = heat._stencil_spec
        domain = (1024,)
    elif ndim == 2:
        @stencil(ndim=2, order=2)
        def lap2d(u, i, j):
            return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]
        spec = lap2d._stencil_spec
        domain = (256, 256)
    else:
        @stencil(ndim=3, order=2)
        def lap3d(u, i, j, k):
            return (
                u[i - 1, j, k] + u[i + 1, j, k]
                + u[i, j - 1, k] + u[i, j + 1, k]
                + u[i, j, k - 1] + u[i, j, k + 1]
                - 6 * u[i, j, k]
            )
        spec = lap3d._stencil_spec
        domain = (64, 64, 64)

    extract_footprint(spec)
    spec.halo_widths = compute_halo(spec.accesses, ndim)
    cfg = compute_tile_config(spec, domain, hw)
    return NumbaCudaCodeGenerator(spec, cfg)


class TestNumbaCudaCodegen:
    def test_1d_valid_python(self):
        """1D generated kernel is valid Python."""
        ast.parse(_make_gen(1).emit())

    def test_2d_valid_python(self):
        """2D generated kernel is valid Python."""
        ast.parse(_make_gen(2).emit())

    def test_3d_valid_python(self):
        """3D generated kernel is valid Python."""
        ast.parse(_make_gen(3).emit())

    def test_has_cuda_jit(self):
        """Generated code uses @cuda.jit decorator."""
        assert "@cuda.jit" in _make_gen(1).emit()

    def test_has_numba_import(self):
        """Generated code imports numba.cuda."""
        assert "from numba import cuda" in _make_gen(1).emit()

    def test_has_cuda_grid(self):
        """Generated code uses cuda.grid() for thread indexing."""
        assert "cuda.grid" in _make_gen(2).emit()

    def test_has_launcher(self):
        """Generated code includes a launch_<name> function."""
        assert "def launch_heat(" in _make_gen(1).emit()

    def test_emit_to_file(self, tmp_path):
        """emit_to_file writes valid Python to disk."""
        gen = _make_gen(1)
        out = tmp_path / "numba_kernel.py"
        gen.emit_to_file(out)
        assert out.exists()
        ast.parse(out.read_text())

    def test_1d_kernel_function_signature(self):
        """1D kernel is named <stencil>_kernel with correct parameters."""
        src = _make_gen(1).emit()
        assert "def heat_kernel(u, output):" in src

    def test_2d_kernel_function_signature(self):
        """2D kernel is named <stencil>_kernel with correct parameters."""
        src = _make_gen(2).emit()
        assert "def lap2d_kernel(u, output):" in src

    def test_3d_kernel_function_signature(self):
        """3D kernel is named <stencil>_kernel with correct parameters."""
        src = _make_gen(3).emit()
        assert "def lap3d_kernel(u, output):" in src

    def test_1d_uses_cuda_grid_1(self):
        """1D kernel uses cuda.grid(1) for thread index."""
        src = _make_gen(1).emit()
        assert "cuda.grid(1)" in src

    def test_2d_uses_cuda_grid_2(self):
        """2D kernel uses cuda.grid(2) for thread indices."""
        src = _make_gen(2).emit()
        assert "cuda.grid(2)" in src

    def test_3d_uses_cuda_grid_3(self):
        """3D kernel uses cuda.grid(3) for thread indices."""
        src = _make_gen(3).emit()
        assert "cuda.grid(3)" in src

    def test_has_numpy_import(self):
        """Generated code imports numpy."""
        assert "import numpy as np" in _make_gen(1).emit()

    def test_2d_launcher(self):
        """2D launcher function is present with correct name."""
        src = _make_gen(2).emit()
        assert "def launch_lap2d(" in src

    def test_3d_launcher(self):
        """3D launcher function is present with correct name."""
        src = _make_gen(3).emit()
        assert "def launch_lap3d(" in src

    def test_emit_to_file_creates_parent_dirs(self, tmp_path):
        """emit_to_file creates parent directories if they don't exist."""
        gen = _make_gen(2)
        out = tmp_path / "sub" / "dir" / "kernel.py"
        gen.emit_to_file(out)
        assert out.exists()

    def test_3d_has_bounds_check_all_dims(self):
        """3D kernel has bounds checks for all three dimensions."""
        src = _make_gen(3).emit()
        # Should check bounds in i, j, and k
        assert "output.shape[0]" in src
        assert "output.shape[1]" in src
        assert "output.shape[2]" in src

    def test_closure_constants_emitted(self):
        """Closure constants from stencil function are emitted as module-level vars."""
        dt = 0.01
        dx = 0.1

        @stencil(ndim=1, order=2)
        def diffusion(u, i):
            return u[i] + dt * (u[i - 1] - 2 * u[i] + u[i + 1]) / dx**2

        spec = diffusion._stencil_spec
        domain = (512,)
        extract_footprint(spec)
        spec.halo_widths = compute_halo(spec.accesses, 1)
        cfg = compute_tile_config(spec, domain, hw)
        gen = NumbaCudaCodeGenerator(spec, cfg)
        src = gen.emit()
        # dt and dx should appear as constants
        assert "dt" in src
        assert "dx" in src
