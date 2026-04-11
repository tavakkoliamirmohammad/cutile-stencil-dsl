"""Tests for Triton backend code generation."""

import ast
import pytest
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import HardwareSpec
from cutile_stencil.dsl.registry import clear
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.analysis.tiling import compute_tile_config
from cutile_stencil.codegen.triton_codegen import TritonCodeGenerator


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


hw = HardwareSpec(shared_mem_bytes=49152, dtype_bytes=8)


def _make_gen(ndim):
    if ndim == 1:
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
        spec = heat._stencil_spec
        domain = (1024,)
    elif ndim == 2:
        @stencil(ndim=2, order=2)
        def lap2d(u, i, j):
            return u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]
        spec = lap2d._stencil_spec
        domain = (256, 256)
    else:
        @stencil(ndim=3, order=2)
        def lap3d(u, i, j, k):
            return (u[i-1,j,k] + u[i+1,j,k] + u[i,j-1,k] + u[i,j+1,k]
                    + u[i,j,k-1] + u[i,j,k+1] - 6*u[i,j,k])
        spec = lap3d._stencil_spec
        domain = (64, 64, 64)

    accesses = extract_footprint(spec)
    spec.halo_widths = compute_halo(accesses, ndim)
    cfg = compute_tile_config(spec, domain, hw)
    return TritonCodeGenerator(spec, cfg)


class TestTritonCodegen:
    def test_1d_valid_python(self):
        code = _make_gen(1).emit()
        ast.parse(code)

    def test_2d_valid_python(self):
        code = _make_gen(2).emit()
        ast.parse(code)

    def test_3d_valid_python(self):
        code = _make_gen(3).emit()
        ast.parse(code)

    def test_has_triton_jit(self):
        code = _make_gen(1).emit()
        assert "@triton.jit" in code

    def test_has_triton_imports(self):
        code = _make_gen(1).emit()
        assert "import triton" in code
        assert "import triton.language as tl" in code

    def test_has_tl_load_store(self):
        code = _make_gen(2).emit()
        assert "tl.load(" in code
        assert "tl.store(" in code

    def test_has_program_id(self):
        code = _make_gen(2).emit()
        assert "tl.program_id(0)" in code
        assert "tl.program_id(1)" in code

    def test_has_launcher(self):
        code = _make_gen(1).emit()
        assert "def launch_heat(" in code

    def test_has_constexpr(self):
        code = _make_gen(1).emit()
        assert "tl.constexpr" in code

    def test_3d_has_three_program_ids(self):
        code = _make_gen(3).emit()
        assert "tl.program_id(0)" in code
        assert "tl.program_id(1)" in code
        assert "tl.program_id(2)" in code

    def test_has_mask(self):
        code = _make_gen(1).emit()
        assert "mask=" in code

    def test_emit_to_file(self, tmp_path):
        gen = _make_gen(1)
        out = tmp_path / "triton_kernel.py"
        gen.emit_to_file(out)
        assert out.exists()
        ast.parse(out.read_text())
