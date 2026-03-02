"""Tests for code generation: generated code must be valid Python (ast.parse)."""

import ast
import pytest
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import HardwareSpec
from cutile_stencil.dsl.registry import clear
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.analysis.tiling import compute_tile_config
from cutile_stencil.analysis.temporal import compute_temporal_config
from cutile_stencil.codegen.stencil_codegen import StencilCodeGenerator
from cutile_stencil.solvers.kernels import (
    generate_dia_spmv, generate_bsr_spmv, generate_axpy,
    generate_dot, generate_norm2,
)
from cutile_stencil.solvers.cg import generate_cg_driver
from cutile_stencil.solvers.persistent_cg import generate_persistent_cg
from cutile_stencil.solvers.mixed_precision import generate_mixed_precision_cg


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


hw = HardwareSpec(shared_mem_bytes=49152, dtype_bytes=8)


def _gen_1d():
    @stencil(ndim=1, order=2)
    def heat(u, i):
        return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
    spec = heat._stencil_spec
    assert spec.ndim == 1
    assert spec.order == 2
    accesses = extract_footprint(spec)
    spec.halo_widths = compute_halo(accesses, 1)
    cfg = compute_tile_config(spec, (1024,), hw)
    return StencilCodeGenerator(spec, cfg)


def _gen_2d():
    @stencil(ndim=2, order=2)
    def lap2d(u, i, j):
        return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]
    spec = lap2d._stencil_spec
    assert spec.ndim == 2
    assert spec.order == 2
    accesses = extract_footprint(spec)
    spec.halo_widths = compute_halo(accesses, 2)
    cfg = compute_tile_config(spec, (256, 256), hw)
    return StencilCodeGenerator(spec, cfg)


def _gen_3d():
    @stencil(ndim=3, order=2)
    def lap3d(u, i, j, k):
        return (u[i - 1, j, k] + u[i + 1, j, k]
                + u[i, j - 1, k] + u[i, j + 1, k]
                + u[i, j, k - 1] + u[i, j, k + 1]
                - 6 * u[i, j, k])
    spec = lap3d._stencil_spec
    assert spec.ndim == 3
    assert spec.order == 2
    accesses = extract_footprint(spec)
    spec.halo_widths = compute_halo(accesses, 3)
    cfg = compute_tile_config(spec, (64, 64, 64), hw)
    return StencilCodeGenerator(spec, cfg)


def _gen_2d_temporal():
    @stencil(ndim=2, order=2)
    def heat2d(u, i, j):
        return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])
    spec = heat2d._stencil_spec
    assert spec.ndim == 2
    assert spec.order == 2
    accesses = extract_footprint(spec)
    spec.halo_widths = compute_halo(accesses, 2)
    cfg = compute_tile_config(spec, (256, 256), hw)
    temp = compute_temporal_config(spec, cfg, hw)
    return StencilCodeGenerator(spec, cfg, temp)


class TestStencilCodegen:
    def test_1d_valid_python(self):
        code = _gen_1d().emit()
        ast.parse(code)

    def test_2d_valid_python(self):
        code = _gen_2d().emit()
        ast.parse(code)

    def test_3d_valid_python(self):
        code = _gen_3d().emit()
        ast.parse(code)

    def test_2d_temporal_valid_python(self):
        code = _gen_2d_temporal().emit()
        ast.parse(code)

    def test_1d_contains_kernel_decorator(self):
        code = _gen_1d().emit()
        assert "@ct.kernel" in code

    def test_1d_contains_ct_load_store(self):
        code = _gen_1d().emit()
        assert "ct.load" in code
        assert "ct.store" in code

    def test_2d_uses_in_kernel_slice(self):
        code = _gen_2d().emit()
        # 2D codegen uses in-kernel .slice() approach
        assert "ct.load" in code
        assert "ct.store" in code
        assert ".slice(axis=" in code

    def test_3d_uses_three_bid(self):
        code = _gen_3d().emit()
        assert "ct.bid(0)" in code
        assert "ct.bid(1)" in code
        assert "ct.bid(2)" in code

    def test_1d_uses_slice_naming(self):
        code = _gen_1d().emit()
        # New naming: u_m1, u_0, u_p1 (not v_u_offm1)
        assert "u_m1" in code
        assert "u_0" in code
        assert "u_p1" in code
        assert ".slice(axis=0" in code
        # Launcher should not create shifted views
        assert "v_u_" not in code

    def test_1d_kernel_params_simple(self):
        code = _gen_1d().emit()
        # Kernel params: just the array + output + TILE + HALO
        assert "heat_kernel(u, output, TILE: ConstInt, HALO: ConstInt)" in code

    def test_2d_kernel_has_halo_params(self):
        code = _gen_2d().emit()
        assert "HX: ConstInt" in code
        assert "HY: ConstInt" in code

    def test_3d_kernel_has_halo_params(self):
        code = _gen_3d().emit()
        assert "HX: ConstInt" in code
        assert "HY: ConstInt" in code
        assert "HZ: ConstInt" in code

    def test_emit_to_file(self, tmp_path):
        gen = _gen_1d()
        out = tmp_path / "test_kernel.py"
        gen.emit_to_file(out)
        assert out.exists()
        code = out.read_text()
        ast.parse(code)


class TestSolverKernelCodegen:
    def test_dia_spmv(self):
        code = generate_dia_spmv()
        ast.parse(code)
        assert "dia_spmv_kernel" in code

    def test_bsr_spmv(self):
        code = generate_bsr_spmv()
        ast.parse(code)
        assert "bsr_spmv_kernel" in code

    def test_axpy(self):
        code = generate_axpy()
        ast.parse(code)
        assert "axpy_kernel" in code

    def test_dot(self):
        code = generate_dot()
        ast.parse(code)
        assert "ct.atomic_add" in code

    def test_norm2(self):
        code = generate_norm2()
        ast.parse(code)
        assert "norm2_kernel" in code

    def test_cg_driver(self):
        code = generate_cg_driver()
        ast.parse(code)
        assert "cg_solve" in code

    def test_persistent_cg(self):
        code = generate_persistent_cg()
        ast.parse(code)
        assert "persistent_cg_kernel" in code
        assert "ct.atomic_cas" in code

    def test_mixed_precision_cg(self):
        code = generate_mixed_precision_cg()
        ast.parse(code)
        assert "ct.astype" in code
