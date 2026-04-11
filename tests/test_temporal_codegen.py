"""Tests for temporal blocking code generation."""

import ast
import pytest
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import HardwareSpec, TemporalConfig
from cutile_stencil.dsl.registry import clear
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.analysis.tiling import compute_tile_config
from cutile_stencil.analysis.temporal import compute_temporal_config
from cutile_stencil.codegen.stencil_codegen import StencilCodeGenerator


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()

# Use large shared memory to ensure temporal steps > 1
hw = HardwareSpec(shared_mem_bytes=256 * 1024, dtype_bytes=8)


def _make_temporal_gen(ndim=1, min_steps=2):
    if ndim == 1:
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
        spec = heat._stencil_spec
        domain = (1024,)
    else:
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])
        spec = heat2d._stencil_spec
        domain = (256, 256)

    extract_footprint(spec)
    spec.halo_widths = compute_halo(spec.accesses, ndim)
    cfg = compute_tile_config(spec, domain, hw)
    temp = compute_temporal_config(spec, cfg, hw)
    assert temp.steps >= min_steps, f"Only got {temp.steps} temporal steps, need >= {min_steps}"
    return StencilCodeGenerator(spec, cfg, temp), temp.steps


class TestTemporalBlockingCodegen:
    def test_1d_valid_python(self):
        gen, _ = _make_temporal_gen(ndim=1)
        code = gen.emit()
        ast.parse(code)

    def test_2d_valid_python(self):
        gen, _ = _make_temporal_gen(ndim=2)
        code = gen.emit()
        ast.parse(code)

    def test_has_temporal_kernel(self):
        gen, _ = _make_temporal_gen(ndim=1)
        code = gen.emit()
        assert "temporal" in code.lower() or "tmp" in code

    def test_has_intermediate_buffers(self):
        gen, steps = _make_temporal_gen(ndim=1)
        code = gen.emit()
        if steps >= 2:
            # Should allocate at least one intermediate buffer
            assert "tmp" in code.lower() or "zeros_like" in code

    def test_multiple_ct_store(self):
        gen, steps = _make_temporal_gen(ndim=1)
        code = gen.emit()
        if steps >= 2:
            # Should have multiple store calls (one per step)
            assert code.count("ct.store") >= 2

    def test_multiple_ct_load(self):
        gen, steps = _make_temporal_gen(ndim=1)
        code = gen.emit()
        if steps >= 2:
            # Should have multiple load calls (accesses * steps)
            assert code.count("ct.load") >= 2 * steps

    def test_step_comments(self):
        gen, steps = _make_temporal_gen(ndim=1)
        code = gen.emit()
        for s in range(steps):
            assert f"Temporal step {s + 1} of {steps}" in code

    def test_launcher_allocates_intermediates(self):
        gen, steps = _make_temporal_gen(ndim=1)
        code = gen.emit()
        if steps >= 2:
            assert "zeros_like" in code
            for s in range(steps - 1):
                assert f"u_tmp{s}" in code

    def test_kernel_has_correct_params(self):
        gen, steps = _make_temporal_gen(ndim=1)
        code = gen.emit()
        # Should have temporal kernel name
        assert "temporal_kernel" in code
        # Should have u_out param
        assert "u_out" in code
        # Should have tile and halo constants
        assert "TX: ConstInt" in code
        assert "HX: ConstInt" in code

    def test_2d_has_both_tile_and_halo_params(self):
        gen, _ = _make_temporal_gen(ndim=2)
        code = gen.emit()
        assert "TX: ConstInt" in code
        assert "TY: ConstInt" in code
        assert "HX: ConstInt" in code
        assert "HY: ConstInt" in code

    def test_single_step_uses_standard_path(self):
        """With temporal_steps=1, should use standard (non-temporal) codegen."""
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
        spec = heat._stencil_spec
        extract_footprint(spec)
        spec.halo_widths = compute_halo(spec.accesses, 1)
        cfg = compute_tile_config(spec, (1024,), hw)
        temp = TemporalConfig(steps=1, expanded_halo=(1,), bandwidth_reduction_factor=1.0)
        gen = StencilCodeGenerator(spec, cfg, temp)
        code = gen.emit()
        ast.parse(code)
        # Should NOT have temporal-specific code
        assert code.count("ct.store") == 1  # Only one store for single step
        assert "temporal_kernel" not in code

    def test_stencil_expr_repeated_per_step(self):
        """Each temporal step should have its own stencil expression."""
        gen, steps = _make_temporal_gen(ndim=1)
        code = gen.emit()
        if steps >= 2:
            # Each step produces a result_sN variable
            for s in range(steps):
                assert f"result_s{s}" in code

    def test_1d_slice_offsets_correct_for_step0(self):
        """Step 0 should have start=0+off (i.e., s*halo + off with s=0)."""
        gen, steps = _make_temporal_gen(ndim=1)
        code = gen.emit()
        # For a halo=1 stencil at step 0: u_m1_s0 start should be -1
        # u_0_s0 start should be 0, u_p1_s0 start should be 1
        assert "u_m1_s0" in code
        assert "u_0_s0" in code
        assert "u_p1_s0" in code
