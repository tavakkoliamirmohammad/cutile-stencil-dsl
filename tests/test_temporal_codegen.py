"""Tests for temporal blocking code generation."""

import ast
import importlib.util
import tempfile
import numpy as np
import pytest
from conftest import gpu_required
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import HardwareSpec, TemporalConfig
from cutile_stencil.dsl.registry import clear
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.analysis.tiling import compute_tile_config
from cutile_stencil.analysis.temporal import compute_temporal_config
from cutile_stencil.codegen.stencil_codegen import StencilCodeGenerator
from cutile_stencil.reference.stencil_ref import apply_stencil


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

    def test_has_kernel(self):
        gen, _ = _make_temporal_gen(ndim=1)
        code = gen.emit()
        assert "_kernel" in code

    def test_has_buffer_allocation(self):
        gen, steps = _make_temporal_gen(ndim=1)
        code = gen.emit()
        assert "zeros_like" in code

    def test_has_launch_loop(self):
        """Temporal launcher should loop over T kernel launches."""
        gen, steps = _make_temporal_gen(ndim=1)
        code = gen.emit()
        assert f"range({steps})" in code

    def test_has_buffer_chain(self):
        gen, steps = _make_temporal_gen(ndim=1)
        code = gen.emit()
        assert "bufs" in code
        assert "bufs[_step]" in code or "bufs[_step + 1]" in code

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
        assert code.count("ct.store") == 1
        assert "bufs" not in code  # No buffer chain for single step

    def test_2d_has_correct_params(self):
        gen, _ = _make_temporal_gen(ndim=2)
        code = gen.emit()
        assert "TX: ConstInt" in code
        assert "TY: ConstInt" in code
        assert "HX: ConstInt" in code
        assert "HY: ConstInt" in code


class TestTemporalBlockingGPUValidation:
    """Run temporal-blocking kernels on GPU and verify against CPU reference."""

    def _run_temporal_validation(self, ndim):
        """Generate temporal kernel, run on GPU, compare T steps vs CPU reference."""
        import cupy as cp

        # Use modest shared mem to get T=2-4 (not huge T)
        test_hw = HardwareSpec(shared_mem_bytes=49152, dtype_bytes=8)

        if ndim == 1:
            @stencil(ndim=1, order=2)
            def heat(u, i):
                return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
            spec = heat._stencil_spec
            domain = (512,)
        elif ndim == 2:
            @stencil(ndim=2, order=2)
            def heat2d(u, i, j):
                return 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])
            spec = heat2d._stencil_spec
            domain = (128, 128)
        else:
            # 3D tiles are huge (32^3 * 8B = 256KB), so temporal blocking
            # needs unrealistic shared memory. Use 2MB to get T>=2.
            test_hw = HardwareSpec(shared_mem_bytes=2048 * 1024, dtype_bytes=8)
            @stencil(ndim=3, order=2)
            def heat3d(u, i, j, k):
                return (u[i-1,j,k] + u[i+1,j,k] + u[i,j-1,k] + u[i,j+1,k] + u[i,j,k-1] + u[i,j,k+1]) / 6.0
            spec = heat3d._stencil_spec
            domain = (32, 32, 32)

        extract_footprint(spec)
        halo = compute_halo(spec.accesses, ndim)
        spec.halo_widths = halo
        cfg = compute_tile_config(spec, domain, test_hw)
        temp = compute_temporal_config(spec, cfg, test_hw)
        T = temp.steps
        assert T >= 2, f"Need T>=2 for temporal test, got {T}"

        gen = StencilCodeGenerator(spec, cfg, temp)
        code = gen.emit()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(code)
            tmp_path = f.name
        mod_spec = importlib.util.spec_from_file_location(
            f"_temporal_test_{ndim}d", tmp_path
        )
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        launch = getattr(mod, f"launch_{spec.name}")

        # Expanded input: n + 2*T*halo per dimension
        np.random.seed(42)
        expanded_shape = tuple(d + 2 * T * h for d, h in zip(domain, halo))
        u_np = np.random.randn(*expanded_shape).astype(np.float64)

        # GPU: temporal launcher does T steps internally
        u_gpu = cp.asarray(u_np)
        out_gpu = cp.zeros_like(u_gpu)
        launch(u_gpu, out_gpu)
        cp.cuda.Device(0).synchronize()
        gpu_result = cp.asnumpy(out_gpu)

        # CPU: apply stencil T times
        u_cpu = u_np.copy()
        for _ in range(T):
            u_cpu = apply_stencil(u_cpu, spec)

        # Compare interior (strip T*halo from each side)
        margins = tuple(T * h for h in halo)
        slices = tuple(slice(m, -m) for m in margins)
        gpu_interior = gpu_result[slices]
        cpu_interior = u_cpu[slices]

        maxdiff = np.max(np.abs(gpu_interior - cpu_interior))
        assert np.allclose(gpu_interior, cpu_interior, atol=1e-10), (
            f"{ndim}D temporal blocking ({T} steps): max diff = {maxdiff:.2e}"
        )

    @gpu_required
    def test_1d_temporal_matches_reference(self):
        self._run_temporal_validation(ndim=1)

    @gpu_required
    def test_2d_temporal_matches_reference(self):
        self._run_temporal_validation(ndim=2)

    @gpu_required
    def test_3d_temporal_matches_reference(self):
        self._run_temporal_validation(ndim=3)
