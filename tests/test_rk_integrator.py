"""Tests for Runge-Kutta time integrator."""

import ast
import pytest
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.registry import clear
from cutile_stencil.dsl.rk_integrator import RKIntegrator, RKCompileResult, _RK_METHODS


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


def _make_heat_fn():
    @stencil(ndim=1, order=2)
    def heat(u, i):
        return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
    return heat


class TestRKIntegrator:
    def test_rk2_creation(self):
        heat = _make_heat_fn()
        rk = RKIntegrator(heat, method="RK2")
        assert rk.method == "RK2"
        assert rk.stages == 2

    def test_rk4_creation(self):
        heat = _make_heat_fn()
        rk = RKIntegrator(heat, method="RK4")
        assert rk.method == "RK4"
        assert rk.stages == 4

    def test_ssprk3_creation(self):
        heat = _make_heat_fn()
        rk = RKIntegrator(heat, method="SSPRK3")
        assert rk.method == "SSPRK3"
        assert rk.stages == 3

    def test_invalid_method(self):
        heat = _make_heat_fn()
        with pytest.raises(ValueError, match="Unknown RK method"):
            RKIntegrator(heat, method="RK99")

    def test_non_stencil_raises(self):
        with pytest.raises(TypeError, match="Expected a @stencil"):
            RKIntegrator(lambda u, i: u, method="RK4")


class TestRKCompile:
    def test_rk4_compiles(self):
        heat = _make_heat_fn()
        rk = RKIntegrator(heat, method="RK4")
        result = rk.compile(domain=(1024,))
        assert isinstance(result, RKCompileResult)
        assert result.method == "RK4"
        assert result.stages == 4

    def test_generated_code_valid_python(self):
        heat = _make_heat_fn()
        rk = RKIntegrator(heat, method="RK4")
        result = rk.compile(domain=(1024,))
        # Both parts should be valid Python
        ast.parse(result.stencil_code)
        ast.parse(result.integrator_code)
        ast.parse(result.full_code())

    def test_rk4_has_4_stages(self):
        heat = _make_heat_fn()
        rk = RKIntegrator(heat, method="RK4")
        result = rk.compile(domain=(1024,))
        # Should reference k0, k1, k2, k3
        for s in range(4):
            assert f"k{s}" in result.integrator_code

    def test_rk2_has_2_stages(self):
        heat = _make_heat_fn()
        rk = RKIntegrator(heat, method="RK2")
        result = rk.compile(domain=(1024,))
        assert "k0" in result.integrator_code
        assert "k1" in result.integrator_code

    def test_integrator_calls_launch(self):
        heat = _make_heat_fn()
        rk = RKIntegrator(heat, method="RK4")
        result = rk.compile(domain=(1024,))
        assert "launch_heat" in result.integrator_code

    def test_all_methods_compile(self):
        """All registered methods should compile successfully."""
        heat = _make_heat_fn()
        for method in _RK_METHODS:
            rk = RKIntegrator(heat, method=method)
            result = rk.compile(domain=(1024,))
            ast.parse(result.full_code())


class TestButcherTableaux:
    def test_rk4_weights_sum_to_1(self):
        _, _, b, _ = _RK_METHODS["RK4"]
        assert abs(sum(b) - 1.0) < 1e-12

    def test_rk2_weights_sum_to_1(self):
        _, _, b, _ = _RK_METHODS["RK2"]
        assert abs(sum(b) - 1.0) < 1e-12

    def test_ssprk3_weights_sum_to_1(self):
        _, _, b, _ = _RK_METHODS["SSPRK3"]
        assert abs(sum(b) - 1.0) < 1e-12


def _has_gpu():
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


class TestRKGPUValidation:
    """Run RK4 integrator on GPU and verify against CPU RK4 reference."""

    @pytest.fixture(autouse=True)
    def _clear(self):
        clear()
        yield
        clear()

    @pytest.mark.skipif(
        not _has_gpu(), reason="No GPU available"
    )
    def test_rk4_matches_cpu_reference(self):
        import cupy as cp
        import numpy as np
        import importlib.util
        import tempfile
        from cutile_stencil.reference.stencil_ref import apply_stencil

        @stencil(ndim=1, order=2)
        def heat_rk(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]

        rk = RKIntegrator(heat_rk, method="RK4")
        result = rk.compile(domain=(256,))
        code = result.full_code()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(code)
            tmp_path = f.name
        mod_spec = importlib.util.spec_from_file_location("_rk_test", tmp_path)
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        rk_step = getattr(mod, f"rk_step_{heat_rk._stencil_spec.name}")

        # Setup
        np.random.seed(42)
        N = 256 + 2  # domain + halo
        u_np = np.random.randn(N).astype(np.float64)
        dt = 0.001
        n_steps = 10

        # GPU: RK4 steps
        u_gpu = cp.asarray(u_np)
        for _ in range(n_steps):
            u_gpu = rk_step(u_gpu, dt)
        cp.cuda.Device(0).synchronize()
        gpu_result = cp.asnumpy(u_gpu)

        # CPU: manual RK4 with apply_stencil as the operator L(u)
        # apply_stencil returns full array with interior=L(u), boundary=u_boundary
        # GPU k arrays have boundary=0, so we zero out boundaries to match
        spec = heat_rk._stencil_spec
        halo_w = spec.halo_widths[0]

        def cpu_L(u_arr):
            """Compute L(u) matching GPU behavior: interior=stencil, boundary=0."""
            result = np.zeros_like(u_arr)
            s = apply_stencil(u_arr, spec)
            result[halo_w:-halo_w] = s[halo_w:-halo_w]
            return result

        u_cpu = u_np.copy()
        for _ in range(n_steps):
            k1 = cpu_L(u_cpu)
            k2 = cpu_L(u_cpu + 0.5 * dt * k1)
            k3 = cpu_L(u_cpu + 0.5 * dt * k2)
            k4 = cpu_L(u_cpu + dt * k3)
            u_cpu = u_cpu + dt * (k1/6 + k2/3 + k3/3 + k4/6)

        # Compare interior
        gpu_int = gpu_result[1:-1]
        cpu_int = u_cpu[1:-1]
        maxdiff = np.max(np.abs(gpu_int - cpu_int))
        assert np.allclose(gpu_int, cpu_int, atol=1e-10), (
            f"RK4 GPU vs CPU mismatch: max_diff={maxdiff:.2e}"
        )
