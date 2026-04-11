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
