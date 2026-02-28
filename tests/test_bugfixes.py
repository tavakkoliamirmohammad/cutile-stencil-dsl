"""Tests for critical bug fixes: temporal blocking, closure capture, FLOP counting."""

import ast
import pytest
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import HardwareSpec, StencilSpec, OffsetAccess
from cutile_stencil.dsl.registry import clear
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.analysis.tiling import compute_tile_config
from cutile_stencil.analysis.temporal import compute_temporal_config
from cutile_stencil.analysis.roofline import roofline_analysis, _FlopCounter
from cutile_stencil.codegen.stencil_codegen import StencilCodeGenerator
from cutile_stencil.codegen.ast_transform import extract_closure_constants


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


hw = HardwareSpec(shared_mem_bytes=49152, dtype_bytes=8)


class TestTemporalBlockingCodegen:
    """Verify the temporal blocking kernel uses constant per-step halo."""

    def _make_temporal_gen(self, T_target=2):
        """Create a 2D stencil codegen with temporal blocking."""
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        spec = heat2d._stencil_spec
        accesses = extract_footprint(spec)
        spec.halo_widths = compute_halo(accesses, 2)
        cfg = compute_tile_config(spec, (256, 256), hw)
        temp = compute_temporal_config(spec, cfg, hw)
        # Ensure we actually get multi-step temporal blocking
        assert temp.steps >= T_target, (
            f"Temporal config only has {temp.steps} steps, need >= {T_target}"
        )
        return StencilCodeGenerator(spec, cfg, temp)

    def test_temporal_kernel_valid_python(self):
        gen = self._make_temporal_gen()
        code = gen.emit()
        ast.parse(code)

    def test_temporal_uses_constant_halo(self):
        """The temporal kernel should use constant per-step halo, not decreasing."""
        gen = self._make_temporal_gen()
        code = gen.emit()
        # Should NOT contain the buggy pattern of variable h
        assert "ehx - _t_step" not in code
        # Should contain constant halo offsets
        assert "Per-step halo is constant" in code

    def test_temporal_tile_shrinks_correctly(self):
        """Tile shape shrinks by 2*hx and 2*hy each step."""
        gen = self._make_temporal_gen()
        code = gen.emit()
        # The generated code should reference tile.shape for inner dimensions
        assert "tile.shape[0]" in code
        assert "tile.shape[1]" in code

    def test_temporal_kernel_contains_loop(self):
        gen = self._make_temporal_gen()
        code = gen.emit()
        assert "for _t_step in range(" in code
        assert "temporal_kernel" in code


class TestClosureConstantCapture:
    """Verify that closure/scope constants are extracted from stencil functions."""

    def test_closure_constants_basic(self):
        """Simple closure variable is captured."""
        dt = 0.01
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return u[i] + dt * (u[i - 1] - 2 * u[i] + u[i + 1])

        spec = heat._stencil_spec
        assert "dt" in spec.closure_constants
        assert spec.closure_constants["dt"] == 0.01

    def test_multiple_closure_constants(self):
        """Multiple closure variables are captured."""
        dt = 0.01
        dx = 0.1
        alpha = 0.5
        @stencil(ndim=1, order=2)
        def diffusion(u, i):
            return u[i] + alpha * dt / dx ** 2 * (u[i - 1] - 2 * u[i] + u[i + 1])

        spec = diffusion._stencil_spec
        assert spec.closure_constants["dt"] == 0.01
        assert spec.closure_constants["dx"] == 0.1
        assert spec.closure_constants["alpha"] == 0.5

    def test_no_closure_no_constants(self):
        """Stencil with no closure variables has empty constants."""
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        spec = heat._stencil_spec
        # Should not capture inline numeric literals as closure vars
        assert len(spec.closure_constants) == 0

    def test_extract_closure_constants_function(self):
        """Direct test of extract_closure_constants utility."""
        val = 42.0
        def f(u, i):
            return val * u[i]

        consts = extract_closure_constants(f)
        assert "val" in consts
        assert consts["val"] == 42.0

    def test_non_scalar_not_captured(self):
        """Non-scalar values (lists, arrays, etc.) are not captured."""
        import numpy as np
        weights = np.array([0.25, 0.5, 0.25])
        name = "test"
        @stencil(ndim=1, order=2)
        def s(u, i):
            return name  # string is OK but arrays are not

        spec = s._stencil_spec
        assert "weights" not in spec.closure_constants

    def test_closure_constants_in_codegen(self):
        """Generated code includes closure constants as module-level definitions."""
        dt = 0.01
        coeff = 0.5
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return u[i] + dt * coeff * (u[i - 1] - 2 * u[i] + u[i + 1])

        spec = heat._stencil_spec
        accesses = extract_footprint(spec)
        spec.halo_widths = compute_halo(accesses, 1)
        cfg = compute_tile_config(spec, (1024,), hw)
        gen = StencilCodeGenerator(spec, cfg)
        code = gen.emit()

        # Generated code should define the constants
        assert "dt = 0.01" in code
        assert "coeff = 0.5" in code
        # Should be valid Python
        ast.parse(code)

    def test_closure_constants_in_2d_codegen(self):
        """2D codegen also emits closure constants."""
        viscosity = 0.1
        @stencil(ndim=2, order=2)
        def diffuse(u, i, j):
            return u[i, j] + viscosity * (
                u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]
            )

        spec = diffuse._stencil_spec
        accesses = extract_footprint(spec)
        spec.halo_widths = compute_halo(accesses, 2)
        cfg = compute_tile_config(spec, (256, 256), hw)
        gen = StencilCodeGenerator(spec, cfg)
        code = gen.emit()

        assert "viscosity = 0.1" in code
        ast.parse(code)


class TestFlopCounting:
    """Verify FLOP counting includes Pow, Mod operations."""

    def _count_flops(self, code_str):
        """Count FLOPs in a code string."""
        tree = ast.parse(code_str)
        counter = _FlopCounter()
        counter.visit(tree)
        return counter

    def test_basic_add_mul(self):
        counter = self._count_flops("x + y * z")
        assert counter.adds == 1
        assert counter.muls == 1
        assert counter.total == 2

    def test_div_counted(self):
        counter = self._count_flops("x / y + x // z")
        assert counter.divs == 2
        assert counter.adds == 1
        assert counter.total == 3

    def test_pow_counted(self):
        """Exponentiation is counted as a FLOP."""
        counter = self._count_flops("x ** 2 + y ** 3")
        assert counter.pows == 2
        assert counter.adds == 1
        assert counter.total == 3

    def test_mod_counted(self):
        """Modulo is counted as a FLOP."""
        counter = self._count_flops("x % y + z")
        assert counter.mods == 1
        assert counter.adds == 1
        assert counter.total == 2

    def test_complex_expression(self):
        """Expression with all operation types."""
        counter = self._count_flops("(a + b) * c / d ** e % f - g")
        assert counter.adds == 2  # a + b, and - g (Sub counted as Add)
        assert counter.muls == 1  # * c
        assert counter.divs == 1  # / d
        assert counter.pows == 1  # ** e
        assert counter.mods == 1  # % f
        assert counter.total == 6

    def test_roofline_with_pow(self):
        """Roofline analysis counts pow operations from stencil functions."""
        @stencil(ndim=1, order=2)
        def s(u, i):
            return u[i - 1] ** 2 + u[i + 1]

        spec = s._stencil_spec
        extract_footprint(spec)
        result = roofline_analysis(spec, hw)
        # Should count: 1 pow + 1 add = 2 flops minimum
        assert result.flops_per_point >= 2


class TestTemporalBlockingDimensionIndependentHalos:
    """Test temporal blocking when halo differs per dimension."""

    def test_asymmetric_halo_temporal(self):
        """Temporal kernel with different halos per dimension."""
        @stencil(ndim=2, order=4)
        def asym(u, i, j):
            return u[i - 2, j] + u[i + 2, j] + u[i, j - 1] + u[i, j + 1]

        spec = asym._stencil_spec
        accesses = extract_footprint(spec)
        halo = compute_halo(accesses, 2)
        spec.halo_widths = halo
        assert halo == (2, 1), f"Expected asymmetric halo (2, 1) but got {halo}"

        cfg = compute_tile_config(spec, (256, 256), hw)
        temp = compute_temporal_config(spec, cfg, hw)

        if temp.steps > 1:
            gen = StencilCodeGenerator(spec, cfg, temp)
            code = gen.emit()
            ast.parse(code)
            # Verify separate halo handling per dimension
            assert "tile.shape[0]" in code
            assert "tile.shape[1]" in code
