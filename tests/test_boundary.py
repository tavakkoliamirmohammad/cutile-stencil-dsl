"""Tests for boundary condition abstractions."""

import numpy as np
import pytest
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import StencilSpec
from cutile_stencil.dsl.registry import clear
from cutile_stencil.dsl.boundary import (
    BoundaryType, BoundaryCondition, BoundarySpec,
)
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.reference.stencil_ref import time_march, _apply_boundary_conditions


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


def _make_heat_spec(boundary=None):
    @stencil(ndim=1, order=2, boundary=boundary)
    def heat(u, i):
        return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

    spec = heat._stencil_spec
    accesses = extract_footprint(spec)
    spec.halo_widths = compute_halo(accesses, 1)
    return spec


class TestBoundaryTypes:
    def test_dirichlet_default(self):
        bc = BoundaryCondition(BoundaryType.DIRICHLET)
        assert bc.value == 0.0

    def test_dirichlet_custom_value(self):
        bc = BoundaryCondition(BoundaryType.DIRICHLET, value=1.0)
        assert bc.value == 1.0

    def test_boundary_spec_uniform(self):
        bs = BoundarySpec.uniform(BoundaryType.PERIODIC, 3)
        assert len(bs.conditions) == 3
        for low, high in bs.conditions:
            assert low.bc_type == BoundaryType.PERIODIC
            assert high.bc_type == BoundaryType.PERIODIC

    def test_dirichlet_shortcut(self):
        bs = BoundarySpec.dirichlet(2, value=5.0)
        assert len(bs.conditions) == 2
        assert bs.conditions[0][0].bc_type == BoundaryType.DIRICHLET
        assert bs.conditions[0][0].value == 5.0

    def test_periodic_shortcut(self):
        bs = BoundarySpec.periodic(1)
        assert bs.conditions[0][0].bc_type == BoundaryType.PERIODIC

    def test_neumann_shortcut(self):
        bs = BoundarySpec.neumann(2)
        assert bs.conditions[0][0].bc_type == BoundaryType.NEUMANN

    def test_padding_mode_mapping(self):
        assert BoundarySpec.dirichlet(1).cuTile_padding_mode(0) == "ZERO"
        assert BoundarySpec.neumann(1).cuTile_padding_mode(0) == "CLAMP"
        assert BoundarySpec.periodic(1).cuTile_padding_mode(0) == "WRAP"
        bs = BoundarySpec.uniform(BoundaryType.REFLECTING, 1)
        assert bs.cuTile_padding_mode(0) == "REFLECT"


class TestDirichletBC:
    def test_1d_dirichlet_zeros(self):
        """Dirichlet BC sets boundaries to zero."""
        spec = _make_heat_spec(boundary=BoundarySpec.dirichlet(1))
        u = np.ones(20)
        _apply_boundary_conditions(u, spec.boundary, spec.halo_widths)
        assert u[0] == 0.0
        assert u[-1] == 0.0
        assert u[5] == 1.0  # Interior unchanged

    def test_1d_dirichlet_nonzero(self):
        """Dirichlet BC with custom value."""
        bc = BoundarySpec.dirichlet(1, value=3.0)
        u = np.zeros(10)
        _apply_boundary_conditions(u, bc, (1,))
        assert u[0] == 3.0
        assert u[-1] == 3.0


class TestNeumannBC:
    def test_1d_neumann(self):
        """Neumann BC copies interior neighbor to boundary."""
        bc = BoundarySpec.neumann(1)
        u = np.array([99.0, 2.0, 3.0, 4.0, 99.0])
        _apply_boundary_conditions(u, bc, (1,))
        # Low: u[0] = u[1] = 2.0
        assert u[0] == 2.0
        # High: u[4] = u[3] = 4.0
        assert u[4] == 4.0


class TestPeriodicBC:
    def test_1d_periodic(self):
        """Periodic BC wraps values around."""
        bc = BoundarySpec.periodic(1)
        # Array: [ghost, 1, 2, 3, 4, ghost] with halo=1
        u = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 0.0])
        _apply_boundary_conditions(u, bc, (1,))
        # Low ghost = last interior value
        assert u[0] == 4.0
        # High ghost = first interior value
        assert u[5] == 1.0


class TestReflectingBC:
    def test_1d_reflecting(self):
        """Reflecting BC mirrors interior values."""
        bc = BoundarySpec.uniform(BoundaryType.REFLECTING, 1)
        u = np.array([0.0, 5.0, 6.0, 7.0, 0.0])
        _apply_boundary_conditions(u, bc, (1,))
        # Low: u[0] = u[1] = 5.0 (reflect at boundary)
        assert u[0] == 5.0
        # High: u[4] = u[3] = 7.0
        assert u[4] == 7.0


class TestTimeMarchWithBC:
    def test_dirichlet_maintained_during_march(self):
        """Dirichlet BCs are maintained throughout time marching."""
        bc = BoundarySpec.dirichlet(1, value=0.0)
        spec = _make_heat_spec(boundary=bc)
        u0 = np.zeros(22)  # N=20, halo=1
        u0[11] = 1.0  # Spike
        history = time_march(u0, spec, steps=10)
        for snap in history:
            assert snap[0] == 0.0
            assert snap[-1] == 0.0

    def test_periodic_conservation(self):
        """Periodic BCs should approximately conserve total mass."""
        bc = BoundarySpec.periodic(1)
        spec = _make_heat_spec(boundary=bc)
        u0 = np.zeros(22)
        u0[5:15] = 1.0
        _apply_boundary_conditions(u0, bc, spec.halo_widths)
        initial_mass = np.sum(u0[1:-1])
        history = time_march(u0, spec, steps=20)
        final_mass = np.sum(history[-1][1:-1])
        np.testing.assert_allclose(final_mass, initial_mass, rtol=0.01)


class TestBoundaryInDecorator:
    def test_spec_has_boundary(self):
        bc = BoundarySpec.periodic(2)

        @stencil(ndim=2, order=2, boundary=bc)
        def lap(u, i, j):
            return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]

        assert lap._stencil_spec.boundary is bc

    def test_spec_boundary_none_by_default(self):
        @stencil(ndim=1, order=2)
        def s(u, i):
            return u[i]

        assert s._stencil_spec.boundary is None


class TestCodegenPaddingMode:
    def test_periodic_generates_valid_shifted_views(self):
        """Codegen with periodic BC generates valid shifted-view code."""
        from cutile_stencil.analysis.tiling import compute_tile_config
        from cutile_stencil.codegen.stencil_codegen import StencilCodeGenerator
        from cutile_stencil.dsl.types import HardwareSpec
        import ast

        bc = BoundarySpec.periodic(2)

        @stencil(ndim=2, order=2, boundary=bc)
        def lap(u, i, j):
            return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]

        spec = lap._stencil_spec
        accesses = extract_footprint(spec)
        spec.halo_widths = compute_halo(accesses, 2)
        hw = HardwareSpec(shared_mem_bytes=49152, dtype_bytes=8)
        cfg = compute_tile_config(spec, (64, 64), hw)
        gen = StencilCodeGenerator(spec, cfg)
        code = gen.emit()
        ast.parse(code)
        # Shifted-view approach uses ct.load with power-of-two tiles
        assert "ct.load" in code
        assert "ct.store" in code
