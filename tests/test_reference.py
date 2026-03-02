"""Tests for NumPy reference stencil executor."""

import numpy as np
import pytest
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.registry import clear
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.reference.stencil_ref import apply_stencil, time_march


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


def test_heat_1d_single_step():
    """Verify one step of 1D heat equation against manual calculation."""
    @stencil(ndim=1, order=2)
    def heat(u, i):
        return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

    spec = heat._stencil_spec
    assert spec.ndim == 1
    assert spec.order == 2
    accesses = extract_footprint(spec)
    spec.halo_widths = compute_halo(accesses, 1)

    # Simple test: [0, 0, 1, 0, 0] with halo=1 → boundary cells are index 0 and 4
    u = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    result = apply_stencil(u, spec)
    # Interior is indices 1,2,3
    # result[1] = 0.25*0 + 0.5*0 + 0.25*1 = 0.25
    # result[2] = 0.25*0 + 0.5*1 + 0.25*0 = 0.5
    # result[3] = 0.25*1 + 0.5*0 + 0.25*0 = 0.25
    np.testing.assert_allclose(result[1], 0.25)
    np.testing.assert_allclose(result[2], 0.5)
    np.testing.assert_allclose(result[3], 0.25)


def test_heat_1d_energy_dissipation():
    """Heat equation should dissipate energy over time."""
    @stencil(ndim=1, order=2)
    def heat(u, i):
        return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

    spec = heat._stencil_spec
    accesses = extract_footprint(spec)
    spec.halo_widths = compute_halo(accesses, 1)

    N = 50
    u0 = np.zeros(N + 2)  # +2 for halo
    u0[N // 2 + 1] = 1.0  # Spike in middle
    initial_energy = np.sum(u0**2)

    history = time_march(u0, spec, steps=20)
    final_energy = np.sum(history[-1]**2)

    assert final_energy < initial_energy


def test_heat_1d_constant_preserved():
    """A constant field should remain constant under heat equation."""
    @stencil(ndim=1, order=2)
    def heat(u, i):
        return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

    spec = heat._stencil_spec
    accesses = extract_footprint(spec)
    spec.halo_widths = compute_halo(accesses, 1)

    u = np.ones(20)  # Constant everywhere including boundaries
    result = apply_stencil(u, spec)
    # Interior should remain 1.0 (weighted average of 1s)
    np.testing.assert_allclose(result[1:-1], 1.0, atol=1e-14)


def test_laplacian_2d():
    """Test 2D Laplacian on a known function."""
    @stencil(ndim=2, order=2)
    def lap2d(u, i, j):
        return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]

    spec = lap2d._stencil_spec
    assert spec.ndim == 2
    assert spec.order == 2
    accesses = extract_footprint(spec)
    spec.halo_widths = compute_halo(accesses, 2)

    # Point source: Laplacian of delta should give -4 at center
    N = 10
    u = np.zeros((N + 2, N + 2))
    c = N // 2 + 1
    u[c, c] = 1.0

    result = apply_stencil(u, spec)
    np.testing.assert_allclose(result[c, c], -4.0, atol=1e-14)
    # Neighbors get +1
    np.testing.assert_allclose(result[c - 1, c], 1.0, atol=1e-14)
    np.testing.assert_allclose(result[c + 1, c], 1.0, atol=1e-14)
    np.testing.assert_allclose(result[c, c - 1], 1.0, atol=1e-14)
    np.testing.assert_allclose(result[c, c + 1], 1.0, atol=1e-14)


def test_laplacian_3d():
    """Test 3D 7-point Laplacian."""
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

    N = 8
    u = np.zeros((N + 2, N + 2, N + 2))
    c = N // 2 + 1
    u[c, c, c] = 1.0

    result = apply_stencil(u, spec)
    np.testing.assert_allclose(result[c, c, c], -6.0, atol=1e-14)


def test_time_march_returns_correct_count():
    @stencil(ndim=1, order=2)
    def heat(u, i):
        return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

    spec = heat._stencil_spec
    accesses = extract_footprint(spec)
    spec.halo_widths = compute_halo(accesses, 1)

    u0 = np.zeros(20)
    history = time_march(u0, spec, steps=10)
    assert len(history) == 11  # initial + 10 steps
