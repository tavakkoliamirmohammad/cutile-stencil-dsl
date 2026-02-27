"""Tests for stencil DSL: spec creation and footprint extraction."""

import pytest
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import StencilSpec, OffsetAccess
from cutile_stencil.dsl.registry import lookup, clear
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo


@pytest.fixture(autouse=True)
def _clear_registry():
    clear()
    yield
    clear()


def test_stencil_decorator_creates_spec():
    @stencil(ndim=1, order=2)
    def my_stencil(u, i):
        return u[i - 1] + u[i] + u[i + 1]

    assert hasattr(my_stencil, "_stencil_spec")
    spec = my_stencil._stencil_spec
    assert isinstance(spec, StencilSpec)
    assert spec.name == "my_stencil"
    assert spec.ndim == 1
    assert spec.order == 2
    assert spec.inputs == ("u",)


def test_stencil_registered():
    @stencil(ndim=1, order=2)
    def registered_stencil(u, i):
        return u[i]

    found = lookup("registered_stencil")
    assert found is not None
    assert found.name == "registered_stencil"


def test_footprint_1d_heat():
    @stencil(ndim=1, order=2)
    def heat(u, i):
        return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

    spec = heat._stencil_spec
    accesses = extract_footprint(spec)
    offsets = {a.offsets for a in accesses}
    assert (-1,) in offsets
    assert (0,) in offsets
    assert (1,) in offsets


def test_footprint_2d():
    @stencil(ndim=2, order=2)
    def lap2d(u, i, j):
        return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]

    spec = lap2d._stencil_spec
    accesses = extract_footprint(spec)
    offsets = {a.offsets for a in accesses}
    assert (-1, 0) in offsets
    assert (1, 0) in offsets
    assert (0, -1) in offsets
    assert (0, 1) in offsets
    assert (0, 0) in offsets


def test_halo_computation():
    @stencil(ndim=2, order=4)
    def fourth_order(u, i, j):
        return u[i - 2, j] + u[i + 2, j] + u[i, j - 1] + u[i, j + 1]

    spec = fourth_order._stencil_spec
    accesses = extract_footprint(spec)
    halo = compute_halo(accesses, 2)
    assert halo == (2, 1)


def test_footprint_3d():
    @stencil(ndim=3, order=2)
    def lap3d(u, i, j, k):
        return (u[i - 1, j, k] + u[i + 1, j, k]
                + u[i, j - 1, k] + u[i, j + 1, k]
                + u[i, j, k - 1] + u[i, j, k + 1]
                - 6 * u[i, j, k])

    spec = lap3d._stencil_spec
    accesses = extract_footprint(spec)
    assert len(accesses) == 7
    halo = compute_halo(accesses, 3)
    assert halo == (1, 1, 1)


def test_callable_preserved():
    @stencil(ndim=1, order=2)
    def my_fn(u, i):
        return u[i]

    # The decorated function should still be callable
    class FakeArray:
        def __getitem__(self, key):
            return 42
    assert my_fn(FakeArray(), 0) == 42
