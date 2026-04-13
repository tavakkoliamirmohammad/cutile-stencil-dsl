"""Tests for the stencil-to-cuTile lowering pass and Python emitter.

Verifies that the lowering from Dialect 1 IR to cuTile Python source
produces syntactically valid code matching the cuTile API.
"""

from __future__ import annotations

import ast

import pytest

from cutile.frontend.decorator import stencil
from cutile.lowering.stencil_to_cutile import lower_stencil_to_python
from cutile.lowering.emitter import CodeEmitter


# -------------------------------------------------------------------- #
# Helpers
# -------------------------------------------------------------------- #


def _assert_valid_python(code: str) -> None:
    """Assert that *code* is valid Python syntax."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        pytest.fail(f"Generated code is not valid Python:\n{exc}\n\n{code}")


# -------------------------------------------------------------------- #
# CodeEmitter unit tests
# -------------------------------------------------------------------- #


class TestCodeEmitter:
    def test_line_and_render(self):
        e = CodeEmitter()
        e.line("x = 1")
        e.line("y = 2")
        result = e.render()
        assert result == "x = 1\ny = 2\n"

    def test_blank(self):
        e = CodeEmitter()
        e.line("a")
        e.blank()
        e.line("b")
        assert e.render() == "a\n\nb\n"

    def test_indent(self):
        e = CodeEmitter()
        e.line("if True:")
        with e.indent():
            e.line("pass")
        e.line("done")
        assert e.render() == "if True:\n    pass\ndone\n"

    def test_nested_indent(self):
        e = CodeEmitter()
        e.line("for i in range(10):")
        with e.indent():
            e.line("if i > 0:")
            with e.indent():
                e.line("print(i)")
        result = e.render()
        assert "        print(i)" in result


# -------------------------------------------------------------------- #
# 2D heat stencil
# -------------------------------------------------------------------- #


@stencil(ndim=2, order=2)
def heat2d(u, i, j):
    return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])


class TestHeat2D:
    def test_basic_lowering(self):
        code = lower_stencil_to_python(
            heat2d._ir,
            tile_sizes=(64, 64),
            halo_widths=(1, 1),
            temporal_steps=1,
        )
        _assert_valid_python(code)
        assert "@ct.kernel" in code
        assert "ct.load" in code
        assert "ct.store" in code
        assert ".slice(" in code
        assert "def launch_heat2d" in code

    def test_kernel_structure(self):
        code = lower_stencil_to_python(
            heat2d._ir,
            tile_sizes=(64, 64),
            halo_widths=(1, 1),
        )
        assert "ct.bid(0)" in code
        assert "ct.bid(1)" in code
        assert "heat2d_kernel" in code
        assert "ConstInt = ct.Constant[int]" in code
        assert "import cuda.tile as ct" in code

    def test_launcher_structure(self):
        code = lower_stencil_to_python(
            heat2d._ir,
            tile_sizes=(64, 64),
            halo_widths=(1, 1),
        )
        assert "def launch_heat2d(u_in, u_out, stream=None):" in code
        assert "ct.cdiv(" in code
        assert "ct.launch(" in code
        assert "grid = " in code

    def test_slice_chains(self):
        code = lower_stencil_to_python(
            heat2d._ir,
            tile_sizes=(64, 64),
            halo_widths=(1, 1),
        )
        # Should have slice chains for the 4-point stencil
        assert "u_m1_0" in code
        assert "u_p1_0" in code
        assert "u_0_m1" in code
        assert "u_0_p1" in code
        assert ".slice(axis=0," in code
        assert ".slice(axis=1," in code


# -------------------------------------------------------------------- #
# 1D heat stencil
# -------------------------------------------------------------------- #


@stencil(ndim=1, order=2)
def heat1d(u, i):
    return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]


class TestHeat1D:
    def test_lowering(self):
        code = lower_stencil_to_python(
            heat1d._ir,
            tile_sizes=(128,),
            halo_widths=(1,),
        )
        _assert_valid_python(code)
        assert "def launch_heat1d" in code
        assert "ct.bid(0)" in code
        assert ".slice(axis=0" in code

    def test_1d_trailing_comma(self):
        """1D grid must have trailing comma for tuple syntax."""
        code = lower_stencil_to_python(
            heat1d._ir,
            tile_sizes=(128,),
            halo_widths=(1,),
        )
        # The grid should be a 1-tuple: (expr,)
        assert "grid = (" in code
        # shape unpacking should also have trailing comma
        assert "Nx, = " in code


# -------------------------------------------------------------------- #
# 3D Laplacian stencil
# -------------------------------------------------------------------- #


@stencil(ndim=3, order=2)
def laplacian3d(u, i, j, k):
    return (
        u[i - 1, j, k]
        + u[i + 1, j, k]
        + u[i, j - 1, k]
        + u[i, j + 1, k]
        + u[i, j, k - 1]
        + u[i, j, k + 1]
        - 6 * u[i, j, k]
    )


class TestLaplacian3D:
    def test_lowering(self):
        code = lower_stencil_to_python(
            laplacian3d._ir,
            tile_sizes=(32, 32, 32),
            halo_widths=(1, 1, 1),
        )
        _assert_valid_python(code)
        assert "def launch_laplacian3d" in code
        assert "ct.bid(2)" in code
        assert "bz" in code
        assert ".slice(axis=2" in code

    def test_3d_grid(self):
        code = lower_stencil_to_python(
            laplacian3d._ir,
            tile_sizes=(32, 32, 32),
            halo_widths=(1, 1, 1),
        )
        assert "Nx, Ny, Nz = " in code
        assert "TX, TY, TZ = " in code


# -------------------------------------------------------------------- #
# Temporal blocking
# -------------------------------------------------------------------- #


class TestTemporalBlocking:
    def test_temporal_loop(self):
        code = lower_stencil_to_python(
            heat2d._ir,
            tile_sizes=(64, 64),
            halo_widths=(1, 1),
            temporal_steps=3,
        )
        _assert_valid_python(code)
        assert "range(3)" in code
        assert "bufs" in code
        assert "def launch_heat2d" in code

    def test_buffer_chain(self):
        code = lower_stencil_to_python(
            heat2d._ir,
            tile_sizes=(64, 64),
            halo_widths=(1, 1),
            temporal_steps=5,
        )
        _assert_valid_python(code)
        assert "range(5)" in code
        assert "bufs.append" in code
        assert "bufs[_step]" in code
        assert "bufs[_step + 1]" in code


# -------------------------------------------------------------------- #
# Boundary conditions
# -------------------------------------------------------------------- #


class TestBoundaryConditions:
    def test_periodic(self):
        code = lower_stencil_to_python(
            heat2d._ir,
            tile_sizes=(64, 64),
            halo_widths=(1, 1),
            boundary_spec={"type": "periodic"},
        )
        _assert_valid_python(code)
        assert "apply_boundary_heat2d" in code
        assert "periodic" in code

    def test_dirichlet(self):
        code = lower_stencil_to_python(
            heat2d._ir,
            tile_sizes=(64, 64),
            halo_widths=(1, 1),
            boundary_spec={"type": "dirichlet", "value": 0.0},
        )
        _assert_valid_python(code)
        assert "apply_boundary_heat2d" in code
        assert "Dirichlet" in code

    def test_neumann(self):
        code = lower_stencil_to_python(
            heat2d._ir,
            tile_sizes=(64, 64),
            halo_widths=(1, 1),
            boundary_spec={"type": "neumann"},
        )
        _assert_valid_python(code)
        assert "Neumann" in code


# -------------------------------------------------------------------- #
# Multi-input stencil
# -------------------------------------------------------------------- #


@stencil(ndim=2, order=2)
def wave2d(u, v, i, j):
    return u[i, j] + 0.1 * (
        v[i - 1, j] + v[i + 1, j] + v[i, j - 1] + v[i, j + 1] - 4 * v[i, j]
    )


class TestMultiInput:
    def test_multi_input_kernel(self):
        code = lower_stencil_to_python(
            wave2d._ir,
            tile_sizes=(64, 64),
            halo_widths=(1, 1),
        )
        _assert_valid_python(code)
        assert "def launch_wave2d" in code
        assert "ct.load" in code

    def test_multi_input_params(self):
        code = lower_stencil_to_python(
            wave2d._ir,
            tile_sizes=(64, 64),
            halo_widths=(1, 1),
        )
        # Kernel should have both input params
        assert "def wave2d_kernel(u, v, output" in code


# -------------------------------------------------------------------- #
# Default tile/halo inference
# -------------------------------------------------------------------- #


class TestDefaults:
    def test_default_tile_sizes(self):
        """When tile_sizes is None, sensible defaults should be chosen."""
        code = lower_stencil_to_python(heat2d._ir)
        _assert_valid_python(code)
        assert "TX, TY = 64, 64" in code

    def test_default_halo_from_order(self):
        """Halo widths should be derived from stencil order when not specified."""
        code = lower_stencil_to_python(heat2d._ir)
        _assert_valid_python(code)
        assert "HX, HY = 1, 1" in code
