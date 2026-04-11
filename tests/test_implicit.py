"""Tests for implicit stencil decorator and CG bridge."""

import ast
import pytest
from cutile_stencil.dsl.registry import clear
from cutile_stencil.dsl.implicit import implicit_stencil, ImplicitCompileResult


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


class TestImplicitStencilDecorator:
    def test_creates_spec(self):
        @implicit_stencil(ndim=1, order=2)
        def heat(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]

        assert hasattr(heat, '_stencil_spec')
        assert heat._stencil_spec.ndim == 1
        assert heat._stencil_spec.order == 2

    def test_has_compile_solver(self):
        @implicit_stencil(ndim=1, order=2)
        def heat(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]

        assert hasattr(heat, 'compile_solver')
        assert callable(heat.compile_solver)

    def test_callable_as_function(self):
        @implicit_stencil(ndim=1, order=2)
        def heat(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]

        # Should still work as a normal function
        class FakeArray:
            def __getitem__(self, idx):
                return 1.0
        result = heat(FakeArray(), 5)
        assert result == 0.0  # 1 - 2 + 1

    def test_infers_params(self):
        @implicit_stencil
        def heat(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]

        assert heat._stencil_spec.ndim == 1
        assert heat._stencil_spec.order == 2

    def test_2d_stencil(self):
        @implicit_stencil(ndim=2, order=2)
        def lap2d(u, i, j):
            return u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]

        assert lap2d._stencil_spec.ndim == 2
        assert hasattr(lap2d, 'compile_solver')


class TestImplicitCompile:
    def test_compile_solver_returns_result(self):
        @implicit_stencil(ndim=1, order=2)
        def heat(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]

        result = heat.compile_solver(domain=(256,))
        assert isinstance(result, ImplicitCompileResult)
        assert result.domain == (256,)
        assert result.spec.name == "heat"

    def test_compile_solver_produces_code(self):
        @implicit_stencil(ndim=1, order=2)
        def heat(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]

        result = heat.compile_solver(domain=(256,))
        # The stencil_cg_result should have generated code
        assert result.stencil_cg_result is not None

    def test_print_summary(self, capsys):
        @implicit_stencil(ndim=1, order=2)
        def heat(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]

        result = heat.compile_solver(domain=(256,))
        result.print_summary()
        captured = capsys.readouterr()
        assert "heat" in captured.out
