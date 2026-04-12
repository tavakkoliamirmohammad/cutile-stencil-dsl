"""Tests for CG solver convergence history integration."""

import ast
import pytest
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.registry import clear


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


class TestCGConvergenceCodegen:
    def test_stencil_cg_with_history_generates_valid_python(self):
        from cutile_stencil.solvers.stencil_cg import generate_stencil_cg

        @stencil(ndim=1, order=2)
        def lap(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]

        code = generate_stencil_cg(lap._stencil_spec, domain=(256,), return_history=True)
        ast.parse(code)

    def test_history_code_contains_convergence_history(self):
        from cutile_stencil.solvers.stencil_cg import generate_stencil_cg

        @stencil(ndim=1, order=2)
        def lap(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]

        code = generate_stencil_cg(lap._stencil_spec, domain=(256,), return_history=True)
        assert "ConvergenceHistory" in code
        assert "residual_norms" in code
        assert "perf_counter" in code

    def test_without_history_no_convergence_history(self):
        from cutile_stencil.solvers.stencil_cg import generate_stencil_cg

        @stencil(ndim=1, order=2)
        def lap(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]

        code = generate_stencil_cg(lap._stencil_spec, domain=(256,))
        assert "ConvergenceHistory" not in code

    def test_history_code_imports_time(self):
        from cutile_stencil.solvers.stencil_cg import generate_stencil_cg

        @stencil(ndim=1, order=2)
        def lap(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]

        code = generate_stencil_cg(lap._stencil_spec, domain=(256,), return_history=True)
        assert "import time" in code

    def test_without_history_no_import_time(self):
        from cutile_stencil.solvers.stencil_cg import generate_stencil_cg

        @stencil(ndim=1, order=2)
        def lap(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]

        code = generate_stencil_cg(lap._stencil_spec, domain=(256,))
        assert "import time" not in code

    def test_history_code_returns_convergence_history_both_paths(self):
        """Both the early-exit and max_iter paths should return ConvergenceHistory."""
        from cutile_stencil.solvers.stencil_cg import generate_stencil_cg

        @stencil(ndim=1, order=2)
        def lap(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]

        code = generate_stencil_cg(lap._stencil_spec, domain=(256,), return_history=True)
        # Should appear at least twice: once for converged path, once for not-converged
        assert code.count("ConvergenceHistory(") >= 2

    def test_compile_stencil_cg_with_history(self):
        """compile_stencil_cg with return_history=True produces valid AST."""
        from cutile_stencil.solvers.stencil_cg import compile_stencil_cg

        @stencil(ndim=1, order=2)
        def lap(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]

        result = compile_stencil_cg(lap._stencil_spec, domain=(256,), return_history=True)
        ast.parse(result.code)
        assert "ConvergenceHistory" in result.code

    def test_convergence_history_dataclass_importable(self):
        from cutile_stencil.solvers.convergence import ConvergenceHistory
        hist = ConvergenceHistory(
            iterations=5,
            residual_norms=[1.0, 0.5, 0.25, 0.1, 0.05],
            converged=True,
            final_residual=0.05,
            tolerance=0.1,
            max_iterations=100,
        )
        assert hist.converged
        assert hist.iterations == 5
        assert hist.convergence_rate is not None
        assert "converged" in hist.summary()

    def test_convergence_history_via_solvers_init(self):
        from cutile_stencil.solvers import ConvergenceHistory
        hist = ConvergenceHistory(
            iterations=3,
            residual_norms=[1.0, 0.1, 0.01],
            converged=True,
            final_residual=0.01,
            tolerance=0.05,
            max_iterations=50,
        )
        assert hist.summary().startswith("converged")

    def test_emit_cg_loop_with_history_exported(self):
        """emit_cg_loop_with_history is importable from solver_base."""
        from cutile_stencil.solvers.solver_base import emit_cg_loop_with_history
        assert callable(emit_cg_loop_with_history)

    def test_emit_cg_function_return_history_flag(self):
        """emit_cg_function with return_history=True produces history-aware code."""
        from cutile_stencil.codegen.emitter import CodeEmitter
        from cutile_stencil.solvers.solver_base import emit_cg_function

        e = CodeEmitter()
        # Need dot_kernel and axpy_kernel in scope for the function body references
        emit_cg_function(
            e,
            func_name="test_cg",
            operator_call="spmv_fn(p, Ap)",
            extra_params="spmv_fn, ",
            return_history=True,
        )
        code = e.render()
        ast.parse(code)
        assert "ConvergenceHistory" in code
        assert "residual_norms" in code
