"""Tests for the unified BLAS kernel emit helpers."""

import ast

import pytest

from cutile_stencil.codegen.emitter import CodeEmitter
from cutile_stencil.solvers.blas_kernels import (
    _ct_dtype,
    emit_dot_kernel,
    emit_axpy_kernel,
    emit_norm2_kernel,
    emit_cast_kernel,
    emit_dot_launch,
    emit_axpy_launch,
    emit_norm2_launch,
    emit_dia_spmv_phase,
)
from cutile_stencil.solvers.kernels import (
    generate_dot,
    generate_axpy,
    generate_norm2,
    generate_dia_spmv,
)
from cutile_stencil.solvers.cg import generate_cg_driver
from cutile_stencil.solvers.mixed_precision import generate_mixed_precision_cg
from cutile_stencil.solvers.persistent_cg import generate_persistent_cg


class TestCtDtype:
    def test_float64(self):
        assert _ct_dtype("float64") == "ct.float64"

    def test_float32(self):
        assert _ct_dtype("float32") == "ct.float32"

    def test_float16(self):
        assert _ct_dtype("float16") == "ct.float16"

    def test_unknown_defaults_to_float64(self):
        assert _ct_dtype("bfloat16") == "ct.float64"


class TestEmitHelpers:
    """Emit helpers produce valid Python code."""

    def _emit_with_header(self, emit_fn, *args):
        """Wrap an emit helper with the required header for parsing."""
        e = CodeEmitter()
        e.line("import cuda.tile as ct")
        e.line("import cupy as cp")
        e.line("ConstInt = ct.Constant[int]")
        e.blank()
        emit_fn(e, *args)
        return e.render()

    def test_emit_dot_kernel(self):
        code = self._emit_with_header(emit_dot_kernel)
        ast.parse(code)

    def test_emit_axpy_kernel(self):
        code = self._emit_with_header(emit_axpy_kernel)
        ast.parse(code)

    def test_emit_norm2_kernel(self):
        code = self._emit_with_header(emit_norm2_kernel)
        ast.parse(code)

    def test_emit_cast_kernel(self):
        code = self._emit_with_header(emit_cast_kernel)
        ast.parse(code)

    def test_emit_dot_launch(self):
        e = CodeEmitter()
        e.line("import cuda.tile as ct")
        e.line("import cupy as cp")
        e.line("ConstInt = ct.Constant[int]")
        e.blank()
        emit_dot_kernel(e)
        e.blank()
        emit_dot_launch(e)
        ast.parse(e.render())

    def test_emit_axpy_launch(self):
        e = CodeEmitter()
        e.line("import cuda.tile as ct")
        e.line("import cupy as cp")
        e.line("ConstInt = ct.Constant[int]")
        e.blank()
        emit_axpy_kernel(e)
        e.blank()
        emit_axpy_launch(e)
        ast.parse(e.render())

    def test_emit_norm2_launch(self):
        e = CodeEmitter()
        e.line("import cuda.tile as ct")
        e.line("import cupy as cp")
        e.line("ConstInt = ct.Constant[int]")
        e.blank()
        emit_norm2_kernel(e)
        e.blank()
        emit_norm2_launch(e)
        ast.parse(e.render())


class TestGeneratorsValidPython:
    """All generators produce parseable Python."""

    def test_generate_dot(self):
        ast.parse(generate_dot())

    def test_generate_axpy(self):
        ast.parse(generate_axpy())

    def test_generate_norm2(self):
        ast.parse(generate_norm2())

    def test_generate_dia_spmv(self):
        ast.parse(generate_dia_spmv())

    def test_generate_cg_driver(self):
        ast.parse(generate_cg_driver())

    def test_generate_mixed_precision_cg(self):
        ast.parse(generate_mixed_precision_cg())

    def test_generate_persistent_cg(self):
        ast.parse(generate_persistent_cg())


class TestDtypeParameterization:
    """Dtype parameter changes generated code."""

    def test_dot_float32(self):
        code = generate_dot(dtype="float32")
        ast.parse(code)

    def test_axpy_float32(self):
        code = generate_axpy(dtype="float32")
        ast.parse(code)

    def test_norm2_float32(self):
        code = generate_norm2(dtype="float32")
        ast.parse(code)

    def test_dia_spmv_float32(self):
        code = generate_dia_spmv(dtype="float32")
        ast.parse(code)
        assert "ct.float32" in code

    def test_cg_driver_float32(self):
        code = generate_cg_driver(dtype="float32")
        ast.parse(code)

    def test_persistent_cg_float32(self):
        code = generate_persistent_cg(dtype="float32")
        ast.parse(code)
        assert "ct.float32" in code

    def test_mixed_precision_float16(self):
        code = generate_mixed_precision_cg(inner_dtype="float16")
        ast.parse(code)


class TestRegressionDot:
    """Regression: generate_dot() output structure unchanged."""

    def test_contains_dot_kernel(self):
        code = generate_dot()
        assert "def dot_kernel(" in code

    def test_contains_launch_dot(self):
        code = generate_dot()
        assert "def launch_dot(" in code

    def test_contains_atomic_add(self):
        code = generate_dot()
        assert "ct.atomic_add" in code

    def test_contains_ct_sum(self):
        code = generate_dot()
        assert "ct.sum" in code
