"""Tests for solver compile results and GPU validation."""

import ast
import os
import tempfile

import numpy as np
import pytest

from cutile_stencil.config import SolverConfig
from cutile_stencil.solvers.cg import compile_cg
from cutile_stencil.solvers.mixed_precision import compile_mixed_precision_cg
from cutile_stencil.solvers.persistent_cg import compile_persistent_cg
from cutile_stencil.solvers.stencil_cg import compile_stencil_cg
from cutile_stencil.solvers.stencil_bridge import laplacian_stencil_spec
from cutile_stencil.solvers.solver_compile import (
    SolverCompileResult,
    CGCompileResult,
    MixedPrecisionCGCompileResult,
    PersistentCGCompileResult,
    StencilCGCompileResult,
)


# ── Helper ────────────────────────────────────────────────────────────

_HAS_GPU = False
try:
    import cupy as cp
    cp.cuda.Device(0).compute_capability
    _HAS_GPU = True
except Exception:
    pass

gpu_required = pytest.mark.skipif(not _HAS_GPU, reason="No GPU available")


# ── CPU-only tests: compile_cg ────────────────────────────────────────

class TestCompileCG:
    def test_returns_correct_type(self):
        r = compile_cg()
        assert isinstance(r, CGCompileResult)
        assert isinstance(r, SolverCompileResult)

    def test_code_is_valid_python(self):
        r = compile_cg()
        ast.parse(r.code)

    def test_auxiliary_code_valid(self):
        r = compile_cg()
        assert "dia_spmv" in r.auxiliary_code
        ast.parse(r.auxiliary_code["dia_spmv"])

    def test_solver_type(self):
        r = compile_cg()
        assert r.solver_type == "CG"

    def test_dtype_default(self):
        r = compile_cg()
        assert r.dtype == "float64"

    def test_dtype_float32(self):
        r = compile_cg(dtype="float32")
        assert r.dtype == "float32"
        ast.parse(r.code)

    def test_custom_config(self):
        cfg = SolverConfig(cg_tol=1e-6, cg_max_iter=100)
        r = compile_cg(solver_config=cfg)
        assert r.config.cg_tol == 1e-6
        assert r.config.cg_max_iter == 100

    def test_print_summary(self, capsys):
        r = compile_cg()
        r.print_summary()
        out = capsys.readouterr().out
        assert "CG" in out
        assert "float64" in out

    def test_emit_to_file(self, tmp_path):
        r = compile_cg()
        p = r.emit_to_file(tmp_path / "cg_solver.py")
        assert p.exists()
        assert (tmp_path / "dia_spmv.py").exists()
        # Both are valid Python
        ast.parse(p.read_text())
        ast.parse((tmp_path / "dia_spmv.py").read_text())

    def test_code_contains_cg_solve(self):
        r = compile_cg()
        assert "def cg_solve(" in r.code


# ── CPU-only tests: compile_mixed_precision_cg ────────────────────────

class TestCompileMixedPrecisionCG:
    def test_returns_correct_type(self):
        r = compile_mixed_precision_cg()
        assert isinstance(r, MixedPrecisionCGCompileResult)

    def test_code_is_valid_python(self):
        r = compile_mixed_precision_cg()
        ast.parse(r.code)

    def test_auxiliary_code_valid(self):
        r = compile_mixed_precision_cg()
        assert "dia_spmv_hp" in r.auxiliary_code
        assert "dia_spmv_lp" in r.auxiliary_code
        ast.parse(r.auxiliary_code["dia_spmv_hp"])
        ast.parse(r.auxiliary_code["dia_spmv_lp"])

    def test_solver_type(self):
        r = compile_mixed_precision_cg()
        assert r.solver_type == "MixedPrecisionCG"

    def test_dtypes(self):
        r = compile_mixed_precision_cg()
        assert r.outer_dtype == "float64"
        assert r.inner_dtype == "float32"

    def test_print_summary(self, capsys):
        r = compile_mixed_precision_cg()
        r.print_summary()
        out = capsys.readouterr().out
        assert "MixedPrecisionCG" in out

    def test_emit_to_file(self, tmp_path):
        r = compile_mixed_precision_cg()
        p = r.emit_to_file(tmp_path / "mp_cg.py")
        assert p.exists()
        assert (tmp_path / "dia_spmv_hp.py").exists()
        assert (tmp_path / "dia_spmv_lp.py").exists()

    def test_code_contains_mixed_precision_cg(self):
        r = compile_mixed_precision_cg()
        assert "def mixed_precision_cg(" in r.code


# ── CPU-only tests: compile_persistent_cg ─────────────────────────────

class TestCompilePersistentCG:
    def test_returns_correct_type(self):
        r = compile_persistent_cg()
        assert isinstance(r, PersistentCGCompileResult)

    def test_code_is_valid_python(self):
        r = compile_persistent_cg()
        ast.parse(r.code)

    def test_no_auxiliary_code(self):
        r = compile_persistent_cg()
        assert r.auxiliary_code == {}

    def test_solver_type(self):
        r = compile_persistent_cg()
        assert r.solver_type == "PersistentCG"

    def test_print_summary(self, capsys):
        r = compile_persistent_cg()
        r.print_summary()
        out = capsys.readouterr().out
        assert "PersistentCG" in out

    def test_emit_to_file(self, tmp_path):
        r = compile_persistent_cg()
        p = r.emit_to_file(tmp_path / "persistent_cg.py")
        assert p.exists()

    def test_code_contains_persistent_kernel(self):
        r = compile_persistent_cg()
        assert "def persistent_cg_kernel(" in r.code
        assert "def launch_persistent_cg(" in r.code


# ── CPU-only tests: compile_stencil_cg ────────────────────────────────

class TestCompileStencilCG:
    def test_1d_returns_correct_type(self):
        spec = laplacian_stencil_spec(1)
        r = compile_stencil_cg(spec, (128,))
        assert isinstance(r, StencilCGCompileResult)

    def test_1d_code_is_valid_python(self):
        spec = laplacian_stencil_spec(1)
        r = compile_stencil_cg(spec, (128,))
        ast.parse(r.code)

    def test_2d_code_is_valid_python(self):
        spec = laplacian_stencil_spec(2)
        r = compile_stencil_cg(spec, (32, 32))
        ast.parse(r.code)

    def test_no_auxiliary_code(self):
        spec = laplacian_stencil_spec(1)
        r = compile_stencil_cg(spec, (128,))
        assert r.auxiliary_code == {}

    def test_solver_type(self):
        spec = laplacian_stencil_spec(1)
        r = compile_stencil_cg(spec, (128,))
        assert r.solver_type == "StencilCG"

    def test_stores_spec_and_domain(self):
        spec = laplacian_stencil_spec(1)
        r = compile_stencil_cg(spec, (128,))
        assert r.spec is spec
        assert r.domain == (128,)

    def test_print_summary(self, capsys):
        spec = laplacian_stencil_spec(1)
        r = compile_stencil_cg(spec, (128,))
        r.print_summary()
        out = capsys.readouterr().out
        assert "StencilCG" in out

    def test_emit_to_file(self, tmp_path):
        spec = laplacian_stencil_spec(1)
        r = compile_stencil_cg(spec, (128,))
        p = r.emit_to_file(tmp_path / "stencil_cg.py")
        assert p.exists()

    def test_code_contains_stencil_cg_solve(self):
        spec = laplacian_stencil_spec(1)
        r = compile_stencil_cg(spec, (128,))
        assert "def stencil_cg_solve(" in r.code


# ── GPU-gated tests ───────────────────────────────────────────────────

class TestGPUValidation:
    @gpu_required
    def test_cg_validate(self):
        r = compile_cg()
        r.validate(N=32, tol=1e-8, atol=1e-5)

    @gpu_required
    def test_mixed_precision_cg_validate(self):
        r = compile_mixed_precision_cg()
        r.validate(N=32, tol=1e-8, atol=1e-4)

    @gpu_required
    def test_persistent_cg_validate(self):
        r = compile_persistent_cg()
        r.validate(N=64, tol=1e-8, atol=1e-5)

    @gpu_required
    def test_stencil_cg_validate(self):
        spec = laplacian_stencil_spec(1)
        r = compile_stencil_cg(spec, (128,))
        r.validate(N=64, tol=1e-8, atol=1e-5)


# ── Test exports ──────────────────────────────────────────────────────

class TestExports:
    def test_solvers_init_exports(self):
        from cutile_stencil.solvers import (
            compile_cg,
            compile_mixed_precision_cg,
            compile_persistent_cg,
            compile_stencil_cg,
            SolverCompileResult,
            CGCompileResult,
            MixedPrecisionCGCompileResult,
            PersistentCGCompileResult,
            StencilCGCompileResult,
        )

    def test_top_level_exports(self):
        from cutile_stencil import compile_stencil_cg
