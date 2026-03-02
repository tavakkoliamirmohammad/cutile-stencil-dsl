"""Solver compile results with GPU validation (template-method pattern).

Mirrors ``CompileResult`` from ``dsl/pipeline.py`` but for solver code:
emit to file, load via importlib, run on GPU, compare against NumPy
reference.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from cutile_stencil.config import SolverConfig
from cutile_stencil.solvers.solver_analysis import SolverAnalysisResult


# ── Internal helpers ──────────────────────────────────────────────────

@dataclass
class _SolverTestProblem:
    """Data bundle for a validation test problem."""
    N: int
    b: np.ndarray
    x0: np.ndarray
    x_exact: np.ndarray
    # DIA-based solvers
    dia_matrix: object = None  # DIAMatrix
    # Stencil-based solver
    spec: object = None        # StencilSpec
    halo: tuple = ()


@dataclass
class _SolverResult:
    """Result of running a solver (GPU or CPU)."""
    x: np.ndarray
    iters: int
    final_residual: float


# ── Base class ────────────────────────────────────────────────────────

@dataclass
class SolverCompileResult:
    """Result of compiling a solver: code + validation infrastructure.

    Subclasses override ``_build_test_problem``, ``_run_gpu_solver``,
    and ``_run_cpu_reference`` for each solver type.
    """
    solver_type: str
    code: str
    auxiliary_code: Dict[str, str] = field(default_factory=dict)
    config: SolverConfig = field(default_factory=SolverConfig)
    dtype: str = "float64"
    analysis: Optional[SolverAnalysisResult] = None
    _emitted_path: Optional[Path] = field(default=None, repr=False)

    def emit_to_file(self, path: str | Path) -> Path:
        """Write main code + auxiliary files to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.code)
        self._emitted_path = path

        # Write auxiliary code as sibling files
        for name, aux_code in self.auxiliary_code.items():
            aux_path = path.parent / f"{name}.py"
            aux_path.write_text(aux_code)

        return path

    def print_summary(self) -> None:
        """Print solver type, config, code stats, optional analysis."""
        lines = self.code.count("\n") + 1
        print(f"Solver: {self.solver_type}")
        print(f"  dtype: {self.dtype}")
        print(f"  code lines: {lines}")
        if self.auxiliary_code:
            for name, aux in self.auxiliary_code.items():
                aux_lines = aux.count("\n") + 1
                print(f"  auxiliary '{name}': {aux_lines} lines")
        print(f"  config: tol={self.config.cg_tol}, max_iter={self.config.cg_max_iter}, "
              f"tile_size={self.config.tile_size}")
        if self.analysis is not None:
            from cutile_stencil.solvers.solver_analysis import print_solver_analysis
            print()
            print_solver_analysis(self.analysis)

    def validate(
        self,
        N: int = 64,
        tol: float = 1e-8,
        atol: float = 1e-6,
        max_iter: int = 500,
        seed: int = 42,
    ) -> None:
        """Validate GPU solver against CPU reference.

        Template method: calls _load_modules, _build_test_problem,
        _run_gpu_solver, _run_cpu_reference, then compares.

        Raises
        ------
        AssertionError
            If GPU result differs from CPU reference beyond tolerance.
        """
        modules = self._load_modules()
        problem = self._build_test_problem(N, seed)
        gpu_result = self._run_gpu_solver(modules, problem, tol, max_iter)
        cpu_result = self._run_cpu_reference(problem, tol, max_iter)

        match = np.allclose(gpu_result.x, cpu_result.x, atol=atol)
        if match:
            max_err = float(np.max(np.abs(gpu_result.x - cpu_result.x)))
            print(f"  \u2713 {self.solver_type} GPU matches CPU reference "
                  f"(max_err={max_err:.2e}, gpu_iters={gpu_result.iters}, "
                  f"cpu_iters={cpu_result.iters})")
        else:
            max_err = float(np.max(np.abs(gpu_result.x - cpu_result.x)))
            raise AssertionError(
                f"{self.solver_type} GPU/CPU mismatch: max_err={max_err:.2e}, "
                f"gpu_iters={gpu_result.iters}, cpu_iters={cpu_result.iters}"
            )

    # ── Template methods (override in subclasses) ─────────────────

    def _load_modules(self) -> dict:
        """Emit code to temp files and load via importlib."""
        tmpdir = Path(tempfile.mkdtemp(prefix="solver_validate_"))

        # Main module
        main_path = tmpdir / "solver_main.py"
        main_path.write_text(self.code)

        mod_spec = importlib.util.spec_from_file_location(
            "solver_main", str(main_path)
        )
        main_mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(main_mod)

        modules = {"main": main_mod}

        # Auxiliary modules
        for name, aux_code in self.auxiliary_code.items():
            aux_path = tmpdir / f"{name}.py"
            aux_path.write_text(aux_code)
            aux_spec = importlib.util.spec_from_file_location(
                name, str(aux_path)
            )
            aux_mod = importlib.util.module_from_spec(aux_spec)
            aux_spec.loader.exec_module(aux_mod)
            modules[name] = aux_mod

        return modules

    def _build_test_problem(self, N: int, seed: int) -> _SolverTestProblem:
        raise NotImplementedError

    def _run_gpu_solver(
        self, modules: dict, problem: _SolverTestProblem,
        tol: float, max_iter: int,
    ) -> _SolverResult:
        raise NotImplementedError

    def _run_cpu_reference(
        self, problem: _SolverTestProblem,
        tol: float, max_iter: int,
    ) -> _SolverResult:
        raise NotImplementedError


# ── Subclass: Standard CG ────────────────────────────────────────────

@dataclass
class CGCompileResult(SolverCompileResult):
    """Compile result for the standard multi-launch CG solver."""

    def _build_test_problem(self, N: int, seed: int) -> _SolverTestProblem:
        from cutile_stencil.solvers.formats import laplacian_1d_dia
        from cutile_stencil.reference.spmv_ref import dia_spmv

        rng = np.random.RandomState(seed)
        A = laplacian_1d_dia(N, dtype=np.float64)

        # x_exact → b = A @ x_exact
        x_exact = rng.randn(N)
        b = dia_spmv(A, x_exact)
        x0 = np.zeros(N, dtype=np.float64)

        return _SolverTestProblem(N=N, b=b, x0=x0, x_exact=x_exact, dia_matrix=A)

    def _run_gpu_solver(self, modules, problem, tol, max_iter):
        import cupy as cp

        main = modules["main"]
        dia_mod = modules["dia_spmv"]

        A = problem.dia_matrix
        diag_vals = cp.asarray(A.data)
        offsets_arr = cp.asarray(A.offsets)
        num_diags = len(A.offsets)

        def spmv_fn(x_gpu, y_gpu):
            dia_mod.launch_dia_spmv(diag_vals, offsets_arr, x_gpu, y_gpu, num_diags)

        b_gpu = cp.asarray(problem.b)
        x0_gpu = cp.asarray(problem.x0)

        x_gpu, iters, final_res = main.cg_solve(spmv_fn, b_gpu, x0_gpu,
                                                  tol=tol, max_iter=max_iter)
        cp.cuda.Device().synchronize()
        x_np = cp.asnumpy(x_gpu)
        return _SolverResult(x=x_np, iters=iters, final_residual=final_res)

    def _run_cpu_reference(self, problem, tol, max_iter):
        from cutile_stencil.reference.cg_ref import cg_solve
        from cutile_stencil.reference.spmv_ref import dia_spmv

        A = problem.dia_matrix
        A_fn = lambda v: dia_spmv(A, v)
        x, iters, residuals = cg_solve(A_fn, problem.b, problem.x0,
                                        tol=tol, max_iter=max_iter)
        return _SolverResult(x=x, iters=iters, final_residual=residuals[-1])


# ── Subclass: Mixed-Precision CG ─────────────────────────────────────

@dataclass
class MixedPrecisionCGCompileResult(SolverCompileResult):
    """Compile result for mixed-precision iterative refinement CG."""
    outer_dtype: str = "float64"
    inner_dtype: str = "float32"

    def _build_test_problem(self, N: int, seed: int) -> _SolverTestProblem:
        from cutile_stencil.solvers.formats import laplacian_1d_dia
        from cutile_stencil.reference.spmv_ref import dia_spmv

        rng = np.random.RandomState(seed)
        A = laplacian_1d_dia(N, dtype=np.float64)

        x_exact = rng.randn(N)
        b = dia_spmv(A, x_exact)
        x0 = np.zeros(N, dtype=np.float64)

        return _SolverTestProblem(N=N, b=b, x0=x0, x_exact=x_exact, dia_matrix=A)

    def _run_gpu_solver(self, modules, problem, tol, max_iter):
        import cupy as cp

        main = modules["main"]
        dia_hp = modules["dia_spmv_hp"]
        dia_lp = modules["dia_spmv_lp"]

        A = problem.dia_matrix
        diag_vals_hp = cp.asarray(A.data, dtype=cp.float64)
        offsets_arr = cp.asarray(A.offsets)
        num_diags = len(A.offsets)

        # FP32 copy for low-precision SpMV
        diag_vals_lp = cp.asarray(A.data, dtype=cp.float32)

        def spmv_hp(x_gpu, y_gpu):
            dia_hp.launch_dia_spmv(diag_vals_hp, offsets_arr, x_gpu, y_gpu, num_diags)

        def spmv_lp(x_gpu, y_gpu):
            dia_lp.launch_dia_spmv(diag_vals_lp, offsets_arr, x_gpu, y_gpu, num_diags)

        b_gpu = cp.asarray(problem.b, dtype=cp.float64)
        x0_gpu = cp.asarray(problem.x0, dtype=cp.float64)

        x_gpu, iters, final_res = main.mixed_precision_cg(
            spmv_hp, spmv_lp, b_gpu, x0_gpu,
            tol=tol, max_outer=max_iter,
        )
        cp.cuda.Device().synchronize()
        x_np = cp.asnumpy(x_gpu)
        return _SolverResult(x=x_np, iters=iters, final_residual=final_res)

    def _run_cpu_reference(self, problem, tol, max_iter):
        from cutile_stencil.reference.cg_ref import mixed_precision_cg
        from cutile_stencil.reference.spmv_ref import dia_spmv

        A = problem.dia_matrix
        A_fn = lambda v: dia_spmv(A, v)
        x, iters, residuals = mixed_precision_cg(
            A_fn, problem.b, problem.x0,
            tol=tol, max_outer=max_iter,
        )
        return _SolverResult(x=x, iters=iters, final_residual=residuals[-1])


# ── Subclass: Persistent CG ──────────────────────────────────────────

@dataclass
class PersistentCGCompileResult(SolverCompileResult):
    """Compile result for the persistent single-kernel CG solver."""

    def _build_test_problem(self, N: int, seed: int) -> _SolverTestProblem:
        from cutile_stencil.solvers.formats import laplacian_1d_dia
        from cutile_stencil.reference.spmv_ref import dia_spmv

        rng = np.random.RandomState(seed)
        A = laplacian_1d_dia(N, dtype=np.float64)

        x_exact = rng.randn(N)
        b = dia_spmv(A, x_exact)
        x0 = np.zeros(N, dtype=np.float64)

        return _SolverTestProblem(N=N, b=b, x0=x0, x_exact=x_exact, dia_matrix=A)

    def _run_gpu_solver(self, modules, problem, tol, max_iter):
        import cupy as cp

        main = modules["main"]
        A = problem.dia_matrix

        diag_vals = cp.asarray(A.data)
        offsets_arr = cp.asarray(A.offsets)
        num_diags = len(A.offsets)

        x_gpu = cp.asarray(problem.x0)
        b_gpu = cp.asarray(problem.b)

        x_result = main.launch_persistent_cg(
            diag_vals, offsets_arr, num_diags,
            x_gpu, b_gpu, tol=tol, max_iter=max_iter,
        )
        cp.cuda.Device().synchronize()
        x_np = cp.asnumpy(x_result)
        # Persistent CG doesn't return iters/residual
        return _SolverResult(x=x_np, iters=-1, final_residual=-1.0)

    def _run_cpu_reference(self, problem, tol, max_iter):
        from cutile_stencil.reference.cg_ref import cg_solve
        from cutile_stencil.reference.spmv_ref import dia_spmv

        A = problem.dia_matrix
        A_fn = lambda v: dia_spmv(A, v)
        x, iters, residuals = cg_solve(A_fn, problem.b, problem.x0,
                                        tol=tol, max_iter=max_iter)
        return _SolverResult(x=x, iters=iters, final_residual=residuals[-1])


# ── Subclass: Stencil CG ─────────────────────────────────────────────

@dataclass
class StencilCGCompileResult(SolverCompileResult):
    """Compile result for the matrix-free stencil CG solver."""
    spec: object = None    # StencilSpec
    domain: tuple = ()

    def _build_test_problem(self, N: int, seed: int) -> _SolverTestProblem:
        from cutile_stencil.reference.stencil_ref import apply_stencil

        spec = self.spec
        halo = spec.halo_widths
        ndim = spec.ndim
        rng = np.random.RandomState(seed)

        # Build full arrays with halo
        if ndim == 1:
            full_shape = (N + 2 * halo[0],)
            interior = (slice(halo[0], N + halo[0]),)
        elif ndim == 2:
            side = int(math.isqrt(N)) or 8
            full_shape = (side + 2 * halo[0], side + 2 * halo[1])
            interior = (slice(halo[0], side + halo[0]),
                        slice(halo[1], side + halo[1]))
        else:
            side = max(int(round(N ** (1.0 / 3))), 4)
            full_shape = tuple(side + 2 * h for h in halo)
            interior = tuple(slice(h, side + h) for h in halo)

        # x_exact with zero boundary
        x_exact = np.zeros(full_shape, dtype=np.float64)
        interior_shape = tuple(s.stop - s.start for s in interior)
        x_exact[interior] = rng.randn(*interior_shape)

        # b = A @ x_exact
        Ax = apply_stencil(x_exact, spec)
        b = np.zeros(full_shape, dtype=np.float64)
        b[interior] = Ax[interior]

        x0 = np.zeros(full_shape, dtype=np.float64)
        return _SolverTestProblem(
            N=N, b=b, x0=x0, x_exact=x_exact, spec=spec, halo=halo
        )

    def _run_gpu_solver(self, modules, problem, tol, max_iter):
        import cupy as cp

        main = modules["main"]
        b_gpu = cp.asarray(problem.b)
        x0_gpu = cp.asarray(problem.x0)

        x_gpu, iters, final_res = main.stencil_cg_solve(
            b_gpu, x0_gpu, tol=tol, max_iter=max_iter,
        )
        cp.cuda.Device().synchronize()
        x_np = cp.asnumpy(x_gpu)
        return _SolverResult(x=x_np, iters=iters, final_residual=final_res)

    def _run_cpu_reference(self, problem, tol, max_iter):
        from cutile_stencil.reference.stencil_cg_ref import stencil_cg_solve

        x, iters, residuals = stencil_cg_solve(
            problem.spec, problem.b, problem.x0,
            tol=tol, max_iter=max_iter,
        )
        return _SolverResult(x=x, iters=iters, final_residual=residuals[-1])
