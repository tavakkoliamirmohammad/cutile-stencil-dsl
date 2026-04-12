"""Tests for boundary condition code generation."""

import ast
import importlib.util
import tempfile
import numpy as np
import pytest
from conftest import gpu_required
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import HardwareSpec
from cutile_stencil.dsl.registry import clear
from cutile_stencil.dsl.boundary import BoundarySpec, BoundaryType
from cutile_stencil.dsl.pipeline import compile
from cutile_stencil.reference.stencil_ref import apply_stencil, _apply_boundary_conditions


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


class TestDirichletCodegen:
    def test_1d_dirichlet_has_padding_mode(self):
        @stencil(ndim=1, order=2, boundary=BoundarySpec.dirichlet(1))
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
        result = compile(heat, domain=(1024,))
        assert "PaddingMode.ZERO" in result.code

    def test_2d_dirichlet_has_padding_mode(self):
        @stencil(ndim=2, order=2, boundary=BoundarySpec.dirichlet(2))
        def lap(u, i, j):
            return u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]
        result = compile(lap, domain=(256, 256))
        assert "PaddingMode.ZERO" in result.code

    def test_dirichlet_valid_python(self):
        @stencil(ndim=1, order=2, boundary=BoundarySpec.dirichlet(1))
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
        result = compile(heat, domain=(1024,))
        ast.parse(result.code)


class TestPeriodicCodegen:
    def test_1d_periodic_has_boundary_fill(self):
        @stencil(ndim=1, order=2, boundary=BoundarySpec.periodic(1))
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
        result = compile(heat, domain=(1024,))
        assert "apply_boundary" in result.code
        assert "periodic" in result.code.lower()

    def test_periodic_valid_python(self):
        @stencil(ndim=1, order=2, boundary=BoundarySpec.periodic(1))
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
        result = compile(heat, domain=(1024,))
        ast.parse(result.code)


class TestNeumannCodegen:
    def test_1d_neumann_has_boundary_fill(self):
        @stencil(ndim=1, order=2, boundary=BoundarySpec.neumann(1))
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
        result = compile(heat, domain=(1024,))
        assert "apply_boundary" in result.code
        assert "Neumann" in result.code or "neumann" in result.code.lower()

    def test_neumann_valid_python(self):
        @stencil(ndim=1, order=2, boundary=BoundarySpec.neumann(1))
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
        result = compile(heat, domain=(1024,))
        ast.parse(result.code)


class TestNoBoundary:
    def test_no_boundary_no_padding_mode(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
        result = compile(heat, domain=(1024,))
        assert "PaddingMode" not in result.code
        assert "apply_boundary" not in result.code


class TestBoundaryGPUValidation:
    """Run stencils with boundary conditions on GPU and verify against CPU reference."""

    def _run_boundary_validation(self, bc_spec, n_steps=5):
        """Compile stencil with BC, run n_steps on GPU, compare to CPU reference."""
        import cupy as cp

        ndim = len(bc_spec.conditions)

        if ndim == 1:
            @stencil(ndim=1, order=2, boundary=bc_spec)
            def heat(u, i):
                return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
            spec = heat._stencil_spec
            domain = (256,)
        else:
            @stencil(ndim=2, order=2, boundary=bc_spec)
            def heat2d(u, i, j):
                return 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])
            spec = heat2d._stencil_spec
            domain = (64, 64)

        result = compile(spec, domain=domain)
        halo = spec.halo_widths

        # Write and import the generated kernel
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(result.code)
            tmp_path = f.name
        mod_spec = importlib.util.spec_from_file_location(
            f"_bc_test_{id(bc_spec)}", tmp_path
        )
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        launch = getattr(mod, f"launch_{spec.name}")
        apply_bc = getattr(mod, f"apply_boundary_{spec.name}", None)

        # Input
        np.random.seed(42)
        shape = tuple(d + 2 * h for d, h in zip(domain, halo))
        u_np = np.random.randn(*shape).astype(np.float64)

        # Apply initial boundary conditions
        _apply_boundary_conditions(u_np, bc_spec, halo)

        # GPU: n_steps of stencil + boundary
        u_gpu = cp.asarray(u_np)
        out_gpu = cp.zeros_like(u_gpu)
        for step in range(n_steps):
            launch(u_gpu, out_gpu)
            cp.cuda.Device(0).synchronize()
            # Apply boundary conditions to output
            if apply_bc is not None:
                apply_bc(out_gpu)
                cp.cuda.Device(0).synchronize()
            # Swap buffers
            u_gpu, out_gpu = out_gpu, u_gpu
        gpu_result = cp.asnumpy(u_gpu)

        # CPU: n_steps of stencil + boundary
        u_cpu = u_np.copy()
        for step in range(n_steps):
            u_cpu = apply_stencil(u_cpu, spec)
            _apply_boundary_conditions(u_cpu, bc_spec, halo)

        # Compare interior
        interior = tuple(slice(h, -h) for h in halo)
        gpu_int = gpu_result[interior]
        cpu_int = u_cpu[interior]

        maxdiff = np.max(np.abs(gpu_int - cpu_int))
        assert np.allclose(gpu_int, cpu_int, atol=1e-10), (
            f"{ndim}D BC validation ({n_steps} steps): max diff = {maxdiff:.2e}"
        )

    @gpu_required
    def test_1d_periodic_matches_reference(self):
        self._run_boundary_validation(BoundarySpec.periodic(1))

    @gpu_required
    def test_1d_neumann_matches_reference(self):
        self._run_boundary_validation(BoundarySpec.neumann(1))

    @gpu_required
    def test_1d_dirichlet_matches_reference(self):
        self._run_boundary_validation(BoundarySpec.dirichlet(1))

    @gpu_required
    def test_2d_periodic_matches_reference(self):
        self._run_boundary_validation(BoundarySpec.periodic(2))

    @gpu_required
    def test_2d_dirichlet_matches_reference(self):
        self._run_boundary_validation(BoundarySpec.dirichlet(2))
