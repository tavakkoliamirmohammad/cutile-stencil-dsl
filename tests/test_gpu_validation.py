"""GPU validation tests for stencil kernels.

Tests that compiled kernels produce correct results on actual GPU hardware
by comparing against NumPy CPU reference implementations.
"""

import ast
import importlib.util
import tempfile
import numpy as np
import pytest

from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.registry import clear
from cutile_stencil.dsl.types import HardwareSpec
from cutile_stencil.dsl.pipeline import compile as stencil_compile
from cutile_stencil.dsl.boundary import BoundarySpec
from cutile_stencil.reference.stencil_ref import apply_stencil

try:
    import cupy as cp
    cp.cuda.Device(0).compute_capability
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False

gpu_required = pytest.mark.skipif(not _HAS_GPU, reason="No GPU available")


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


def _load_module(result):
    """Write generated code to a temp file and import it as a module."""
    tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
    tmp.write(result.code)
    tmp.close()
    spec = importlib.util.spec_from_file_location("_kernel", tmp.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_and_run(result, *inputs):
    """Load compiled kernel and run it on GPU inputs.

    Handles single-input stencils (1D/2D/3D) and multi-input stencils.
    The launcher is invoked as launch_<name>(*gpu_inputs, gpu_out).
    """
    mod = _load_module(result)
    launch = getattr(mod, f"launch_{result.spec.name}")

    gpu_inputs = [cp.asarray(inp) for inp in inputs]
    gpu_out = cp.zeros_like(gpu_inputs[0])

    launch(*gpu_inputs, gpu_out)
    cp.cuda.Device().synchronize()
    return cp.asnumpy(gpu_out)


@gpu_required
class TestGPUStencilCorrectness:
    """Validate generated stencil kernels against NumPy reference on GPU."""

    def test_1d_heat_gpu(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        result = stencil_compile(heat, domain=(1024,))
        np.random.seed(42)
        u = np.random.randn(1024 + 2).astype(np.float64)

        gpu_out = _load_and_run(result, u)
        cpu_ref = apply_stencil(u, heat._stencil_spec)

        halo = 1
        assert np.allclose(gpu_out[halo:-halo], cpu_ref[halo:-halo], atol=1e-10)

    def test_2d_laplacian_gpu(self):
        @stencil(ndim=2, order=2)
        def lap(u, i, j):
            return u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]

        result = stencil_compile(lap, domain=(128, 128))
        np.random.seed(42)
        u = np.random.randn(130, 130).astype(np.float64)

        gpu_out = _load_and_run(result, u)
        cpu_ref = apply_stencil(u, lap._stencil_spec)

        assert np.allclose(gpu_out[1:-1, 1:-1], cpu_ref[1:-1, 1:-1], atol=1e-10)

    def test_3d_laplacian_gpu(self):
        @stencil(ndim=3, order=2)
        def lap3d(u, i, j, k):
            return (u[i-1,j,k] + u[i+1,j,k] + u[i,j-1,k] + u[i,j+1,k]
                    + u[i,j,k-1] + u[i,j,k+1] - 6*u[i,j,k])

        result = stencil_compile(lap3d, domain=(32, 32, 32))
        np.random.seed(42)
        u = np.random.randn(34, 34, 34).astype(np.float64)

        gpu_out = _load_and_run(result, u)
        cpu_ref = apply_stencil(u, lap3d._stencil_spec)

        assert np.allclose(gpu_out[1:-1,1:-1,1:-1], cpu_ref[1:-1,1:-1,1:-1], atol=1e-10)

    def test_multi_input_gpu(self):
        @stencil(ndim=2, order=2)
        def diffusion(D, u, i, j):
            lap = u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]
            return D[i,j] * lap

        result = stencil_compile(diffusion, domain=(64, 64))
        np.random.seed(42)
        D = np.ones((66, 66), dtype=np.float64) * 0.5
        u = np.random.randn(66, 66).astype(np.float64)

        # launch_diffusion(D, u, u_out, stream=None)
        gpu_out = _load_and_run(result, D, u)

        # CPU reference using _ArrayProxy for multi-input
        from cutile_stencil.reference.stencil_ref import _ArrayProxy
        halo = diffusion._stencil_spec.halo_widths
        proxies = [_ArrayProxy(D, halo), _ArrayProxy(u, halo)]
        cpu_interior = diffusion._stencil_spec.update_fn(*proxies, 0, 0)
        cpu_ref = u.copy()
        interior = tuple(slice(h, s - h) for h, s in zip(halo, cpu_ref.shape))
        cpu_ref[interior] = cpu_interior

        assert np.allclose(gpu_out[1:-1,1:-1], cpu_ref[1:-1,1:-1], atol=1e-10)

    def test_4th_order_gpu(self):
        dx = 1.0
        @stencil(ndim=1, order=4)
        def lap4(u, i):
            return (-u[i-2] + 16*u[i-1] - 30*u[i] + 16*u[i+1] - u[i+2]) / (12 * dx**2)

        result = stencil_compile(lap4, domain=(512,))
        np.random.seed(42)
        u = np.random.randn(516).astype(np.float64)  # 512 + 2*2 halo

        gpu_out = _load_and_run(result, u)
        cpu_ref = apply_stencil(u, lap4._stencil_spec)

        assert np.allclose(gpu_out[2:-2], cpu_ref[2:-2], atol=1e-10)


class TestGPUBoundaryConditions:
    """Validate boundary condition handling on GPU.

    These tests exercise the BoundarySpec integration with the compile pipeline.
    The full GPU-side boundary fill (apply_boundary) is part of
    feat/boundary-codegen which is not yet merged here; we validate that the
    compile pipeline accepts BoundarySpec and emits valid Python, and run the
    stencil kernel on GPU where possible.
    """

    def test_dirichlet_bc_compiles(self):
        """BoundarySpec.dirichlet compiles to valid Python without error."""
        @stencil(ndim=1, order=2, boundary=BoundarySpec.dirichlet(1))
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        result = stencil_compile(heat, domain=(256,))
        # Must be valid Python syntax at minimum
        ast.parse(result.code)
        # Spec must carry the boundary info
        assert heat._stencil_spec.boundary is not None

    @gpu_required
    def test_dirichlet_bc_gpu_kernel_runs(self):
        """A stencil with Dirichlet BCs still produces a runnable GPU kernel.

        On batch-b-base the boundary fill happens outside the kernel (caller
        responsibility).  We zero the halo manually and verify the interior
        stencil result matches the CPU reference.
        """
        @stencil(ndim=1, order=2, boundary=BoundarySpec.dirichlet(1))
        def heat_dir(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        result = stencil_compile(heat_dir, domain=(256,))
        np.random.seed(0)
        u = np.random.randn(258).astype(np.float64)
        # Apply Dirichlet (zero) boundary on CPU before sending to GPU
        u[0] = 0.0
        u[-1] = 0.0

        gpu_out = _load_and_run(result, u)
        cpu_ref = apply_stencil(u, heat_dir._stencil_spec)

        assert np.allclose(gpu_out[1:-1], cpu_ref[1:-1], atol=1e-10)
