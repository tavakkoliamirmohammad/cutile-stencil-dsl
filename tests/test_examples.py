"""Smoke tests: run each example's main() to ensure the full pipeline works.
"""

import shutil
import pytest

from conftest import gpu_required


@gpu_required
def test_heat_1d():
    from examples.heat_1d import main
    main()


@gpu_required
def test_wave_2d():
    from examples.wave_2d import main
    main()


@gpu_required
def test_laplacian_3d():
    from examples.laplacian_3d import main
    main()


def test_poisson_cg():
    from examples.poisson_cg import main
    main()


def test_mixed_precision_cg():
    from examples.mixed_precision_cg import main
    main()


@gpu_required
def test_heat_2d_bricked():
    from examples.heat_2d_bricked import main
    main()


def test_benchmark_suite():
    from examples.benchmark_suite import main
    main()


def test_laplacian_4th_order_example():
    """Smoke test for 4th-order Laplacian example."""
    from examples.laplacian_4th_order import main
    main()


def test_variable_diffusion_example():
    """Smoke test for variable-coefficient diffusion example."""
    from examples.variable_diffusion import main
    main()


@gpu_required
def test_4th_order_laplacian_gpu_convergence():
    """Run 4th-order 1D Laplacian on GPU and verify against CPU reference."""
    import numpy as np
    import cupy as cp
    import importlib.util
    import tempfile
    from cutile_stencil import stencil, compile as stencil_compile
    from cutile_stencil.dsl.registry import clear
    from cutile_stencil.reference.stencil_ref import apply_stencil

    clear()

    dx = 1.0
    @stencil(ndim=1, order=4)
    def lap4(u, i):
        return (-u[i-2] + 16*u[i-1] - 30*u[i] + 16*u[i+1] - u[i+2]) / (12 * dx**2)

    spec = lap4._stencil_spec
    domain = (256,)
    result = stencil_compile(lap4, domain=domain)

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write(result.code)
        tmp_path = f.name
    mod_spec = importlib.util.spec_from_file_location("_lap4", tmp_path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    launch = getattr(mod, f"launch_{spec.name}")

    np.random.seed(42)
    halo = spec.halo_widths[0]  # 2 for 4th-order
    N = domain[0] + 2 * halo
    u_np = np.sin(np.linspace(0, 4*np.pi, N)).astype(np.float64)

    # GPU
    u_gpu = cp.asarray(u_np)
    out_gpu = cp.zeros_like(u_gpu)
    launch(u_gpu, out_gpu)
    cp.cuda.Device(0).synchronize()
    gpu_result = cp.asnumpy(out_gpu)

    # CPU reference
    cpu_result = apply_stencil(u_np, spec)

    # Compare interior
    gpu_int = gpu_result[halo:-halo]
    cpu_int = cpu_result[halo:-halo]
    maxdiff = np.max(np.abs(gpu_int - cpu_int))
    assert np.allclose(gpu_int, cpu_int, atol=1e-10), (
        f"4th-order Laplacian GPU vs CPU: max_diff={maxdiff:.2e}"
    )
