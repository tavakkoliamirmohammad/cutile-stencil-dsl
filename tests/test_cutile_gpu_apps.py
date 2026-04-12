"""GPU convergence tests for all stencil applications.

Each test compiles a stencil, runs the generated kernel on GPU, and
compares the result against a CPU reference computed with _ArrayProxy.

Covers:
  - Single-input stencils: heat_1d, heat_2d, laplacian_3d, 4th-order
    Laplacian, advection_upwind, wave_2d (4th-order)
  - Two-input stencils: FDTD Maxwell update_E and update_H,
    Gray-Scott u and v
  - Three-input stencil: Shallow water h, hu, hv
"""

import numpy as np
import pytest

from cutile.frontend.decorator import stencil
from cutile.runtime.launcher import compile as stencil_compile
from cutile.reference.stencil_ref import apply_stencil, _ArrayProxy

# Guard cupy import for environments without GPU
try:
    import cupy as cp
    cp.cuda.Device(0).compute_capability
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False
    cp = None

gpu_required = pytest.mark.skipif(not _HAS_GPU, reason="No GPU available")


# ===================================================================
# Helpers
# ===================================================================


def _interior_slices(halo_widths):
    """Build interior slices that handle zero-width halos correctly."""
    slices = []
    for h in halo_widths:
        if h == 0:
            slices.append(slice(None))
        else:
            slices.append(slice(h, -h))
    return tuple(slices)


def _apply_multi_input(arrays, fn, ndim, halo_widths):
    """Apply a multi-input stencil function using _ArrayProxy.

    Parameters
    ----------
    arrays : list of np.ndarray
        Input arrays (all same shape, all with halo padding).
    fn : callable
        The raw stencil function (e.g. ``stencil_fn._fn``).
    ndim : int
        Number of spatial dimensions.
    halo_widths : tuple of int
        Halo width per dimension.

    Returns
    -------
    np.ndarray
        Copy of the first array with only the interior updated.
    """
    proxies = [_ArrayProxy(a, halo_widths) for a in arrays]
    idx_args = [0] * ndim
    result = fn(*proxies, *idx_args)

    out = arrays[0].copy()
    interior = tuple(
        slice(h, s - h) for h, s in zip(halo_widths, arrays[0].shape)
    )
    out[interior] = result
    return out


# ===================================================================
# GPU convergence tests: single-input stencils
# ===================================================================


@gpu_required
class TestSingleInputGPUConvergence:
    """GPU convergence tests for single-input stencils."""

    def _validate(self, stencil_fn, domain, atol=1e-10):
        """Compile stencil, run on GPU, compare to CPU reference."""
        result = stencil_compile(stencil_fn, temporal_blocking=False)
        mod = result.load_module()
        launch_fn = getattr(mod, f"launch_{result.name}")

        ndim = result.ndim
        halo = result.halo_widths

        shape = tuple(d + 2 * h for d, h in zip(domain, halo))
        np.random.seed(42)
        u_np = np.random.randn(*shape).astype(np.float64)

        # GPU
        u_gpu = cp.asarray(u_np)
        out_gpu = cp.zeros_like(u_gpu)
        launch_fn(u_gpu, out_gpu)
        cp.cuda.Device(0).synchronize()
        gpu_result = cp.asnumpy(out_gpu)

        # CPU reference
        cpu_result = apply_stencil(u_np, stencil_fn._fn, ndim=ndim, halo_widths=halo)

        # Compare interior
        slices = _interior_slices(halo)
        maxdiff = np.max(np.abs(gpu_result[slices] - cpu_result[slices]))
        assert maxdiff < atol, f"max_diff={maxdiff:.2e}"

    def test_heat_1d(self):
        @stencil(ndim=1, order=2)
        def heat1d(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        self._validate(heat1d, (512,))

    def test_heat_2d(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        self._validate(heat2d, (128, 128))

    def test_laplacian_3d(self):
        @stencil(ndim=3, order=2)
        def lap3d(u, i, j, k):
            return (
                u[i - 1, j, k] + u[i + 1, j, k]
                + u[i, j - 1, k] + u[i, j + 1, k]
                + u[i, j, k - 1] + u[i, j, k + 1]
            ) / 6.0

        self._validate(lap3d, (32, 32, 32))

    def test_4th_order_laplacian_1d(self):
        @stencil(ndim=1, order=4)
        def lap4(u, i):
            return (-u[i - 2] + 16 * u[i - 1] - 30 * u[i] + 16 * u[i + 1] - u[i + 2]) / 12.0

        self._validate(lap4, (256,))

    def test_advection_upwind(self):
        CFL = 0.5

        @stencil(ndim=1, order=2)
        def advect(u, i):
            return u[i] - CFL * (u[i] - u[i - 1])

        self._validate(advect, (512,))

    def test_wave_2d_4th_order(self):
        """4th-order 2D Laplacian with pre-distributed scaling factor.

        The constant 0.1 is distributed to each term to avoid a known
        codegen operator-precedence limitation with ``c * (sum_of_terms)``
        expressions.
        """
        c = 0.1

        @stencil(ndim=2, order=4)
        def wave(u, i, j):
            return c * (-1 / 12) * u[i - 2, j] + c * (4 / 3) * u[i - 1, j] + c * (-5 / 2) * u[i, j] + c * (4 / 3) * u[i + 1, j] + c * (-1 / 12) * u[i + 2, j] + c * (-1 / 12) * u[i, j - 2] + c * (4 / 3) * u[i, j - 1] + c * (-5 / 2) * u[i, j] + c * (4 / 3) * u[i, j + 1] + c * (-1 / 12) * u[i, j + 2]

        self._validate(wave, (64, 64))


# ===================================================================
# GPU convergence tests: two-input stencils (FDTD Maxwell)
# ===================================================================


@gpu_required
class TestFDTDMaxwellGPUConvergence:
    """GPU convergence tests for 1D FDTD Maxwell (two-input stencils)."""

    def _validate_two_input(self, stencil_fn, domain, atol=1e-10):
        """Compile a two-input stencil, run on GPU, compare to CPU."""
        result = stencil_compile(stencil_fn, temporal_blocking=False)
        mod = result.load_module()
        launch_fn = getattr(mod, f"launch_{result.name}")

        ndim = result.ndim
        halo = result.halo_widths

        shape = tuple(d + 2 * h for d, h in zip(domain, halo))
        np.random.seed(42)
        a_np = np.random.randn(*shape).astype(np.float64)
        b_np = np.random.randn(*shape).astype(np.float64)

        # GPU: launch_fn(u, v, u_out) for two-input stencils
        a_gpu = cp.asarray(a_np)
        b_gpu = cp.asarray(b_np)
        out_gpu = cp.zeros_like(a_gpu)
        launch_fn(a_gpu, b_gpu, out_gpu)
        cp.cuda.Device(0).synchronize()
        gpu_result = cp.asnumpy(out_gpu)

        # CPU reference using _ArrayProxy
        cpu_result = _apply_multi_input(
            [a_np, b_np], stencil_fn._fn, ndim=ndim, halo_widths=halo
        )

        # Compare interior
        slices = _interior_slices(halo)
        maxdiff = np.max(np.abs(gpu_result[slices] - cpu_result[slices]))
        assert maxdiff < atol, f"max_diff={maxdiff:.2e}"

    def test_update_E(self):
        dx = 0.01
        dt = 0.005
        eps = 1.0

        @stencil(ndim=1, order=2)
        def update_E(E, H, i):
            return E[i] + (dt / (eps * dx)) * (H[i] - H[i - 1])

        self._validate_two_input(update_E, (512,))

    def test_update_H(self):
        dx = 0.01
        dt = 0.005
        mu = 1.0

        @stencil(ndim=1, order=2)
        def update_H(E, H, i):
            return H[i] + (dt / (mu * dx)) * (E[i + 1] - E[i])

        self._validate_two_input(update_H, (512,))


# ===================================================================
# GPU convergence tests: two-input stencils (Gray-Scott)
# ===================================================================


@gpu_required
class TestGrayScottGPUConvergence:
    """GPU convergence tests for Gray-Scott reaction-diffusion (two-input)."""

    def _validate_two_input(self, stencil_fn, domain, atol=1e-10):
        """Compile a two-input stencil, run on GPU, compare to CPU."""
        result = stencil_compile(stencil_fn, temporal_blocking=False)
        mod = result.load_module()
        launch_fn = getattr(mod, f"launch_{result.name}")

        ndim = result.ndim
        halo = result.halo_widths

        shape = tuple(d + 2 * h for d, h in zip(domain, halo))
        # Use uniform [0.2, 0.8] to keep values in physically reasonable range
        np.random.seed(42)
        u_np = 0.2 + 0.6 * np.random.rand(*shape).astype(np.float64)
        v_np = 0.2 + 0.6 * np.random.rand(*shape).astype(np.float64)

        # GPU
        u_gpu = cp.asarray(u_np)
        v_gpu = cp.asarray(v_np)
        out_gpu = cp.zeros_like(u_gpu)
        launch_fn(u_gpu, v_gpu, out_gpu)
        cp.cuda.Device(0).synchronize()
        gpu_result = cp.asnumpy(out_gpu)

        # CPU reference
        cpu_result = _apply_multi_input(
            [u_np, v_np], stencil_fn._fn, ndim=ndim, halo_widths=halo
        )

        # Compare interior
        slices = _interior_slices(halo)
        maxdiff = np.max(np.abs(gpu_result[slices] - cpu_result[slices]))
        assert maxdiff < atol, f"max_diff={maxdiff:.2e}"

    def test_gray_scott_u(self):
        Du = 0.16
        F = 0.035
        dt = 1.0

        @stencil(ndim=2, order=2)
        def gs_u(u, v, i, j):
            return u[i, j] + dt * (Du * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]) - u[i, j] * v[i, j] * v[i, j] + F * (1.0 - u[i, j]))

        self._validate_two_input(gs_u, (64, 64))

    def test_gray_scott_v(self):
        Dv = 0.08
        F = 0.035
        k = 0.065
        dt = 1.0

        @stencil(ndim=2, order=2)
        def gs_v(u, v, i, j):
            return v[i, j] + dt * (Dv * (v[i - 1, j] + v[i + 1, j] + v[i, j - 1] + v[i, j + 1] - 4 * v[i, j]) + u[i, j] * v[i, j] * v[i, j] - (F + k) * v[i, j])

        self._validate_two_input(gs_v, (64, 64))


# ===================================================================
# GPU convergence tests: multi-input stencils (Shallow Water)
# ===================================================================


@gpu_required
class TestShallowWaterGPUConvergence:
    """GPU convergence tests for shallow water equations (up to three inputs).

    The stencil expressions pre-distribute constant factors to each term
    to avoid a known codegen operator-precedence limitation with
    ``c * (sum_of_terms)`` expressions.
    """

    def _validate_three_input(self, stencil_fn, domain, atol=1e-10):
        """Compile a three-input stencil, run on GPU, compare to CPU."""
        result = stencil_compile(stencil_fn, temporal_blocking=False)
        mod = result.load_module()
        launch_fn = getattr(mod, f"launch_{result.name}")

        ndim = result.ndim
        halo = result.halo_widths

        shape = tuple(d + 2 * h for d, h in zip(domain, halo))
        np.random.seed(42)
        a_np = np.random.randn(*shape).astype(np.float64)
        b_np = np.random.randn(*shape).astype(np.float64)
        c_np = np.random.randn(*shape).astype(np.float64)

        # GPU: launch_fn(u, v, w, u_out) for three-input stencils
        a_gpu = cp.asarray(a_np)
        b_gpu = cp.asarray(b_np)
        c_gpu = cp.asarray(c_np)
        out_gpu = cp.zeros_like(a_gpu)
        launch_fn(a_gpu, b_gpu, c_gpu, out_gpu)
        cp.cuda.Device(0).synchronize()
        gpu_result = cp.asnumpy(out_gpu)

        # CPU reference
        cpu_result = _apply_multi_input(
            [a_np, b_np, c_np], stencil_fn._fn, ndim=ndim, halo_widths=halo
        )

        # Compare interior
        slices = _interior_slices(halo)
        maxdiff = np.max(np.abs(gpu_result[slices] - cpu_result[slices]))
        assert maxdiff < atol, f"max_diff={maxdiff:.2e}"

    def _validate_two_input(self, stencil_fn, domain, atol=1e-10):
        """Compile a two-input stencil, run on GPU, compare to CPU."""
        result = stencil_compile(stencil_fn, temporal_blocking=False)
        mod = result.load_module()
        launch_fn = getattr(mod, f"launch_{result.name}")

        ndim = result.ndim
        halo = result.halo_widths

        shape = tuple(d + 2 * h for d, h in zip(domain, halo))
        np.random.seed(42)
        a_np = np.random.randn(*shape).astype(np.float64)
        b_np = np.random.randn(*shape).astype(np.float64)

        # GPU: launch_fn(u, v, u_out) for two-input stencils
        a_gpu = cp.asarray(a_np)
        b_gpu = cp.asarray(b_np)
        out_gpu = cp.zeros_like(a_gpu)
        launch_fn(a_gpu, b_gpu, out_gpu)
        cp.cuda.Device(0).synchronize()
        gpu_result = cp.asnumpy(out_gpu)

        # CPU reference
        cpu_result = _apply_multi_input(
            [a_np, b_np], stencil_fn._fn, ndim=ndim, halo_widths=halo
        )

        # Compare interior
        slices = _interior_slices(halo)
        maxdiff = np.max(np.abs(gpu_result[slices] - cpu_result[slices]))
        assert maxdiff < atol, f"max_diff={maxdiff:.2e}"

    def test_shallow_water_h(self):
        """Shallow water h-equation with distributed constant factor."""
        H0 = 1.0
        dt = 0.01
        dx = 0.1
        # Pre-computed: dt * H0 / (2 * dx)
        coeff = dt * H0 / (2 * dx)

        @stencil(ndim=2, order=2)
        def sw_h(h, hu, hv, i, j):
            return h[i, j] - coeff * (hu[i + 1, j] - hu[i - 1, j]) - coeff * (hv[i, j + 1] - hv[i, j - 1])

        self._validate_three_input(sw_h, (64, 64))

    def test_shallow_water_hu(self):
        """Shallow water hu-equation with pre-computed coefficient."""
        g = 9.81
        dt = 0.01
        dx = 0.1
        coeff = dt * g / (2 * dx)

        @stencil(ndim=2, order=2)
        def sw_hu(h, hu, i, j):
            return hu[i, j] - coeff * (h[i + 1, j] - h[i - 1, j])

        self._validate_two_input(sw_hu, (64, 64))

    def test_shallow_water_hv(self):
        """Shallow water hv-equation with pre-computed coefficient."""
        g = 9.81
        dt = 0.01
        dx = 0.1
        coeff = dt * g / (2 * dx)

        @stencil(ndim=2, order=2)
        def sw_hv(h, hv, i, j):
            return hv[i, j] - coeff * (h[i, j + 1] - h[i, j - 1])

        self._validate_two_input(sw_hv, (64, 64))
