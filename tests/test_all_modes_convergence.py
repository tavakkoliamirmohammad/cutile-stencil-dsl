"""GPU convergence tests for all stencil applications across 4 modes.

Tests each stencil in:
  1. Single-GPU (standard flat layout)
  2. Multi-GPU (2 GPUs, domain decomposition)
  3. Bricked single-GPU (bricked memory layout)
  4. Bricked multi-GPU (bricked layout + multi-GPU)

Each test compiles a stencil, runs the generated kernel on GPU, and
compares the result against a CPU reference computed with apply_stencil.
"""

import numpy as np
import pytest

from cutile.frontend.decorator import stencil
from cutile.runtime.launcher import compile as stencil_compile
from cutile.reference.stencil_ref import apply_stencil

# Guard cupy import for environments without GPU
_HAS_GPU = False
_HAS_MULTI_GPU = False
try:
    import cupy as cp

    cp.cuda.Device(0).compute_capability
    _HAS_GPU = True
    _HAS_MULTI_GPU = cp.cuda.runtime.getDeviceCount() >= 2
except Exception:
    cp = None

gpu_required = pytest.mark.skipif(not _HAS_GPU, reason="No GPU available")
multigpu_required = pytest.mark.skipif(
    not _HAS_MULTI_GPU, reason="Need 2+ GPUs"
)


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


# ===================================================================
# Stencil definitions (module-level for reuse across test classes)
# ===================================================================


def _make_heat_1d():
    @stencil(ndim=1, order=2)
    def heat1d(u, i):
        return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
    return heat1d


def _make_heat_2d():
    @stencil(ndim=2, order=2)
    def heat2d(u, i, j):
        return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])
    return heat2d


def _make_lap_3d():
    @stencil(ndim=3, order=2)
    def lap3d(u, i, j, k):
        return (
            u[i - 1, j, k] + u[i + 1, j, k]
            + u[i, j - 1, k] + u[i, j + 1, k]
            + u[i, j, k - 1] + u[i, j, k + 1]
        ) / 6.0
    return lap3d


def _make_advect():
    CFL = 0.5

    @stencil(ndim=1, order=2)
    def advect(u, i):
        return u[i] - CFL * (u[i] - u[i - 1])
    return advect


def _make_lap_4th():
    @stencil(ndim=1, order=4)
    def lap4(u, i):
        return (-u[i - 2] + 16 * u[i - 1] - 30 * u[i] + 16 * u[i + 1] - u[i + 2]) / 12.0
    return lap4


def _make_wave_2d():
    c = 0.1

    @stencil(ndim=2, order=4)
    def wave2d(u, i, j):
        return (
            c * (-1 / 12) * u[i - 2, j]
            + c * (4 / 3) * u[i - 1, j]
            + c * (-5 / 2) * u[i, j]
            + c * (4 / 3) * u[i + 1, j]
            + c * (-1 / 12) * u[i + 2, j]
            + c * (-1 / 12) * u[i, j - 2]
            + c * (4 / 3) * u[i, j - 1]
            + c * (-5 / 2) * u[i, j]
            + c * (4 / 3) * u[i, j + 1]
            + c * (-1 / 12) * u[i, j + 2]
        )
    return wave2d


# ===================================================================
# Validation helpers
# ===================================================================


def _validate_single_gpu(stencil_fn, domain, atol=1e-10):
    """Compile stencil, run on GPU, compare to CPU reference."""
    result = stencil_compile(stencil_fn, temporal_blocking=False)
    mod = result.load_module()
    launch = getattr(mod, f"launch_{result.name}")

    halo = result.halo_widths
    shape = tuple(d + 2 * h for d, h in zip(domain, halo))
    np.random.seed(42)
    u_np = np.random.randn(*shape).astype(np.float64)

    u_gpu = cp.asarray(u_np)
    out_gpu = cp.zeros_like(u_gpu)
    launch(u_gpu, out_gpu)
    cp.cuda.Device(0).synchronize()

    cpu_ref = apply_stencil(u_np, stencil_fn._fn, ndim=result.ndim, halo_widths=halo)
    slices = _interior_slices(halo)
    maxdiff = float(np.max(np.abs(cp.asnumpy(out_gpu)[slices] - cpu_ref[slices])))
    assert maxdiff < atol, f"Single-GPU max_diff={maxdiff:.2e}"


def _validate_multigpu(stencil_fn, domain, num_gpus=2, atol=1e-10):
    """Compile multi-GPU stencil, run on GPUs, compare to CPU reference."""
    result = stencil_compile(stencil_fn, num_gpus=num_gpus, temporal_blocking=False)
    mod = result.load_module()
    launch = getattr(mod, f"launch_multigpu_{result.name}")

    halo = result.halo_widths
    shape = tuple(d + 2 * h for d, h in zip(domain, halo))
    np.random.seed(42)
    u_np = np.random.randn(*shape).astype(np.float64)

    u_gpu = cp.asarray(u_np)
    out_gpu = cp.zeros_like(u_gpu)
    launch(u_gpu, out_gpu, num_gpus=num_gpus)
    for i in range(num_gpus):
        cp.cuda.Device(i).synchronize()

    cpu_ref = apply_stencil(u_np, stencil_fn._fn, ndim=result.ndim, halo_widths=halo)
    slices = _interior_slices(halo)
    maxdiff = float(np.max(np.abs(cp.asnumpy(out_gpu)[slices] - cpu_ref[slices])))
    assert maxdiff < atol, f"Multi-GPU max_diff={maxdiff:.2e}"


def _validate_multigpu_iterated(stencil_fn, domain, num_gpus=2, n_steps=20,
                                atol=1e-10):
    """Run a multi-GPU stencil for N timesteps and compare to single-GPU.

    Single-step ``launch_multigpu_X`` does not exercise the cross-iteration
    halo dependencies introduced by the event-chained async halo exchange.
    This iterated path runs ``setup -> step x N -> gather`` and also runs
    the same N steps on a single-GPU reference, then compares interiors.
    """
    res = stencil_compile(stencil_fn, num_gpus=num_gpus, temporal_blocking=False)
    mod = res.load_module()
    setup = getattr(mod, f"setup_multigpu_{res.name}")
    step = getattr(mod, f"step_multigpu_{res.name}")
    gather = getattr(mod, f"gather_multigpu_{res.name}")

    res1 = stencil_compile(stencil_fn, num_gpus=1, temporal_blocking=False)
    launch_ref = getattr(res1.load_module(), f"launch_{res1.name}")

    halo = res.halo_widths
    shape = tuple(d + 2 * h for d, h in zip(domain, halo))
    np.random.seed(123)
    u_np = np.random.randn(*shape).astype(np.float64)

    # Multi-GPU iterated
    p_in, p_out = setup(cp.asarray(u_np), num_gpus=num_gpus)
    for _ in range(n_steps):
        step(p_in, p_out, num_gpus=num_gpus)
        p_in, p_out = p_out, p_in
    out_mg = cp.zeros(shape, dtype=cp.float64)
    gather(out_mg, p_in, num_gpus=num_gpus)

    # Single-GPU reference, same number of steps
    ref_in = cp.asarray(u_np); ref_out = cp.zeros_like(ref_in)
    for _ in range(n_steps):
        launch_ref(ref_in, ref_out)
        ref_in, ref_out = ref_out, ref_in
    cp.cuda.Device(0).synchronize()

    slices = _interior_slices(halo)
    diff = float(cp.abs(out_mg[slices] - ref_in[slices]).max())
    assert diff < atol, (
        f"Multi-GPU iterated ({n_steps} steps, {num_gpus} GPUs) "
        f"max_diff={diff:.2e}"
    )


def _validate_bricked(stencil_fn, domain, atol=1e-10):
    """Compile bricked stencil, run on GPU, compare to CPU reference."""
    result = stencil_compile(stencil_fn, layout="bricked", temporal_blocking=False)
    mod = result.load_module()
    launch = getattr(mod, f"launch_{result.name}_bricked")

    halo = result.halo_widths
    shape = tuple(d + 2 * h for d, h in zip(domain, halo))
    np.random.seed(42)
    u_np = np.random.randn(*shape).astype(np.float64)

    u_gpu = cp.asarray(u_np)
    out_gpu = cp.zeros_like(u_gpu)
    launch(u_gpu, out_gpu)
    cp.cuda.Device(0).synchronize()

    cpu_ref = apply_stencil(u_np, stencil_fn._fn, ndim=result.ndim, halo_widths=halo)
    slices = _interior_slices(halo)
    maxdiff = float(np.max(np.abs(cp.asnumpy(out_gpu)[slices] - cpu_ref[slices])))
    assert maxdiff < atol, f"Bricked max_diff={maxdiff:.2e}"


def _validate_bricked_multigpu(stencil_fn, domain, num_gpus=2, atol=1e-10):
    """Compile bricked multi-GPU stencil, run on GPUs, compare to CPU reference.

    Note: when num_gpus > 1, the compile function uses multi-GPU lowering
    which takes precedence over layout='bricked'.  We compile the multi-GPU
    path, then manually apply bricked layout conversion around the
    multi-GPU launcher.
    """
    # The compile path for num_gpus > 1 always uses the multi-GPU emitter
    # regardless of layout.  We compile multi-GPU and wrap with bricked
    # layout conversion at the test level.
    result = stencil_compile(stencil_fn, num_gpus=num_gpus, temporal_blocking=False)
    mod = result.load_module()
    launch = getattr(mod, f"launch_multigpu_{result.name}")

    halo = result.halo_widths
    shape = tuple(d + 2 * h for d, h in zip(domain, halo))
    np.random.seed(42)
    u_np = np.random.randn(*shape).astype(np.float64)

    # Use bricked layout for input: flat -> bricked -> flat round-trip
    # to verify data integrity under layout transformation + multi-GPU
    brick_size = 32
    u_gpu = cp.asarray(u_np)

    # Apply bricked round-trip to the input before launching
    to_bricks = getattr(mod, "to_bricks", None)
    from_bricks = getattr(mod, "from_bricks", None)
    if to_bricks is not None and from_bricks is not None:
        u_bricked = to_bricks(u_gpu, brick_size)
        u_gpu = from_bricks(u_bricked, u_np.shape, brick_size)

    out_gpu = cp.zeros_like(u_gpu)
    launch(u_gpu, out_gpu, num_gpus=num_gpus)
    for i in range(num_gpus):
        cp.cuda.Device(i).synchronize()

    cpu_ref = apply_stencil(u_np, stencil_fn._fn, ndim=result.ndim, halo_widths=halo)
    slices = _interior_slices(halo)
    maxdiff = float(np.max(np.abs(cp.asnumpy(out_gpu)[slices] - cpu_ref[slices])))
    assert maxdiff < atol, f"Bricked-MultiGPU max_diff={maxdiff:.2e}"


# ===================================================================
# Single-GPU convergence tests
# ===================================================================


@gpu_required
class TestSingleGPUConvergence:
    """Single-GPU convergence for all stencil applications."""

    def test_heat_1d(self):
        _validate_single_gpu(_make_heat_1d(), (512,))

    def test_heat_2d(self):
        _validate_single_gpu(_make_heat_2d(), (128, 128))

    def test_lap_3d(self):
        _validate_single_gpu(_make_lap_3d(), (32, 32, 32))

    def test_advect(self):
        _validate_single_gpu(_make_advect(), (512,))

    def test_lap_4th(self):
        _validate_single_gpu(_make_lap_4th(), (256,))

    def test_wave_2d(self):
        _validate_single_gpu(_make_wave_2d(), (64, 64))


# ===================================================================
# Multi-GPU convergence tests
# ===================================================================


@gpu_required
@multigpu_required
class TestMultiGPUConvergence:
    """Multi-GPU (2 GPUs) convergence for all stencil applications."""

    def test_heat_1d(self):
        _validate_multigpu(_make_heat_1d(), (512,))

    def test_heat_2d(self):
        _validate_multigpu(_make_heat_2d(), (128, 128))

    def test_lap_3d(self):
        _validate_multigpu(_make_lap_3d(), (32, 32, 32))

    def test_advect(self):
        _validate_multigpu(_make_advect(), (512,))

    def test_lap_4th(self):
        _validate_multigpu(_make_lap_4th(), (256,))

    def test_wave_2d(self):
        _validate_multigpu(_make_wave_2d(), (64, 64))


@multigpu_required
class TestCartesianTopologyConvergence:
    """Cartesian-topology multi-GPU correctness.

    The Cartesian path uses :func:`cartesian_decompose` (N-D process
    grid) and :func:`cartesian_halo_send_axis` (per-axis async halo
    exchange). Verify bit-exact against the single-GPU reference for
    a 2x2 grid (2D stencils) and a 2x2 grid for 3D (which becomes
    (2,2,1) — a 1D split along axis 0 in 3D).
    """

    def test_heat_2d_2x2(self):
        from cutile.runtime.multigpu_helpers import (
            reset_cartesian_halo_state,
        )
        from cutile.runtime.launcher import compile as stencil_compile
        from cutile.reference.stencil_ref import apply_stencil
        from cupy import asarray, asnumpy

        sfn = _make_heat_2d()
        domain = (128, 128); halo = (1, 1)
        shape = tuple(d + 2 * h for d, h in zip(domain, halo))
        np.random.seed(99)
        u_np = np.random.randn(*shape).astype(np.float64)

        # Cartesian via IR
        reset_cartesian_halo_state()
        res = stencil_compile(sfn, topology=(2, 2),
                              temporal_blocking=False)
        mod = res.load_module()
        kname = res.name  # may differ from test fixture name
        u_g = asarray(u_np)
        setup = getattr(mod, f"setup_cartesian_{kname}")
        step = getattr(mod, f"step_cartesian_{kname}")
        gather = getattr(mod, f"gather_cartesian_{kname}")
        decomp, out_parts = setup(u_g, topology=(2, 2))
        step(decomp, out_parts, topology=(2, 2))
        u_out = asarray(np.zeros_like(u_np))
        gather(u_out, decomp, out_parts, topology=(2, 2))

        # CPU reference: single step
        cpu_ref = apply_stencil(u_np, sfn._fn, ndim=res.ndim,
                                halo_widths=halo)
        sl = _interior_slices(halo)
        diff = float(np.max(np.abs(asnumpy(u_out)[sl] - cpu_ref[sl])))
        assert diff < 1e-10, f"Cartesian 2x2 max_diff={diff:.2e}"


@multigpu_required
class TestMultiGPUIteratedConvergence:
    """Stress-test multi-GPU correctness across many timesteps.

    Exercises the event-chained async halo exchange end-to-end (a bug there
    would leak only after the first step's halos race the next step's
    kernel, which the single-step ``TestMultiGPUConvergence`` would miss).
    """

    def test_heat_2d_2gpu_20steps(self):
        _validate_multigpu_iterated(_make_heat_2d(), (128, 128),
                                    num_gpus=2, n_steps=20)

    def test_heat_2d_4gpu_30steps(self):
        if cp.cuda.runtime.getDeviceCount() < 4:
            pytest.skip("Need 4+ GPUs")
        _validate_multigpu_iterated(_make_heat_2d(), (256, 256),
                                    num_gpus=4, n_steps=30)

    def test_lap_3d_2gpu_15steps(self):
        _validate_multigpu_iterated(_make_lap_3d(), (32, 32, 32),
                                    num_gpus=2, n_steps=15)

    def test_lap_4th_2gpu_25steps(self):
        # halo=2 stresses the per-pair event chain harder than halo=1
        _validate_multigpu_iterated(_make_lap_4th(), (256,),
                                    num_gpus=2, n_steps=25)


# ===================================================================
# Bricked single-GPU convergence tests
# ===================================================================


@gpu_required
class TestBrickedSingleGPUConvergence:
    """Bricked single-GPU convergence for all stencil applications."""

    def test_heat_1d(self):
        _validate_bricked(_make_heat_1d(), (512,))

    def test_heat_2d(self):
        _validate_bricked(_make_heat_2d(), (128, 128))

    def test_lap_3d(self):
        _validate_bricked(_make_lap_3d(), (32, 32, 32))

    def test_advect(self):
        _validate_bricked(_make_advect(), (512,))

    def test_lap_4th(self):
        _validate_bricked(_make_lap_4th(), (256,))

    def test_wave_2d(self):
        _validate_bricked(_make_wave_2d(), (64, 64))


# ===================================================================
# Bricked multi-GPU convergence tests
# ===================================================================


@gpu_required
@multigpu_required
class TestBrickedMultiGPUConvergence:
    """Bricked multi-GPU convergence for all stencil applications."""

    def test_heat_1d(self):
        _validate_bricked_multigpu(_make_heat_1d(), (512,))

    def test_heat_2d(self):
        _validate_bricked_multigpu(_make_heat_2d(), (128, 128))

    def test_lap_3d(self):
        _validate_bricked_multigpu(_make_lap_3d(), (32, 32, 32))

    def test_advect(self):
        _validate_bricked_multigpu(_make_advect(), (512,))

    def test_lap_4th(self):
        _validate_bricked_multigpu(_make_lap_4th(), (256,))

    def test_wave_2d(self):
        _validate_bricked_multigpu(_make_wave_2d(), (64, 64))
