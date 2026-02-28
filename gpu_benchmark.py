#!/usr/bin/env python3
"""Real GPU benchmarks using Numba CUDA on your RTX PRO 6000 Blackwell.

Runs actual CUDA kernels for each stencil pattern and measures real throughput,
then compares against NumPy CPU and Numba CPU baselines.
"""

import time
import math
import numpy as np
from numba import cuda, njit, prange, float32, float64

# ============================================================================
# GPU KERNELS (Numba CUDA)
# ============================================================================

# --- 1D Heat (3-point stencil, order 2) ---
@cuda.jit
def heat_1d_gpu_f64(u, out):
    i = cuda.grid(1)
    if i >= 1 and i < u.shape[0] - 1:
        out[i] = 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

@cuda.jit
def heat_1d_gpu_f32(u, out):
    i = cuda.grid(1)
    if i >= 1 and i < u.shape[0] - 1:
        out[i] = float32(0.25) * u[i - 1] + float32(0.5) * u[i] + float32(0.25) * u[i + 1]

# --- 1D Heat order 4 (5-point stencil) ---
@cuda.jit
def heat_1d_o4_gpu_f64(u, out):
    i = cuda.grid(1)
    if i >= 2 and i < u.shape[0] - 2:
        out[i] = (-1.0/12.0)*u[i-2] + (4.0/3.0)*u[i-1] + (-5.0/2.0)*u[i] + (4.0/3.0)*u[i+1] + (-1.0/12.0)*u[i+2]

@cuda.jit
def heat_1d_o4_gpu_f32(u, out):
    i = cuda.grid(1)
    if i >= 2 and i < u.shape[0] - 2:
        out[i] = float32(-1.0/12.0)*u[i-2] + float32(4.0/3.0)*u[i-1] + float32(-5.0/2.0)*u[i] + float32(4.0/3.0)*u[i+1] + float32(-1.0/12.0)*u[i+2]

# --- 2D Laplacian (5-point stencil) ---
@cuda.jit
def laplacian_2d_gpu_f64(u, out):
    i, j = cuda.grid(2)
    if i >= 1 and i < u.shape[0] - 1 and j >= 1 and j < u.shape[1] - 1:
        out[i, j] = u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4.0*u[i,j]

@cuda.jit
def laplacian_2d_gpu_f32(u, out):
    i, j = cuda.grid(2)
    if i >= 1 and i < u.shape[0] - 1 and j >= 1 and j < u.shape[1] - 1:
        out[i, j] = u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - float32(4.0)*u[i,j]

# --- 3D Laplacian (7-point stencil) ---
@cuda.jit
def laplacian_3d_gpu_f64(u, out):
    i, j, k = cuda.grid(3)
    if (i >= 1 and i < u.shape[0] - 1 and
        j >= 1 and j < u.shape[1] - 1 and
        k >= 1 and k < u.shape[2] - 1):
        out[i,j,k] = (u[i-1,j,k] + u[i+1,j,k] +
                       u[i,j-1,k] + u[i,j+1,k] +
                       u[i,j,k-1] + u[i,j,k+1] - 6.0*u[i,j,k])

@cuda.jit
def laplacian_3d_gpu_f32(u, out):
    i, j, k = cuda.grid(3)
    if (i >= 1 and i < u.shape[0] - 1 and
        j >= 1 and j < u.shape[1] - 1 and
        k >= 1 and k < u.shape[2] - 1):
        out[i,j,k] = (u[i-1,j,k] + u[i+1,j,k] +
                       u[i,j-1,k] + u[i,j+1,k] +
                       u[i,j,k-1] + u[i,j,k+1] - float32(6.0)*u[i,j,k])

# --- 2D Wave (multi-statement, the codegen bug case) ---
@cuda.jit
def wave_2d_gpu_f64(u, out):
    i, j = cuda.grid(2)
    if i >= 1 and i < u.shape[0] - 1 and j >= 1 and j < u.shape[1] - 1:
        lap_x = u[i-1,j] + u[i+1,j] - 2.0*u[i,j]
        lap_y = u[i,j-1] + u[i,j+1] - 2.0*u[i,j]
        out[i,j] = u[i,j] + 0.1 * (lap_x + lap_y)

@cuda.jit
def wave_2d_gpu_f32(u, out):
    i, j = cuda.grid(2)
    if i >= 1 and i < u.shape[0] - 1 and j >= 1 and j < u.shape[1] - 1:
        lap_x = u[i-1,j] + u[i+1,j] - float32(2.0)*u[i,j]
        lap_y = u[i,j-1] + u[i,j+1] - float32(2.0)*u[i,j]
        out[i,j] = u[i,j] + float32(0.1) * (lap_x + lap_y)


# ============================================================================
# CPU KERNELS (Numba JIT, parallel)
# ============================================================================

@njit(parallel=True)
def heat_1d_cpu(u, out):
    for i in prange(1, u.shape[0] - 1):
        out[i] = 0.25 * u[i-1] + 0.5 * u[i] + 0.25 * u[i+1]

@njit(parallel=True)
def heat_1d_o4_cpu(u, out):
    for i in prange(2, u.shape[0] - 2):
        out[i] = (-1.0/12.0)*u[i-2] + (4.0/3.0)*u[i-1] + (-5.0/2.0)*u[i] + (4.0/3.0)*u[i+1] + (-1.0/12.0)*u[i+2]

@njit(parallel=True)
def laplacian_2d_cpu(u, out):
    for i in prange(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            out[i,j] = u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4.0*u[i,j]

@njit(parallel=True)
def laplacian_3d_cpu(u, out):
    for i in prange(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            for k in range(1, u.shape[2] - 1):
                out[i,j,k] = (u[i-1,j,k] + u[i+1,j,k] +
                               u[i,j-1,k] + u[i,j+1,k] +
                               u[i,j,k-1] + u[i,j,k+1] - 6.0*u[i,j,k])

@njit(parallel=True)
def wave_2d_cpu(u, out):
    for i in prange(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            lap_x = u[i-1,j] + u[i+1,j] - 2.0*u[i,j]
            lap_y = u[i,j-1] + u[i,j+1] - 2.0*u[i,j]
            out[i,j] = u[i,j] + 0.1 * (lap_x + lap_y)


# ============================================================================
# Benchmark harness
# ============================================================================

def bench_gpu(kernel, u_dev, out_dev, grid, block, iters=50, warmup=10):
    """Time a GPU kernel with proper CUDA synchronization."""
    # Warmup
    for _ in range(warmup):
        kernel[grid, block](u_dev, out_dev)
    cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        kernel[grid, block](u_dev, out_dev)
    cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters


def bench_cpu_numba(fn, u, out, iters=20, warmup=5):
    """Time a Numba CPU parallel kernel."""
    for _ in range(warmup):
        fn(u, out)
    start = time.perf_counter()
    for _ in range(iters):
        fn(u, out)
    elapsed = time.perf_counter() - start
    return elapsed / iters


def bench_numpy(fn_np, u, out, iters=20, warmup=5):
    """Time a NumPy stencil."""
    for _ in range(warmup):
        fn_np(u, out)
    start = time.perf_counter()
    for _ in range(iters):
        fn_np(u, out)
    elapsed = time.perf_counter() - start
    return elapsed / iters


# NumPy reference implementations
def heat_1d_np(u, out):
    out[1:-1] = 0.25*u[:-2] + 0.5*u[1:-1] + 0.25*u[2:]

def heat_1d_o4_np(u, out):
    out[2:-2] = (-1/12)*u[:-4] + (4/3)*u[1:-3] + (-5/2)*u[2:-2] + (4/3)*u[3:-1] + (-1/12)*u[4:]

def laplacian_2d_np(u, out):
    out[1:-1,1:-1] = u[:-2,1:-1] + u[2:,1:-1] + u[1:-1,:-2] + u[1:-1,2:] - 4*u[1:-1,1:-1]

def laplacian_3d_np(u, out):
    out[1:-1,1:-1,1:-1] = (u[:-2,1:-1,1:-1] + u[2:,1:-1,1:-1] +
                            u[1:-1,:-2,1:-1] + u[1:-1,2:,1:-1] +
                            u[1:-1,1:-1,:-2] + u[1:-1,1:-1,2:] - 6*u[1:-1,1:-1,1:-1])

def wave_2d_np(u, out):
    c = u[1:-1,1:-1]
    lap_x = u[:-2,1:-1] + u[2:,1:-1] - 2*c
    lap_y = u[1:-1,:-2] + u[1:-1,2:] - 2*c
    out[1:-1,1:-1] = c + 0.1*(lap_x + lap_y)


def sep(c="=", w=95):
    print(c * w)

def header(title):
    print()
    sep()
    print(f"  {title}")
    sep()


# ============================================================================
# Main
# ============================================================================

def run_all():
    header("REAL GPU BENCHMARK — NVIDIA RTX PRO 6000 Blackwell Max-Q")
    print("  Measuring actual kernel execution times with CUDA synchronization")
    print("  Comparing: NumPy (1 core) vs Numba CPU (all cores) vs Numba CUDA (GPU)")
    sep()

    BLOCK_1D = 256
    BLOCK_2D = (16, 16)
    BLOCK_3D = (8, 8, 8)

    benchmarks = [
        # (name, ndim, halo, interior_points, flops/pt, bytes/pt_f64, bytes/pt_f32,
        #  gpu_f64, gpu_f32, cpu_numba, numpy_fn, grids)
        {
            "name": "heat_1d (3-pt)",
            "ndim": 1, "halo": 1,
            "flops": 7, "loads": 3,
            "gpu_f64": heat_1d_gpu_f64, "gpu_f32": heat_1d_gpu_f32,
            "cpu": heat_1d_cpu, "numpy": heat_1d_np,
            "grids": [(65536,), (262144,), (1048576,), (4194304,), (16777216,)],
        },
        {
            "name": "heat_1d_o4 (5-pt)",
            "ndim": 1, "halo": 2,
            "flops": 18, "loads": 5,
            "gpu_f64": heat_1d_o4_gpu_f64, "gpu_f32": heat_1d_o4_gpu_f32,
            "cpu": heat_1d_o4_cpu, "numpy": heat_1d_o4_np,
            "grids": [(65536,), (262144,), (1048576,), (4194304,), (16777216,)],
        },
        {
            "name": "laplacian_2d (5-pt)",
            "ndim": 2, "halo": 1,
            "flops": 9, "loads": 5,
            "gpu_f64": laplacian_2d_gpu_f64, "gpu_f32": laplacian_2d_gpu_f32,
            "cpu": laplacian_2d_cpu, "numpy": laplacian_2d_np,
            "grids": [(256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)],
        },
        {
            "name": "laplacian_3d (7-pt)",
            "ndim": 3, "halo": 1,
            "flops": 13, "loads": 7,
            "gpu_f64": laplacian_3d_gpu_f64, "gpu_f32": laplacian_3d_gpu_f32,
            "cpu": laplacian_3d_cpu, "numpy": laplacian_3d_np,
            "grids": [(64, 64, 64), (128, 128, 128), (256, 256, 256)],
        },
        {
            "name": "wave_2d (9-pt)",
            "ndim": 2, "halo": 1,
            "flops": 13, "loads": 5,
            "gpu_f64": wave_2d_gpu_f64, "gpu_f32": wave_2d_gpu_f32,
            "cpu": wave_2d_cpu, "numpy": wave_2d_np,
            "grids": [(256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)],
        },
    ]

    grand_summary = []

    for bench in benchmarks:
        name = bench["name"]
        ndim = bench["ndim"]
        halo = bench["halo"]
        flops = bench["flops"]
        loads = bench["loads"]

        header(f"{name}  |  {flops} FLOPs/pt  |  {loads} loads/pt")

        print(f"\n  {'Grid':<16} {'Points':<12} {'Prec':<7} "
              f"{'NumPy':<12} {'Numba CPU':<12} {'GPU':<12} "
              f"{'GPU GB/s':<10} {'GPU/NumPy':<10} {'GPU/CPU':<10}")
        print(f"  {'':<16} {'':<12} {'':<7} "
              f"{'ms':<12} {'ms':<12} {'ms':<12} "
              f"{'':<10} {'speedup':<10} {'speedup':<10}")
        sep("-")

        best_gpu_f32 = None
        best_gpu_f64 = None
        best_cpu = None
        best_np = None

        for grid_interior in bench["grids"]:
            total_pts = 1
            for g in grid_interior:
                total_pts *= g

            # Skip grids that would use too much GPU memory
            bytes_f64 = total_pts * 8 * 2  # u + out
            if bytes_f64 > 80e9:
                continue

            full_shape = tuple(g + 2 * halo for g in grid_interior)
            grid_label = "x".join(str(g) for g in grid_interior)

            for dtype_name, np_dtype, gpu_kernel, bytes_per_elem in [
                ("f64", np.float64, bench["gpu_f64"], 8),
                ("f32", np.float32, bench["gpu_f32"], 4),
            ]:
                bytes_per_pt = (loads + 1) * bytes_per_elem

                # Allocate
                u_host = np.random.randn(*full_shape).astype(np_dtype)
                out_host = np.zeros_like(u_host)

                # --- NumPy ---
                iters_np = max(3, min(50, int(2e7 / max(total_pts, 1))))
                try:
                    ms_np = bench_numpy(bench["numpy"], u_host, out_host,
                                        iters=iters_np, warmup=3) * 1000
                except Exception:
                    ms_np = float('inf')

                # --- Numba CPU (parallel) ---
                iters_cpu = max(3, min(50, int(2e7 / max(total_pts, 1))))
                try:
                    ms_cpu = bench_cpu_numba(bench["cpu"], u_host, out_host,
                                            iters=iters_cpu, warmup=3) * 1000
                except Exception:
                    ms_cpu = float('inf')

                # --- GPU ---
                u_dev = cuda.to_device(u_host)
                out_dev = cuda.device_array_like(u_host)

                if ndim == 1:
                    block = (BLOCK_1D,)
                    grid_dim = (math.ceil(full_shape[0] / BLOCK_1D),)
                elif ndim == 2:
                    block = BLOCK_2D
                    grid_dim = (math.ceil(full_shape[0] / BLOCK_2D[0]),
                                math.ceil(full_shape[1] / BLOCK_2D[1]))
                else:
                    block = BLOCK_3D
                    grid_dim = (math.ceil(full_shape[0] / BLOCK_3D[0]),
                                math.ceil(full_shape[1] / BLOCK_3D[1]),
                                math.ceil(full_shape[2] / BLOCK_3D[2]))

                iters_gpu = max(10, min(200, int(5e8 / max(total_pts, 1))))
                try:
                    ms_gpu = bench_gpu(gpu_kernel, u_dev, out_dev,
                                       grid_dim, block,
                                       iters=iters_gpu, warmup=20) * 1000
                except Exception as e:
                    ms_gpu = float('inf')
                    print(f"  GPU ERROR: {e}")

                # Metrics
                gpts_gpu = (total_pts / ms_gpu / 1e6) if ms_gpu > 0 else 0  # Gpts/s
                gbs_gpu = (total_pts * bytes_per_pt / ms_gpu / 1e6) if ms_gpu > 0 else 0
                sp_np = ms_np / ms_gpu if ms_gpu > 0 else 0
                sp_cpu = ms_cpu / ms_gpu if ms_gpu > 0 else 0

                print(f"  {grid_label:<16} {total_pts:<12,} {dtype_name:<7} "
                      f"{ms_np:<12.3f} {ms_cpu:<12.3f} {ms_gpu:<12.4f} "
                      f"{gbs_gpu:<10.1f} {sp_np:<10.1f}x {sp_cpu:<10.1f}x")

                # Track bests
                if dtype_name == "f32" and (best_gpu_f32 is None or gbs_gpu > best_gpu_f32[0]):
                    best_gpu_f32 = (gbs_gpu, gpts_gpu, total_pts, ms_gpu)
                if dtype_name == "f64" and (best_gpu_f64 is None or gbs_gpu > best_gpu_f64[0]):
                    best_gpu_f64 = (gbs_gpu, gpts_gpu, total_pts, ms_gpu)
                if best_cpu is None or (ms_cpu < float('inf') and total_pts / ms_cpu > (best_cpu[2] / best_cpu[1] if best_cpu[1] > 0 else 0)):
                    best_cpu = (0, ms_cpu, total_pts)
                if best_np is None or (ms_np < float('inf') and total_pts / ms_np > (best_np[2] / best_np[1] if best_np[1] > 0 else 0)):
                    best_np = (0, ms_np, total_pts)

                # Free GPU memory
                del u_dev, out_dev

        if best_gpu_f32 and best_gpu_f64:
            print(f"\n  Peak GPU bandwidth: f64={best_gpu_f64[0]:.1f} GB/s  "
                  f"f32={best_gpu_f32[0]:.1f} GB/s  "
                  f"(theoretical max: 1792 GB/s)")
            grand_summary.append((name, best_gpu_f64[0], best_gpu_f32[0]))

    # Grand summary
    header("GRAND SUMMARY — Measured GPU Bandwidth")
    print(f"\n  {'Stencil':<25} {'GPU f64 GB/s':<15} {'GPU f32 GB/s':<15} "
          f"{'% of peak f64':<15} {'% of peak f32':<15}")
    sep("-")
    for name, gbs_f64, gbs_f32 in grand_summary:
        pct_f64 = 100 * gbs_f64 / 1792
        pct_f32 = 100 * gbs_f32 / 1792
        print(f"  {name:<25} {gbs_f64:<15.1f} {gbs_f32:<15.1f} "
              f"{pct_f64:<15.1f}% {pct_f32:<15.1f}%")

    print(f"""
  Theoretical peak memory bandwidth: 1792 GB/s

  What the numbers mean:
  - "GPU GB/s" = actual measured memory throughput on your RTX PRO 6000
  - "% of peak" = how close to the 1792 GB/s theoretical max
  - Typical well-optimized stencil kernels achieve 60-80% of peak
  - These are naive global-memory kernels (no shared memory tiling)
  - With cuTile shared-memory tiling + temporal blocking, expect higher %
""")
    sep()


if __name__ == "__main__":
    run_all()
