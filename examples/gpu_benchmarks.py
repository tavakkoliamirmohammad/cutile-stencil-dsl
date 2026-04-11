"""GPU Performance Benchmarks: cuTile vs NumPy.

Compiles stencil kernels, runs on GPU with precise event timing,
compares against NumPy CPU baseline.

Usage:
    python examples/gpu_benchmarks.py
"""

import math
import time
import importlib.util
import tempfile
import numpy as np

from cutile_stencil import stencil, compile as stencil_compile
from cutile_stencil.reference.stencil_ref import apply_stencil


# --- Stencil definitions ---

@stencil(ndim=1, order=2)
def heat_1d(u, i):
    return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]


@stencil(ndim=2, order=2)
def laplacian_2d(u, i, j):
    return u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]


@stencil(ndim=3, order=2)
def laplacian_3d(u, i, j, k):
    return (u[i-1,j,k] + u[i+1,j,k] + u[i,j-1,k] + u[i,j+1,k]
            + u[i,j,k-1] + u[i,j,k+1] - 6*u[i,j,k])


BENCHMARKS = [
    ("heat_1d",      heat_1d,      (1024*1024,)),
    ("heat_1d_4M",   heat_1d,      (4*1024*1024,)),
    ("lap_2d_1k",    laplacian_2d, (1024, 1024)),
    ("lap_2d_2k",    laplacian_2d, (2048, 2048)),
    ("lap_3d_128",   laplacian_3d, (128, 128, 128)),
    ("lap_3d_256",   laplacian_3d, (256, 256, 256)),
]


def _load_kernel(result):
    """Load compiled kernel from CompileResult."""
    tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
    tmp.write(result.code)
    tmp.close()
    spec = importlib.util.spec_from_file_location("_k", tmp.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, f"launch_{result.spec.name}")


def benchmark_gpu(launch_fn, u_gpu, out_gpu, warmup=10, iterations=50):
    """Time GPU kernel using cupy.cuda.Event."""
    import cupy as cp

    stream = cp.cuda.get_current_stream()

    # Warmup
    for _ in range(warmup):
        launch_fn(u_gpu, out_gpu)
    stream.synchronize()

    # Timed
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record(stream)
    for _ in range(iterations):
        launch_fn(u_gpu, out_gpu)
    end.record(stream)
    end.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(start, end)
    return elapsed_ms / iterations  # avg ms per iteration


def benchmark_numpy(stencil_fn, u_np, iterations=10):
    """Time NumPy reference implementation."""
    spec = stencil_fn._stencil_spec
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = apply_stencil(u_np, spec)
    t1 = time.perf_counter()
    return (t1 - t0) / iterations * 1000  # ms


def compute_throughput(domain, elapsed_ms):
    """Compute throughput in Gpoints/s."""
    total_points = math.prod(domain)
    return total_points / (elapsed_ms / 1000) / 1e9


def main():
    import cupy as cp

    device = cp.cuda.Device(0)
    device.use()

    print("=" * 80)
    print("GPU Stencil Benchmarks: cuTile vs NumPy")
    print(f"Device: {device.id} (compute capability {device.compute_capability})")
    print("=" * 80)
    print()

    results = []

    for name, stencil_fn, domain in BENCHMARKS:
        spec = stencil_fn._stencil_spec
        halo = spec.order // 2

        # Compile
        result = stencil_compile(stencil_fn, domain)
        launch = _load_kernel(result)

        # Create arrays with halo
        padded_shape = tuple(d + 2 * halo for d in domain)
        np.random.seed(42)
        u_np = np.random.randn(*padded_shape).astype(np.float64)
        u_gpu = cp.asarray(u_np)
        out_gpu = cp.zeros_like(u_gpu)

        # GPU benchmark
        gpu_ms = benchmark_gpu(launch, u_gpu, out_gpu)
        gpu_gpts = compute_throughput(domain, gpu_ms)

        # NumPy benchmark (fewer iterations for large domains)
        np_iters = max(1, min(10, int(5000 / (math.prod(domain) / 1e6))))
        np_ms = benchmark_numpy(stencil_fn, u_np, iterations=np_iters)
        np_gpts = compute_throughput(domain, np_ms)

        speedup = np_ms / gpu_ms if gpu_ms > 0 else float('inf')

        results.append({
            "name": name,
            "domain": domain,
            "gpu_ms": gpu_ms,
            "gpu_gpts": gpu_gpts,
            "np_ms": np_ms,
            "np_gpts": np_gpts,
            "speedup": speedup,
            "roofline_gpts": result.analysis.roofline.peak_gpoints_s,
        })

        print(f"{name}: GPU={gpu_ms:.3f}ms ({gpu_gpts:.2f} Gpts/s) | "
              f"NumPy={np_ms:.1f}ms ({np_gpts:.4f} Gpts/s) | "
              f"Speedup={speedup:.1f}x")

    # Print summary table
    print()
    print("## Benchmark Summary")
    print()
    print("| Stencil | Domain | GPU (ms) | GPU (Gpts/s) | NumPy (ms) | Speedup | Roofline (Gpts/s) | Efficiency |")
    print("|---------|--------|----------|-------------|------------|---------|-------------------|------------|")
    for r in results:
        eff = r["gpu_gpts"] / r["roofline_gpts"] * 100 if r["roofline_gpts"] > 0 else 0
        print(f"| {r['name']} | {r['domain']} | {r['gpu_ms']:.3f} | "
              f"{r['gpu_gpts']:.2f} | {r['np_ms']:.1f} | "
              f"{r['speedup']:.1f}x | {r['roofline_gpts']:.2f} | {eff:.0f}% |")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
