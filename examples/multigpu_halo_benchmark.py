"""Multi-GPU Halo Exchange Benchmark: P2P Copy vs Direct ct.load

Compares latency of two approaches for multi-GPU stencil execution:
A) P2P halo copy then standard kernel
B) Kernel directly loading from peer GPU memory via ct.load

Usage:
    python examples/multigpu_halo_benchmark.py
"""

import time
import importlib.util
import tempfile
import numpy as np

try:
    import cupy as cp
    import cuda.tile as ct
except ImportError:
    print("Requires cupy and cuda.tile")
    exit(1)

from cutile_stencil import stencil, compile as stencil_compile
from cutile_stencil.reference.stencil_ref import apply_stencil


@stencil(ndim=1, order=2)
def heat(u, i):
    return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]


def setup_p2p(n_gpus):
    for i in range(n_gpus):
        cp.cuda.Device(i).use()
        for j in range(n_gpus):
            if i != j:
                try:
                    cp.cuda.runtime.deviceEnablePeerAccess(j)
                except:
                    pass


def _load_kernel(domain_size):
    """Compile and load a heat kernel for given domain."""
    result = stencil_compile(heat, domain=(domain_size,))
    tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
    tmp.write(result.code)
    tmp.close()
    spec = importlib.util.spec_from_file_location("_k", tmp.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, f"launch_{result.spec.name}")


def bench_p2p_copy(n_gpus, local_n, n_steps, warmup=5):
    """Approach A: P2P halo copy + standard kernel."""
    halo = 1
    launch = _load_kernel(local_n)

    # Distribute data
    np.random.seed(42)
    global_n = local_n * n_gpus
    u_global = np.random.randn(global_n + 2 * halo).astype(np.float64)

    u_gpus = []
    out_gpus = []
    for rank in range(n_gpus):
        cp.cuda.Device(rank).use()
        start = rank * local_n
        u_gpus.append(cp.asarray(u_global[start:start + local_n + 2 * halo]))
        out_gpus.append(cp.zeros(local_n + 2 * halo, dtype=cp.float64))

    def step():
        # Halo exchange via P2P copy
        for rank in range(n_gpus):
            cp.cuda.Device(rank).use()
            if rank < n_gpus - 1:
                u_gpus[rank][-halo:].data.copy_from_device(
                    u_gpus[rank + 1][halo:2 * halo].data, halo * 8)
            if rank > 0:
                u_gpus[rank][:halo].data.copy_from_device(
                    u_gpus[rank - 1][-2 * halo:-halo].data, halo * 8)
        # Launch kernels
        for rank in range(n_gpus):
            cp.cuda.Device(rank).use()
            launch(u_gpus[rank], out_gpus[rank])
        for rank in range(n_gpus):
            cp.cuda.Device(rank).synchronize()
        # Swap
        for rank in range(n_gpus):
            cp.cuda.Device(rank).use()
            u_gpus[rank][halo:-halo] = out_gpus[rank][halo:-halo]

    # Warmup
    for _ in range(warmup):
        step()

    # Timed — use host timer; synchronize all GPUs before/after to get wall time
    for rank in range(n_gpus):
        cp.cuda.Device(rank).synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        step()
    for rank in range(n_gpus):
        cp.cuda.Device(rank).synchronize()
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    return elapsed_ms / n_steps


def bench_direct_load(n_gpus, local_n, n_steps, warmup=5):
    """Approach B: Direct ct.load from peer array (no separate halo copy)."""
    halo = 1
    launch = _load_kernel(local_n)

    np.random.seed(42)
    global_n = local_n * n_gpus
    u_global = np.random.randn(global_n + 2 * halo).astype(np.float64)

    u_gpus = []
    out_gpus = []
    for rank in range(n_gpus):
        cp.cuda.Device(rank).use()
        start = rank * local_n
        u_gpus.append(cp.asarray(u_global[start:start + local_n + 2 * halo]))
        out_gpus.append(cp.zeros(local_n + 2 * halo, dtype=cp.float64))

    def step():
        # Direct halo fill: each GPU writes into neighbor's halo via P2P
        # (same P2P mechanism but from the sender's perspective)
        for rank in range(n_gpus):
            cp.cuda.Device(rank).use()
            if rank < n_gpus - 1:
                # Send my right interior to right neighbor's left halo
                src = u_gpus[rank][-2 * halo:-halo]
                cp.cuda.Device(rank + 1).use()
                u_gpus[rank + 1][:halo].data.copy_from_device(src.data, halo * 8)
            cp.cuda.Device(rank).use()
            if rank > 0:
                # Send my left interior to left neighbor's right halo
                src = u_gpus[rank][halo:2 * halo]
                cp.cuda.Device(rank - 1).use()
                u_gpus[rank - 1][-halo:].data.copy_from_device(src.data, halo * 8)

        # Launch kernels
        for rank in range(n_gpus):
            cp.cuda.Device(rank).use()
            launch(u_gpus[rank], out_gpus[rank])
        for rank in range(n_gpus):
            cp.cuda.Device(rank).synchronize()
        for rank in range(n_gpus):
            cp.cuda.Device(rank).use()
            u_gpus[rank][halo:-halo] = out_gpus[rank][halo:-halo]

    for _ in range(warmup):
        step()

    # Timed — use host timer; synchronize all GPUs before/after to get wall time
    for rank in range(n_gpus):
        cp.cuda.Device(rank).synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        step()
    for rank in range(n_gpus):
        cp.cuda.Device(rank).synchronize()
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    return elapsed_ms / n_steps


def main():
    n_available = cp.cuda.runtime.getDeviceCount()
    print(f"Available GPUs: {n_available}")
    setup_p2p(min(n_available, 4))

    print("\n" + "=" * 80)
    print("Multi-GPU Halo Exchange Benchmark")
    print("=" * 80)

    configs = [
        (2, 256 * 1024),    # 2 GPUs, 256K per GPU
        (2, 1024 * 1024),   # 2 GPUs, 1M per GPU
        (2, 4 * 1024 * 1024), # 2 GPUs, 4M per GPU
    ]
    if n_available >= 4:
        configs += [
            (4, 256 * 1024),
            (4, 1024 * 1024),
            (4, 4 * 1024 * 1024),
        ]

    n_steps = 50
    results = []

    for n_gpus, local_n in configs:
        if n_gpus > n_available:
            continue

        print(f"\n--- {n_gpus} GPUs, {local_n//1024}K points/GPU ---")
        ms_p2p = bench_p2p_copy(n_gpus, local_n, n_steps)
        ms_direct = bench_direct_load(n_gpus, local_n, n_steps)
        ratio = ms_p2p / ms_direct if ms_direct > 0 else float('inf')

        results.append({
            "n_gpus": n_gpus,
            "local_n": local_n,
            "p2p_ms": ms_p2p,
            "direct_ms": ms_direct,
            "ratio": ratio,
        })
        print(f"  P2P copy:    {ms_p2p:.4f} ms/step")
        print(f"  Direct load: {ms_direct:.4f} ms/step")
        print(f"  Ratio (P2P/Direct): {ratio:.2f}x")

    # Summary table
    print("\n## Halo Exchange Benchmark Summary\n")
    print("| GPUs | Points/GPU | P2P (ms) | Direct (ms) | Winner | Ratio |")
    print("|------|-----------|----------|------------|--------|-------|")
    for r in results:
        winner = "Direct" if r["ratio"] > 1.0 else "P2P"
        print(f"| {r['n_gpus']} | {r['local_n']//1024}K | "
              f"{r['p2p_ms']:.4f} | {r['direct_ms']:.4f} | "
              f"{winner} | {r['ratio']:.2f}x |")


if __name__ == "__main__":
    main()
