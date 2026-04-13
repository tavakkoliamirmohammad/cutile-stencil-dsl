#!/usr/bin/env python3
"""Comprehensive stencil benchmark: cuTile-DSL vs JAX/XLA.

Run on any machine with a GPU:

    module load cuda/13.2.0       # or appropriate CUDA version
    python run_benchmarks.py                  # cuTile only
    python run_benchmarks.py --jax            # + JAX comparison
    python run_benchmarks.py --autotune       # autotune tile sizes
    python run_benchmarks.py --autotune --jax # full comparison

Options:
    --autotune    Empirically autotune tile sizes on GPU (cached after first run)
    --jax         Include JAX/XLA comparison
    --verbose     Print autotuning progress
    --output FILE JSON output file (default: benchmark_results.json)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cupy as cp
import numpy as np

from cutile import stencil, compile


# ====================================================================
# GPU info
# ====================================================================

def gpu_info():
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode()
    n_gpus = cp.cuda.runtime.getDeviceCount()

    # Measure peak DRAM bandwidth (STREAM copy)
    n = 256 * 1024 * 1024 // 8
    a = cp.random.randn(n).astype(cp.float64)
    b = cp.zeros_like(a)
    for _ in range(10):
        b[:] = a
    cp.cuda.Device(0).synchronize()
    e1, e2 = cp.cuda.Event(), cp.cuda.Event()
    e1.record()
    for _ in range(100):
        b[:] = a
    e2.record()
    e2.synchronize()
    peak_bw = 2 * n * 8 / (cp.cuda.get_elapsed_time(e1, e2) / 100 / 1000) / 1e9
    del a, b

    return name, n_gpus, peak_bw


# ====================================================================
# Benchmark helpers
# ====================================================================

def bench_cutile(launch_fn, shape, warmup=50, iters=200):
    """Benchmark a cuTile kernel using CUDA events. Returns time in ms."""
    u = cp.random.randn(*shape).astype(cp.float64)
    out = cp.zeros_like(u)
    for _ in range(warmup):
        launch_fn(u, out)
    cp.cuda.Device(0).synchronize()
    e1, e2 = cp.cuda.Event(), cp.cuda.Event()
    e1.record()
    for _ in range(iters):
        launch_fn(u, out)
    e2.record()
    e2.synchronize()
    return cp.cuda.get_elapsed_time(e1, e2) / iters


def bench_jax_fn(jax_fn, shape, warmup=50, iters=200):
    """Benchmark a JAX function. Returns time in ms."""
    import jax.numpy as jnp
    u = jnp.array(np.random.randn(*shape).astype(np.float64))
    for _ in range(warmup):
        _ = jax_fn(u).block_until_ready()
    times = []
    for _ in range(iters):
        s = time.perf_counter()
        _ = jax_fn(u).block_until_ready()
        times.append(time.perf_counter() - s)
    return np.median(times) * 1000


# ====================================================================
# Stencil definitions
# ====================================================================

@stencil(ndim=1, order=2)
def heat_1d(u, i):
    return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

@stencil(ndim=2, order=2)
def heat_2d(u, i, j):
    return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

@stencil(ndim=2, order=2)
def lap_2d(u, i, j):
    return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4.0 * u[i, j]

@stencil(ndim=2, order=4)
def wave_2d(u, i, j):
    return (-u[i-2,j] + 16*u[i-1,j] - 30*u[i,j] + 16*u[i+1,j] - u[i+2,j]
            - u[i,j-2] + 16*u[i,j-1] - 30*u[i,j] + 16*u[i,j+1] - u[i,j+2]) / 12.0

@stencil(ndim=3, order=2)
def lap_3d(u, i, j, k):
    return (u[i-1,j,k] + u[i+1,j,k] + u[i,j-1,k] + u[i,j+1,k]
            + u[i,j,k-1] + u[i,j,k+1]) / 6.0

@stencil(ndim=1, order=2)
def advect_1d(u, i):
    return u[i] - 0.5 * (u[i] - u[i - 1])

@stencil(ndim=1, order=4)
def lap_1d_4th(u, i):
    return (-u[i-2] + 16*u[i-1] - 30*u[i] + 16*u[i+1] - u[i+2]) / 12.0


STENCILS = {
    "heat_1d":     (heat_1d,    1, [(2**16,), (2**18,), (2**20,), (2**22,)]),
    "heat_2d":     (heat_2d,    2, [(512,512), (1024,1024), (2048,2048), (4096,4096)]),
    "lap_2d_5pt":  (lap_2d,     2, [(512,512), (1024,1024), (2048,2048), (4096,4096)]),
    "lap_1d_4th":  (lap_1d_4th, 1, [(2**16,), (2**18,), (2**20,), (2**22,)]),
    "wave_2d_4th": (wave_2d,    2, [(512,512), (1024,1024), (2048,2048), (4096,4096)]),
    "lap_3d_7pt":  (lap_3d,     3, [(32,32,32), (64,64,64), (128,128,128)]),
    "advect_1d":   (advect_1d,  1, [(2**16,), (2**18,), (2**20,), (2**22,)]),
}


def _init_jax():
    """Initialize JAX functions."""
    import jax

    # Auto-detect CUDA path
    for path in [
        os.environ.get("CUDA_HOME", ""),
        "/uufs/chpc.utah.edu/sys/spack/v10/linux-rocky8-x86_64/gcc-8.5.0/cuda-13.2.0-hc3gf4kzlz2qzpq6r5kekfbdxgjccyou",
        "/usr/local/cuda",
        "/usr/local/cuda-12",
    ]:
        if path and os.path.exists(path):
            os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={path}"
            break

    return {
        "heat_1d":     jax.jit(lambda u: 0.25*u[:-2] + 0.5*u[1:-1] + 0.25*u[2:]),
        "heat_2d":     jax.jit(lambda u: 0.25*(u[:-2,1:-1]+u[2:,1:-1]+u[1:-1,:-2]+u[1:-1,2:])),
        "lap_2d_5pt":  jax.jit(lambda u: u[:-2,1:-1]+u[2:,1:-1]+u[1:-1,:-2]+u[1:-1,2:]-4.0*u[1:-1,1:-1]),
        "lap_1d_4th":  jax.jit(lambda u: (-u[:-4]+16*u[1:-3]-30*u[2:-2]+16*u[3:-1]-u[4:])/12.0),
        "wave_2d_4th": jax.jit(lambda u: (-u[:-4,2:-2]+16*u[1:-3,2:-2]-30*u[2:-2,2:-2]+16*u[3:-1,2:-2]-u[4:,2:-2]-u[2:-2,:-4]+16*u[2:-2,1:-3]-30*u[2:-2,2:-2]+16*u[2:-2,3:-1]-u[2:-2,4:])/12.0),
        "lap_3d_7pt":  jax.jit(lambda u: (u[:-2,1:-1,1:-1]+u[2:,1:-1,1:-1]+u[1:-1,:-2,1:-1]+u[1:-1,2:,1:-1]+u[1:-1,1:-1,:-2]+u[1:-1,1:-1,2:])/6.0),
        "advect_1d":   jax.jit(lambda u: u[1:-1]-0.5*(u[1:-1]-u[:-2])),
    }


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="cuTile-DSL stencil benchmarks")
    parser.add_argument("--autotune", action="store_true", help="Autotune tile sizes on GPU")
    parser.add_argument("--jax", action="store_true", help="Include JAX/XLA comparison")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default="benchmark_results.json", help="JSON output")
    args = parser.parse_args()

    gpu_name, n_gpus, peak_bw = gpu_info()

    print("=" * 110)
    print(f"  cuTile-DSL Stencil Benchmark Suite")
    print(f"  GPU: {gpu_name} ({n_gpus} available)")
    print(f"  Peak DRAM BW: {peak_bw:.0f} GB/s")
    print(f"  Autotuning: {'ON' if args.autotune else 'OFF'}")
    print(f"  JAX comparison: {'ON' if args.jax else 'OFF'}")
    print("=" * 110)
    print()

    jax_fns = _init_jax() if args.jax else {}

    all_results = []

    # Header
    hdr = f"{'Stencil':>15s} | {'Domain':>15s} | {'Tiles':>12s} | {'cuTile GP/s':>12s} | {'cuTile ms':>10s} | {'Eff BW%':>7s}"
    if args.jax:
        hdr += f" | {'JAX GP/s':>10s} | {'JAX ms':>8s} | {'vs JAX':>10s}"
    print(hdr)
    print("=" * len(hdr))

    for name, (sfn, ndim, domains) in STENCILS.items():
        halo_val = sfn._order // 2
        halo = tuple(halo_val for _ in range(ndim))

        for domain in domains:
            npoints = math.prod(domain)
            shape = tuple(d + 2 * h for d, h in zip(domain, halo))

            # Compile (with optional autotuning)
            r = compile(sfn, autotune=args.autotune, domain=domain, temporal_blocking=False)
            m = r.load_module()
            l = getattr(m, f"launch_{r.name}")

            ct_ms = bench_cutile(l, shape)
            ct_gp = npoints / (ct_ms / 1000) / 1e9
            ct_bw = npoints * 8 * 2 / (ct_ms / 1000) / 1e9
            bw_pct = ct_bw / peak_bw * 100
            tile_str = "x".join(str(t) for t in r.tile_sizes)

            row = {"stencil": name, "domain": list(domain), "tiles": list(r.tile_sizes),
                   "cutile_gps": round(ct_gp, 2), "cutile_ms": round(ct_ms, 4),
                   "eff_bw_pct": round(bw_pct, 1)}

            line = (f"{name:>15s} | {'x'.join(str(d) for d in domain):>15s} | "
                    f"{tile_str:>12s} | {ct_gp:12.1f} | {ct_ms:10.4f} | {bw_pct:6.1f}%")

            if args.jax and name in jax_fns:
                jms = bench_jax_fn(jax_fns[name], shape)
                jgp = npoints / (jms / 1000) / 1e9
                ratio = ct_gp / jgp
                label = f"{ratio:.2f}x" if ratio >= 1 else f"{1/ratio:.2f}x JAX"
                line += f" | {jgp:10.1f} | {jms:8.4f} | {label:>10s}"
                row["jax_gps"] = round(jgp, 2)
                row["jax_ms"] = round(jms, 4)

            print(line)
            all_results.append(row)
        print("-" * len(hdr))

    # Save
    output = {
        "gpu": gpu_name,
        "n_gpus": n_gpus,
        "peak_bw_gbs": round(peak_bw, 1),
        "autotuned": args.autotune,
        "results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
