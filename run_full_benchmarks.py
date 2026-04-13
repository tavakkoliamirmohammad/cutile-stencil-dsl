#!/usr/bin/env python3
"""Full benchmark suite: all stencils × all modes × many problem sizes.

Modes: 1-GPU flat, 1-GPU bricked, 2-GPU flat (step-only), JAX
Problem sizes: small → very large per dimensionality

Usage:
    module load cuda/13.2.0
    python run_full_benchmarks.py [--output results.json]
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

# JAX setup
for cuda_path in [
    os.environ.get("CUDA_HOME", ""),
    "/uufs/chpc.utah.edu/sys/spack/v10/linux-rocky8-x86_64/gcc-8.5.0/cuda-13.2.0-hc3gf4kzlz2qzpq6r5kekfbdxgjccyou",
    "/usr/local/cuda",
]:
    if cuda_path and os.path.exists(cuda_path):
        os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={cuda_path}"
        break

import jax
import jax.numpy as jnp
from cutile import stencil, compile

# ====================================================================
# GPU info
# ====================================================================

def gpu_info():
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode()
    n_gpus = cp.cuda.runtime.getDeviceCount()
    n = 256 * 1024 * 1024 // 8
    a = cp.random.randn(n).astype(cp.float64); b = cp.zeros_like(a)
    for _ in range(10): b[:] = a
    cp.cuda.Device(0).synchronize()
    e1, e2 = cp.cuda.Event(), cp.cuda.Event()
    e1.record()
    for _ in range(100): b[:] = a
    e2.record(); e2.synchronize()
    peak = 2 * n * 8 / (cp.cuda.get_elapsed_time(e1, e2) / 100 / 1000) / 1e9
    del a, b
    return name, n_gpus, peak


# ====================================================================
# Timing helpers
# ====================================================================

def bench_1gpu(launch, shape, warmup=30, iters=100):
    u = cp.random.randn(*shape).astype(cp.float64)
    o = cp.zeros_like(u)
    for _ in range(warmup): launch(u, o)
    cp.cuda.Device(0).synchronize()
    e1, e2 = cp.cuda.Event(), cp.cuda.Event()
    e1.record()
    for _ in range(iters): launch(u, o)
    e2.record(); e2.synchronize()
    return cp.cuda.get_elapsed_time(e1, e2) / iters


def bench_2gpu(step_fn, setup_fn, shape, n_gpus=2, warmup=20, iters=50):
    u = cp.random.randn(*shape).astype(cp.float64)
    p_in, p_out = setup_fn(u, num_gpus=n_gpus)
    for _ in range(warmup):
        step_fn(p_in, p_out, num_gpus=n_gpus)
        p_in, p_out = p_out, p_in
    for g in range(n_gpus): cp.cuda.Device(g).synchronize()
    s = time.perf_counter()
    for _ in range(iters):
        step_fn(p_in, p_out, num_gpus=n_gpus)
        p_in, p_out = p_out, p_in
    for g in range(n_gpus): cp.cuda.Device(g).synchronize()
    return (time.perf_counter() - s) / iters * 1000


def bench_bricked(launch_br, shape, warmup=30, iters=100):
    u = cp.random.randn(*shape).astype(cp.float64)
    o = cp.zeros_like(u)
    for _ in range(warmup): launch_br(u, o)
    cp.cuda.Device(0).synchronize()
    e1, e2 = cp.cuda.Event(), cp.cuda.Event()
    e1.record()
    for _ in range(iters): launch_br(u, o)
    e2.record(); e2.synchronize()
    return cp.cuda.get_elapsed_time(e1, e2) / iters


def bench_jax(jfn, shape, warmup=30, iters=100):
    u = jnp.array(np.random.randn(*shape).astype(np.float64))
    for _ in range(warmup): _ = jfn(u).block_until_ready()
    times = []
    for _ in range(iters):
        s = time.perf_counter()
        _ = jfn(u).block_until_ready()
        times.append(time.perf_counter() - s)
    return np.median(times) * 1000


# ====================================================================
# Stencil definitions
# ====================================================================

@stencil(ndim=1, order=2)
def heat_1d(u, i): return 0.25 * u[i-1] + 0.5 * u[i] + 0.25 * u[i+1]

@stencil(ndim=2, order=2)
def heat_2d(u, i, j): return 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])

@stencil(ndim=2, order=2)
def lap_2d(u, i, j): return u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4.0*u[i,j]

@stencil(ndim=2, order=4)
def wave_2d(u, i, j):
    return (-u[i-2,j]+16*u[i-1,j]-30*u[i,j]+16*u[i+1,j]-u[i+2,j]
            -u[i,j-2]+16*u[i,j-1]-30*u[i,j]+16*u[i,j+1]-u[i,j+2]) / 12.0

@stencil(ndim=3, order=2)
def lap_3d(u, i, j, k):
    return (u[i-1,j,k]+u[i+1,j,k]+u[i,j-1,k]+u[i,j+1,k]+u[i,j,k-1]+u[i,j,k+1]) / 6.0

@stencil(ndim=1, order=2)
def advect_1d(u, i): return u[i] - 0.5 * (u[i] - u[i-1])

@stencil(ndim=1, order=4)
def lap_1d_4th(u, i): return (-u[i-2]+16*u[i-1]-30*u[i]+16*u[i+1]-u[i+2]) / 12.0

# JAX equivalents
JAX = {
    "heat_1d":     jax.jit(lambda u: 0.25*u[:-2]+0.5*u[1:-1]+0.25*u[2:]),
    "heat_2d":     jax.jit(lambda u: 0.25*(u[:-2,1:-1]+u[2:,1:-1]+u[1:-1,:-2]+u[1:-1,2:])),
    "lap_2d":      jax.jit(lambda u: u[:-2,1:-1]+u[2:,1:-1]+u[1:-1,:-2]+u[1:-1,2:]-4.0*u[1:-1,1:-1]),
    "wave_2d":     jax.jit(lambda u: (-u[:-4,2:-2]+16*u[1:-3,2:-2]-30*u[2:-2,2:-2]+16*u[3:-1,2:-2]-u[4:,2:-2]-u[2:-2,:-4]+16*u[2:-2,1:-3]-30*u[2:-2,2:-2]+16*u[2:-2,3:-1]-u[2:-2,4:])/12.0),
    "lap_3d":      jax.jit(lambda u: (u[:-2,1:-1,1:-1]+u[2:,1:-1,1:-1]+u[1:-1,:-2,1:-1]+u[1:-1,2:,1:-1]+u[1:-1,1:-1,:-2]+u[1:-1,1:-1,2:])/6.0),
    "advect_1d":   jax.jit(lambda u: u[1:-1]-0.5*(u[1:-1]-u[:-2])),
    "lap_1d_4th":  jax.jit(lambda u: (-u[:-4]+16*u[1:-3]-30*u[2:-2]+16*u[3:-1]-u[4:])/12.0),
}

# Problem sizes per ndim
DOMAINS_1D = [(2**14,), (2**16,), (2**18,), (2**20,), (2**22,), (2**24,)]
DOMAINS_2D = [(256,256), (512,512), (1024,1024), (2048,2048), (4096,4096), (8192,8192)]
DOMAINS_3D = [(16,16,16), (32,32,32), (64,64,64), (128,128,128), (256,256,256)]

STENCILS = [
    ("heat_1d",    heat_1d,    1, DOMAINS_1D),
    ("heat_2d",    heat_2d,    2, DOMAINS_2D),
    ("lap_2d",     lap_2d,     2, DOMAINS_2D),
    ("wave_2d",    wave_2d,    2, DOMAINS_2D),
    ("lap_3d",     lap_3d,     3, DOMAINS_3D),
    ("advect_1d",  advect_1d,  1, DOMAINS_1D),
    ("lap_1d_4th", lap_1d_4th, 1, DOMAINS_1D),
]


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="full_benchmark_results.json")
    args = parser.parse_args()

    gpu_name, n_gpus, peak_bw = gpu_info()

    print("=" * 140)
    print(f"  FULL BENCHMARK: {gpu_name} × {n_gpus} | Peak BW: {peak_bw:.0f} GB/s")
    print("=" * 140)
    print()

    hdr = (f"{'Stencil':>12s} | {'Domain':>12s} | {'Points':>10s} | "
           f"{'1-GPU GP/s':>10s} | {'Bricked':>10s} | {'2-GPU GP/s':>10s} | {'Scaling':>7s} | "
           f"{'JAX GP/s':>10s} | {'vs JAX':>10s} | {'Tiles':>12s}")
    print(hdr)
    print("=" * len(hdr))

    all_results = []

    for name, sfn, ndim, domains in STENCILS:
        halo_val = sfn._order // 2
        halo = tuple(halo_val for _ in range(ndim))
        jfn = JAX.get(name)

        for domain in domains:
            npoints = math.prod(domain)
            shape = tuple(d + 2*h for d, h in zip(domain, halo))
            dom_str = "x".join(str(d) for d in domain)
            pts_str = f"{npoints/1e6:.1f}M" if npoints >= 1e6 else f"{npoints/1e3:.0f}K"

            row = {"stencil": name, "domain": list(domain), "npoints": npoints}

            # Skip very large domains that would OOM
            data_bytes = math.prod(shape) * 8
            if data_bytes > 8e9:  # 8 GB limit per array
                print(f"{name:>12s} | {dom_str:>12s} | {pts_str:>10s} | {'SKIP (OOM)':>10s} |")
                continue

            # --- 1-GPU flat (autotuned) ---
            try:
                r1 = compile(sfn, domain=domain, temporal_blocking=False)
                m1 = r1.load_module()
                l1 = getattr(m1, f"launch_{r1.name}")
                ms1 = bench_1gpu(l1, shape)
                gp1 = npoints / (ms1/1000) / 1e9
                tiles = "x".join(str(t) for t in r1.tile_sizes)
                row["gpu1_gps"] = round(gp1, 2)
                row["gpu1_ms"] = round(ms1, 4)
                row["tiles"] = list(r1.tile_sizes)
            except Exception as e:
                gp1 = 0; tiles = "ERR"
                row["gpu1_err"] = str(e)[:50]

            # --- 1-GPU bricked ---
            try:
                rb = compile(sfn, domain=domain, layout="bricked", temporal_blocking=False)
                mb = rb.load_module()
                lb = getattr(mb, f"launch_{rb.name}_bricked")
                msb = bench_bricked(lb, shape)
                gpb = npoints / (msb/1000) / 1e9
                row["bricked_gps"] = round(gpb, 2)
            except Exception as e:
                gpb = 0
                row["bricked_err"] = str(e)[:50]

            # --- 2-GPU (step only) ---
            gp2 = 0; scaling_str = "N/A"
            if n_gpus >= 2:
                try:
                    r2 = compile(sfn, num_gpus=2, domain=domain, temporal_blocking=False)
                    m2 = r2.load_module()
                    setup = getattr(m2, f"setup_multigpu_{r2.name}")
                    step = getattr(m2, f"step_multigpu_{r2.name}")
                    ms2 = bench_2gpu(step, setup, shape)
                    gp2 = npoints / (ms2/1000) / 1e9
                    scaling_str = f"{gp2/gp1:.2f}x" if gp1 > 0 else "N/A"
                    row["gpu2_gps"] = round(gp2, 2)
                    row["gpu2_ms"] = round(ms2, 4)
                    row["scaling"] = round(gp2/gp1, 2) if gp1 > 0 else 0
                except Exception as e:
                    scaling_str = "ERR"
                    row["gpu2_err"] = str(e)[:50]

            # --- JAX ---
            jgp = 0
            if jfn is not None:
                try:
                    jms = bench_jax(jfn, shape)
                    jgp = npoints / (jms/1000) / 1e9
                    row["jax_gps"] = round(jgp, 2)
                    row["jax_ms"] = round(jms, 4)
                except Exception as e:
                    row["jax_err"] = str(e)[:50]

            # vs JAX
            if gp1 > 0 and jgp > 0:
                ratio = gp1 / jgp
                vs = f"{ratio:.2f}x" if ratio >= 1 else f"{1/ratio:.2f}x JAX"
            else:
                vs = "N/A"

            gp1_s = f"{gp1:.1f}" if gp1 > 0 else "ERR"
            gpb_s = f"{gpb:.1f}" if gpb > 0 else "ERR"
            gp2_s = f"{gp2:.1f}" if gp2 > 0 else "N/A"
            jgp_s = f"{jgp:.1f}" if jgp > 0 else "N/A"

            print(f"{name:>12s} | {dom_str:>12s} | {pts_str:>10s} | "
                  f"{gp1_s:>10s} | {gpb_s:>10s} | {gp2_s:>10s} | {scaling_str:>7s} | "
                  f"{jgp_s:>10s} | {vs:>10s} | {tiles:>12s}")

            all_results.append(row)
        print("-" * len(hdr))

    # Save
    output = {
        "gpu": gpu_name,
        "n_gpus": n_gpus,
        "peak_bw_gbs": round(peak_bw, 1),
        "results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
