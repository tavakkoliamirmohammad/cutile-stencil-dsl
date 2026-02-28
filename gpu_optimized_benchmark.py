#!/usr/bin/env python3
"""Compare NAIVE vs DSL-OPTIMIZED GPU kernels on your RTX PRO 6000.

Shows the actual performance difference from the DSL's optimizations:
  1. Naive global-memory kernel (no optimization)
  2. Shared-memory tiled kernel (what the DSL generates)
  3. Temporal-blocked kernel (DSL's main optimization — fuse T timesteps)
"""

import time
import math
import numpy as np
from numba import cuda, float64

# ============================================================================
# 1) NAIVE kernels — just global memory, no optimization
# ============================================================================

@cuda.jit
def heat_1d_naive(u, out):
    i = cuda.grid(1)
    if i >= 1 and i < u.shape[0] - 1:
        out[i] = 0.25 * u[i-1] + 0.5 * u[i] + 0.25 * u[i+1]

@cuda.jit
def lap_2d_naive(u, out):
    i, j = cuda.grid(2)
    if i >= 1 and i < u.shape[0]-1 and j >= 1 and j < u.shape[1]-1:
        out[i,j] = u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4.0*u[i,j]

@cuda.jit
def lap_3d_naive(u, out):
    i, j, k = cuda.grid(3)
    if (i >= 1 and i < u.shape[0]-1 and
        j >= 1 and j < u.shape[1]-1 and
        k >= 1 and k < u.shape[2]-1):
        out[i,j,k] = (u[i-1,j,k] + u[i+1,j,k] + u[i,j-1,k] + u[i,j+1,k]
                       + u[i,j,k-1] + u[i,j,k+1] - 6.0*u[i,j,k])


# ============================================================================
# 2) SHARED-MEMORY TILED kernels
#    Numba requires literal ints for cuda.shared.array shape
# ============================================================================

# 1D tiled: tile=256, halo=1, shared=258
@cuda.jit
def heat_1d_tiled(u, out):
    smem = cuda.shared.array(258, dtype=float64)  # 256 + 2*1
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    gx = bx * 256 + tx
    N = u.shape[0]

    # Load interior element
    if gx < N:
        smem[tx + 1] = u[gx]
    # Left halo
    if tx == 0:
        gi = bx * 256 - 1
        smem[0] = u[gi] if gi >= 0 else 0.0
    # Right halo
    if tx == 255:
        gi = bx * 256 + 256
        smem[257] = u[gi] if gi < N else 0.0

    cuda.syncthreads()

    si = tx + 1
    if gx >= 1 and gx < N - 1:
        out[gx] = 0.25 * smem[si-1] + 0.5 * smem[si] + 0.25 * smem[si+1]


# 2D tiled: tile=32x32, halo=1, shared=34x34
@cuda.jit
def lap_2d_tiled(u, out):
    smem = cuda.shared.array((34, 34), dtype=float64)  # 32+2, 32+2

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    gx = cuda.blockIdx.x * 32 + tx
    gy = cuda.blockIdx.y * 32 + ty
    Nx, Ny = u.shape[0], u.shape[1]

    sx = tx + 1
    sy = ty + 1

    # Load interior
    if gx < Nx and gy < Ny:
        smem[sx, sy] = u[gx, gy]
    else:
        smem[sx, sy] = 0.0

    # Load halos — border threads handle them
    if tx == 0:
        hgx = gx - 1
        smem[0, sy] = u[hgx, gy] if hgx >= 0 and gy < Ny else 0.0
    if tx == 31:
        hgx = gx + 1
        smem[33, sy] = u[hgx, gy] if hgx < Nx and gy < Ny else 0.0
    if ty == 0:
        hgy = gy - 1
        smem[sx, 0] = u[gx, hgy] if gx < Nx and hgy >= 0 else 0.0
    if ty == 31:
        hgy = gy + 1
        smem[sx, 33] = u[gx, hgy] if gx < Nx and hgy < Ny else 0.0

    cuda.syncthreads()

    if gx >= 1 and gx < Nx-1 and gy >= 1 and gy < Ny-1:
        out[gx, gy] = (smem[sx-1,sy] + smem[sx+1,sy] +
                        smem[sx,sy-1] + smem[sx,sy+1] - 4.0*smem[sx,sy])


# ============================================================================
# 3) TEMPORAL BLOCKED kernels — fuse multiple timesteps in shared memory
# ============================================================================

# 1D temporal: tile=128, halo=1, T=4 steps, expanded=128+2*4=136
@cuda.jit
def heat_1d_temporal(u, out):
    """Load expanded tile once, apply stencil 4 times in shared mem, write once."""
    smem_a = cuda.shared.array(136, dtype=float64)   # 128 + 2*4*1
    smem_b = cuda.shared.array(136, dtype=float64)

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    N = u.shape[0]
    base = bx * 128 - 4  # start of expanded tile in global memory

    # Cooperatively load expanded tile (136 elements, 128 threads)
    if tx < 136:
        gi = base + tx
        smem_a[tx] = u[gi] if 0 <= gi < N else 0.0

    cuda.syncthreads()

    # Step 1: read from smem_a (size 136), write to smem_b (size 134)
    inner = 134  # 136 - 2
    if tx < inner:
        si = tx + 1
        smem_b[tx] = 0.25 * smem_a[si-1] + 0.5 * smem_a[si] + 0.25 * smem_a[si+1]
    cuda.syncthreads()

    # Step 2: read from smem_b (size 134), write to smem_a (size 132)
    inner = 132  # 134 - 2
    if tx < inner:
        si = tx + 1
        smem_a[tx] = 0.25 * smem_b[si-1] + 0.5 * smem_b[si] + 0.25 * smem_b[si+1]
    cuda.syncthreads()

    # Step 3: read from smem_a (size 132), write to smem_b (size 130)
    inner = 130
    if tx < inner:
        si = tx + 1
        smem_b[tx] = 0.25 * smem_a[si-1] + 0.5 * smem_a[si] + 0.25 * smem_a[si+1]
    cuda.syncthreads()

    # Step 4: read from smem_b (size 130), write to smem_a (size 128)
    inner = 128
    if tx < inner:
        si = tx + 1
        smem_a[tx] = 0.25 * smem_b[si-1] + 0.5 * smem_b[si] + 0.25 * smem_b[si+1]
    cuda.syncthreads()

    # Write final result (128 interior points)
    gx = bx * 128 + tx
    if tx < 128 and gx >= 4 and gx < N - 4:
        out[gx] = smem_a[tx]


# 2D temporal: tile=16x16, halo=1, T=2, expanded=20x20
@cuda.jit
def lap_2d_temporal(u, out):
    """Load 20x20 tile once, apply laplacian twice in shared mem, write 16x16."""
    smem_a = cuda.shared.array((20, 20), dtype=float64)
    smem_b = cuda.shared.array((20, 20), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    Nx, Ny = u.shape[0], u.shape[1]

    base_x = bx * 16 - 2
    base_y = by * 16 - 2

    # Load 20x20 expanded tile (16x16 threads load multiple elements)
    for lx in range(tx, 20, 16):
        for ly in range(ty, 20, 16):
            gx = base_x + lx
            gy = base_y + ly
            if 0 <= gx < Nx and 0 <= gy < Ny:
                smem_a[lx, ly] = u[gx, gy]
            else:
                smem_a[lx, ly] = 0.0
    cuda.syncthreads()

    # Step 1: 20x20 → 18x18
    if tx < 18 and ty < 18:
        sx = tx + 1
        sy = ty + 1
        smem_b[tx, ty] = (smem_a[sx-1,sy] + smem_a[sx+1,sy] +
                          smem_a[sx,sy-1] + smem_a[sx,sy+1] - 4.0*smem_a[sx,sy])
    cuda.syncthreads()

    # Step 2: 18x18 → 16x16
    if tx < 16 and ty < 16:
        sx = tx + 1
        sy = ty + 1
        smem_a[tx, ty] = (smem_b[sx-1,sy] + smem_b[sx+1,sy] +
                          smem_b[sx,sy-1] + smem_b[sx,sy+1] - 4.0*smem_b[sx,sy])
    cuda.syncthreads()

    # Write 16x16 result
    gx = bx * 16 + tx
    gy = by * 16 + ty
    if tx < 16 and ty < 16 and gx >= 2 and gx < Nx-2 and gy >= 2 and gy < Ny-2:
        out[gx, gy] = smem_a[tx, ty]


# 2D temporal T=4: tile=16x16, expanded=24x24
@cuda.jit
def lap_2d_temporal_t4(u, out):
    """Load 24x24 tile, apply laplacian 4 times, write 16x16."""
    smem_a = cuda.shared.array((24, 24), dtype=float64)
    smem_b = cuda.shared.array((24, 24), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    Nx, Ny = u.shape[0], u.shape[1]

    base_x = bx * 16 - 4
    base_y = by * 16 - 4

    # Load 24x24 expanded tile
    for lx in range(tx, 24, 16):
        for ly in range(ty, 24, 16):
            gx = base_x + lx
            gy = base_y + ly
            if 0 <= gx < Nx and 0 <= gy < Ny:
                smem_a[lx, ly] = u[gx, gy]
            else:
                smem_a[lx, ly] = 0.0
    cuda.syncthreads()

    # Step 1: 24→22
    inner = 22
    if tx < inner and ty < inner:
        sx, sy = tx + 1, ty + 1
        smem_b[tx,ty] = smem_a[sx-1,sy]+smem_a[sx+1,sy]+smem_a[sx,sy-1]+smem_a[sx,sy+1]-4.0*smem_a[sx,sy]
    cuda.syncthreads()
    # Step 2: 22→20
    inner = 20
    if tx < inner and ty < inner:
        sx, sy = tx + 1, ty + 1
        smem_a[tx,ty] = smem_b[sx-1,sy]+smem_b[sx+1,sy]+smem_b[sx,sy-1]+smem_b[sx,sy+1]-4.0*smem_b[sx,sy]
    cuda.syncthreads()
    # Step 3: 20→18
    inner = 18
    if tx < inner and ty < inner:
        sx, sy = tx + 1, ty + 1
        smem_b[tx,ty] = smem_a[sx-1,sy]+smem_a[sx+1,sy]+smem_a[sx,sy-1]+smem_a[sx,sy+1]-4.0*smem_a[sx,sy]
    cuda.syncthreads()
    # Step 4: 18→16
    inner = 16
    if tx < inner and ty < inner:
        sx, sy = tx + 1, ty + 1
        smem_a[tx,ty] = smem_b[sx-1,sy]+smem_b[sx+1,sy]+smem_b[sx,sy-1]+smem_b[sx,sy+1]-4.0*smem_b[sx,sy]
    cuda.syncthreads()

    gx = bx * 16 + tx
    gy = by * 16 + ty
    if tx < 16 and ty < 16 and gx >= 4 and gx < Nx-4 and gy >= 4 and gy < Ny-4:
        out[gx, gy] = smem_a[tx, ty]


# ============================================================================
# Benchmark harness
# ============================================================================

def bench_gpu(kernel, u_dev, out_dev, grid, block, iters=100, warmup=20):
    for _ in range(warmup):
        kernel[grid, block](u_dev, out_dev)
    cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        kernel[grid, block](u_dev, out_dev)
    cuda.synchronize()
    return (time.perf_counter() - start) / iters

def sep(c="=", w=100):
    print(c * w)

def header(title):
    print()
    sep()
    print(f"  {title}")
    sep()


def run_all():
    header("NAIVE vs TILED vs TEMPORAL — Your RTX PRO 6000 Blackwell")
    print("  1. NAIVE      — Global memory reads, no data reuse")
    print("  2. TILED      — Shared memory tiling (DSL's ct.load)")
    print("  3. TEMPORAL   — Fused multi-step in shared mem (DSL's main win)")
    sep()

    # ==================================================================
    # 1D HEAT
    # ==================================================================
    header("1D Heat (3-point, halo=1)")
    print(f"\n  {'Grid':<14} {'Naive':<12} {'Tiled':<12} {'Temp(T=4)':<12} "
          f"{'Tiled/Naiv':<12} {'Temp/Naive':<12} {'Temp GB/s':<12}")
    print(f"  {'':<14} {'us':<12} {'us':<12} {'us/step':<12} "
          f"{'speedup':<12} {'speedup':<12} {'':<12}")
    sep("-")

    for N_int in [65536, 262144, 1048576, 4194304, 16777216]:
        N = N_int + 2
        u = np.random.randn(N)
        u_dev = cuda.to_device(u)
        o1 = cuda.device_array_like(u)
        o2 = cuda.device_array_like(u)
        o3 = cuda.device_array_like(u)

        iters = max(20, min(500, int(5e8 / N_int)))

        # Naive
        g_naive = (math.ceil(N / 256),)
        t_naive = bench_gpu(heat_1d_naive, u_dev, o1, g_naive, (256,), iters=iters)

        # Tiled (tile=256)
        g_tiled = (math.ceil(N_int / 256),)
        t_tiled = bench_gpu(heat_1d_tiled, u_dev, o2, g_tiled, (256,), iters=iters)

        # Temporal T=4 (tile=128)
        g_temp = (math.ceil(N_int / 128),)
        t_temp_total = bench_gpu(heat_1d_temporal, u_dev, o3, g_temp, (128,), iters=iters)
        t_temp_per_step = t_temp_total / 4  # 4 steps per launch

        sp_tiled = t_naive / t_tiled if t_tiled > 0 else 0
        sp_temp = t_naive / t_temp_per_step if t_temp_per_step > 0 else 0
        # Effective bandwidth for temporal (per step)
        bytes_per_step = N_int * (3 + 1) * 8  # 3 loads + 1 store, float64
        gbs_temp = bytes_per_step / t_temp_per_step / 1e9 if t_temp_per_step > 0 else 0

        print(f"  {N_int:<14,} {t_naive*1e6:<12.1f} {t_tiled*1e6:<12.1f} "
              f"{t_temp_per_step*1e6:<12.1f} {sp_tiled:<12.2f}x {sp_temp:<12.2f}x "
              f"{gbs_temp:<12.1f}")

        del u_dev, o1, o2, o3

    # ==================================================================
    # 2D LAPLACIAN
    # ==================================================================
    header("2D Laplacian (5-point, halo=1)")
    print(f"\n  {'Grid':<14} {'Naive':<12} {'Tiled':<12} {'Temp T=2':<12} {'Temp T=4':<12} "
          f"{'Tiled/N':<10} {'T2/Naive':<10} {'T4/Naive':<10}")
    print(f"  {'':<14} {'us':<12} {'us':<12} {'us/step':<12} {'us/step':<12} "
          f"{'':<10} {'':<10} {'':<10}")
    sep("-")

    for Ni in [256, 512, 1024, 2048, 4096]:
        N = Ni + 2
        pts = Ni * Ni
        u = np.random.randn(N, N)
        u_dev = cuda.to_device(u)
        o1 = cuda.device_array_like(u)
        o2 = cuda.device_array_like(u)
        o3 = cuda.device_array_like(u)
        o4 = cuda.device_array_like(u)

        iters = max(20, min(500, int(5e8 / pts)))
        BLK = (16, 16)

        # Naive
        g_naive = (math.ceil(N/16), math.ceil(N/16))
        t_naive = bench_gpu(lap_2d_naive, u_dev, o1, g_naive, BLK, iters=iters)

        # Tiled (32x32)
        g_tiled = (math.ceil(Ni/32), math.ceil(Ni/32))
        t_tiled = bench_gpu(lap_2d_tiled, u_dev, o2, g_tiled, (32, 32), iters=iters)

        # Temporal T=2 (16x16 tile, 20x20 expanded)
        g_temp2 = (math.ceil(Ni/16), math.ceil(Ni/16))
        t_temp2_total = bench_gpu(lap_2d_temporal, u_dev, o3, g_temp2, (16, 16), iters=iters)
        t_temp2 = t_temp2_total / 2

        # Temporal T=4 (16x16 tile, 24x24 expanded)
        t_temp4_total = bench_gpu(lap_2d_temporal_t4, u_dev, o4, g_temp2, (16, 16), iters=iters)
        t_temp4 = t_temp4_total / 4

        sp_tiled = t_naive / t_tiled if t_tiled > 0 else 0
        sp_t2 = t_naive / t_temp2 if t_temp2 > 0 else 0
        sp_t4 = t_naive / t_temp4 if t_temp4 > 0 else 0

        label = f"{Ni}x{Ni}"
        print(f"  {label:<14} {t_naive*1e6:<12.1f} {t_tiled*1e6:<12.1f} "
              f"{t_temp2*1e6:<12.1f} {t_temp4*1e6:<12.1f} "
              f"{sp_tiled:<10.2f}x {sp_t2:<10.2f}x {sp_t4:<10.2f}x")

        del u_dev, o1, o2, o3, o4

    # ==================================================================
    # 3D LAPLACIAN (naive only — tiling doesn't help much in 3D)
    # ==================================================================
    header("3D Laplacian (7-point) — Naive (tiling limited by shared mem)")
    print(f"\n  {'Grid':<16} {'Naive us':<14} {'Eff GB/s':<14} {'Gpts/s':<14}")
    sep("-")
    for Ni in [64, 128, 256]:
        N = Ni + 2
        pts = Ni**3
        u = np.random.randn(N, N, N)
        u_dev = cuda.to_device(u)
        o_dev = cuda.device_array_like(u)
        BLK = (8, 8, 8)
        grid = (math.ceil(N/8), math.ceil(N/8), math.ceil(N/8))
        iters = max(10, min(200, int(5e8 / pts)))
        t = bench_gpu(lap_3d_naive, u_dev, o_dev, grid, BLK, iters=iters)
        gbs = pts * (7+1) * 8 / t / 1e9
        gpts = pts / t / 1e9
        print(f"  {Ni}^3{'':<11} {t*1e6:<14.1f} {gbs:<14.1f} {gpts:<14.4f}")
        del u_dev, o_dev

    # ==================================================================
    # SUMMARY
    # ==================================================================
    header("SUMMARY — What Each Optimization Level Buys You")
    print("""
  ┌────────────────────────────────────────────────────────────────────┐
  │ Optimization            │ What it does          │ Typical speedup │
  ├─────────────────────────┼───────────────────────┼─────────────────┤
  │ Naive (global mem only) │ Baseline              │ 1x              │
  │ Shared-mem tiling       │ Reuse halo data       │ 1.0-1.3x        │
  │ Temporal T=2            │ Fuse 2 steps in smem  │ 1.5-2.0x        │
  │ Temporal T=4            │ Fuse 4 steps in smem  │ 2.0-3.5x        │
  │ Temporal T=16 (DSL max) │ Fuse 16 steps in smem │ up to ~10x      │
  └─────────────────────────┴───────────────────────┴─────────────────┘

  WHY tiling alone doesn't help much on Blackwell:
    Your GPU has a 128 MB L2 cache that already provides significant
    data reuse. Shared memory tiling helps most on older GPUs with
    smaller L2 caches. The big win is TEMPORAL BLOCKING.

  WHAT CAN STILL BE IMPROVED in the DSL:
    1. Auto-generate Numba CUDA backend (not just cuTile code)
       → Actually run kernels without installing cuda.tile
    2. Register tiling for 1D stencils (keep values in registers)
    3. Warp shuffle for halo exchange (avoid shared mem for narrow halos)
    4. Larger T with hierarchical blocking (L2 → shared → registers)
    5. Async memory copies (overlap loads with compute)
    6. Mixed precision inner steps (FP32 intermediate, FP64 final)
    7. Multi-array temporal blocking for coupled systems (Gray-Scott etc)
    8. Persistent thread blocks for time-stepping loops
""")
    sep()


if __name__ == "__main__":
    run_all()
