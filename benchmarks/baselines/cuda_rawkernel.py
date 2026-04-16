"""CUDA C baseline via CuPy RawKernel: hand-tuned GPU stencils.

Two variants:
  - 'naive': one thread per point, direct global memory (baseline floor)
  - 'smem':  shared-memory tiled, the expert-level GPU implementation

This is the strongest GPU baseline — raw CUDA C with manual optimization.
"""

from __future__ import annotations

import cupy as cp

from benchmarks.stencils import STENCIL_META, interior_size


# ── 1D Heat ────────────────────────────────────────────────────────

_heat_1d_naive = cp.RawKernel(r'''
extern "C" __global__
void heat_1d_naive(const double* __restrict__ u, double* __restrict__ out,
                   int N, int H) {
    int n = N - 2*H;
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gx < n) {
        int ci = gx + H;
        out[ci] = 0.25 * u[ci-1] + 0.5 * u[ci] + 0.25 * u[ci+1];
    }
}
''', 'heat_1d_naive')


# ── 2D Heat ────────────────────────────────────────────────────────

_heat_2d_naive = cp.RawKernel(r'''
extern "C" __global__
void heat_2d_naive(const double* __restrict__ u, double* __restrict__ out,
                   int Nx, int Ny, int H) {
    int nx = Nx - 2*H;
    int ny = Ny - 2*H;
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx < nx && gy < ny) {
        int ci = (gx + H) * Ny + (gy + H);
        out[ci] = 0.25 * (u[ci - Ny] + u[ci + Ny] + u[ci - 1] + u[ci + 1]);
    }
}
''', 'heat_2d_naive')

_heat_2d_smem = cp.RawKernel(r'''
extern "C" __global__
void heat_2d_smem(const double* __restrict__ u, double* __restrict__ out,
                  int Nx, int Ny, int H) {
    // Block tile: 32x32 interior + 1-wide halo = 34x34 shared mem
    const int BX = 32, BY = 32;
    __shared__ double s[(32+2)*(32+2)];
    const int SP = BY + 2;  // shared memory pitch

    int nx = Nx - 2*H;
    int ny = Ny - 2*H;
    int bx = blockIdx.x * BX;
    int by = blockIdx.y * BY;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = bx + tx;
    int gy = by + ty;

    // Load interior
    if (gx < nx && gy < ny) {
        s[(tx+1)*SP + (ty+1)] = u[(gx+H)*Ny + (gy+H)];
    }
    // Load halos
    if (tx == 0 && gx > 0 && gy < ny)
        s[0*SP + (ty+1)] = u[(gx+H-1)*Ny + (gy+H)];
    if (tx == BX-1 && gx+1 < nx && gy < ny)
        s[(BX+1)*SP + (ty+1)] = u[(gx+H+1)*Ny + (gy+H)];
    if (ty == 0 && gy > 0 && gx < nx)
        s[(tx+1)*SP + 0] = u[(gx+H)*Ny + (gy+H-1)];
    if (ty == BY-1 && gy+1 < ny && gx < nx)
        s[(tx+1)*SP + (BY+1)] = u[(gx+H)*Ny + (gy+H+1)];

    // Edge blocks: load halo even if tx/ty != boundary of block
    if (bx + BX >= nx && tx == min(BX-1, nx-1-bx) && gx+1 < nx && gy < ny)
        s[(tx+2)*SP + (ty+1)] = u[(gx+H+1)*Ny + (gy+H)];
    if (by + BY >= ny && ty == min(BY-1, ny-1-by) && gy+1 < ny && gx < nx)
        s[(tx+1)*SP + (ty+2)] = u[(gx+H)*Ny + (gy+H+1)];

    __syncthreads();

    if (gx < nx && gy < ny) {
        double result = 0.25 * (
            s[(tx  )*SP + (ty+1)] +  // west
            s[(tx+2)*SP + (ty+1)] +  // east
            s[(tx+1)*SP + (ty  )] +  // south
            s[(tx+1)*SP + (ty+2)]    // north
        );
        out[(gx+H)*Ny + (gy+H)] = result;
    }
}
''', 'heat_2d_smem')


# ── 2D Laplacian 5-point ──────────────────────────────────────────

_lap_2d_5pt_naive = cp.RawKernel(r'''
extern "C" __global__
void lap_2d_5pt_naive(const double* __restrict__ u, double* __restrict__ out,
                      int Nx, int Ny, int H) {
    int nx = Nx - 2*H;
    int ny = Ny - 2*H;
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx < nx && gy < ny) {
        int ci = (gx + H) * Ny + (gy + H);
        double c = u[ci];
        out[ci] = u[ci-Ny] + u[ci+Ny] + u[ci-1] + u[ci+1] - 4.0*c;
    }
}
''', 'lap_2d_5pt_naive')

_lap_2d_5pt_smem = cp.RawKernel(r'''
extern "C" __global__
void lap_2d_5pt_smem(const double* __restrict__ u, double* __restrict__ out,
                     int Nx, int Ny, int H) {
    const int BX = 32, BY = 32;
    __shared__ double s[(32+2)*(32+2)];
    const int SP = BY + 2;

    int nx = Nx - 2*H;
    int ny = Ny - 2*H;
    int bx = blockIdx.x * BX;
    int by = blockIdx.y * BY;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = bx + tx;
    int gy = by + ty;

    if (gx < nx && gy < ny)
        s[(tx+1)*SP + (ty+1)] = u[(gx+H)*Ny + (gy+H)];
    if (tx == 0 && gx > 0 && gy < ny)
        s[0*SP + (ty+1)] = u[(gx+H-1)*Ny + (gy+H)];
    if (tx == BX-1 && gx+1 < nx && gy < ny)
        s[(BX+1)*SP + (ty+1)] = u[(gx+H+1)*Ny + (gy+H)];
    if (ty == 0 && gy > 0 && gx < nx)
        s[(tx+1)*SP + 0] = u[(gx+H)*Ny + (gy+H-1)];
    if (ty == BY-1 && gy+1 < ny && gx < nx)
        s[(tx+1)*SP + (BY+1)] = u[(gx+H)*Ny + (gy+H+1)];

    if (bx + BX >= nx && tx == min(BX-1, nx-1-bx) && gx+1 < nx && gy < ny)
        s[(tx+2)*SP + (ty+1)] = u[(gx+H+1)*Ny + (gy+H)];
    if (by + BY >= ny && ty == min(BY-1, ny-1-by) && gy+1 < ny && gx < nx)
        s[(tx+1)*SP + (ty+2)] = u[(gx+H)*Ny + (gy+H+1)];

    __syncthreads();

    if (gx < nx && gy < ny) {
        out[(gx+H)*Ny + (gy+H)] =
            s[(tx  )*SP + (ty+1)] +
            s[(tx+2)*SP + (ty+1)] +
            s[(tx+1)*SP + (ty  )] +
            s[(tx+1)*SP + (ty+2)] -
            4.0 * s[(tx+1)*SP + (ty+1)];
    }
}
''', 'lap_2d_5pt_smem')


# ── 3D Laplacian 7-point ──────────────────────────────────────────

_lap_3d_7pt_naive = cp.RawKernel(r'''
extern "C" __global__
void lap_3d_7pt_naive(const double* __restrict__ u, double* __restrict__ out,
                      int Nx, int Ny, int Nz, int H) {
    int nx = Nx - 2*H;
    int ny = Ny - 2*H;
    int nz = Nz - 2*H;
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gz = blockIdx.z * blockDim.z + threadIdx.z;
    if (gx < nx && gy < ny && gz < nz) {
        int yz = Ny * Nz;
        int ci = (gx+H)*yz + (gy+H)*Nz + (gz+H);
        double c = u[ci];
        out[ci] = u[ci-yz] + u[ci+yz] + u[ci-Nz] + u[ci+Nz]
                 + u[ci-1] + u[ci+1] - 6.0*c;
    }
}
''', 'lap_3d_7pt_naive')


# ── Dispatch tables ────────────────────────────────────────────────

_KERNELS_NAIVE = {
    "heat_1d":          (_heat_1d_naive,      1),
    "heat_2d":          (_heat_2d_naive,      2),
    "laplacian_2d_5pt": (_lap_2d_5pt_naive,   2),
    "laplacian_3d_7pt": (_lap_3d_7pt_naive,   3),
}

_KERNELS_SMEM = {
    "heat_2d":          (_heat_2d_smem,       2),
    "laplacian_2d_5pt": (_lap_2d_5pt_smem,    2),
}

# Block sizes
_BLOCK_1D = (256,)
_BLOCK_2D = (32, 8)
_BLOCK_2D_SMEM = (32, 32)
_BLOCK_3D = (8, 8, 4)


def _launch(kernel, u, out, shape, halo, block, ndim):
    """Launch a CUDA kernel with the right grid/block dims."""
    h = halo[0]  # all our stencils have uniform halo
    if ndim == 1:
        n = shape[0] - 2 * h
        grid = ((n + block[0] - 1) // block[0],)
        kernel(grid, block, (u, out, shape[0], h))
    elif ndim == 2:
        nx, ny = shape[0] - 2*h, shape[1] - 2*h
        grid = ((nx + block[0] - 1) // block[0],
                (ny + block[1] - 1) // block[1])
        kernel(grid, block, (u, out, shape[0], shape[1], h))
    elif ndim == 3:
        nx = shape[0] - 2*h
        ny = shape[1] - 2*h
        nz = shape[2] - 2*h
        grid = ((nx + block[0] - 1) // block[0],
                (ny + block[1] - 1) // block[1],
                (nz + block[2] - 1) // block[2])
        kernel(grid, block, (u, out, shape[0], shape[1], shape[2], h))


def bench_cuda(
    name: str,
    shape: tuple[int, ...],
    variant: str = "naive",
    warmup: int = 30,
    iters: int = 100,
) -> dict:
    """Benchmark a CUDA C stencil kernel.

    Args:
        variant: "naive" (direct global memory) or "smem" (shared-memory tiled)
    """
    meta = STENCIL_META[name]
    ndim = meta["ndim"]

    if variant == "smem" and name in _KERNELS_SMEM:
        kernel, _ = _KERNELS_SMEM[name]
        block = _BLOCK_2D_SMEM
    elif name in _KERNELS_NAIVE:
        kernel, _ = _KERNELS_NAIVE[name]
        if ndim == 1:
            block = _BLOCK_1D
        elif ndim == 2:
            block = _BLOCK_2D
        else:
            block = _BLOCK_3D
    else:
        raise ValueError(f"No CUDA kernel for {name!r}/{variant}")

    u = cp.random.randn(*shape).astype(cp.float64)
    out = cp.zeros_like(u)

    for _ in range(warmup):
        _launch(kernel, u, out, shape, meta["halo"], block, ndim)
    cp.cuda.Device(0).synchronize()

    e1, e2 = cp.cuda.Event(), cp.cuda.Event()
    e1.record()
    for _ in range(iters):
        _launch(kernel, u, out, shape, meta["halo"], block, ndim)
    e2.record()
    e2.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(e1, e2) / iters
    domain = tuple(s - 2 * h for s, h in zip(shape, meta["halo"]))
    npts = interior_size(domain)
    gps = npts / elapsed_ms * 1e-6
    gbytes = npts * (meta["loads_per_point"] + meta["stores_per_point"]) * meta["dtype_bytes"] / elapsed_ms * 1e-6

    fw_name = "CUDA-naive" if variant == "naive" else "CUDA-smem"
    return {
        "name": name,
        "framework": fw_name,
        "time_ms": elapsed_ms,
        "gpoints_per_s": gps,
        "gbytes_per_s": gbytes,
    }
