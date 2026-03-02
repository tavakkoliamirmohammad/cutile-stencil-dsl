#!/usr/bin/env python3
"""Comprehensive benchmark suite: CPU + GPU for every example stencil.

Covers every stencil from examples/:
  Single-input:  heat_1d, advection_upwind, laplacian_2d, wave_2d, laplacian_3d
  Multi-input:   gray_scott (u,v), fdtd_maxwell (E,H), shallow_water (h,hu,hv)
  Solvers:       poisson_cg, mixed_precision_cg

For each benchmark:
  CPU perf      — measured (NumPy, FP64 & FP32)
  GPU perf      — predicted via roofline model (FP64 & FP32)
  Precision err — |FP64 result - FP32 result|  (L-inf and relative)
  Speedup       — GPU predicted / CPU measured
  Generated     — cuTile kernel path

Target GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q

Usage:
    python run_benchmarks.py
"""

import os
import time
import numpy as np

from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.registry import clear
from cutile_stencil.dsl.types import HardwareSpec, StencilSpec
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.analysis.roofline import roofline_analysis
from cutile_stencil.analysis.tiling import compute_tile_config
from cutile_stencil.analysis.temporal import compute_temporal_config
from cutile_stencil.codegen.stencil_codegen import StencilCodeGenerator
from cutile_stencil.reference.stencil_ref import apply_stencil, _ArrayProxy
from cutile_stencil.config import GPU_PRESETS


# ── Configuration ─────────────────────────────────────────────────────────────

TARGET_GPU = "RTX_PRO_6000_MAXQ"
GEN_DIR = os.path.join(os.path.dirname(__file__), "examples", "generated")


# ── Formatting helpers ────────────────────────────────────────────────────────

def sep(ch="=", w=94):
    print(ch * w)

def header(title):
    print()
    sep()
    print(f"  {title}")
    sep()

def sub(title):
    print(f"\n  -- {title} {'-' * max(1, 72 - len(title))}")

def gs(g):
    return "x".join(str(x) for x in g)

def npts(g):
    p = 1
    for x in g:
        p *= x
    return p


# ── Stencil definitions (mirror examples/) ───────────────────────────────────

def define_stencils():
    """Return list of all benchmark stencils."""
    clear()
    specs = []

    # -- heat_1d (examples/heat_1d.py) --
    @stencil(ndim=1, order=2, dtype="float64")
    def heat_1d(u, i):
        return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
    specs.append(("heat_1d", heat_1d._stencil_spec,
                   [(4096,), (65536,), (262144,), (1048576,)],
                   "single", 1))

    # -- advection_upwind (examples/advection_upwind.py) --
    CFL = 0.5
    @stencil(ndim=1, order=2, dtype="float64")
    def advection_upwind(u, i):
        return u[i] - CFL * (u[i] - u[i - 1])
    specs.append(("advection_upwind", advection_upwind._stencil_spec,
                   [(4096,), (65536,), (262144,), (1048576,)],
                   "single", 1))

    # -- laplacian_2d (from examples/benchmark_suite.py) --
    @stencil(ndim=2, order=2, dtype="float64")
    def laplacian_2d(u, i, j):
        return u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]
    specs.append(("laplacian_2d", laplacian_2d._stencil_spec,
                   [(128, 128), (256, 256), (512, 512), (1024, 1024)],
                   "single", 1))

    # -- wave_2d (examples/wave_2d.py) --
    @stencil(ndim=2, order=4, dtype="float64")
    def wave_2d(u, i, j):
        lx = (-1/12)*u[i-2,j] + (4/3)*u[i-1,j] + (-5/2)*u[i,j] \
             + (4/3)*u[i+1,j] + (-1/12)*u[i+2,j]
        ly = (-1/12)*u[i,j-2] + (4/3)*u[i,j-1] + (-5/2)*u[i,j] \
             + (4/3)*u[i,j+1] + (-1/12)*u[i,j+2]
        return 0.1 * (lx + ly)
    specs.append(("wave_2d", wave_2d._stencil_spec,
                   [(128, 128), (256, 256), (512, 512)],
                   "single", 1))

    # -- laplacian_3d (examples/laplacian_3d.py) --
    @stencil(ndim=3, order=2, dtype="float64")
    def laplacian_3d(u, i, j, k):
        return (u[i-1,j,k] + u[i+1,j,k] + u[i,j-1,k] + u[i,j+1,k]
                + u[i,j,k-1] + u[i,j,k+1] - 6*u[i,j,k])
    specs.append(("laplacian_3d", laplacian_3d._stencil_spec,
                   [(32, 32, 32), (64, 64, 64), (128, 128, 128)],
                   "single", 1))

    # -- gray_scott_u (examples/gray_scott.py) --
    Du, Dv, F_gs, k_gs, dt_gs = 0.16, 0.08, 0.035, 0.065, 1.0
    @stencil(ndim=2, order=2, dtype="float64")
    def gray_scott_u(u, v, i, j):
        lap = u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]
        return u[i,j] + dt_gs*(Du*lap - u[i,j]*v[i,j]*v[i,j] + F_gs*(1.0 - u[i,j]))
    specs.append(("gray_scott_u", gray_scott_u._stencil_spec,
                   [(64, 64), (128, 128), (256, 256)],
                   "multi", 2))

    # -- fdtd_update_E (examples/fdtd_maxwell_1d.py) --
    dx_f, dt_f, eps_f = 0.01, 0.005, 1.0
    @stencil(ndim=1, order=2, dtype="float64")
    def fdtd_update_E(E, H, i):
        return E[i] + (dt_f / (eps_f * dx_f)) * (H[i] - H[i - 1])
    specs.append(("fdtd_update_E", fdtd_update_E._stencil_spec,
                   [(4096,), (65536,), (262144,), (1048576,)],
                   "multi", 2))

    # -- shallow_water_h (examples/shallow_water.py) --
    g_sw, H0_sw, dt_sw, dx_sw = 9.81, 1.0, 0.01, 0.1
    @stencil(ndim=2, order=2, dtype="float64")
    def shallow_water_h(h, hu, hv, i, j):
        dhu_dx = (hu[i+1,j] - hu[i-1,j]) / (2*dx_sw)
        dhv_dy = (hv[i,j+1] - hv[i,j-1]) / (2*dx_sw)
        return h[i,j] - dt_sw * H0_sw * (dhu_dx + dhv_dy)
    specs.append(("shallow_water_h", shallow_water_h._stencil_spec,
                   [(64, 64), (128, 128), (256, 256)],
                   "multi", 3))

    # Extract footprints
    for _, spec, _, _, _ in specs:
        if not spec.accesses:
            extract_footprint(spec)
        if spec.halo_widths is None:
            spec.halo_widths = compute_halo(spec.accesses, spec.ndim)

    return specs


# ── Benchmarking primitives ───────────────────────────────────────────────────

def _choose_iters(pts):
    if pts < 500_000:
        return 30, 5
    if pts < 5_000_000:
        return 20, 5
    if pts < 50_000_000:
        return 10, 3
    return 5, 2


def bench_single(spec, grid_size, dtype="float64"):
    """Benchmark a single-input stencil.  Returns (Gpts/s, GB/s, ms/iter)."""
    np_dtype = getattr(np, dtype)
    halo = spec.halo_widths
    full = tuple(g + 2*h for g, h in zip(grid_size, halo))
    u = np.random.randn(*full).astype(np_dtype)
    iters, warmup = _choose_iters(npts(grid_size))

    for _ in range(warmup):
        apply_stencil(u, spec)

    t0 = time.perf_counter()
    for _ in range(iters):
        apply_stencil(u, spec)
    elapsed = time.perf_counter() - t0

    pts = npts(grid_size)
    pps = (pts * iters) / elapsed
    hw_local = HardwareSpec(dtype_bytes=np.dtype(np_dtype).itemsize)
    roof = roofline_analysis(spec, hw_local)
    return pps/1e9, (pps * roof.bytes_per_point)/1e9, (elapsed/iters)*1000


def bench_multi(spec, grid_size, n_inputs, dtype="float64"):
    """Benchmark a multi-input stencil.  Returns (Gpts/s, GB/s, ms/iter)."""
    np_dtype = getattr(np, dtype)
    halo = spec.halo_widths
    full = tuple(g + 2*h for g, h in zip(grid_size, halo))
    arrays = [np.random.randn(*full).astype(np_dtype) for _ in range(n_inputs)]
    proxies = [_ArrayProxy(a, halo) for a in arrays]
    idx = [0] * spec.ndim
    iters, warmup = _choose_iters(npts(grid_size))

    for _ in range(warmup):
        spec.update_fn(*proxies, *idx)

    t0 = time.perf_counter()
    for _ in range(iters):
        spec.update_fn(*proxies, *idx)
    elapsed = time.perf_counter() - t0

    pts = npts(grid_size)
    pps = (pts * iters) / elapsed
    hw_local = HardwareSpec(dtype_bytes=np.dtype(np_dtype).itemsize)
    roof = roofline_analysis(spec, hw_local)
    return pps/1e9, (pps * roof.bytes_per_point)/1e9, (elapsed/iters)*1000


def precision_error_single(spec, grid_size):
    """Return (L_inf, relative L_inf) between FP64 and FP32 results."""
    halo = spec.halo_widths
    full = tuple(g + 2*h for g, h in zip(grid_size, halo))
    np.random.seed(42)
    u64 = np.random.randn(*full)
    u32 = u64.astype(np.float32)
    r64 = apply_stencil(u64, spec)
    r32 = apply_stencil(u32, spec)
    interior = tuple(slice(h, s-h) for h, s in zip(halo, full))
    diff = np.abs(r64[interior] - r32[interior].astype(np.float64))
    linf = float(np.max(diff))
    ref = float(np.max(np.abs(r64[interior])))
    return linf, linf / max(ref, 1e-30)


def precision_error_multi(spec, grid_size, n_inputs):
    """Return (L_inf, relative L_inf) between FP64 and FP32 results."""
    halo = spec.halo_widths
    full = tuple(g + 2*h for g, h in zip(grid_size, halo))
    np.random.seed(42)
    a64 = [np.random.randn(*full) for _ in range(n_inputs)]
    a32 = [a.astype(np.float32) for a in a64]
    p64 = [_ArrayProxy(a, halo) for a in a64]
    p32 = [_ArrayProxy(a, halo) for a in a32]
    idx = [0] * spec.ndim
    r64 = np.asarray(spec.update_fn(*p64, *idx), dtype=np.float64)
    r32 = np.asarray(spec.update_fn(*p32, *idx), dtype=np.float64)
    diff = np.abs(r64 - r32)
    linf = float(np.max(diff))
    ref = float(np.max(np.abs(r64)))
    return linf, linf / max(ref, 1e-30)


# ── Main benchmark loop ──────────────────────────────────────────────────────

def run_stencil_benchmarks():
    header("cuTile Stencil DSL  --  Comprehensive Benchmark Suite")
    print(f"  Target GPU : {TARGET_GPU}  ({GPU_PRESETS[TARGET_GPU].name})")

    specs = define_stencils()
    os.makedirs(GEN_DIR, exist_ok=True)

    summary_rows = []

    for name, spec, grids, stype, n_in in specs:

        header(f"{name}  (ndim={spec.ndim}, order={spec.order}, "
               f"inputs={spec.inputs})")

        # ── Roofline ─────────────────────────────────────────────────
        hw64 = HardwareSpec.from_preset(TARGET_GPU, dtype="float64")
        hw32 = HardwareSpec.from_preset(TARGET_GPU, dtype="float32")
        r64 = roofline_analysis(spec, hw64)
        r32 = roofline_analysis(spec, hw32)

        sub("Roofline Analysis")
        fmt = "  {:<30} {:<18} {:<18}"
        print(fmt.format("Metric", "FP64", "FP32"))
        print(fmt.format("-"*28, "-"*16, "-"*16))
        print(fmt.format("FLOPs/point",    str(r64.flops_per_point),
                          str(r32.flops_per_point)))
        print(fmt.format("Bytes/point",    str(r64.bytes_per_point),
                          str(r32.bytes_per_point)))
        print(fmt.format("Arith. intensity (F/B)",
                          f"{r64.arithmetic_intensity:.3f}",
                          f"{r32.arithmetic_intensity:.3f}"))
        print(fmt.format("Bottleneck", r64.bound, r32.bound))
        print(fmt.format("GPU peak (Gpts/s)",
                          f"{r64.peak_gpoints_s:.2f}",
                          f"{r32.peak_gpoints_s:.2f}"))

        # ── Tiling ───────────────────────────────────────────────────
        tile64 = compute_tile_config(spec, grids[-1], hw64)
        tile32 = compute_tile_config(spec, grids[-1], hw32)
        print(f"\n  Tile (FP64): {tile64.tile_sizes}  "
              f"overhead={tile64.overhead_fraction:.1%}")
        print(f"  Tile (FP32): {tile32.tile_sizes}  "
              f"overhead={tile32.overhead_fraction:.1%}")

        # ── Code generation (GPU version) ────────────────────────────
        try:
            codegen = StencilCodeGenerator(spec, tile64)
            kpath = os.path.join(GEN_DIR, f"{name}_kernel.py")
            codegen.emit_to_file(kpath)
            import ast
            ast.parse(open(kpath).read())
            print(f"  Generated GPU kernel: {kpath}  (valid Python)")
        except Exception as e:
            kpath = None
            print(f"  Code generation skipped: {e}")

        # ── CPU benchmarks ───────────────────────────────────────────
        sub("CPU Performance (NumPy) vs GPU Predicted")
        hdr = (f"  {'Grid':<16} {'Dtype':<8} {'CPU Gpts/s':<12} "
               f"{'CPU GB/s':<10} {'CPU ms':<10} "
               f"{'GPU peak':<10} {'Speedup':<10}")
        print(hdr)
        print(f"  {'-'*14:<16} {'-'*6:<8} {'-'*10:<12} "
              f"{'-'*8:<10} {'-'*8:<10} "
              f"{'-'*8:<10} {'-'*8:<10}")

        best_f64_gpts = 0.0
        best_f32_gpts = 0.0

        for grid in grids:
            for dtype in ("float64", "float32"):
                try:
                    if stype == "single":
                        gpts, gbs, ms = bench_single(spec, grid, dtype)
                    else:
                        gpts, gbs, ms = bench_multi(spec, grid, n_in, dtype)

                    roof = r64 if dtype == "float64" else r32
                    gpu_peak = roof.peak_gpoints_s
                    speedup = gpu_peak / gpts if gpts > 0 else 0

                    print(f"  {gs(grid):<16} {dtype:<8} {gpts:<12.4f} "
                          f"{gbs:<10.2f} {ms:<10.2f} "
                          f"{gpu_peak:<10.2f} "
                          f"{speedup:<10.0f}x")

                    if dtype == "float64" and gpts > best_f64_gpts:
                        best_f64_gpts = gpts
                    if dtype == "float32" and gpts > best_f32_gpts:
                        best_f32_gpts = gpts

                except Exception as e:
                    print(f"  {gs(grid):<16} {dtype:<8} ERROR: {e}")

        # ── Precision error ──────────────────────────────────────────
        sub("FP64 vs FP32 Precision Error (GPU trade-off)")
        err_grid = grids[min(1, len(grids)-1)]
        try:
            if stype == "single":
                linf, rel = precision_error_single(spec, err_grid)
            else:
                linf, rel = precision_error_multi(spec, err_grid, n_in)
            print(f"  Grid:         {gs(err_grid)}")
            print(f"  Absolute L-inf error:  {linf:.2e}")
            print(f"  Relative L-inf error:  {rel:.2e}")
        except Exception as e:
            linf, rel = float('nan'), float('nan')
            print(f"  Error: {e}")

        # ── Per-stencil summary ──────────────────────────────────────
        sub("Summary")
        if best_f64_gpts > 0:
            sp64 = r64.peak_gpoints_s / best_f64_gpts
            sp32 = r32.peak_gpoints_s / best_f64_gpts
            print(f"  Best CPU (FP64):       {best_f64_gpts:.4f} Gpts/s")
            if best_f32_gpts > 0:
                print(f"  Best CPU (FP32):       {best_f32_gpts:.4f} Gpts/s")
            print(f"  GPU peak  (FP64):      {r64.peak_gpoints_s:.2f} Gpts/s"
                  f"  = {sp64:.0f}x over CPU")
            print(f"  GPU peak  (FP32):      {r32.peak_gpoints_s:.2f} Gpts/s"
                  f"  = {sp32:.0f}x over CPU")
            print(f"  FP32 precision loss:   {rel:.2e}  (relative L-inf)")

            summary_rows.append({
                "name": name, "ndim": spec.ndim,
                "cpu64": best_f64_gpts, "cpu32": best_f32_gpts,
                "gpu64": r64.peak_gpoints_s, "gpu32": r32.peak_gpoints_s,
                "err": rel, "speedup": sp32,
            })

    # ── Grand summary ─────────────────────────────────────────────────
    header("Grand Summary: CPU (NumPy) vs GPU (Roofline-Predicted)")
    print(f"  GPU: {TARGET_GPU}\n")
    hdr = (f"  {'Stencil':<20} {'dim':<5} {'CPU FP64':<12} {'CPU FP32':<12} "
           f"{'GPU FP64':<12} {'GPU FP32':<12} {'FP32 err':<12} {'Speedup':<10}")
    print(hdr)
    unit = (f"  {'':<20} {'':<5} {'Gpts/s':<12} {'Gpts/s':<12} "
            f"{'Gpts/s':<12} {'Gpts/s':<12} {'(rel)':<12} {'(FP32)':<10}")
    print(unit)
    print(f"  {'-'*18:<20} {'-'*3:<5} {'-'*10:<12} {'-'*10:<12} "
          f"{'-'*10:<12} {'-'*10:<12} {'-'*10:<12} {'-'*8:<10}")

    for s in summary_rows:
        c32 = f"{s['cpu32']:.4f}" if s['cpu32'] > 0 else "---"
        print(f"  {s['name']:<20} {s['ndim']:<5} {s['cpu64']:<12.4f} {c32:<12} "
              f"{s['gpu64']:<12.2f} {s['gpu32']:<12.2f} "
              f"{s['err']:<12.2e} {s['speedup']:<10.0f}x")

    sep()
    return summary_rows


# ── Solver benchmarks ─────────────────────────────────────────────────────────

def run_solver_benchmarks():
    from cutile_stencil.solvers.formats import laplacian_1d_dia, laplacian_2d_dia, DIAMatrix
    from cutile_stencil.reference.spmv_ref import dia_spmv
    from cutile_stencil.reference.cg_ref import cg_solve, mixed_precision_cg

    header("Solver Benchmarks: CG and Mixed-Precision CG")

    # ── 1D Poisson, FP64 vs FP32 CG ─────────────────────────────────
    sub("Poisson 1D  (N=1000, standard CG)")
    N = 1000
    h = 1.0 / (N + 1)
    A = laplacian_1d_dia(N)
    xg = np.linspace(h, 1-h, N)
    exact = np.sin(np.pi * xg) / (np.pi**2)
    f = np.sin(np.pi * xg) * h**2
    A_fn = lambda v: dia_spmv(A, v)

    # FP64
    t0 = time.perf_counter()
    x64, it64, res64 = cg_solve(A_fn, f, tol=1e-12, max_iter=2000)
    t64 = time.perf_counter() - t0
    err64 = float(np.max(np.abs(x64 - exact)))

    # FP32 (cast matrix and RHS)
    A32 = DIAMatrix(A.data.astype(np.float32), A.offsets, A.shape)
    f32 = f.astype(np.float32)
    A_fn32 = lambda v: dia_spmv(A32, v)
    t0 = time.perf_counter()
    x32, it32, res32 = cg_solve(A_fn32, f32, tol=1e-6, max_iter=2000)
    t32 = time.perf_counter() - t0
    err32 = float(np.max(np.abs(x32.astype(np.float64) - exact)))

    fmt = "  {:<28} {:<20} {:<20}"
    print(fmt.format("Metric", "CG (FP64)", "CG (FP32)"))
    print(fmt.format("-"*26, "-"*18, "-"*18))
    print(fmt.format("Solve time (ms)", f"{t64*1000:.2f}", f"{t32*1000:.2f}"))
    print(fmt.format("Iterations", str(it64), str(it32)))
    print(fmt.format("Error vs exact (L-inf)", f"{err64:.2e}", f"{err32:.2e}"))
    print(fmt.format("Final residual", f"{res64[-1]:.2e}", f"{res32[-1]:.2e}"))
    sp = t64 / max(t32, 1e-30)
    print(fmt.format("FP32 time speedup", "", f"{sp:.2f}x"))

    # ── Mixed-precision CG ───────────────────────────────────────────
    sub("Mixed-Precision CG  (N=1000, FP64 outer + FP32 inner)")
    exact2 = np.sin(2 * np.pi * xg)
    f_mp = dia_spmv(A, exact2)

    t0 = time.perf_counter()
    x_std, it_std, res_std = cg_solve(A_fn, f_mp, tol=1e-12, max_iter=2000)
    t_std = time.perf_counter() - t0
    err_std = float(np.max(np.abs(x_std - exact2)))

    t0 = time.perf_counter()
    x_mp, it_mp, res_mp = mixed_precision_cg(
        A_fn, f_mp, inner_dtype=np.float32,
        tol=1e-12, max_outer=30, max_inner=500, inner_tol=1e-3,
    )
    t_mp = time.perf_counter() - t0
    err_mp = float(np.max(np.abs(x_mp - exact2)))

    print(fmt.format("Metric", "Standard CG", "Mixed-Prec CG"))
    print(fmt.format("-"*26, "-"*18, "-"*18))
    print(fmt.format("Solve time (ms)", f"{t_std*1000:.2f}", f"{t_mp*1000:.2f}"))
    print(fmt.format("Iterations", str(it_std), str(it_mp)))
    print(fmt.format("Error vs exact (L-inf)", f"{err_std:.2e}", f"{err_mp:.2e}"))
    print(fmt.format("Final residual", f"{res_std[-1]:.2e}", f"{res_mp[-1]:.2e}"))

    # ── 2D Poisson ───────────────────────────────────────────────────
    sub("Poisson 2D  (40x40 grid, CG FP64 vs FP32)")
    Nx, Ny = 40, 40
    h2 = 1.0 / (Nx + 1)
    A2 = laplacian_2d_dia(Nx, Ny)
    N2 = Nx * Ny
    f2_64 = np.ones(N2) * h2**2
    A2_fn = lambda v: dia_spmv(A2, v)

    t0 = time.perf_counter()
    x2_64, it2_64, res2_64 = cg_solve(A2_fn, f2_64, tol=1e-10, max_iter=5000)
    t2_64 = time.perf_counter() - t0

    A2_32 = DIAMatrix(A2.data.astype(np.float32), A2.offsets, A2.shape)
    f2_32 = f2_64.astype(np.float32)
    A2_fn32 = lambda v: dia_spmv(A2_32, v)

    t0 = time.perf_counter()
    x2_32, it2_32, res2_32 = cg_solve(A2_fn32, f2_32, tol=1e-6, max_iter=5000)
    t2_32 = time.perf_counter() - t0

    # Compare solutions
    sol_diff = float(np.max(np.abs(x2_64 - x2_32.astype(np.float64))))
    Ax = dia_spmv(A2, x2_64)
    rel64 = float(np.linalg.norm(Ax - f2_64) / np.linalg.norm(f2_64))
    Ax32 = dia_spmv(A2_32, x2_32)
    rel32 = float(np.linalg.norm(Ax32.astype(np.float64) - f2_64) / np.linalg.norm(f2_64))

    print(f"\n  Grid: {Nx}x{Ny} = {N2} unknowns")
    print(fmt.format("Metric", "CG (FP64)", "CG (FP32)"))
    print(fmt.format("-"*26, "-"*18, "-"*18))
    print(fmt.format("Solve time (ms)", f"{t2_64*1000:.2f}", f"{t2_32*1000:.2f}"))
    print(fmt.format("Iterations", str(it2_64), str(it2_32)))
    print(fmt.format("Relative residual", f"{rel64:.2e}", f"{rel32:.2e}"))
    print(fmt.format("FP64 vs FP32 sol diff", f"{sol_diff:.2e}", ""))
    print(fmt.format("FP32 time speedup", "", f"{t2_64/max(t2_32,1e-30):.2f}x"))

    sep()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_stencil_benchmarks()
    run_solver_benchmarks()
