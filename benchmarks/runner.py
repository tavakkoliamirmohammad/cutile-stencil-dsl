"""Unified benchmark runner: all frameworks x all stencils x all domains.

Usage:
    python -m benchmarks.runner                         # cuTile only
    python -m benchmarks.runner --all                   # all baselines
    python -m benchmarks.runner --baselines cupy jax    # specific baselines
    python -m benchmarks.runner --output results.json   # custom output
    python -m benchmarks.runner --scaling 1 2 3 4 5     # multi-GPU scaling
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

from benchmarks.stencils import STENCIL_META, DOMAINS, full_shape, interior_size
from cutile import stencil, compile as stencil_compile


def gpu_info() -> dict:
    """Detect GPU and measure peak DRAM bandwidth."""
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode()
    n_gpus = cp.cuda.runtime.getDeviceCount()

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

    return {"name": name, "n_gpus": n_gpus, "peak_bw_gbs": peak_bw}


def bench_cutile_stencil(name: str, domain: tuple[int, ...],
                         warmup: int = 30, iters: int = 100) -> dict:
    """Compile and benchmark a cuTile stencil."""
    meta = STENCIL_META[name]
    sfn = meta["cutile_fn"]
    result = stencil_compile(sfn, domain=domain, temporal_blocking=False)
    mod = result.load_module()
    launch = getattr(mod, f"launch_{result.name}")
    shape = full_shape(domain, meta["halo"])

    u = cp.random.randn(*shape).astype(cp.float64)
    out = cp.zeros_like(u)
    for _ in range(warmup):
        launch(u, out)
    cp.cuda.Device(0).synchronize()

    e1, e2 = cp.cuda.Event(), cp.cuda.Event()
    e1.record()
    for _ in range(iters):
        launch(u, out)
    e2.record()
    e2.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(e1, e2) / iters
    npts = interior_size(domain)
    gps = npts / elapsed_ms * 1e-6
    gbytes = npts * (meta["loads_per_point"] + meta["stores_per_point"]) * meta["dtype_bytes"] / elapsed_ms * 1e-6

    return {
        "framework": "cuTile-DSL",
        "time_ms": elapsed_ms,
        "gpoints_per_s": gps,
        "gbytes_per_s": gbytes,
        "tile_sizes": list(result.tile_sizes),
        "temporal_steps": result.temporal_steps,
    }


def bench_cutile_graph(name: str, domain: tuple[int, ...],
                       warmup: int = 20, iters: int = 200,
                       n_per_graph: int = 50) -> dict:
    """Bench a cuTile stencil via CUDA-Graph captured loops.

    Captures *n_per_graph* alternating launches into a graph, then
    measures throughput by replaying the graph. This amortises the
    ~5-10 µs of per-launch host overhead over many iterations and
    restores DRAM-bound throughput at small domains.
    """
    from cutile.runtime.graph_helpers import capture_loop

    meta = STENCIL_META[name]
    sfn = meta["cutile_fn"]
    result = stencil_compile(sfn, domain=domain, temporal_blocking=False)
    mod = result.load_module()
    launch = getattr(mod, f"launch_{result.name}")
    shape = full_shape(domain, meta["halo"])

    u = cp.random.randn(*shape).astype(cp.float64)
    out = cp.zeros_like(u)
    captured = capture_loop(launch, u, out, n_iters=n_per_graph)

    for _ in range(warmup):
        captured.replay()
    captured.synchronize()

    n_replays = max(iters // n_per_graph, 1)
    e1, e2 = cp.cuda.Event(), cp.cuda.Event()
    e1.record(captured.stream)
    captured.replay(n_replays)
    e2.record(captured.stream)
    e2.synchronize()
    total_iters = n_replays * n_per_graph
    elapsed_ms = cp.cuda.get_elapsed_time(e1, e2) / total_iters

    npts = interior_size(domain)
    gps = npts / elapsed_ms * 1e-6
    gbytes = npts * (meta["loads_per_point"] + meta["stores_per_point"]) * meta["dtype_bytes"] / elapsed_ms * 1e-6

    return {
        "framework": "cuTile-Graph",
        "time_ms": elapsed_ms,
        "gpoints_per_s": gps,
        "gbytes_per_s": gbytes,
        "n_per_graph": n_per_graph,
        "tile_sizes": list(result.tile_sizes),
    }


def bench_cutile_multigpu(name: str, domain: tuple[int, ...], num_gpus: int,
                          warmup: int = 20, iters: int = 50) -> dict:
    """Benchmark cuTile with multi-GPU decomposition."""
    meta = STENCIL_META[name]
    sfn = meta["cutile_fn"]
    result = stencil_compile(sfn, domain=domain, num_gpus=num_gpus, temporal_blocking=False)
    mod = result.load_module()
    setup = getattr(mod, f"setup_multigpu_{result.name}")
    step = getattr(mod, f"step_multigpu_{result.name}")
    shape = full_shape(domain, meta["halo"])

    u = cp.random.randn(*shape).astype(cp.float64)
    p_in, p_out = setup(u, num_gpus=num_gpus)

    for _ in range(warmup):
        step(p_in, p_out, num_gpus=num_gpus)
        p_in, p_out = p_out, p_in
    for g in range(num_gpus):
        cp.cuda.Device(g).synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        step(p_in, p_out, num_gpus=num_gpus)
        p_in, p_out = p_out, p_in
    for g in range(num_gpus):
        cp.cuda.Device(g).synchronize()
    elapsed_ms = (time.perf_counter() - t0) / iters * 1000

    npts = interior_size(domain)
    gps = npts / elapsed_ms * 1e-6

    return {
        "framework": f"cuTile-{num_gpus}GPU",
        "time_ms": elapsed_ms,
        "gpoints_per_s": gps,
        "num_gpus": num_gpus,
    }


def _run_baseline(baseline_name: str, stencil_name: str, shape: tuple[int, ...],
                  domain: tuple[int, ...], warmup: int, iters: int) -> dict | None:
    """Run a single baseline, return result dict or None on error."""
    try:
        if baseline_name == "cupy":
            from benchmarks.baselines.cupy_baseline import bench_cupy
            return bench_cupy(stencil_name, shape, warmup, iters)
        elif baseline_name == "jax":
            from benchmarks.baselines.jax_baseline import bench_jax
            return bench_jax(stencil_name, shape, warmup, iters)
        elif baseline_name == "devito":
            from benchmarks.baselines.devito_baseline import bench_devito
            return bench_devito(stencil_name, domain, warmup, iters)
        elif baseline_name == "handwritten":
            from benchmarks.baselines.handwritten_cutile import bench_handwritten
            return bench_handwritten(stencil_name, shape, warmup, iters)
        elif baseline_name == "cuda":
            from benchmarks.baselines.cuda_rawkernel import bench_cuda
            return bench_cuda(stencil_name, shape, variant="naive", warmup=warmup, iters=iters)
        elif baseline_name == "cuda_smem":
            from benchmarks.baselines.cuda_rawkernel import bench_cuda
            return bench_cuda(stencil_name, shape, variant="smem", warmup=warmup, iters=iters)
    except Exception as e:
        return {"framework": baseline_name, "error": str(e)}
    return None


def run_benchmarks(
    stencils: list[str] | None = None,
    baselines: list[str] | None = None,
    scaling_gpus: list[int] | None = None,
    warmup: int = 30,
    iters: int = 100,
    graph_n: int | None = None,
) -> dict:
    """Run the full benchmark suite. Returns structured results dict."""
    if stencils is None:
        stencils = list(STENCIL_META.keys())
    if baselines is None:
        baselines = []

    gpu = gpu_info()
    print(f"GPU: {gpu['name']} ({gpu['n_gpus']} available)")
    print(f"Peak DRAM bandwidth: {gpu['peak_bw_gbs']:.0f} GB/s")
    print()

    all_results = []

    for sname in stencils:
        meta = STENCIL_META[sname]
        domains = DOMAINS[meta["ndim"]]

        for domain in domains:
            shape = full_shape(domain, meta["halo"])
            data_bytes = math.prod(shape) * meta["dtype_bytes"]
            if data_bytes > 8e9:
                print(f"  SKIP {sname} {domain} (>{data_bytes / 1e9:.1f} GB)")
                continue

            print(f"  {sname} domain={domain} ... ", end="", flush=True)

            row = {
                "stencil": sname,
                "ndim": meta["ndim"],
                "domain": list(domain),
                "npoints": interior_size(domain),
            }

            # cuTile-DSL
            try:
                ct_result = bench_cutile_stencil(sname, domain, warmup, iters)
                row["cutile"] = ct_result
                print(f"cuTile={ct_result['gpoints_per_s']:.2f} GP/s", end=" ")
            except Exception as e:
                row["cutile_error"] = str(e)
                print("cuTile=ERR", end=" ")

            # cuTile + CUDA Graph (amortises per-launch host overhead)
            if graph_n:
                try:
                    g_result = bench_cutile_graph(
                        sname, domain, warmup=warmup, iters=iters,
                        n_per_graph=graph_n,
                    )
                    row["cutile_graph"] = g_result
                    print(f"graph={g_result['gpoints_per_s']:.2f} GP/s",
                          end=" ")
                except Exception as e:
                    row["cutile_graph_error"] = str(e)
                    print("graph=ERR", end=" ")

            # Baselines
            for bl in baselines:
                bl_result = _run_baseline(bl, sname, shape, domain, warmup, iters)
                if bl_result:
                    row[bl] = bl_result
                    if "error" not in bl_result:
                        print(f"{bl}={bl_result['gpoints_per_s']:.2f} GP/s", end=" ")
                    else:
                        print(f"{bl}=ERR", end=" ")

            # Multi-GPU scaling. ng=1 reuses the single-GPU result (the
            # multi-GPU codegen path requires num_gpus>=2).
            if scaling_gpus and meta["ndim"] >= 2:
                row["scaling"] = []
                for ng in scaling_gpus:
                    if ng > gpu["n_gpus"]:
                        continue
                    if ng == 1 and "cutile" in row and "gpoints_per_s" in row["cutile"]:
                        row["scaling"].append({
                            "framework": "cuTile-1GPU",
                            "time_ms": row["cutile"]["time_ms"],
                            "gpoints_per_s": row["cutile"]["gpoints_per_s"],
                            "num_gpus": 1,
                        })
                        continue
                    try:
                        sg = bench_cutile_multigpu(sname, domain, ng, warmup=20, iters=50)
                        row["scaling"].append(sg)
                    except Exception as e:
                        row["scaling"].append({"num_gpus": ng, "error": str(e)})

            print()
            all_results.append(row)

    return {
        "gpu": gpu,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warmup": warmup,
        "iters": iters,
        "baselines": baselines,
        "results": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="cuTile Stencil DSL benchmark suite")
    parser.add_argument("--all", action="store_true", help="Run all baselines")
    parser.add_argument("--baselines", nargs="+", default=[],
                        choices=["cupy", "jax", "devito", "handwritten", "cuda", "cuda_smem"],
                        help="Specific baselines to run")
    parser.add_argument("--stencils", nargs="+", default=None,
                        help="Specific stencils to benchmark")
    parser.add_argument("--scaling", nargs="+", type=int, default=None,
                        help="GPU counts for scaling study (e.g., 1 2 3 4 5)")
    parser.add_argument("--output", default="benchmarks/results/results.json",
                        help="Output JSON file")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--graph", type=int, metavar="N", default=None,
                        help="Also bench cuTile via CUDA-Graph capture, "
                             "recording N kernel launches per graph "
                             "(amortises ~5-10 us host overhead).")

    args = parser.parse_args()

    baselines = args.baselines
    if args.all:
        baselines = ["cupy", "jax", "handwritten", "cuda", "cuda_smem"]

    results = run_benchmarks(
        stencils=args.stencils,
        baselines=baselines,
        scaling_gpus=args.scaling,
        warmup=args.warmup,
        iters=args.iters,
        graph_n=args.graph,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
