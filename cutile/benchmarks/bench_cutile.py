"""Benchmark cuTile-generated stencil kernels."""

import numpy as np

from cutile import compile as stencil_compile
from cutile.benchmarks.runner import BenchmarkResult, BenchmarkReport
from cutile.benchmarks.stencils import (
    heat_1d, laplacian_2d_5pt, laplacian_2d_9pt, laplacian_3d_7pt, heat_2d,
    BENCHMARK_DOMAINS,
)


def bench_stencil(stencil_fn, domains=None, warmup=10, iters=50):
    """Benchmark a stencil on various domain sizes."""
    results = []
    ndim = stencil_fn._ndim
    if domains is None:
        domains = BENCHMARK_DOMAINS.get(ndim, [(256,) * ndim])

    for domain in domains:
        try:
            result = stencil_compile(stencil_fn, domain=domain)
            hw = result.halo_widths
            # Build a full-size input array (domain + halo on each side)
            full_shape = tuple(d + 2 * h for d, h in zip(domain, hw))
            u_input = np.random.rand(*full_shape)
            metrics = result.benchmark(u_input=u_input, warmup=warmup, iters=iters)
            dom_str = "x".join(str(d) for d in domain)
            print(f"  {dom_str}: {metrics['gpoints_per_s']:.2f} GPoints/s, "
                  f"{metrics['time_ms']:.3f} ms")
            results.append(BenchmarkResult(
                framework="cuTile-DSL",
                stencil_name=result.name,
                ndim=ndim,
                domain=domain,
                gpoints_per_s=metrics.get("gpoints_per_s", 0),
                gbytes_per_s=metrics.get("gbytes_per_s", 0),
                time_ms=metrics.get("time_ms", 0),
                temporal_steps=result.temporal_steps,
            ))
        except Exception as e:
            print(f"  Benchmark failed for {domain}: {e}")

    return results


def run_all_benchmarks():
    report = BenchmarkReport()
    stencils = [heat_1d, heat_2d, laplacian_2d_5pt, laplacian_3d_7pt]

    for s in stencils:
        print(f"\nBenchmarking {s._fn.__name__}...")
        results = bench_stencil(s)
        for r in results:
            report.add(r)

    return report


if __name__ == "__main__":
    report = run_all_benchmarks()
    print("\n" + report.to_table())
    report.to_json("benchmark_results.json")
