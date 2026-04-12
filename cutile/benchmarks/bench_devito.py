"""Benchmark Devito stencil kernels for comparison."""

import time

from cutile.benchmarks.runner import BenchmarkResult, BenchmarkReport


def _has_devito():
    try:
        import devito  # noqa: F401
        return True
    except ImportError:
        return False


def bench_devito_heat_2d(domain, warmup=10, iters=50):
    """Benchmark 2D heat equation in Devito."""
    try:
        from devito import Grid, TimeFunction, Eq, Operator
        import numpy as np

        grid = Grid(shape=domain, dtype=np.float64)
        u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2)
        eq = Eq(u.forward, 0.25 * (u[u.indices[0]+1, u.indices[1]-1, u.indices[2]]
                                   + u[u.indices[0]+1, u.indices[1]+1, u.indices[2]]
                                   + u[u.indices[0]+1, u.indices[1], u.indices[2]-1]
                                   + u[u.indices[0]+1, u.indices[1], u.indices[2]+1]))
        op = Operator([eq])

        # Warmup
        for _ in range(warmup):
            op.apply(time_M=1)

        # Benchmark
        start = time.perf_counter()
        for _ in range(iters):
            op.apply(time_M=1)
        elapsed = (time.perf_counter() - start) / iters

        npoints = domain[0] * domain[1]
        gpoints = npoints / elapsed / 1e9
        gbytes = npoints * 8 * 2 / elapsed / 1e9  # read + write

        return {
            "gpoints_per_s": gpoints,
            "gbytes_per_s": gbytes,
            "time_ms": elapsed * 1000,
        }
    except ImportError:
        print("Devito not installed. Skipping.")
        return None


def bench_devito_laplacian_2d(domain, warmup=10, iters=50):
    """Benchmark 2D 5-point Laplacian in Devito."""
    try:
        from devito import Grid, TimeFunction, Eq, Operator
        import numpy as np

        grid = Grid(shape=domain, dtype=np.float64)
        u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2)
        t, x, y = u.indices
        eq = Eq(u.forward, u[t+1, x-1, y] + u[t+1, x+1, y]
                          + u[t+1, x, y-1] + u[t+1, x, y+1]
                          - 4*u[t+1, x, y])
        op = Operator([eq])

        # Warmup
        for _ in range(warmup):
            op.apply(time_M=1)

        # Benchmark
        start = time.perf_counter()
        for _ in range(iters):
            op.apply(time_M=1)
        elapsed = (time.perf_counter() - start) / iters

        npoints = domain[0] * domain[1]
        gpoints = npoints / elapsed / 1e9
        gbytes = npoints * 8 * 2 / elapsed / 1e9

        return {
            "gpoints_per_s": gpoints,
            "gbytes_per_s": gbytes,
            "time_ms": elapsed * 1000,
        }
    except ImportError:
        print("Devito not installed. Skipping.")
        return None


def bench_devito_laplacian_3d(domain, warmup=10, iters=50):
    """Benchmark 3D 7-point Laplacian in Devito."""
    try:
        from devito import Grid, TimeFunction, Eq, Operator
        import numpy as np

        grid = Grid(shape=domain, dtype=np.float64)
        u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2)
        t, x, y, z = u.indices
        eq = Eq(u.forward, u[t+1, x-1, y, z] + u[t+1, x+1, y, z]
                          + u[t+1, x, y-1, z] + u[t+1, x, y+1, z]
                          + u[t+1, x, y, z-1] + u[t+1, x, y, z+1]
                          - 6*u[t+1, x, y, z])
        op = Operator([eq])

        # Warmup
        for _ in range(warmup):
            op.apply(time_M=1)

        # Benchmark
        start = time.perf_counter()
        for _ in range(iters):
            op.apply(time_M=1)
        elapsed = (time.perf_counter() - start) / iters

        npoints = domain[0] * domain[1] * domain[2]
        gpoints = npoints / elapsed / 1e9
        gbytes = npoints * 8 * 2 / elapsed / 1e9

        return {
            "gpoints_per_s": gpoints,
            "gbytes_per_s": gbytes,
            "time_ms": elapsed * 1000,
        }
    except ImportError:
        print("Devito not installed. Skipping.")
        return None


_DEVITO_BENCHMARKS = {
    "heat_2d": {
        "fn": bench_devito_heat_2d,
        "ndim": 2,
        "name": "heat_2d",
    },
    "laplacian_2d_5pt": {
        "fn": bench_devito_laplacian_2d,
        "ndim": 2,
        "name": "laplacian_2d_5pt",
    },
    "laplacian_3d_7pt": {
        "fn": bench_devito_laplacian_3d,
        "ndim": 3,
        "name": "laplacian_3d_7pt",
    },
}

# Standard domains matching those in stencils.py
_DOMAINS = {
    2: [(256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)],
    3: [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)],
}


def run_all_benchmarks():
    """Run all Devito benchmarks and return a report."""
    if not _has_devito():
        print("Devito not installed. Skipping all Devito benchmarks.")
        return BenchmarkReport()

    report = BenchmarkReport()
    for key, spec in _DEVITO_BENCHMARKS.items():
        ndim = spec["ndim"]
        name = spec["name"]
        bench_fn = spec["fn"]
        domains = _DOMAINS.get(ndim, [])

        print(f"\nBenchmarking Devito {name}...")
        for domain in domains:
            metrics = bench_fn(domain)
            if metrics is not None:
                dom_str = "x".join(str(d) for d in domain)
                print(f"  {dom_str}: {metrics['gpoints_per_s']:.2f} GPoints/s, "
                      f"{metrics['time_ms']:.3f} ms")
                report.add(BenchmarkResult(
                    framework="Devito",
                    stencil_name=name,
                    ndim=ndim,
                    domain=domain,
                    gpoints_per_s=metrics["gpoints_per_s"],
                    gbytes_per_s=metrics["gbytes_per_s"],
                    time_ms=metrics["time_ms"],
                ))
    return report


if __name__ == "__main__":
    report = run_all_benchmarks()
    print("\n" + report.to_table())
