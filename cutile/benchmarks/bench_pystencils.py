"""Benchmark PyStencils stencil kernels for comparison."""

import time

from cutile.benchmarks.runner import BenchmarkResult, BenchmarkReport


def _has_pystencils():
    try:
        import pystencils  # noqa: F401
        return True
    except ImportError:
        return False


def bench_pystencils_heat_2d(domain, warmup=10, iters=50):
    """Benchmark 2D heat equation in PyStencils."""
    try:
        import pystencils as ps
        import numpy as np

        src = np.random.rand(*domain).astype(np.float64)
        dst = np.zeros_like(src)

        src_field = ps.fields(f"src: float64[{domain[0]}, {domain[1]}]")
        dst_field = ps.fields(f"dst: float64[{domain[0]}, {domain[1]}]")

        update = ps.Assignment(
            dst_field.center,
            0.25 * (src_field[1, 0] + src_field[-1, 0]
                    + src_field[0, 1] + src_field[0, -1])
        )
        kernel = ps.create_kernel(update).compile()

        # Warmup
        for _ in range(warmup):
            kernel(src=src, dst=dst)

        # Benchmark
        start = time.perf_counter()
        for _ in range(iters):
            kernel(src=src, dst=dst)
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
        print("PyStencils not installed. Skipping.")
        return None
    except Exception as e:
        print(f"PyStencils heat_2d failed: {e}")
        return None


def bench_pystencils_laplacian_2d(domain, warmup=10, iters=50):
    """Benchmark 2D 5-point Laplacian in PyStencils."""
    try:
        import pystencils as ps
        import numpy as np

        src = np.random.rand(*domain).astype(np.float64)
        dst = np.zeros_like(src)

        src_field = ps.fields(f"src: float64[{domain[0]}, {domain[1]}]")
        dst_field = ps.fields(f"dst: float64[{domain[0]}, {domain[1]}]")

        update = ps.Assignment(
            dst_field.center,
            src_field[1, 0] + src_field[-1, 0]
            + src_field[0, 1] + src_field[0, -1]
            - 4 * src_field.center
        )
        kernel = ps.create_kernel(update).compile()

        # Warmup
        for _ in range(warmup):
            kernel(src=src, dst=dst)

        # Benchmark
        start = time.perf_counter()
        for _ in range(iters):
            kernel(src=src, dst=dst)
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
        print("PyStencils not installed. Skipping.")
        return None
    except Exception as e:
        print(f"PyStencils laplacian_2d failed: {e}")
        return None


def bench_pystencils_laplacian_3d(domain, warmup=10, iters=50):
    """Benchmark 3D 7-point Laplacian in PyStencils."""
    try:
        import pystencils as ps
        import numpy as np

        src = np.random.rand(*domain).astype(np.float64)
        dst = np.zeros_like(src)

        src_field = ps.fields(
            f"src: float64[{domain[0]}, {domain[1]}, {domain[2]}]"
        )
        dst_field = ps.fields(
            f"dst: float64[{domain[0]}, {domain[1]}, {domain[2]}]"
        )

        update = ps.Assignment(
            dst_field.center,
            src_field[1, 0, 0] + src_field[-1, 0, 0]
            + src_field[0, 1, 0] + src_field[0, -1, 0]
            + src_field[0, 0, 1] + src_field[0, 0, -1]
            - 6 * src_field.center
        )
        kernel = ps.create_kernel(update).compile()

        # Warmup
        for _ in range(warmup):
            kernel(src=src, dst=dst)

        # Benchmark
        start = time.perf_counter()
        for _ in range(iters):
            kernel(src=src, dst=dst)
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
        print("PyStencils not installed. Skipping.")
        return None
    except Exception as e:
        print(f"PyStencils laplacian_3d failed: {e}")
        return None


_PYSTENCILS_BENCHMARKS = {
    "heat_2d": {
        "fn": bench_pystencils_heat_2d,
        "ndim": 2,
        "name": "heat_2d",
    },
    "laplacian_2d_5pt": {
        "fn": bench_pystencils_laplacian_2d,
        "ndim": 2,
        "name": "laplacian_2d_5pt",
    },
    "laplacian_3d_7pt": {
        "fn": bench_pystencils_laplacian_3d,
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
    """Run all PyStencils benchmarks and return a report."""
    if not _has_pystencils():
        print("PyStencils not installed. Skipping all PyStencils benchmarks.")
        return BenchmarkReport()

    report = BenchmarkReport()
    for key, spec in _PYSTENCILS_BENCHMARKS.items():
        ndim = spec["ndim"]
        name = spec["name"]
        bench_fn = spec["fn"]
        domains = _DOMAINS.get(ndim, [])

        print(f"\nBenchmarking PyStencils {name}...")
        for domain in domains:
            metrics = bench_fn(domain)
            if metrics is not None:
                dom_str = "x".join(str(d) for d in domain)
                print(f"  {dom_str}: {metrics['gpoints_per_s']:.2f} GPoints/s, "
                      f"{metrics['time_ms']:.3f} ms")
                report.add(BenchmarkResult(
                    framework="PyStencils",
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
