"""Run all benchmarks and generate comparison report."""

from cutile.benchmarks.runner import BenchmarkReport
from cutile.benchmarks.bench_cutile import run_all_benchmarks as bench_cutile


def run_comparison():
    report = BenchmarkReport()

    # cuTile benchmarks
    print("=" * 60)
    print("Benchmarking cuTile-DSL")
    print("=" * 60)
    cutile_report = bench_cutile()
    for r in cutile_report.results:
        report.add(r)

    # Try Devito
    try:
        from cutile.benchmarks.bench_devito import run_all_benchmarks as bench_devito
        print("\n" + "=" * 60)
        print("Benchmarking Devito")
        print("=" * 60)
        devito_report = bench_devito()
        for r in devito_report.results:
            report.add(r)
    except ImportError:
        print("\nDevito not installed, skipping.")

    # Try PyStencils
    try:
        from cutile.benchmarks.bench_pystencils import run_all_benchmarks as bench_pystencils
        print("\n" + "=" * 60)
        print("Benchmarking PyStencils")
        print("=" * 60)
        pystencils_report = bench_pystencils()
        for r in pystencils_report.results:
            report.add(r)
    except ImportError:
        print("\nPyStencils not installed, skipping.")

    print("\n\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(report.to_table())
    report.to_json("benchmark_comparison.json")

    return report


if __name__ == "__main__":
    run_comparison()
