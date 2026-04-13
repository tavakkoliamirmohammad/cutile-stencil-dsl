"""Run cuTile benchmarks and generate report."""

from cutile.benchmarks.runner import BenchmarkReport
from cutile.benchmarks.bench_cutile import run_all_benchmarks as bench_cutile


def run_comparison():
    report = BenchmarkReport()

    print("=" * 60)
    print("Benchmarking cuTile-DSL")
    print("=" * 60)
    cutile_report = bench_cutile()
    for r in cutile_report.results:
        report.add(r)

    print("\n\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(report.to_table())
    report.to_json("benchmark_results.json")

    return report


if __name__ == "__main__":
    run_comparison()
