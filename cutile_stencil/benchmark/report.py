"""Benchmark reporting: tables, CSV, JSON output."""

from __future__ import annotations

import csv
import io
import json
from typing import List, Optional

from cutile_stencil.benchmark.runner import BenchmarkResult


class BenchmarkReport:
    """Collects benchmark results and produces formatted reports.

    Parameters
    ----------
    results : list of BenchmarkResult
        Results to include in the report.
    """

    def __init__(self, results: List[BenchmarkResult] | None = None):
        self.results: List[BenchmarkResult] = results or []

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def to_table(self, title: str = "Benchmark Results") -> str:
        """Format results as an ASCII table.

        Uses tabulate if available, otherwise a simple fixed-width format.
        """
        headers = ["Name", "Backend", "Grid", "Dtype", "Iters",
                    "Time(s)", "Gpts/s", "GB/s", "ms/iter"]

        rows = []
        for r in self.results:
            grid_str = "x".join(str(g) for g in r.grid_size)
            rows.append([
                r.name,
                r.backend,
                grid_str,
                r.dtype,
                r.iterations,
                f"{r.elapsed_seconds:.4f}",
                f"{r.gpoints_per_sec:.4f}",
                f"{r.gbytes_per_sec:.2f}",
                f"{r.ms_per_iteration:.2f}",
            ])

        try:
            from tabulate import tabulate
            table = tabulate(rows, headers=headers, tablefmt="grid")
            return f"{title}\n{table}"
        except ImportError:
            # Simple fallback
            lines = [title, "=" * len(title)]
            col_widths = [max(len(h), max((len(str(row[i])) for row in rows), default=0))
                          for i, h in enumerate(headers)]
            header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
            lines.append(header_line)
            lines.append("-" * len(header_line))
            for row in rows:
                lines.append("  ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
            return "\n".join(lines)

    def to_csv(self) -> str:
        """Export results as CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["name", "backend", "grid_size", "dtype", "iterations",
                          "elapsed_seconds", "gpoints_per_sec", "gbytes_per_sec",
                          "flops_per_sec", "ms_per_iteration"])
        for r in self.results:
            grid_str = "x".join(str(g) for g in r.grid_size)
            writer.writerow([
                r.name, r.backend, grid_str, r.dtype, r.iterations,
                r.elapsed_seconds, r.gpoints_per_sec, r.gbytes_per_sec,
                r.flops_per_sec, r.ms_per_iteration,
            ])
        return output.getvalue()

    def to_json(self, indent: int = 2) -> str:
        """Export results as JSON string."""
        data = []
        for r in self.results:
            data.append({
                "name": r.name,
                "backend": r.backend,
                "grid_size": list(r.grid_size),
                "dtype": r.dtype,
                "iterations": r.iterations,
                "elapsed_seconds": r.elapsed_seconds,
                "gpoints_per_sec": r.gpoints_per_sec,
                "gbytes_per_sec": r.gbytes_per_sec,
                "flops_per_sec": r.flops_per_sec,
                "ms_per_iteration": r.ms_per_iteration,
            })
        return json.dumps(data, indent=indent)

    def roofline_comparison(self, prediction: dict) -> str:
        """Compare measured throughput against roofline prediction."""
        lines = ["Roofline Comparison"]
        lines.append("=" * 40)
        lines.append(f"  Theoretical peak: {prediction['peak_gpoints_s']:.4f} Gpts/s")
        lines.append(f"  Bound: {prediction['bound']}")
        lines.append(f"  Arithmetic intensity: {prediction['arithmetic_intensity']:.3f} FLOPs/byte")
        lines.append("")
        for r in self.results:
            pct = (r.gpoints_per_sec / prediction['peak_gpoints_s']) * 100 if prediction['peak_gpoints_s'] > 0 else 0
            lines.append(f"  {r.name} ({r.backend}, {r.dtype}): "
                         f"{r.gpoints_per_sec:.4f} Gpts/s ({pct:.1f}% of peak)")
        return "\n".join(lines)
