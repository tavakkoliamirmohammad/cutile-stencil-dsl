"""Benchmark runner for stencil DSL comparison."""

import time
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class BenchmarkResult:
    framework: str
    stencil_name: str
    ndim: int
    domain: tuple
    gpoints_per_s: float
    gbytes_per_s: float
    time_ms: float
    temporal_steps: int = 1
    num_gpus: int = 1

    def to_dict(self):
        return asdict(self)


@dataclass
class BenchmarkReport:
    results: list[BenchmarkResult] = field(default_factory=list)

    def add(self, result: BenchmarkResult):
        self.results.append(result)

    def to_table(self) -> str:
        """Format results as a markdown table."""
        if not self.results:
            return "No results"
        header = "| Framework | Stencil | Domain | GPoints/s | GBytes/s | Time(ms) | T |"
        sep =    "|-----------|---------|--------|-----------|----------|----------|---|"
        rows = [header, sep]
        for r in self.results:
            dom = "x".join(str(d) for d in r.domain)
            rows.append(f"| {r.framework:9s} | {r.stencil_name:7s} | {dom:6s} | "
                       f"{r.gpoints_per_s:9.2f} | {r.gbytes_per_s:8.2f} | "
                       f"{r.time_ms:8.3f} | {r.temporal_steps} |")
        return "\n".join(rows)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)

    def scaling_table(self) -> str:
        """Multi-GPU scaling table."""
        if not self.results:
            return "No results"
        # Group by (framework, stencil_name, domain) and show scaling across num_gpus
        from collections import defaultdict
        groups = defaultdict(list)
        for r in self.results:
            key = (r.framework, r.stencil_name, r.domain)
            groups[key].append(r)

        header = "| Framework | Stencil | Domain | GPUs | GPoints/s | Speedup |"
        sep =    "|-----------|---------|--------|------|-----------|---------|"
        rows = [header, sep]
        for key, group in sorted(groups.items()):
            group.sort(key=lambda r: r.num_gpus)
            base = group[0].gpoints_per_s if group[0].gpoints_per_s > 0 else 1.0
            for r in group:
                dom = "x".join(str(d) for d in r.domain)
                speedup = r.gpoints_per_s / base if base > 0 else 0.0
                rows.append(f"| {r.framework:9s} | {r.stencil_name:7s} | {dom:6s} | "
                           f"{r.num_gpus:4d} | {r.gpoints_per_s:9.2f} | "
                           f"{speedup:7.2f} |")
        return "\n".join(rows)
