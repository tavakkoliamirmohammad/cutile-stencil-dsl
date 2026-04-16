"""Compare two ``results.json`` runs and flag throughput regressions.

Use after a refactor or perf-affecting change: run the benchmark, save
results.json, make the change, run again, then ::

    python -m benchmarks.compare_runs old.json new.json
    python -m benchmarks.compare_runs old.json new.json --threshold 5

For each (stencil, domain, framework) tuple present in both files,
prints the GP/s before / after / delta. Rows whose delta exceeds the
threshold (default 10%) are flagged as regressions or wins.
Multi-GPU scaling rows are compared too.

Exit code is non-zero if any framework regresses by more than the
threshold — handy for CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_FRAMEWORKS = (
    "cutile", "cupy", "jax", "devito", "handwritten", "cuda", "cuda_smem",
)


def _index(data: dict) -> dict:
    """Map (stencil, tuple(domain), framework) -> gpoints_per_s."""
    out: dict = {}
    for row in data["results"]:
        key_base = (row["stencil"], tuple(row["domain"]))
        for fw in _FRAMEWORKS:
            entry = row.get(fw)
            if isinstance(entry, dict) and "error" not in entry \
                    and "gpoints_per_s" in entry:
                out[key_base + (fw,)] = entry["gpoints_per_s"]
        # Multi-GPU scaling rows
        for s in row.get("scaling", []) or []:
            if "error" in s:
                continue
            ng = s["num_gpus"]
            out[key_base + (f"scaling_{ng}gpu",)] = s["gpoints_per_s"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("old", type=Path, help="baseline results.json")
    parser.add_argument("new", type=Path, help="new results.json")
    parser.add_argument(
        "--threshold", type=float, default=10.0,
        help="Flag rows whose change exceeds +/- this %% (default 10)",
    )
    parser.add_argument(
        "--framework", default=None,
        help="Filter to a single framework (e.g. cutile, scaling_2gpu)",
    )
    args = parser.parse_args()

    old = _index(json.loads(args.old.read_text()))
    new = _index(json.loads(args.new.read_text()))

    common = sorted(set(old) & set(new))
    if args.framework:
        common = [k for k in common if k[2] == args.framework]
    if not common:
        print("No comparable rows found.", file=sys.stderr)
        return 1

    print(f"Comparing  old: {args.old}  ->  new: {args.new}")
    print(f"Threshold: {args.threshold:+.0f}%")
    print()
    print(f"{'stencil':22s} {'domain':16s} {'framework':14s} "
          f"{'old GP/s':>10s} {'new GP/s':>10s} {'delta':>9s}  flag")
    print("-" * 92)

    regressions: list[tuple] = []
    wins: list[tuple] = []
    for key in common:
        sname, dom, fw = key
        a = old[key]; b = new[key]
        if a <= 0:
            pct = 0.0
        else:
            pct = (b - a) / a * 100.0
        flag = ""
        if pct < -args.threshold:
            flag = "REGRESSION"
            regressions.append((key, a, b, pct))
        elif pct > args.threshold:
            flag = "win"
            wins.append((key, a, b, pct))
        print(f"{sname:22s} {str(dom):16s} {fw:14s} "
              f"{a:>10.2f} {b:>10.2f} {pct:>+8.1f}%  {flag}")

    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))
    if only_old or only_new:
        print()
        if only_old:
            print(f"Only in old ({len(only_old)}): "
                  f"{', '.join(f'{s}/{d}/{f}' for s, d, f in only_old[:5])}"
                  f"{'...' if len(only_old) > 5 else ''}")
        if only_new:
            print(f"Only in new ({len(only_new)}): "
                  f"{', '.join(f'{s}/{d}/{f}' for s, d, f in only_new[:5])}"
                  f"{'...' if len(only_new) > 5 else ''}")

    print()
    print(f"Regressions: {len(regressions)}   Wins: {len(wins)}   "
          f"Unchanged (within {args.threshold:.0f}%): "
          f"{len(common) - len(regressions) - len(wins)}")

    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
