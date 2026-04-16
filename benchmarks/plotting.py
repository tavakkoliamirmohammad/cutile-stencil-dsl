"""Publication-quality figure generation from benchmark results.

Usage:
    python -m benchmarks.plotting benchmarks/results/results.json --outdir paper/figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.plot_config import (
    COLORS, MARKERS, STENCIL_LABELS, FONT_SIZES,
    FIG_SINGLE_COL, FIG_DOUBLE_COL, FIG_ROOFLINE,
)
from benchmarks.stencils import STENCIL_META, arithmetic_intensity


def _setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_SIZES["label"],
        "axes.titlesize": FONT_SIZES["title"],
        "axes.labelsize": FONT_SIZES["label"],
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
        "legend.fontsize": FONT_SIZES["legend"],
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def plot_roofline(data: dict, outdir: Path):
    """Roofline: achieved performance vs DRAM-bound arithmetic intensity.

    Two ceilings: peak DRAM BW (STREAM-measured) and an estimated L2 BW.
    Cache-resident workloads (working set < L2 capacity) can land between
    the two ceilings; we mark those points to make the regime explicit.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=FIG_ROOFLINE)

    peak_bw = data["gpu"]["peak_bw_gbs"]
    peak_gflops = 2000.0
    # Blackwell L2 BW is roughly 5x DRAM (architecture rule of thumb);
    # gives a soft upper bound for cache-resident regimes.
    l2_bw = peak_bw * 5.0
    l2_capacity_bytes = 96 * 1024 * 1024  # RTX PRO 6000 Blackwell

    ai_range = np.logspace(-2, 2, 200)
    dram_roof = np.minimum(peak_bw * ai_range, peak_gflops)
    l2_roof = np.minimum(l2_bw * ai_range, peak_gflops)
    ax.loglog(ai_range, l2_roof, "k--", linewidth=0.8,
              label=f"L2 (~{l2_bw / 1000:.1f} TB/s)", zorder=1)
    ax.loglog(ai_range, dram_roof, "k-", linewidth=1.5,
              label=f"DRAM ({peak_bw:.0f} GB/s)", zorder=1)
    ax.fill_between(ai_range, dram_roof, alpha=0.05, color="gray")

    cache_resident_pts = []
    dram_pts = []
    for row in data["results"]:
        sname = row["stencil"]
        if "cutile" not in row or "error" in row.get("cutile", {}):
            continue
        ai = arithmetic_intensity(sname)
        gflops = row["cutile"]["gpoints_per_s"] * STENCIL_META[sname]["flops_per_point"]
        npts = row["npoints"]
        # Working set: 2 arrays (in + out) at dtype_bytes per point
        ws = 2 * npts * STENCIL_META[sname]["dtype_bytes"]
        target = cache_resident_pts if ws < l2_capacity_bytes else dram_pts
        target.append((ai, gflops, npts))

    if dram_pts:
        x, y, n = zip(*dram_pts)
        ax.scatter(x, y, c=COLORS["cuTile-DSL"], marker="o",
                   s=[20 + ni / 1e5 for ni in n],
                   edgecolors="black", linewidths=0.3, zorder=3,
                   label="cuTile (DRAM-resident)")
    if cache_resident_pts:
        x, y, n = zip(*cache_resident_pts)
        ax.scatter(x, y, c=COLORS["cuTile-DSL"], marker="o",
                   s=[20 + ni / 1e5 for ni in n],
                   edgecolors="black", linewidths=0.3, zorder=3,
                   alpha=0.45, label="cuTile (L2-resident)")

    ax.set_xlabel("Arithmetic Intensity (FLOPs/Byte)")
    ax.set_ylabel("Performance (GFLOPS)")
    ax.set_title("Roofline Analysis")
    ax.set_xlim(0.01, 100)
    ax.legend(loc="lower right", fontsize=6)
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)

    fig.savefig(outdir / "roofline.pdf")
    fig.savefig(outdir / "roofline.png")
    plt.close(fig)
    print("  Saved roofline.pdf")


def plot_throughput_vs_size(data: dict, outdir: Path):
    """Throughput (GP/s) vs domain size for each stencil, all frameworks."""
    _setup_style()

    by_stencil: dict[str, list] = {}
    for row in data["results"]:
        by_stencil.setdefault(row["stencil"], []).append(row)

    for sname, rows in by_stencil.items():
        fig, ax = plt.subplots(figsize=FIG_SINGLE_COL)
        rows.sort(key=lambda r: r["npoints"])

        # cuTile-DSL
        ct_sizes = [r["npoints"] for r in rows
                     if "cutile" in r and "error" not in r.get("cutile", {})]
        ct_gps = [r["cutile"]["gpoints_per_s"] for r in rows
                   if "cutile" in r and "error" not in r.get("cutile", {})]
        if ct_gps:
            ax.semilogx(ct_sizes, ct_gps, "-o", color=COLORS["cuTile-DSL"],
                        label="cuTile-DSL", markersize=4)

        # Baselines
        for fw_key, display in [("cupy", "CuPy"), ("jax", "JAX"),
                                 ("devito", "Devito"), ("handwritten", "Hand-cuTile"),
                                 ("cuda", "CUDA-naive"), ("cuda_smem", "CUDA-smem")]:
            fw_sizes, fw_gps = [], []
            for r in rows:
                if fw_key in r and isinstance(r[fw_key], dict) and "error" not in r[fw_key]:
                    fw_sizes.append(r["npoints"])
                    fw_gps.append(r[fw_key]["gpoints_per_s"])
            if fw_gps:
                ax.semilogx(fw_sizes, fw_gps,
                            f"-{MARKERS.get(display, 'x')}",
                            color=COLORS.get(display, "gray"),
                            label=display, markersize=4)

        ax.set_xlabel("Domain Size (points)")
        ax.set_ylabel("Throughput (GP/s)")
        ax.set_title(STENCIL_LABELS.get(sname, sname))
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        fig.savefig(outdir / f"throughput_{sname}.pdf")
        fig.savefig(outdir / f"throughput_{sname}.png")
        plt.close(fig)
        print(f"  Saved throughput_{sname}.pdf")


def plot_speedup_bars(data: dict, outdir: Path):
    """Speedup of cuTile-DSL over each baseline (largest domain per stencil)."""
    _setup_style()

    largest: dict[str, dict] = {}
    for row in data["results"]:
        sname = row["stencil"]
        if sname not in largest or row["npoints"] > largest[sname]["npoints"]:
            largest[sname] = row

    stencil_names = list(largest.keys())
    baselines_present = [
        bl for bl in ["cupy", "jax", "devito", "handwritten", "cuda", "cuda_smem"]
        if any(bl in largest[s] for s in stencil_names)
    ]

    if not baselines_present:
        print("  No baselines to plot speedup bars")
        return

    fig, ax = plt.subplots(figsize=FIG_DOUBLE_COL)
    x = np.arange(len(stencil_names))
    width = 0.8 / len(baselines_present)

    display_map = {"cupy": "CuPy", "jax": "JAX",
                   "devito": "Devito", "handwritten": "Hand-cuTile",
                   "cuda": "CUDA-naive", "cuda_smem": "CUDA-smem"}

    for i, bl in enumerate(baselines_present):
        speedups = []
        for sname in stencil_names:
            row = largest[sname]
            ct_gps = row.get("cutile", {}).get("gpoints_per_s", 0)
            bl_data = row.get(bl, {})
            bl_gps = bl_data.get("gpoints_per_s", 0) if isinstance(bl_data, dict) else 0
            speedups.append(ct_gps / bl_gps if bl_gps > 0 and ct_gps > 0 else 0)

        display = display_map[bl]
        ax.bar(x + i * width, speedups, width, label=f"vs {display}",
               color=COLORS.get(display, "gray"), edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Stencil")
    ax.set_ylabel("Speedup (cuTile-DSL / baseline)")
    ax.set_title("Performance Comparison")
    ax.set_xticks(x + width * (len(baselines_present) - 1) / 2)
    ax.set_xticklabels([STENCIL_LABELS.get(s, s) for s in stencil_names],
                       rotation=15, ha="right")
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.5)
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(outdir / "speedup_bars.pdf")
    fig.savefig(outdir / "speedup_bars.png")
    plt.close(fig)
    print("  Saved speedup_bars.pdf")


def plot_scaling(data: dict, outdir: Path):
    """Strong scaling: throughput vs GPU count."""
    _setup_style()

    has_scaling = any("scaling" in r and r["scaling"] for r in data["results"])
    if not has_scaling:
        print("  No scaling data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE_COL)

    largest_scaling: dict[str, dict] = {}
    for row in data["results"]:
        if "scaling" not in row or not row["scaling"]:
            continue
        sname = row["stencil"]
        if sname not in largest_scaling or row["npoints"] > largest_scaling[sname]["npoints"]:
            largest_scaling[sname] = row

    # Left: absolute throughput
    ax = axes[0]
    for sname, row in largest_scaling.items():
        gpus = [sg["num_gpus"] for sg in row["scaling"] if "error" not in sg]
        gps_vals = [sg["gpoints_per_s"] for sg in row["scaling"] if "error" not in sg]
        if gpus:
            ax.plot(gpus, gps_vals, "-o",
                    label=STENCIL_LABELS.get(sname, sname), markersize=4)

    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Throughput (GP/s)")
    ax.set_title("Strong Scaling")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Right: parallel efficiency
    ax = axes[1]
    for sname, row in largest_scaling.items():
        sorted_sg = sorted(
            [sg for sg in row["scaling"] if "error" not in sg],
            key=lambda s: s["num_gpus"],
        )
        if not sorted_sg:
            continue
        base_gps = sorted_sg[0]["gpoints_per_s"]
        gpus = [sg["num_gpus"] for sg in sorted_sg]
        efficiency = [
            sg["gpoints_per_s"] / (base_gps * sg["num_gpus"]) * 100
            for sg in sorted_sg
        ]
        ax.plot(gpus, efficiency, "-o",
                label=STENCIL_LABELS.get(sname, sname), markersize=4)

    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_title("Scaling Efficiency")
    ax.axhline(y=100, color="black", linestyle="--", linewidth=0.5)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.savefig(outdir / "scaling.pdf")
    fig.savefig(outdir / "scaling.png")
    plt.close(fig)
    print("  Saved scaling.pdf")


def plot_bandwidth_utilization(data: dict, outdir: Path):
    """Effective bandwidth as % of peak for each stencil (largest domain)."""
    _setup_style()

    peak_bw = data["gpu"]["peak_bw_gbs"]

    largest: dict[str, dict] = {}
    for row in data["results"]:
        sname = row["stencil"]
        if sname not in largest or row["npoints"] > largest[sname]["npoints"]:
            largest[sname] = row

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL)

    stencil_names = list(largest.keys())
    x = np.arange(len(stencil_names))
    pcts = []
    for sname in stencil_names:
        gbytes = largest[sname].get("cutile", {}).get("gbytes_per_s", 0)
        pcts.append(gbytes / peak_bw * 100 if peak_bw > 0 else 0)

    ax.bar(x, pcts, color=COLORS["cuTile-DSL"], edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Stencil")
    ax.set_ylabel("Bandwidth Utilization (%)")
    ax.set_title("Effective Bandwidth (% of Peak)")
    ax.set_xticks(x)
    ax.set_xticklabels([STENCIL_LABELS.get(s, s) for s in stencil_names],
                       rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(outdir / "bandwidth_util.pdf")
    fig.savefig(outdir / "bandwidth_util.png")
    plt.close(fig)
    print("  Saved bandwidth_util.pdf")


def generate_all_figures(data: dict, outdir: Path):
    """Generate all publication figures from benchmark results."""
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Generating figures in {outdir}/")
    plot_roofline(data, outdir)
    plot_throughput_vs_size(data, outdir)
    plot_speedup_bars(data, outdir)
    plot_scaling(data, outdir)
    plot_bandwidth_utilization(data, outdir)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("input", help="Benchmark results JSON file")
    parser.add_argument("--outdir", default="paper/figures/", help="Output directory")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    generate_all_figures(data, Path(args.outdir))


if __name__ == "__main__":
    main()
