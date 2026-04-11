"""Tiling visualization: render domain decomposition with halo overlap coloring."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from cutile_stencil.dsl.types import TileConfig, TemporalConfig


def visualize_tiling(
    tile_config: TileConfig,
    domain: Tuple[int, ...],
    halo_widths: Tuple[int, ...] = None,
    temporal_config: TemporalConfig = None,
    ax=None,
):
    """Render domain decomposition with halo overlap coloring.

    Parameters
    ----------
    tile_config : TileConfig
        Tile configuration from analysis.
    domain : tuple of int
        Domain size per dimension.
    halo_widths : tuple of int, optional
        Halo widths. Defaults to tile_config.halo_widths.
    temporal_config : TemporalConfig, optional
        If provided, show expanded halo from temporal blocking.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the visualization.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install cutile-stencil[viz]"
        )

    if halo_widths is None:
        halo_widths = tile_config.halo_widths

    ndim = len(domain)

    if ndim == 1:
        return _visualize_1d(tile_config, domain, halo_widths, temporal_config, ax)
    elif ndim == 2:
        return _visualize_2d(tile_config, domain, halo_widths, temporal_config, ax)
    else:
        # 3D: show a 2D slice at z=0
        tc_2d = TileConfig(
            tile_sizes=tile_config.tile_sizes[:2],
            halo_widths=tile_config.halo_widths[:2],
            num_tiles=tile_config.num_tiles[:2],
            overhead_fraction=tile_config.overhead_fraction,
        )
        domain_2d = domain[:2]
        halo_2d = halo_widths[:2]
        temp_2d = None
        if temporal_config is not None:
            temp_2d = TemporalConfig(
                steps=temporal_config.steps,
                expanded_halo=temporal_config.expanded_halo[:2],
                bandwidth_reduction_factor=temporal_config.bandwidth_reduction_factor,
            )
        fig = _visualize_2d(tc_2d, domain_2d, halo_2d, temp_2d, ax)
        fig.axes[0].set_title(f"3D domain slice (z=0), tile={tile_config.tile_sizes}")
        return fig


def _visualize_1d(tile_config, domain, halo_widths, temporal_config, ax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 2))
    else:
        fig = ax.get_figure()

    ts = tile_config.tile_sizes[0]
    hw = halo_widths[0]
    n_tiles = tile_config.num_tiles[0]
    total = domain[0]

    for t in range(n_tiles):
        x_start = t * ts
        # Tile interior
        rect = mpatches.Rectangle(
            (x_start, 0), ts, 1,
            linewidth=1, edgecolor="black", facecolor="#4a90d9", alpha=0.7,
        )
        ax.add_patch(rect)
        # Left halo
        if x_start - hw >= 0:
            halo_rect = mpatches.Rectangle(
                (x_start - hw, 0), hw, 1,
                linewidth=0.5, edgecolor="red", facecolor="red", alpha=0.2,
            )
            ax.add_patch(halo_rect)
        # Right halo
        if x_start + ts + hw <= total:
            halo_rect = mpatches.Rectangle(
                (x_start + ts, 0), hw, 1,
                linewidth=0.5, edgecolor="red", facecolor="red", alpha=0.2,
            )
            ax.add_patch(halo_rect)

    ax.set_xlim(-hw, total + hw)
    ax.set_ylim(-0.2, 1.5)
    ax.set_xlabel("Domain position")
    ax.set_title(
        f"1D tiling: {n_tiles} tiles of size {ts}, halo={hw}, "
        f"overhead={tile_config.overhead_fraction:.1%}"
    )
    ax.set_yticks([])

    # Legend
    interior = mpatches.Patch(facecolor="#4a90d9", alpha=0.7, label="Tile interior")
    halo_patch = mpatches.Patch(facecolor="red", alpha=0.2, label="Halo overlap")
    ax.legend(handles=[interior, halo_patch], loc="upper right")

    fig.tight_layout()
    return fig


def _visualize_2d(tile_config, domain, halo_widths, temporal_config, ax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.get_figure()

    tx, ty = tile_config.tile_sizes
    hx, hy = halo_widths
    ntx, nty = tile_config.num_tiles
    dx, dy = domain

    for ti in range(ntx):
        for tj in range(nty):
            x0 = ti * tx
            y0 = tj * ty

            # Tile interior
            rect = mpatches.Rectangle(
                (x0, y0), tx, ty,
                linewidth=1, edgecolor="black", facecolor="#4a90d9", alpha=0.6,
            )
            ax.add_patch(rect)

            # Halo region (expanded rectangle minus interior)
            halo_x0 = max(0, x0 - hx)
            halo_y0 = max(0, y0 - hy)
            halo_x1 = min(dx, x0 + tx + hx)
            halo_y1 = min(dy, y0 + ty + hy)
            halo_rect = mpatches.Rectangle(
                (halo_x0, halo_y0), halo_x1 - halo_x0, halo_y1 - halo_y0,
                linewidth=0.5, edgecolor="red", facecolor="red", alpha=0.1,
            )
            ax.add_patch(halo_rect)

    # Temporal expansion dashed outlines
    if temporal_config is not None and temporal_config.steps > 1:
        ehx, ehy = temporal_config.expanded_halo
        for ti in range(ntx):
            for tj in range(nty):
                x0 = ti * tx
                y0 = tj * ty
                exp_rect = mpatches.Rectangle(
                    (x0 - ehx, y0 - ehy), tx + 2 * ehx, ty + 2 * ehy,
                    linewidth=1, edgecolor="orange", facecolor="none",
                    linestyle="--", alpha=0.8,
                )
                ax.add_patch(exp_rect)

    ax.set_xlim(-hx - 5, dx + hx + 5)
    ax.set_ylim(-hy - 5, dy + hy + 5)
    ax.set_xlabel("Dimension 0")
    ax.set_ylabel("Dimension 1")
    ax.set_title(
        f"2D tiling: {ntx}x{nty} tiles of size {tx}x{ty}, "
        f"halo=({hx},{hy}), overhead={tile_config.overhead_fraction:.1%}"
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Legend
    handles = [
        mpatches.Patch(facecolor="#4a90d9", alpha=0.6, label="Tile interior"),
        mpatches.Patch(facecolor="red", alpha=0.1, label="Halo overlap"),
    ]
    if temporal_config is not None and temporal_config.steps > 1:
        handles.append(mpatches.Patch(
            facecolor="none", edgecolor="orange", linestyle="--",
            label=f"Temporal expansion (T={temporal_config.steps})",
        ))
    ax.legend(handles=handles, loc="upper right")

    fig.tight_layout()
    return fig
