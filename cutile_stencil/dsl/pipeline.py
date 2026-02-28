"""One-call analysis pipeline for stencil specifications.

Eliminates the 6-step boilerplate in every example by providing a single
``analyze()`` function that runs the full analysis chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from cutile_stencil.dsl.types import (
    StencilSpec, HardwareSpec, TileConfig, TemporalConfig, RooflineResult,
)
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.analysis.tiling import compute_tile_config
from cutile_stencil.analysis.temporal import compute_temporal_config
from cutile_stencil.analysis.roofline import roofline_analysis


@dataclass
class AnalysisResult:
    """Complete analysis result for a stencil."""
    spec: StencilSpec
    tile_config: TileConfig
    temporal_config: TemporalConfig
    roofline: RooflineResult
    warnings: list


def analyze(
    spec: StencilSpec,
    domain: Tuple[int, ...],
    hw: HardwareSpec | None = None,
) -> AnalysisResult:
    """One-call full analysis: footprint -> halo -> tiling -> temporal -> roofline.

    Parameters
    ----------
    spec : StencilSpec
        The stencil specification (from @stencil decorator).
    domain : tuple of int
        Domain size per dimension (interior points).
    hw : HardwareSpec, optional
        GPU hardware parameters. Defaults to HardwareSpec().

    Returns
    -------
    AnalysisResult
        Complete analysis including tile config, temporal blocking, and roofline.
    """
    if hw is None:
        hw = HardwareSpec()

    # Step 1: Extract footprint
    accesses = extract_footprint(spec)

    # Step 2: Compute halo widths
    halo = compute_halo(accesses, spec.ndim)
    spec.halo_widths = halo

    # Step 3: Tile configuration
    tile_cfg = compute_tile_config(spec, domain, hw)

    # Step 4: Temporal blocking
    temp_cfg = compute_temporal_config(spec, tile_cfg, hw)
    spec.temporal_steps = temp_cfg.steps

    # Step 5: Roofline analysis
    roof = roofline_analysis(spec, hw)

    # Step 6: Validation warnings
    warnings = spec.validate()

    return AnalysisResult(
        spec=spec,
        tile_config=tile_cfg,
        temporal_config=temp_cfg,
        roofline=roof,
        warnings=warnings,
    )
