"""Bayesian auto-tuning for stencil tile configuration.

Uses Optuna to search over tile sizes and temporal blocking steps,
scoring configurations via the analytical roofline model (no GPU needed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from cutile_stencil.dsl.types import (
    HardwareSpec, StencilSpec, TileConfig, TemporalConfig,
)
from cutile_stencil.analysis.tiling import _tile_memory, _overhead
from cutile_stencil.analysis.temporal import compute_temporal_config
from cutile_stencil.analysis.roofline import roofline_analysis


@dataclass
class AutotuneResult:
    """Result of auto-tuning a stencil configuration."""
    best_tile_sizes: Tuple[int, ...]
    best_temporal_steps: int
    best_score: float
    best_tile_config: TileConfig
    best_temporal_config: TemporalConfig
    n_trials: int
    n_feasible: int


# Default search candidates per dimension
_TILE_CANDIDATES = [16, 32, 48, 64, 96, 128, 192, 256, 512]


def autotune(
    stencil_fn,
    domain: Tuple[int, ...],
    hw: Optional[HardwareSpec] = None,
    n_trials: int = 100,
    tile_candidates: Optional[List[int]] = None,
    temporal_candidates: Optional[List[int]] = None,
) -> AutotuneResult:
    """Search for optimal tile configuration using Optuna.

    Uses the roofline model as the objective function — no GPU required.

    Parameters
    ----------
    stencil_fn : decorated stencil function or StencilSpec
        The stencil to tune.
    domain : tuple of int
        Domain size per dimension.
    hw : HardwareSpec, optional
        GPU hardware parameters. Defaults to HardwareSpec().
    n_trials : int
        Number of Optuna trials.
    tile_candidates : list of int, optional
        Tile size candidates per dimension.
    temporal_candidates : list of int, optional
        Temporal blocking step candidates.

    Returns
    -------
    AutotuneResult
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError(
            "optuna is required for auto-tuning. "
            "Install with: pip install cutile-stencil[tune]"
        )

    # Extract spec
    if isinstance(stencil_fn, StencilSpec):
        spec = stencil_fn
    elif hasattr(stencil_fn, '_stencil_spec'):
        spec = stencil_fn._stencil_spec
    else:
        raise TypeError(
            f"Expected a @stencil-decorated function or StencilSpec, "
            f"got {type(stencil_fn).__name__}"
        )

    if hw is None:
        hw = HardwareSpec()

    ndim = spec.ndim
    halo = spec.halo_widths or (spec.order // 2,) * ndim

    if tile_candidates is None:
        tile_candidates = _TILE_CANDIDATES
    if temporal_candidates is None:
        temporal_candidates = [1, 2, 4, 8, 16]

    budget = hw.shared_mem_bytes
    num_arrays = len(spec.inputs) + 1

    # Roofline is independent of tiling — compute once
    roof = roofline_analysis(spec, hw)

    best_result: dict = {"score": -1.0, "config": None, "temporal": None, "feasible": 0}

    def objective(trial):
        # Sample tile sizes per dimension
        tile_sizes = tuple(
            trial.suggest_categorical(f"tile_{d}", tile_candidates)
            for d in range(ndim)
        )
        t_steps = trial.suggest_categorical("temporal_steps", temporal_candidates)

        # Check shared memory feasibility
        mem = _tile_memory(tile_sizes, halo, hw.dtype_bytes, num_arrays)
        if mem > budget:
            return float("-inf")

        # Compute overhead
        oh = _overhead(tile_sizes, halo)

        # Compute temporal config
        nt = tuple(math.ceil(d / t) for d, t in zip(domain, tile_sizes))
        tile_cfg = TileConfig(tile_sizes, halo, nt, oh)
        temp_cfg = compute_temporal_config(spec, tile_cfg, hw, max_steps=t_steps)

        # Score: throughput * efficiency * temporal reduction
        score = roof.peak_gpoints_s * (1.0 - oh) * temp_cfg.bandwidth_reduction_factor
        best_result["feasible"] += 1

        if score > best_result["score"]:
            best_result["score"] = score
            best_result["config"] = tile_cfg
            best_result["temporal"] = temp_cfg

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    if best_result["config"] is None:
        # Fallback: use smallest candidates
        tile_sizes = (tile_candidates[0],) * ndim
        nt = tuple(math.ceil(d / t) for d, t in zip(domain, tile_sizes))
        oh = _overhead(tile_sizes, halo)
        best_result["config"] = TileConfig(tile_sizes, halo, nt, oh)
        best_result["temporal"] = TemporalConfig(1, halo, 1.0)
        best_result["score"] = 0.0
        best_result["feasible"] = 0

    return AutotuneResult(
        best_tile_sizes=best_result["config"].tile_sizes,
        best_temporal_steps=best_result["temporal"].steps,
        best_score=best_result["score"],
        best_tile_config=best_result["config"],
        best_temporal_config=best_result["temporal"],
        n_trials=n_trials,
        n_feasible=best_result["feasible"],
    )
