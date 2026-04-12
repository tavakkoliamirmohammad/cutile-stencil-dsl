"""Model-guided autotuning for stencil tile and temporal blocking parameters.

Uses a two-phase approach:

1. **Analytical model** -- evaluate candidate tile configurations based on
   shared-memory budget, halo overhead ratio, and estimated occupancy.
2. **Empirical validation** -- (when GPU is available) benchmark the top
   candidates and pick the fastest.

Without GPU access the analytical model alone selects the configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class AutotuneResult:
    """Result of the autotuning process."""

    tile_sizes: tuple[int, ...]
    temporal_steps: int
    throughput_gpoints: float
    bandwidth_gbs: float


def autotune(
    stencil_fn,
    domain: tuple[int, ...],
    hw=None,
    n_candidates: int = 20,
) -> AutotuneResult:
    """Model-guided autotuning with optional empirical validation.

    Parameters
    ----------
    stencil_fn
        ``@stencil``-decorated function with ``_ir``, ``_ndim``, ``_order``.
    domain
        Domain shape (including halos).
    hw
        Optional :class:`~cutile.config.HardwareSpec`.
    n_candidates
        Maximum number of candidates to evaluate analytically.

    Returns
    -------
    AutotuneResult
    """
    from cutile.config import HardwareSpec

    if hw is None:
        dtype_str = getattr(stencil_fn, "_dtype", "float64")
        try:
            hw = HardwareSpec.auto_detect(dtype_str)
        except Exception:
            hw = HardwareSpec()

    ndim = stencil_fn._ndim or len(domain)
    order = stencil_fn._order or 2
    halo = order // 2 if order else 1
    dtype_bytes = hw.dtype_bytes
    shared_mem = hw.shared_mem_bytes

    # ------------------------------------------------------------------ #
    # Phase 1: Analytical model -- enumerate and score candidates
    # ------------------------------------------------------------------ #
    candidate_tile_widths = [32, 64, 128, 256, 512, 1024]
    max_temporal = 16

    scored: list[tuple[float, tuple[int, ...], int]] = []

    for tw in candidate_tile_widths:
        tile = (tw,) * ndim

        for t in range(max_temporal, 0, -1):
            # Expanded tile with temporal blocking halos
            expanded = tuple(tw + 2 * t * halo for _ in range(ndim))
            smem = math.prod(expanded) * dtype_bytes * 2  # double-buffer

            if smem > shared_mem:
                continue

            # Halo overhead: fraction of expanded tile that is pure halo
            prod_tile = math.prod(tile)
            prod_exp = math.prod(expanded)
            overhead = 1.0 - prod_tile / prod_exp

            # Temporal reuse factor: more steps = fewer kernel launches
            reuse_score = t

            # Combined score: lower overhead + higher temporal reuse is better
            # The score is in "goodness" (higher = better)
            score = reuse_score * (1.0 - overhead)

            scored.append((score, tile, t))

    # Sort by score descending, take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:n_candidates]

    if not top:
        # Fallback
        return AutotuneResult(
            tile_sizes=(32,) * ndim,
            temporal_steps=1,
            throughput_gpoints=0.0,
            bandwidth_gbs=0.0,
        )

    # ------------------------------------------------------------------ #
    # Phase 2: Empirical validation (if GPU available)
    # ------------------------------------------------------------------ #
    best_score, best_tile, best_t = top[0]

    try:
        import cupy as cp
        import numpy as np

        from cutile.lowering.stencil_to_cutile import lower_stencil_to_python

        best_throughput = 0.0
        best_result = None

        for _score, tile, t in top[:5]:  # benchmark top 5
            code = lower_stencil_to_python(
                stencil_fn._ir.clone(),
                domain=domain,
                tile_sizes=tile,
                halo_widths=(halo,) * ndim,
                temporal_steps=t,
            )

            try:
                import importlib.util
                import tempfile

                with tempfile.NamedTemporaryFile(
                    suffix=".py", delete=False, mode="w"
                ) as f:
                    f.write(code)
                    tmp_path = f.name
                spec = importlib.util.spec_from_file_location(
                    "_autotune_candidate", tmp_path
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                launcher = getattr(
                    mod, f"launch_{stencil_fn._fn.__name__}"
                )
                u_gpu = cp.random.rand(*domain)
                out_gpu = cp.zeros_like(u_gpu)

                # Warmup
                for _ in range(3):
                    launcher(u_gpu, out_gpu)
                cp.cuda.Device().synchronize()

                # Time
                start = cp.cuda.Event()
                end = cp.cuda.Event()
                start.record()
                for _ in range(10):
                    launcher(u_gpu, out_gpu)
                end.record()
                end.synchronize()
                elapsed_ms = float(
                    cp.cuda.get_elapsed_time(start, end)
                ) / 10

                interior = 1
                for h_dim, s in zip((halo,) * ndim, domain):
                    interior *= s - 2 * h_dim
                throughput = interior / elapsed_ms * 1e-6  # GPoints/s

                if throughput > best_throughput:
                    best_throughput = throughput
                    gbytes = interior * 2 * dtype_bytes / elapsed_ms * 1e-6
                    best_result = AutotuneResult(
                        tile_sizes=tile,
                        temporal_steps=t,
                        throughput_gpoints=throughput,
                        bandwidth_gbs=gbytes,
                    )
            except Exception:
                continue

        if best_result is not None:
            return best_result

    except ImportError:
        pass  # No CuPy; use analytical result

    # Analytical-only result
    return AutotuneResult(
        tile_sizes=best_tile,
        temporal_steps=best_t,
        throughput_gpoints=0.0,
        bandwidth_gbs=0.0,
    )
