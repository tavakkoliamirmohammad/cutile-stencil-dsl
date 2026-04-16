"""Empirical autotuning for stencil tile sizes and temporal blocking.

Generates candidate configurations, benchmarks each on the actual GPU,
and returns the fastest.  Results are cached by (stencil_name, ndim,
halo_widths, gpu_name) so subsequent calls are instant.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from dataclasses import dataclass, asdict
from itertools import product as iterproduct
from pathlib import Path
from typing import Optional


@dataclass
class AutotuneResult:
    """Result of the autotuning process."""

    tile_sizes: tuple[int, ...]
    temporal_steps: int
    throughput_gpoints: float
    bandwidth_gbs: float
    time_ms: float = 0.0


# ------------------------------------------------------------------ #
# Cache
# ------------------------------------------------------------------ #

_CACHE_DIR = Path.home() / ".cache" / "cutile" / "autotune"


def _cache_key(name: str, ndim: int, halo: tuple[int, ...], gpu: str,
               domain: tuple[int, ...] | None = None,
               max_temporal: int = 8) -> str:
    raw = f"{name}|{ndim}|{halo}|{gpu}|{domain}|T{max_temporal}|v2"
    return hashlib.md5(raw.encode()).hexdigest()


def _load_cache(key: str) -> Optional[AutotuneResult]:
    path = _CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            d = json.loads(path.read_text())
            return AutotuneResult(**d)
        except Exception:
            pass
    return None


def _save_cache(key: str, result: AutotuneResult) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (_CACHE_DIR / f"{key}.json").write_text(json.dumps(asdict(result)))


# ------------------------------------------------------------------ #
# Candidate generation
# ------------------------------------------------------------------ #

def _generate_candidates(
    ndim: int,
    halo: tuple[int, ...],
    shared_mem: int = 49152,
    dtype_bytes: int = 8,
    max_temporal: int = 8,
) -> list[tuple[tuple[int, ...], int]]:
    """Generate (tile_sizes, temporal_steps) candidates that fit smem.

    Thin wrapper around the shared candidate-generation logic in
    :mod:`cutile.passes.analysis.tile_selection` so the autotuner and
    the static :class:`TilingPass` cannot drift apart on the smem
    model. ``num_loads = 2*ndim + 1`` is a conservative upper bound
    used pre-IR-construction.
    """
    from cutile.passes.analysis.tile_selection import generate_candidates

    return generate_candidates(
        ndim, halo,
        num_loads=2 * ndim + 1,
        shared_mem_bytes=shared_mem,
        dtype_bytes=dtype_bytes,
        max_temporal=max_temporal,
    )


# ------------------------------------------------------------------ #
# Empirical benchmarking
# ------------------------------------------------------------------ #


def _benchmark_candidate(
    stencil_fn,
    tile_sizes: tuple[int, ...],
    halo: tuple[int, ...],
    temporal_steps: int,
    domain: tuple[int, ...],
    warmup: int = 10,
    iters: int = 30,
) -> Optional[float]:
    """Benchmark a single candidate on GPU. Returns time in ms or None."""
    import cupy as cp
    from cutile.lowering.stencil_to_cutile import lower_stencil_to_python

    try:
        code = lower_stencil_to_python(
            stencil_fn._ir.clone(),
            tile_sizes=tile_sizes,
            halo_widths=halo,
            temporal_steps=temporal_steps,
        )
        ast.parse(code)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(code)
            tmp = f.name

        spec = importlib.util.spec_from_file_location("_at", tmp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        launcher = getattr(mod, f"launch_{stencil_fn._fn.__name__}")

        T = temporal_steps
        shape = tuple(d + 2 * T * h for d, h in zip(domain, halo))
        u = cp.random.randn(*shape).astype(cp.float64)
        out = cp.zeros_like(u)

        for _ in range(warmup):
            launcher(u, out)
        cp.cuda.Device(0).synchronize()

        e1, e2 = cp.cuda.Event(), cp.cuda.Event()
        e1.record()
        for _ in range(iters):
            launcher(u, out)
        e2.record()
        e2.synchronize()

        ms = cp.cuda.get_elapsed_time(e1, e2) / iters
        os.unlink(tmp)
        return ms

    except Exception:
        return None


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #


def autotune(
    stencil_fn,
    domain: tuple[int, ...] | None = None,
    hw=None,
    max_candidates: int = 30,
    warmup: int = 10,
    iters: int = 30,
    verbose: bool = False,
    max_temporal: int = 8,
) -> AutotuneResult:
    """Empirically autotune tile sizes and temporal blocking.

    Tries candidate configurations on the GPU and picks the fastest.
    Results are cached so subsequent calls with the same stencil + GPU
    are instant.
    """
    from cutile.config import HardwareSpec

    if hw is None:
        try:
            hw = HardwareSpec.auto_detect("float64")
        except Exception:
            hw = HardwareSpec()

    ndim = stencil_fn._ndim or 2
    order = stencil_fn._order or 2
    halo = tuple(order // 2 for _ in range(ndim))
    name = stencil_fn._fn.__name__

    # Check cache
    try:
        import cupy as cp
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        gpu_name = "unknown"

    cache_key = _cache_key(name, ndim, halo, gpu_name, domain, max_temporal)
    cached = _load_cache(cache_key)
    if cached is not None:
        if verbose:
            print(f"autotune: cache hit for {name} on {gpu_name}")
        return cached

    # Default domain for benchmarking if not provided
    if domain is None:
        if ndim == 1:
            domain = (2**18,)
        elif ndim == 2:
            domain = (1024, 1024)
        else:
            domain = (64,) * ndim

    # Generate candidates
    candidates = _generate_candidates(
        ndim, halo,
        shared_mem=hw.shared_mem_bytes,
        dtype_bytes=hw.dtype_bytes,
        max_temporal=max_temporal,
    )

    # Limit candidates
    if len(candidates) > max_candidates:
        # Prioritize: diverse tile sizes, prefer larger temporal steps
        candidates.sort(key=lambda x: (x[1], math.prod(x[0])), reverse=True)
        candidates = candidates[:max_candidates]

    if verbose:
        print(f"autotune: testing {len(candidates)} candidates for {name} ({ndim}D, halo={halo})")

    # Benchmark each
    best_ms = float("inf")
    best_tile = (32,) * ndim
    best_T = 1

    for tile, T in candidates:
        ms = _benchmark_candidate(stencil_fn, tile, halo, T, domain, warmup, iters)
        if ms is not None and ms < best_ms:
            best_ms = ms
            best_tile = tile
            best_T = T
            if verbose:
                tile_str = "x".join(str(t) for t in tile)
                print(f"  {tile_str} T={T}: {ms:.4f} ms {'*best*' if ms == best_ms else ''}")

    # Compute throughput
    npoints = math.prod(domain)
    effective_points = best_T * npoints
    throughput = effective_points / (best_ms / 1000) / 1e9 if best_ms > 0 else 0
    bandwidth = npoints * hw.dtype_bytes * 2 / (best_ms / 1000) / 1e9 if best_ms > 0 else 0

    result = AutotuneResult(
        tile_sizes=best_tile,
        temporal_steps=best_T,
        throughput_gpoints=throughput,
        bandwidth_gbs=bandwidth,
        time_ms=best_ms,
    )

    # Cache result
    _save_cache(cache_key, result)

    if verbose:
        tile_str = "x".join(str(t) for t in best_tile)
        print(f"autotune: best = {tile_str} T={best_T} ({throughput:.1f} GP/s, {best_ms:.4f} ms)")

    return result
