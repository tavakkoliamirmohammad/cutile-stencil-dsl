"""Unified tile-selection candidate generation and strategies.

Two consumers used to maintain near-identical copies of "given a stencil
shape, return tile sizes that fit in shared memory":

- :class:`cutile.passes.tiling.TilingPass` — static heuristic that picks
  the candidate with minimum halo bandwidth overhead.
- :func:`cutile.runtime.autotune.autotune` — empirical search that
  benchmarks every candidate on the GPU and keeps the fastest.

When the shared-memory model was wrong, *both* over-rejected viable
candidates the same way and stayed in lock-step. Centralising the
candidate generation removes that drift surface and gives both
consumers the same notion of "what's possible".

Public API
----------
:func:`generate_candidates`
    Pure function: enumerate ``(tile, temporal_steps)`` pairs that fit
    a given shared-memory budget under the corrected ``T=1`` model
    (each ``ct.load`` is a ``(tile,)``-shaped smem allocation; halo
    only contributes to bandwidth-redundancy, not capacity) and the
    standard double-buffered halo'd model for ``T>1``.

:func:`overhead_score`
    The bandwidth-redundancy metric: fraction of the loaded tile that
    is halo overlap with neighbouring CTAs. Used by the static pass.

:class:`OverheadHeuristic`
    Wraps ``generate_candidates`` + ``overhead_score`` so the static
    tiling pass becomes a 3-line consumer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product as iterproduct
from typing import Iterable


# Default per-dimension candidate widths. These are the same powers of 2
# that both TilingPass and autotune used pre-unification.
DEFAULT_CANDIDATE_WIDTHS_1D = (32, 64, 128, 256, 512)
DEFAULT_CANDIDATE_WIDTHS_2D = (8, 16, 32, 64, 128)
DEFAULT_CANDIDATE_WIDTHS_3D = (4, 8, 16, 32)


def _default_widths(ndim: int) -> tuple[int, ...]:
    if ndim == 1:
        return DEFAULT_CANDIDATE_WIDTHS_1D
    if ndim == 2:
        return DEFAULT_CANDIDATE_WIDTHS_2D
    return DEFAULT_CANDIDATE_WIDTHS_3D


def generate_candidates(
    ndim: int,
    halo: tuple[int, ...],
    num_loads: int,
    *,
    shared_mem_bytes: int = 49152,
    dtype_bytes: int = 8,
    max_temporal: int = 8,
    candidate_widths: tuple[int, ...] | None = None,
) -> list[tuple[tuple[int, ...], int]]:
    """Enumerate viable ``(tile, temporal_steps)`` pairs.

    ``T=1`` (no temporal blocking): each ``ct.load`` allocates a
    ``(tile,)``-shaped smem tile, total ``smem = (num_loads + 1) *
    prod(tile) * dtype_bytes``. Halo contributes only to bandwidth
    overhead (redundant loads at CTA boundaries), not to smem.

    ``T>1`` (temporal blocking): single double-buffered halo'd tile per
    buffer, ``smem = 2 * prod(tile + 2*T*halo) * dtype_bytes``. Matches
    :class:`cutile.passes.temporal.TemporalBlockingPass`.

    Returns a list of ``((d0, d1, ...), T)`` pairs that fit the budget.
    """
    widths = candidate_widths or _default_widths(ndim)
    candidates: list[tuple[tuple[int, ...], int]] = []

    # T>1: pick the largest T that fits per tile shape (TBing favours
    # bigger T = more cache reuse).
    for combo in iterproduct(widths, repeat=ndim):
        for T in range(max_temporal, 1, -1):
            expanded = tuple(c + 2 * T * h for c, h in zip(combo, halo))
            smem = 2 * math.prod(expanded) * dtype_bytes
            if smem > shared_mem_bytes:
                continue
            candidates.append((combo, T))
            break

    # T=1: every tile that fits the corrected model.
    for combo in iterproduct(widths, repeat=ndim):
        prod_tile = math.prod(combo)
        smem = (num_loads + 1) * prod_tile * dtype_bytes
        if smem <= shared_mem_bytes and (combo, 1) not in candidates:
            candidates.append((combo, 1))

    return candidates


def overhead_score(tile: tuple[int, ...], halo: tuple[int, ...]) -> float:
    """Bandwidth-redundancy metric in ``[0, 1)``.

    Lower = less halo waste relative to compute (more useful work per
    byte fetched from DRAM into smem).
    """
    expanded = tuple(t + 2 * h for t, h in zip(tile, halo))
    return 1.0 - math.prod(tile) / math.prod(expanded)


@dataclass
class OverheadHeuristic:
    """Static tile-selection strategy used by :class:`TilingPass`.

    Pick the smallest-halo-overhead candidate that fits in shared
    memory at ``T=1``. Returns ``None`` if no candidate fits — the
    caller is responsible for falling back.
    """

    shared_mem_bytes: int = 49152
    dtype_bytes: int = 8
    candidate_widths: tuple[int, ...] | None = None

    def select(
        self,
        ndim: int,
        halo: tuple[int, ...],
        num_loads: int,
    ) -> tuple[int, ...] | None:
        """Return the chosen tile or ``None`` if nothing fits."""
        cands = generate_candidates(
            ndim, halo, num_loads,
            shared_mem_bytes=self.shared_mem_bytes,
            dtype_bytes=self.dtype_bytes,
            max_temporal=1,
            candidate_widths=self.candidate_widths,
        )
        t1_only = [tile for tile, T in cands if T == 1]
        if not t1_only:
            return None
        return min(t1_only, key=lambda t: overhead_score(t, halo))


def fallback_tile(ndim: int) -> tuple[int, ...]:
    """Last-resort tile when no candidate fits the smem budget."""
    return {1: (32,), 2: (16, 16), 3: (8, 8, 8)}.get(ndim, (8,) * ndim)
