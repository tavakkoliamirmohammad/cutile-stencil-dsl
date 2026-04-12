"""Pre-built stencil templates for common patterns."""

from __future__ import annotations

import itertools
from typing import Optional, Tuple

from cutile_stencil.dsl.types import StencilSpec, OffsetAccess


def _make_update_fn(name: str, ndim: int, accesses, weights):
    """Create a synthetic update_fn via exec, matching stencil_bridge.py pattern."""
    idx_names = ["i", "j", "k"][:ndim]
    idx_str = ", ".join(idx_names)

    terms = []
    for acc, w in zip(accesses, weights):
        sub_parts = []
        for d, off in enumerate(acc.offsets):
            if off == 0:
                sub_parts.append(idx_names[d])
            elif off > 0:
                sub_parts.append(f"{idx_names[d]} + {off}")
            else:
                sub_parts.append(f"{idx_names[d]} - {abs(off)}")
        subscript = ", ".join(sub_parts)
        if w == 1.0:
            terms.append(f"u[{subscript}]")
        elif w == -1.0:
            terms.append(f"-u[{subscript}]")
        else:
            terms.append(f"{w} * u[{subscript}]")

    body = " + ".join(terms) if terms else "0.0"
    # Clean up double signs: "+ -" -> "- "
    body = body.replace("+ -", "- ")
    src = f"def {name}(u, {idx_str}):\n    return {body}\n"
    globs = {}
    exec(compile(src, f"<template:{name}>", "exec"), globs)  # noqa: S102
    func = globs[name]
    func._source = src
    return func


def compact_stencil(ndim: int, order: int = 2, dtype: str = "float64") -> StencilSpec:
    """Standard compact stencil (3-point 1D, 5-point 2D, 7-point 3D).

    Axis-aligned neighbors only, equal weights summing center + neighbors.
    """
    radius = order // 2
    accesses = []
    # Center point
    center = (0,) * ndim
    accesses.append(OffsetAccess("u", center))
    # Axis-aligned neighbors
    for d in range(ndim):
        for sign in [-1, 1]:
            off = [0] * ndim
            off[d] = sign * radius
            accesses.append(OffsetAccess("u", tuple(off)))

    n_neighbors = 2 * ndim
    weights = [-float(n_neighbors)] + [1.0] * n_neighbors  # Laplacian-style
    halo = (radius,) * ndim

    name = f"compact_{ndim}d_order{order}"
    update_fn = _make_update_fn(name, ndim, accesses, weights)

    return StencilSpec(
        name=name, ndim=ndim, order=order,
        inputs=("u",), output="result",
        update_fn=update_fn, accesses=accesses,
        dtype=dtype, halo_widths=halo,
    )


def cross_stencil(ndim: int, order: int = 2, dtype: str = "float64") -> StencilSpec:
    """Cross/star stencil — all axis-aligned neighbors up to radius.

    For order=4 in 1D: u[i-2], u[i-1], u[i], u[i+1], u[i+2]
    """
    radius = order // 2
    accesses = []
    # Center
    accesses.append(OffsetAccess("u", (0,) * ndim))
    # All axis-aligned offsets up to radius
    for d in range(ndim):
        for r in range(1, radius + 1):
            for sign in [-1, 1]:
                off = [0] * ndim
                off[d] = sign * r
                accesses.append(OffsetAccess("u", tuple(off)))

    weights = [1.0] * len(accesses)
    weights[0] = -2.0 * ndim * radius  # Center weight for Laplacian
    halo = (radius,) * ndim

    name = f"cross_{ndim}d_order{order}"
    update_fn = _make_update_fn(name, ndim, accesses, weights)

    return StencilSpec(
        name=name, ndim=ndim, order=order,
        inputs=("u",), output="result",
        update_fn=update_fn, accesses=accesses,
        dtype=dtype, halo_widths=halo,
    )


def diamond_stencil(ndim: int, order: int = 2, dtype: str = "float64") -> StencilSpec:
    """Diamond-shaped stencil (L1-norm ball of given radius).

    Includes all points where |off_0| + |off_1| + ... <= radius.
    """
    radius = order // 2
    accesses = []

    # Generate all offsets within L1 ball
    ranges = [range(-radius, radius + 1)] * ndim
    for combo in itertools.product(*ranges):
        if sum(abs(c) for c in combo) <= radius:
            accesses.append(OffsetAccess("u", combo))

    weights = [1.0] * len(accesses)
    # Center gets negative sum of others
    center_idx = next(i for i, a in enumerate(accesses) if all(o == 0 for o in a.offsets))
    weights[center_idx] = -float(len(accesses) - 1)
    halo = (radius,) * ndim

    name = f"diamond_{ndim}d_order{order}"
    update_fn = _make_update_fn(name, ndim, accesses, weights)

    return StencilSpec(
        name=name, ndim=ndim, order=order,
        inputs=("u",), output="result",
        update_fn=update_fn, accesses=accesses,
        dtype=dtype, halo_widths=halo,
    )


def box_stencil(ndim: int, order: int = 2, dtype: str = "float64") -> StencilSpec:
    """Full box stencil (all neighbors within radius in each dimension).

    Includes all points where max(|off_d|) <= radius (L-infinity ball).
    """
    radius = order // 2
    accesses = []

    ranges = [range(-radius, radius + 1)] * ndim
    for combo in itertools.product(*ranges):
        accesses.append(OffsetAccess("u", combo))

    weights = [1.0] * len(accesses)
    center_idx = next(i for i, a in enumerate(accesses) if all(o == 0 for o in a.offsets))
    weights[center_idx] = -float(len(accesses) - 1)
    halo = (radius,) * ndim

    name = f"box_{ndim}d_order{order}"
    update_fn = _make_update_fn(name, ndim, accesses, weights)

    return StencilSpec(
        name=name, ndim=ndim, order=order,
        inputs=("u",), output="result",
        update_fn=update_fn, accesses=accesses,
        dtype=dtype, halo_widths=halo,
    )
