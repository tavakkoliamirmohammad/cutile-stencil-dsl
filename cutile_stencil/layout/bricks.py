"""Convert between flat and bricked memory layouts for stencil arrays.

Bricked layout stores the domain as a (2*ndim)-dimensional array where the
outer dimensions index bricks and the inner dimensions hold the padded brick
data (interior + halo from neighbours).

    flat shape:    (N0 + 2*h0, N1 + 2*h1, ...)
    bricked shape: (nb0, nb1, ..., B0 + 2*h0, B1 + 2*h1, ...)
"""

from __future__ import annotations

from itertools import product

import numpy as np


def to_bricks(
    flat: np.ndarray,
    brick_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
) -> np.ndarray:
    """Convert a flat array (with halo) to bricked layout with per-brick halo.

    Parameters
    ----------
    flat : ndarray of shape (N0 + 2*h0, N1 + 2*h1, ...)
        Flat array including halo padding.  The interior size N_d must be
        divisible by ``brick_sizes[d]``.
    brick_sizes : tuple of int
        Interior size of each brick per dimension.
    halo_widths : tuple of int
        Halo width per dimension.

    Returns
    -------
    bricked : ndarray of shape (nb0, nb1, ..., B0+2*h0, B1+2*h1, ...)
    """
    ndim = len(brick_sizes)
    if len(halo_widths) != ndim:
        raise ValueError("brick_sizes and halo_widths must have same length")
    if flat.ndim != ndim:
        raise ValueError(f"flat has {flat.ndim} dims, expected {ndim}")

    # Interior sizes and number of bricks
    interior = tuple(flat.shape[d] - 2 * halo_widths[d] for d in range(ndim))
    num_bricks = []
    for d in range(ndim):
        if interior[d] % brick_sizes[d] != 0:
            raise ValueError(
                f"interior[{d}]={interior[d]} not divisible by "
                f"brick_sizes[{d}]={brick_sizes[d]}"
            )
        num_bricks.append(interior[d] // brick_sizes[d])

    padded = tuple(brick_sizes[d] + 2 * halo_widths[d] for d in range(ndim))
    out_shape = tuple(num_bricks) + padded
    bricked = np.empty(out_shape, dtype=flat.dtype)

    # Iterate over all brick indices
    for brick_idx in product(*(range(nb) for nb in num_bricks)):
        # Source region in flat array: brick interior starts at
        # brick_idx[d] * brick_sizes[d] (in interior coords), which in the
        # flat array (which has halo offset) is at the same position since
        # flat[0..h-1] is the left halo.  The padded source region extends
        # halo_widths on each side of the brick interior.
        src_slices = tuple(
            slice(
                brick_idx[d] * brick_sizes[d],
                brick_idx[d] * brick_sizes[d] + padded[d],
            )
            for d in range(ndim)
        )
        dst_slices = tuple(brick_idx) + tuple(slice(None) for _ in range(ndim))
        bricked[dst_slices] = flat[src_slices]

    return bricked


def from_bricks(
    bricked: np.ndarray,
    brick_sizes: tuple[int, ...],
    halo_widths: tuple[int, ...],
) -> np.ndarray:
    """Convert bricked layout back to flat array (with halo).

    The interior portion of each brick is copied to the correct position in
    the flat array.  Boundary halos in the flat output are populated from the
    first/last bricks' halo regions.

    Parameters
    ----------
    bricked : ndarray of shape (nb0, nb1, ..., B0+2*h0, B1+2*h1, ...)
    brick_sizes : tuple of int
    halo_widths : tuple of int

    Returns
    -------
    flat : ndarray of shape (N0 + 2*h0, N1 + 2*h1, ...)
    """
    ndim = len(brick_sizes)
    if len(halo_widths) != ndim:
        raise ValueError("brick_sizes and halo_widths must have same length")
    if bricked.ndim != 2 * ndim:
        raise ValueError(
            f"bricked has {bricked.ndim} dims, expected {2 * ndim}"
        )

    num_bricks = tuple(bricked.shape[d] for d in range(ndim))
    interior = tuple(num_bricks[d] * brick_sizes[d] for d in range(ndim))
    flat_shape = tuple(interior[d] + 2 * halo_widths[d] for d in range(ndim))
    flat = np.empty(flat_shape, dtype=bricked.dtype)

    # Copy interior of each brick to the flat array
    for brick_idx in product(*(range(nb) for nb in num_bricks)):
        src_slices = (
            tuple(brick_idx)
            + tuple(
                slice(halo_widths[d], halo_widths[d] + brick_sizes[d])
                for d in range(ndim)
            )
        )
        dst_slices = tuple(
            slice(
                halo_widths[d] + brick_idx[d] * brick_sizes[d],
                halo_widths[d] + (brick_idx[d] + 1) * brick_sizes[d],
            )
            for d in range(ndim)
        )
        flat[dst_slices] = bricked[src_slices]

    # Populate boundary halos from boundary bricks (iterate per brick)
    for d in range(ndim):
        h = halo_widths[d]
        if h == 0:
            continue

        # Left halo along dim d: from bricks with index 0 in dim d
        ranges_left = [range(num_bricks[dd]) for dd in range(ndim)]
        ranges_left[d] = [0]
        for brick_idx in product(*ranges_left):
            src_slices = (
                tuple(brick_idx)
                + tuple(
                    slice(None, h) if dd == d
                    else slice(halo_widths[dd], halo_widths[dd] + brick_sizes[dd])
                    for dd in range(ndim)
                )
            )
            dst_slices = tuple(
                slice(None, h) if dd == d
                else slice(
                    halo_widths[dd] + brick_idx[dd] * brick_sizes[dd],
                    halo_widths[dd] + (brick_idx[dd] + 1) * brick_sizes[dd],
                )
                for dd in range(ndim)
            )
            flat[dst_slices] = bricked[src_slices]

        # Right halo along dim d: from bricks with last index in dim d
        ranges_right = [range(num_bricks[dd]) for dd in range(ndim)]
        ranges_right[d] = [num_bricks[d] - 1]
        for brick_idx in product(*ranges_right):
            src_slices = (
                tuple(brick_idx)
                + tuple(
                    slice(h + brick_sizes[dd], h + brick_sizes[dd] + h) if dd == d
                    else slice(halo_widths[dd], halo_widths[dd] + brick_sizes[dd])
                    for dd in range(ndim)
                )
            )
            dst_slices = tuple(
                slice(flat_shape[dd] - h, flat_shape[dd]) if dd == d
                else slice(
                    halo_widths[dd] + brick_idx[dd] * brick_sizes[dd],
                    halo_widths[dd] + (brick_idx[dd] + 1) * brick_sizes[dd],
                )
                for dd in range(ndim)
            )
            flat[dst_slices] = bricked[src_slices]

    return flat
