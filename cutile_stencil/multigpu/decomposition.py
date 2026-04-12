"""Domain decomposition for multi-GPU stencil computations.

Pure Python module -- no GPU imports at module level. Supports:
- 1D decomposition: splits a global domain along a single axis.
- Cartesian decomposition: splits across a 2D or 3D grid of GPUs.

Each GPU owns an interior subdomain and its neighbors provide halo ghost cells.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Partition:
    """A single GPU's subdomain within a domain decomposition.

    Attributes
    ----------
    rank : int
        GPU rank (0-based) within the process group.
    global_domain : tuple of int
        Full domain size (interior points per dimension).
    local_domain : tuple of int
        Interior points this GPU owns per dimension.
    local_offset : tuple of int
        Start index of this partition in the global domain.
    neighbors : dict
        Mapping from direction label to neighbor rank.
        Keys are ``"left"`` and ``"right"`` (along the split axis).
        Values are the neighbor rank or ``None`` at domain boundaries.
    halo_widths : tuple of int
        Halo width per dimension (same as the global stencil halo).
    """

    rank: int
    global_domain: Tuple[int, ...]
    local_domain: Tuple[int, ...]
    local_offset: Tuple[int, ...]
    neighbors: Dict[str, Optional[int]]
    halo_widths: Tuple[int, ...]


@dataclass(frozen=True)
class DomainDecomposition:
    """Complete decomposition of a domain across multiple GPUs.

    Attributes
    ----------
    global_domain : tuple of int
        Full domain size (interior points per dimension).
    num_gpus : int
        Number of GPUs the domain is split across.
    partitions : list of Partition
        One partition per GPU, ordered by rank.
    split_axis : int
        The axis along which the domain is split.
    halo_widths : tuple of int
        Halo widths per dimension (from the stencil spec).
    """

    global_domain: Tuple[int, ...]
    num_gpus: int
    partitions: List[Partition]
    split_axis: int
    halo_widths: Tuple[int, ...]


def decompose(
    domain: Tuple[int, ...],
    num_gpus: int,
    halo_widths: Tuple[int, ...],
    split_axis: Optional[int] = None,
    periodic: bool = False,
) -> DomainDecomposition:
    """Partition a domain across GPUs along one axis.

    Parameters
    ----------
    domain : tuple of int
        Global domain size (interior points per dimension).
    num_gpus : int
        Number of GPUs to split across.
    halo_widths : tuple of int
        Halo width per dimension (from the stencil specification).
    split_axis : int, optional
        Which axis to split along. Defaults to the longest dimension.
    periodic : bool, optional
        If True, the domain wraps around: the first and last partitions
        become neighbors of each other (periodic boundary conditions).

    Returns
    -------
    DomainDecomposition
        The complete decomposition with one ``Partition`` per GPU.

    Raises
    ------
    ValueError
        If ``num_gpus < 1``, dimensions mismatch, or the domain is not
        evenly divisible along the split axis.
    """
    ndim = len(domain)

    # --- Validation ---
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    if len(halo_widths) != ndim:
        raise ValueError(
            f"halo_widths has {len(halo_widths)} dims, expected {ndim}"
        )
    if any(d <= 0 for d in domain):
        raise ValueError(f"All domain dimensions must be > 0, got {domain}")

    # --- Choose split axis ---
    if split_axis is None:
        split_axis = _choose_split_axis(domain)
    if not (0 <= split_axis < ndim):
        raise ValueError(
            f"split_axis={split_axis} out of range for {ndim}D domain"
        )

    # --- Check divisibility ---
    extent = domain[split_axis]
    if extent % num_gpus != 0:
        raise ValueError(
            f"domain[{split_axis}]={extent} is not divisible by "
            f"num_gpus={num_gpus}"
        )

    local_extent = extent // num_gpus

    # Ensure each local subdomain is at least as wide as the halo
    split_halo = halo_widths[split_axis]
    if local_extent < split_halo and num_gpus > 1:
        raise ValueError(
            f"Local extent {local_extent} along split axis is smaller "
            f"than halo width {split_halo}. Use fewer GPUs or a larger domain."
        )

    # --- Build partitions ---
    partitions: List[Partition] = []
    for rank in range(num_gpus):
        # Local domain: same as global except along split axis
        local_domain = list(domain)
        local_domain[split_axis] = local_extent
        local_domain = tuple(local_domain)

        # Offset in global domain
        local_offset = [0] * ndim
        local_offset[split_axis] = rank * local_extent
        local_offset = tuple(local_offset)

        # Neighbors along split axis
        if rank > 0:
            left_rank = rank - 1
        elif periodic and num_gpus > 1:
            left_rank = num_gpus - 1
        else:
            left_rank = None

        if rank < num_gpus - 1:
            right_rank = rank + 1
        elif periodic and num_gpus > 1:
            right_rank = 0
        else:
            right_rank = None

        neighbors = {"left": left_rank, "right": right_rank}

        partitions.append(
            Partition(
                rank=rank,
                global_domain=domain,
                local_domain=local_domain,
                local_offset=local_offset,
                neighbors=neighbors,
                halo_widths=halo_widths,
            )
        )

    return DomainDecomposition(
        global_domain=domain,
        num_gpus=num_gpus,
        partitions=partitions,
        split_axis=split_axis,
        halo_widths=halo_widths,
    )


def _choose_split_axis(domain: Tuple[int, ...]) -> int:
    """Choose the axis with the largest extent for splitting."""
    return max(range(len(domain)), key=lambda d: domain[d])


# =====================================================================
# Cartesian (2D / 3D) Domain Decomposition
# =====================================================================


@dataclass(frozen=True)
class CartesianPartition:
    """A single GPU's subdomain in a Cartesian decomposition.

    Attributes
    ----------
    rank : int
        Linear GPU rank (0-based) within the Cartesian grid.
    coords : tuple of int
        Position of this partition in the Cartesian topology,
        e.g. ``(0, 1)`` for row 0, column 1 in a 2-D grid.
    global_domain : tuple of int
        Full domain size (interior points per dimension).
    local_domain : tuple of int
        Interior points this GPU owns per dimension.
    local_offset : tuple of int
        Start index of this partition in the global domain.
    neighbors : dict
        Mapping ``(axis, direction) -> rank | None``.
        *axis* is the dimension index; *direction* is ``-1`` (left/up)
        or ``+1`` (right/down).  ``None`` indicates a domain boundary.
    halo_widths : tuple of int
        Halo width per dimension (same as the global stencil halo).
    """

    rank: int
    coords: Tuple[int, ...]
    global_domain: Tuple[int, ...]
    local_domain: Tuple[int, ...]
    local_offset: Tuple[int, ...]
    neighbors: Dict[Tuple[int, int], Optional[int]]
    halo_widths: Tuple[int, ...]


@dataclass(frozen=True)
class CartesianDecomposition:
    """Domain decomposition across a Cartesian grid of GPUs.

    Attributes
    ----------
    global_domain : tuple of int
        Full domain size (interior points per dimension).
    topology : tuple of int
        Number of GPU partitions per axis, e.g. ``(2, 2)`` for a
        2-by-2 grid of 4 GPUs in 2-D.
    num_gpus : int
        Total number of GPUs (equals ``math.prod(topology)``).
    partitions : list of CartesianPartition
        One partition per GPU, ordered by their linearised rank.
    halo_widths : tuple of int
        Halo widths per dimension (from the stencil spec).
    """

    global_domain: Tuple[int, ...]
    topology: Tuple[int, ...]
    num_gpus: int
    partitions: List[CartesianPartition]
    halo_widths: Tuple[int, ...]


def _factorize_gpus(num_gpus: int, ndim: int) -> Tuple[int, ...]:
    """Find the most balanced Cartesian grid for *num_gpus* across *ndim* axes.

    The returned topology is sorted in ascending order so that the
    smallest number of partitions is along axis 0.

    Examples
    --------
    >>> _factorize_gpus(4, 2)
    (2, 2)
    >>> _factorize_gpus(8, 3)
    (2, 2, 2)
    >>> _factorize_gpus(6, 2)
    (2, 3)
    >>> _factorize_gpus(6, 3)
    (1, 2, 3)
    """
    if ndim == 1:
        return (num_gpus,)

    if ndim == 2:
        # Find factor pair (i, j) with i <= j closest to sqrt(num_gpus)
        best = (1, num_gpus)
        for i in range(1, int(num_gpus ** 0.5) + 1):
            if num_gpus % i == 0:
                j = num_gpus // i
                if max(i, j) - min(i, j) < max(best) - min(best):
                    best = (i, j)
        return best

    if ndim == 3:
        best = (1, 1, num_gpus)
        best_diff = num_gpus
        for i in range(1, num_gpus + 1):
            if num_gpus % i != 0:
                continue
            rem = num_gpus // i
            for j in range(1, int(rem ** 0.5) + 1):
                if rem % j == 0:
                    k = rem // j
                    diff = max(i, j, k) - min(i, j, k)
                    if diff < best_diff:
                        best_diff = diff
                        best = tuple(sorted([i, j, k]))
        return best

    raise ValueError(f"ndim={ndim} not supported for Cartesian decomposition")


def decompose_cartesian(
    domain: Tuple[int, ...],
    num_gpus: int,
    halo_widths: Tuple[int, ...],
    topology: Optional[Tuple[int, ...]] = None,
    periodic: bool = False,
) -> CartesianDecomposition:
    """Decompose *domain* across a Cartesian grid of GPUs.

    Parameters
    ----------
    domain : tuple of int
        Global domain size (interior points per dimension).
    num_gpus : int
        Total number of GPUs.
    halo_widths : tuple of int
        Halo width per dimension.
    topology : tuple of int, optional
        Number of GPU partitions per axis.  If ``None``, the topology
        is auto-computed for maximum balance.  The product of all
        elements must equal *num_gpus*.
    periodic : bool
        If ``True``, boundary partitions wrap around to the opposite
        side (periodic / toroidal boundary conditions).

    Returns
    -------
    CartesianDecomposition

    Raises
    ------
    ValueError
        If topology product != num_gpus, a domain dimension is not
        evenly divisible by the corresponding topology entry, or a
        local subdomain is too small to accommodate the halo.
    """
    ndim = len(domain)

    if len(halo_widths) != ndim:
        raise ValueError(
            f"halo_widths has {len(halo_widths)} dims, expected {ndim}"
        )

    if topology is None:
        topology = _factorize_gpus(num_gpus, ndim)

    if len(topology) != ndim:
        raise ValueError(
            f"topology has {len(topology)} dims, expected {ndim}"
        )

    if math.prod(topology) != num_gpus:
        raise ValueError(
            f"Product of topology {topology} = {math.prod(topology)}, "
            f"expected {num_gpus}"
        )

    # Check even divisibility per axis
    for d in range(ndim):
        if domain[d] % topology[d] != 0:
            raise ValueError(
                f"domain[{d}]={domain[d]} not divisible by "
                f"topology[{d}]={topology[d]}"
            )

    local_sizes = tuple(domain[d] // topology[d] for d in range(ndim))

    # Ensure local subdomains are large enough for halos
    for d in range(ndim):
        if topology[d] > 1 and local_sizes[d] < 2 * halo_widths[d]:
            raise ValueError(
                f"local_size[{d}]={local_sizes[d]} < 2*halo={2 * halo_widths[d]}"
            )

    # Enumerate all Cartesian coordinates in row-major order so that
    # rank == flat index of coords in topology.
    all_coords = list(itertools.product(*(range(t) for t in topology)))

    partitions: List[CartesianPartition] = []
    for rank, coords in enumerate(all_coords):
        offset = tuple(coords[d] * local_sizes[d] for d in range(ndim))

        # Build neighbor map: (axis, direction) -> rank | None
        neighbors: Dict[Tuple[int, int], Optional[int]] = {}
        for d in range(ndim):
            for direction in (-1, +1):
                neighbor_coords = list(coords)
                neighbor_coords[d] += direction

                if 0 <= neighbor_coords[d] < topology[d]:
                    n_rank = all_coords.index(tuple(neighbor_coords))
                    neighbors[(d, direction)] = n_rank
                elif periodic:
                    neighbor_coords[d] = neighbor_coords[d] % topology[d]
                    n_rank = all_coords.index(tuple(neighbor_coords))
                    neighbors[(d, direction)] = n_rank
                else:
                    neighbors[(d, direction)] = None

        partitions.append(
            CartesianPartition(
                rank=rank,
                coords=coords,
                global_domain=domain,
                local_domain=local_sizes,
                local_offset=offset,
                neighbors=neighbors,
                halo_widths=halo_widths,
            )
        )

    return CartesianDecomposition(
        global_domain=domain,
        topology=topology,
        num_gpus=num_gpus,
        partitions=partitions,
        halo_widths=halo_widths,
    )
