"""Domain decomposition for multi-GPU stencil computations.

Pure Python module -- no GPU imports at module level. Splits a global
domain across GPUs along a single axis (1D decomposition). Each GPU
owns an interior subdomain and its neighbors provide halo ghost cells.
"""

from __future__ import annotations

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
        left_rank = rank - 1 if rank > 0 else None
        right_rank = rank + 1 if rank < num_gpus - 1 else None
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
