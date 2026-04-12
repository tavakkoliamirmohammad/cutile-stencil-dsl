"""AMR level management and hierarchy."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from cutile_stencil.dsl.types import BrickLayout


@dataclass
class RefinementLevel:
    """A single refinement level in an AMR hierarchy."""
    level: int                          # 0 = coarsest
    domain: Tuple[int, ...]             # Points at this level
    refinement_ratio: int = 2           # Ratio to next finer level
    brick_layout: Optional[BrickLayout] = None

    @property
    def spacing(self) -> float:
        """Relative grid spacing (1.0 at level 0, 0.5 at level 1 with ratio 2, etc.)."""
        return 1.0 / (self.refinement_ratio ** self.level)

    @property
    def ndim(self) -> int:
        return len(self.domain)


@dataclass
class AMRHierarchy:
    """Multi-level AMR hierarchy.

    Manages refinement levels and provides methods for level interaction.
    """
    levels: List[RefinementLevel]
    halo_widths: Tuple[int, ...]

    def __post_init__(self):
        if len(self.levels) < 1:
            raise ValueError("AMR hierarchy needs at least one level")
        # Validate consistency
        ndim = self.levels[0].ndim
        for lvl in self.levels:
            if lvl.ndim != ndim:
                raise ValueError(f"Level {lvl.level} has {lvl.ndim} dims, expected {ndim}")
        # Validate levels are ordered
        for i, lvl in enumerate(self.levels):
            if lvl.level != i:
                raise ValueError(f"Level {i} has level={lvl.level}, expected {i}")

    @property
    def ndim(self) -> int:
        return self.levels[0].ndim

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def coarsest(self) -> RefinementLevel:
        return self.levels[0]

    @property
    def finest(self) -> RefinementLevel:
        return self.levels[-1]

    @classmethod
    def create(
        cls,
        base_domain: Tuple[int, ...],
        num_levels: int,
        halo_widths: Tuple[int, ...],
        refinement_ratio: int = 2,
        brick_sizes: Optional[Tuple[int, ...]] = None,
    ) -> AMRHierarchy:
        """Create an AMR hierarchy with uniform refinement.

        Parameters
        ----------
        base_domain : tuple of int
            Domain size at the coarsest level.
        num_levels : int
            Number of refinement levels (1 = no refinement).
        halo_widths : tuple of int
            Halo width per dimension (same for all levels).
        refinement_ratio : int
            Refinement ratio between levels (typically 2 or 4).
        brick_sizes : tuple of int, optional
            Brick sizes for bricked layout. If None, no bricking.
        """
        levels = []
        for i in range(num_levels):
            domain = tuple(d * (refinement_ratio ** i) for d in base_domain)
            layout = BrickLayout(brick_sizes) if brick_sizes is not None else None
            levels.append(RefinementLevel(
                level=i,
                domain=domain,
                refinement_ratio=refinement_ratio,
                brick_layout=layout,
            ))
        return cls(levels=levels, halo_widths=halo_widths)

    def coarse_to_fine_shape(self, coarse_level: int) -> Tuple[int, ...]:
        """Shape of fine level corresponding to coarse level."""
        if coarse_level >= self.num_levels - 1:
            raise ValueError(f"No finer level than {coarse_level}")
        return self.levels[coarse_level + 1].domain

    def subcycling_steps(self, level: int) -> int:
        """Number of fine timesteps per coarse timestep at given level.

        Level 0 takes 1 step. Level 1 takes ratio steps. Level 2 takes ratio^2 steps.
        """
        if level == 0:
            return 1
        return self.levels[level].refinement_ratio ** level
