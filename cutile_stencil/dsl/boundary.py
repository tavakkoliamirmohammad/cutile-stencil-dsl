"""Boundary condition abstractions for stencil computations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class BoundaryType(Enum):
    """Types of boundary conditions."""
    DIRICHLET = "dirichlet"     # Fixed value (default 0)
    NEUMANN = "neumann"         # Zero-gradient (copy neighbor)
    PERIODIC = "periodic"       # Wrap-around
    REFLECTING = "reflecting"   # Symmetric reflection


@dataclass(frozen=True)
class BoundaryCondition:
    """A single boundary condition specification."""
    bc_type: BoundaryType
    value: float = 0.0


@dataclass
class BoundarySpec:
    """Boundary conditions per dimension (low, high) for each spatial axis.

    conditions[d] = (low_bc, high_bc) for dimension d.
    """
    conditions: List[Tuple[BoundaryCondition, BoundaryCondition]]

    @classmethod
    def uniform(cls, bc_type: BoundaryType, ndim: int, value: float = 0.0) -> BoundarySpec:
        """Create uniform BCs on all faces."""
        bc = BoundaryCondition(bc_type, value)
        return cls(conditions=[(bc, bc)] * ndim)

    @classmethod
    def dirichlet(cls, ndim: int, value: float = 0.0) -> BoundarySpec:
        return cls.uniform(BoundaryType.DIRICHLET, ndim, value)

    @classmethod
    def periodic(cls, ndim: int) -> BoundarySpec:
        return cls.uniform(BoundaryType.PERIODIC, ndim)

    @classmethod
    def neumann(cls, ndim: int) -> BoundarySpec:
        return cls.uniform(BoundaryType.NEUMANN, ndim)

    def cuTile_padding_mode(self, dim: int = 0) -> str:
        """Return cuTile padding mode string for a given dimension.

        Maps BC types to cuTile PaddingMode enum values.
        """
        low_bc, high_bc = self.conditions[dim]
        # Use the low boundary type as the representative
        bc = low_bc.bc_type
        mapping = {
            BoundaryType.DIRICHLET: "ZERO",
            BoundaryType.NEUMANN: "CLAMP",
            BoundaryType.PERIODIC: "WRAP",
            BoundaryType.REFLECTING: "REFLECT",
        }
        return mapping.get(bc, "ZERO")
