"""Boundary condition types for the cuTile Stencil DSL user API."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union


class BoundaryType(Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    PERIODIC = "periodic"
    REFLECTING = "reflecting"


@dataclass(frozen=True)
class BoundaryCondition:
    """A single boundary condition (type + optional value)."""

    bc_type: BoundaryType
    value: float = 0.0


@dataclass
class BoundarySpec:
    """Boundary specification per dimension.

    Each entry in *conditions* is a ``(low, high)`` pair of
    :class:`BoundaryCondition` objects -- one per dimension.
    """

    conditions: list[tuple[BoundaryCondition, BoundaryCondition]]

    @classmethod
    def uniform(
        cls, bc_type: Union[str, BoundaryType], value: float = 0.0
    ) -> BoundarySpec:
        """Same boundary condition on all sides, all dimensions.

        The number of dimensions is left unspecified -- an empty
        *conditions* list is used as a sentinel meaning "apply this BC
        uniformly everywhere".  Downstream code should expand it to the
        actual number of dimensions when the stencil is compiled.
        """
        if isinstance(bc_type, str):
            bc_type = BoundaryType(bc_type)
        bc = BoundaryCondition(bc_type=bc_type, value=value)
        # Store a single-entry list as a sentinel for "uniform on all dims"
        return cls(conditions=[(bc, bc)])

    @classmethod
    def periodic(cls) -> BoundarySpec:
        """Shorthand for uniform periodic boundaries."""
        return cls.uniform(BoundaryType.PERIODIC)

    @classmethod
    def dirichlet(cls, value: float = 0.0) -> BoundarySpec:
        """Shorthand for uniform Dirichlet boundaries."""
        return cls.uniform(BoundaryType.DIRICHLET, value)

    @classmethod
    def neumann(cls) -> BoundarySpec:
        """Shorthand for uniform Neumann (zero-gradient) boundaries."""
        return cls.uniform(BoundaryType.NEUMANN)
