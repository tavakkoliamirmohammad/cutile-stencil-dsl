"""cuTile Stencil Dialect — first stage of the three-dialect lowering stack."""

from cutile.dialects.cutile_stencil.dialect import (
    AccessOp,
    BoundaryAttr,
    ComputeOp,
    CutileStencilDialect,
    FuncOp,
    StencilSpecAttr,
    YieldOp,
)

__all__ = [
    "CutileStencilDialect",
    "FuncOp",
    "AccessOp",
    "ComputeOp",
    "YieldOp",
    "BoundaryAttr",
    "StencilSpecAttr",
]
