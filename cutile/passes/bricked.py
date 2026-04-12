"""
Bricked layout pass: annotate ``stencil.ApplyOp`` for bricked memory layout.

Attaches ``layout`` and ``brick_size`` attributes so the lowering/emitter
generates divmod addressing instead of standard row-major indexing.

Attached attributes (on each ``stencil.ApplyOp``):
    ``layout``     -- ``StringAttr("bricked")``.
    ``brick_size`` -- ``IntAttr``: brick side length (elements).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from xdsl.context import Context
from xdsl.dialects.builtin import IntAttr, ModuleOp, StringAttr
from xdsl.dialects.stencil import ApplyOp
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class BrickedLayoutPass(ModulePass):
    """Tag stencil ops for bricked (space-filling-curve) memory layout.

    Parameters
    ----------
    brick_size : int
        Brick side length in elements.  Must be a power of 2.
    """

    name: ClassVar[str] = "bricked-layout"

    brick_size: int = 32

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for child in op.walk():
            if isinstance(child, ApplyOp):
                self._annotate(child)

    # ------------------------------------------------------------------

    def _annotate(self, apply_op: ApplyOp) -> None:
        apply_op.attributes["layout"] = StringAttr("bricked")
        apply_op.attributes["brick_size"] = IntAttr(self.brick_size)
