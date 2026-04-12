"""
Analysis pass: compute stencil footprint and attach ``halo_widths``.

Walks each ``stencil.ApplyOp``, inspects all ``stencil.AccessOp`` offsets
inside its body, and attaches an ``ArrayAttr`` of ``IntAttr`` giving the
maximum absolute offset in each dimension (i.e. the halo width).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from xdsl.context import Context
from xdsl.dialects.builtin import ArrayAttr, IntAttr, ModuleOp
from xdsl.dialects.stencil import AccessOp, ApplyOp
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class AnalysisPass(ModulePass):
    """Compute per-dimension halo widths from stencil access patterns.

    Attaches ``halo_widths`` (``ArrayAttr[IntAttr]``) on each
    ``stencil.ApplyOp``.
    """

    name: ClassVar[str] = "stencil-analysis"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for apply_op in op.walk():
            if not isinstance(apply_op, ApplyOp):
                continue
            self._analyze_apply(apply_op)

    # ------------------------------------------------------------------

    @staticmethod
    def _analyze_apply(apply_op: ApplyOp) -> None:
        ndim: int | None = None
        max_offsets: list[int] = []

        for inner_op in apply_op.walk():
            if not isinstance(inner_op, AccessOp):
                continue

            offsets = list(inner_op.offset)

            if ndim is None:
                ndim = len(offsets)
                max_offsets = [0] * ndim
            # else: ndim should be consistent; silently take max

            for d, off in enumerate(offsets):
                if d < len(max_offsets):
                    max_offsets[d] = max(max_offsets[d], abs(off))

        if ndim is None:
            # No access ops found -- nothing to annotate
            return

        apply_op.attributes["halo_widths"] = ArrayAttr(
            [IntAttr(w) for w in max_offsets]
        )
