"""
Boundary pass: propagate boundary-condition metadata to ``stencil.ApplyOp``.

Reads boundary specifications from the enclosing ``func.func`` (set during
normalisation from the original ``cutile_stencil.FuncOp``'s boundary
property) and attaches appropriate attributes on each ``stencil.ApplyOp``
so that the lowering / emitter can generate the correct boundary-handling
code.

* **Dirichlet** -> ``padding_mode = "zero"`` (GPU-side zero-fill of halo
  elements).
* **Periodic / Neumann / Reflecting** -> ``boundary_fn = "host_side"``
  (boundary handling is done host-side before/after each kernel launch).

This is a metadata-attaching pass; actual code generation happens later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.dialects.func import FuncOp
from xdsl.dialects.stencil import ApplyOp
from xdsl.passes import ModulePass

from cutile.dialects.cutile_stencil.dialect import BoundaryAttr


@dataclass(frozen=True)
class BoundaryPass(ModulePass):
    """Attach boundary-handling attributes on each ``stencil.ApplyOp``."""

    name: ClassVar[str] = "stencil-boundary"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for child in op.walk():
            if isinstance(child, FuncOp):
                self._process_func(child)

    # ------------------------------------------------------------------

    @staticmethod
    def _process_func(func_op: FuncOp) -> None:
        # Look for boundary attribute on the func (set by NormalizePass)
        boundary_attr = func_op.attributes.get("boundary")
        if boundary_attr is None:
            return

        # Propagate to every stencil.apply inside this func
        for inner_op in func_op.walk():
            if not isinstance(inner_op, ApplyOp):
                continue

            # Attach the raw boundary spec so lowering can inspect it
            inner_op.attributes["boundary_spec"] = boundary_attr

            # Determine the handling strategy based on BC type
            if isinstance(boundary_attr, BoundaryAttr):
                bc_type = boundary_attr.bc_type_str
            elif isinstance(boundary_attr, StringAttr):
                bc_type = boundary_attr.data
            else:
                # Fallback: try to read as a string-like attr
                bc_type = str(boundary_attr)

            if bc_type == "dirichlet":
                inner_op.attributes["padding_mode"] = StringAttr("zero")
            elif bc_type in ("periodic", "neumann", "reflecting"):
                inner_op.attributes["boundary_fn"] = StringAttr("host_side")
