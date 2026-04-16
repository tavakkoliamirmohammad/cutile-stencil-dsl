"""IRVisitor: shared dispatch base for cutile_target / comm IR walkers.

Every backend (single-GPU Python emitter, multi-GPU lowering, the
upcoming Triton/Numba paths) needs to walk the same IR and produce
target-specific code. They used to do this with hand-rolled
``isinstance`` chains repeated per backend. ``IRVisitor`` factors the
common dispatch pattern out so each backend just overrides the methods
for the ops it cares about.

Convention
----------
For an op class ``FooBarOp`` with ``name = "dialect.foo_bar"`` define
a method ``visit_foo_bar`` on the visitor. ``IRVisitor.visit(op)`` looks
up the method by name and dispatches; unknown ops invoke
``generic_visit`` which by default raises so silent gaps are visible.

Subclasses override only the ops they handle. If two backends share an
op handler (rare) they can share a mixin. Each ``visit_*`` returns
nothing — emission is recorded into a backend-supplied :class:`CodeEmitter`.

Why a class instead of ``@dispatch`` decorators
-----------------------------------------------
Backends carry state — the active :class:`CodeEmitter`, kernel name,
tile sizes — that's natural to keep on ``self``. Per-method dispatch
also means a backend's surface area is one class file the reader can
skim top-to-bottom.
"""

from __future__ import annotations

import re
from typing import Iterable

from xdsl.ir import Operation, Region


# Convert "dialect.foo_bar" -> "visit_foo_bar"; "DialectOp" class names
# are not consulted (the canonical key is the op's `name` attribute).
_OP_NAME_RE = re.compile(r"[^a-zA-Z0-9_]")


def _method_name(op_name: str) -> str:
    """Map an op's IR name (``dialect.foo_bar``) to ``visit_foo_bar``."""
    base = op_name.split(".", 1)[-1]
    return "visit_" + _OP_NAME_RE.sub("_", base)


class IRVisitor:
    """Visitor base. Override ``visit_<op_short_name>`` per op."""

    def visit(self, op: Operation) -> None:
        """Dispatch to the matching ``visit_*`` method for ``op``."""
        method_name = _method_name(op.name)
        handler = getattr(self, method_name, None)
        if handler is None:
            return self.generic_visit(op)
        return handler(op)

    def generic_visit(self, op: Operation) -> None:
        """Fallback when no specific handler exists.

        Default: raise so missing handlers are loud. Subclasses can
        override to silently skip unknown ops if that is desirable for
        their backend.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no handler for {op.name!r} "
            f"(expected method {_method_name(op.name)!r})"
        )

    def visit_block(self, ops: Iterable[Operation]) -> None:
        """Visit each op in a block in order."""
        for op in ops:
            self.visit(op)

    def visit_region(self, region: Region) -> None:
        """Visit the (single) entry block of a region in order."""
        for block in region.blocks:
            self.visit_block(block.ops)
