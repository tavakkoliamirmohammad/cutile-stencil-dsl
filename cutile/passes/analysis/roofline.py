"""
Roofline analysis pass.

Counts arithmetic operations and unique loads within each
``stencil.ApplyOp`` to compute arithmetic intensity and classify the
kernel as memory-bound or compute-bound.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith
from xdsl.dialects.builtin import FloatAttr, IntAttr, ModuleOp, StringAttr, f64
from xdsl.dialects.stencil import AccessOp, ApplyOp
from xdsl.passes import ModulePass

# Arithmetic intensity threshold (flops / byte).
# Below this value the kernel is classified as memory-bound.
# The value is approximate and corresponds roughly to the crossover on
# modern GPUs with ~1 TB/s bandwidth and ~10 TFLOPS peak FP64.
_COMPUTE_BOUND_THRESHOLD = 10.0

# Bytes per element for common floating-point types.
_DTYPE_BYTES: dict[str, int] = {
    "float64": 8,
    "float32": 4,
    "f64": 8,
    "f32": 4,
}


@dataclass(frozen=True)
class RooflinePass(ModulePass):
    """Compute roofline-model metrics for each ``stencil.ApplyOp``.

    The pass counts:

    * **flops** — number of floating-point arithmetic operations
      (``arith.addf``, ``arith.subf``, ``arith.mulf``, ``arith.divf``).
      ``arith.negf`` is considered free (typically a sign-bit flip).
      ``arith.constant`` is not counted.
    * **unique_loads** — number of distinct ``stencil.access`` offset
      patterns (de-duplicated by offset tuple).
    * **arithmetic_intensity** — ``flops / ((unique_loads + 1) * dtype_bytes)``.
      The ``+1`` accounts for the single output store per stencil point.
    * **bound** — ``"memory"`` or ``"compute"`` classification.

    Results are stored as attributes on each ``ApplyOp``.
    """

    name = "roofline-analysis"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for apply_op in _collect_apply_ops(op):
            self._analyze_apply(apply_op)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _analyze_apply(apply_op: ApplyOp) -> None:
        flops = 0
        unique_offsets: set[tuple[int, ...]] = set()

        for child in apply_op.region.walk():
            # Count arithmetic operations.
            if isinstance(child, (arith.AddfOp, arith.SubfOp)):
                flops += 1
            elif isinstance(child, arith.MulfOp):
                flops += 1
            elif isinstance(child, arith.DivfOp):
                flops += 1
            # arith.negf and arith.constant are free.

            # Collect unique loads.
            if isinstance(child, AccessOp):
                unique_offsets.add(tuple(child.offset))

        unique_loads = len(unique_offsets)

        # Determine dtype bytes from the element type of the result.
        dtype_bytes = _infer_dtype_bytes(apply_op)

        # Arithmetic intensity: flops / memory traffic per stencil point.
        # Memory traffic = (unique_loads + 1 store) * dtype_bytes.
        mem_traffic = (unique_loads + 1) * dtype_bytes
        if mem_traffic > 0:
            ai = flops / mem_traffic
        else:
            ai = 0.0

        bound = "compute" if ai >= _COMPUTE_BOUND_THRESHOLD else "memory"

        # Attach attributes.
        apply_op.attributes["flops"] = IntAttr(flops)
        apply_op.attributes["unique_loads"] = IntAttr(unique_loads)
        apply_op.attributes["arithmetic_intensity"] = FloatAttr(ai, f64)
        apply_op.attributes["bound"] = StringAttr(bound)


# ---------------------------------------------------------------------- #
# Utilities
# ---------------------------------------------------------------------- #


def _collect_apply_ops(module: ModuleOp) -> list[ApplyOp]:
    """Return all ``stencil.ApplyOp`` operations in the module."""
    return [op for op in module.walk() if isinstance(op, ApplyOp)]


def _infer_dtype_bytes(apply_op: ApplyOp) -> int:
    """Best-effort inference of element-type byte width from an ApplyOp."""
    # Try to read from result types.
    if apply_op.res:
        res_type = apply_op.res[0].type
        elem = res_type.element_type
        type_name = elem.name if hasattr(elem, "name") else str(elem)
        if type_name in _DTYPE_BYTES:
            return _DTYPE_BYTES[type_name]

    # Fallback: look for a cutile.dtype attribute on the parent func.
    parent = apply_op.parent_op()
    while parent is not None:
        dtype_attr = parent.attributes.get("cutile.dtype")
        if dtype_attr is not None and isinstance(dtype_attr, StringAttr):
            return _DTYPE_BYTES.get(dtype_attr.data, 8)
        parent = parent.parent_op()

    # Default: float64
    return 8
