"""
Test script for the normalization pass (Dialect 1 → Dialect 2) and
the footprint / roofline analysis passes.
"""

from cutile.frontend.decorator import stencil
from cutile.lowering.normalize import NormalizePass
from cutile.passes.analysis.footprint import AnalysisPass
from cutile.passes.analysis.roofline import RooflinePass
from xdsl.context import Context
from xdsl.dialects.stencil import Stencil
from xdsl.dialects.func import Func
from xdsl.dialects.arith import Arith
from cutile.dialects.cutile_stencil.dialect import CutileStencilDialect


def _make_context() -> Context:
    """Create an xDSL context with all required dialects loaded."""
    ctx = Context(allow_unregistered=True)
    ctx.load_dialect(Stencil)
    ctx.load_dialect(Func)
    ctx.load_dialect(Arith)
    ctx.load_dialect(CutileStencilDialect)
    return ctx


def main():
    # ----- define stencil via the DSL frontend ----- #
    @stencil(ndim=2, order=2)
    def heat2d(u, i, j):
        return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

    # The frontend attaches _ir (a ModuleOp) to the decorated function.
    module = heat2d._ir.clone()

    print("=" * 60)
    print("DIALECT 1 IR (before normalize)")
    print("=" * 60)
    print(module)
    print()

    # ----- Step 1: Normalize ----- #
    ctx = _make_context()
    NormalizePass().apply(ctx, module)

    print("=" * 60)
    print("DIALECT 2 IR (after normalize)")
    print("=" * 60)
    print(module)
    print()

    # ----- Step 2: Footprint analysis ----- #
    AnalysisPass().apply(ctx, module)

    print("=" * 60)
    print("After footprint analysis")
    print("=" * 60)
    print(module)
    print()

    # ----- Step 3: Roofline analysis ----- #
    RooflinePass().apply(ctx, module)

    print("=" * 60)
    print("After roofline analysis")
    print("=" * 60)
    print(module)
    print()

    # ----- Validate attributes ----- #
    from xdsl.dialects.stencil import ApplyOp
    from xdsl.dialects.builtin import IntAttr, FloatAttr, StringAttr, ArrayAttr

    for op in module.walk():
        if isinstance(op, ApplyOp):
            # Footprint
            halo = op.attributes.get("halo_widths")
            assert halo is not None, "halo_widths not found"
            assert isinstance(halo, ArrayAttr)
            halo_vals = [e.data for e in halo.data]
            print(f"Halo widths: {halo_vals}")
            assert halo_vals == [1, 1], f"Expected [1, 1], got {halo_vals}"

            n_unique = op.attributes.get("num_unique_accesses")
            assert n_unique is not None, "num_unique_accesses not found"
            print(f"Unique accesses: {n_unique.data}")
            assert n_unique.data == 4, f"Expected 4, got {n_unique.data}"

            # Roofline
            flops_attr = op.attributes.get("flops")
            assert flops_attr is not None, "flops not found"
            print(f"Flops: {flops_attr.data}")
            # 0.25 * (a + b + c + d)  → 3 addf + 1 mulf = 4 flops
            assert flops_attr.data == 4, f"Expected 4, got {flops_attr.data}"

            ul = op.attributes.get("unique_loads")
            assert ul is not None, "unique_loads not found"
            print(f"Unique loads: {ul.data}")
            assert ul.data == 4, f"Expected 4, got {ul.data}"

            ai = op.attributes.get("arithmetic_intensity")
            assert ai is not None, "arithmetic_intensity not found"
            print(f"Arithmetic intensity: {ai.value.data}")
            # AI = 4 / ((4 + 1) * 8) = 4/40 = 0.1
            expected_ai = 4.0 / (5 * 8)
            assert abs(ai.value.data - expected_ai) < 1e-9, (
                f"Expected AI={expected_ai}, got {ai.value.data}"
            )

            bound = op.attributes.get("bound")
            assert bound is not None, "bound not found"
            print(f"Bound: {bound.data}")
            assert bound.data == "memory", f"Expected 'memory', got {bound.data}"

    print()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
