"""
Normalization pass: Dialect 1 (cutile_stencil.*) -> Standard stencil dialect.

Converts ``cutile_stencil.func`` / ``cutile_stencil.access`` /
``cutile_stencil.yield`` into ``func.func`` wrapping ``stencil.apply`` /
``stencil.access`` / ``stencil.return``, preserving metadata as attributes
on the resulting ops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from xdsl.context import Context
from xdsl.dialects import arith
from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IntAttr,
    ModuleOp,
    StringAttr,
    FunctionType,
    f32,
    f64,
)
from xdsl.dialects.func import FuncOp as StdFuncOp
from xdsl.dialects.stencil import (
    AccessOp as StdAccessOp,
    ApplyOp,
    IndexAttr,
    ReturnOp,
    TempType,
)
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.passes import ModulePass

from cutile.dialects.cutile_stencil.dialect import (
    AccessOp as D1AccessOp,
    FuncOp as D1FuncOp,
    YieldOp as D1YieldOp,
    BoundaryAttr,
)


@dataclass(frozen=True)
class NormalizePass(ModulePass):
    """Convert Dialect 1 IR to the standard xDSL stencil dialect.

    After this pass the module contains ``func.func`` ops wrapping
    ``stencil.apply`` / ``stencil.access`` / ``stencil.return``.  Stencil
    metadata (ndim, order, dtype, boundary) is preserved as attributes.
    """

    name: ClassVar[str] = "normalize"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        d1_funcs: list[D1FuncOp] = []
        for child in op.walk():
            if isinstance(child, D1FuncOp):
                d1_funcs.append(child)

        for d1_func in d1_funcs:
            self._convert_func(op, d1_func)

    # ------------------------------------------------------------------

    def _convert_func(self, module: ModuleOp, d1_func: D1FuncOp) -> None:
        func_name: str = d1_func.func_name.data
        ndim: int = d1_func.ndim.data
        order: int = d1_func.order.data
        dtype_str = (
            d1_func.dtype.data if d1_func.dtype is not None else "float64"
        )
        float_type = f32 if dtype_str == "float32" else f64

        # Number of input field arrays from the Dialect 1 body block args
        d1_body = d1_func.body.blocks[0]
        num_arrays: int = len(d1_body.args)

        # ---- build the stencil.apply body block ----
        temp_type = TempType(ndim, float_type)
        apply_block = Block()
        value_map: dict[SSAValue, SSAValue] = {}

        for idx in range(num_arrays):
            new_arg = apply_block.insert_arg(temp_type, idx)
            value_map[d1_body.args[idx]] = new_arg

        for d1_op in list(d1_body.ops):
            if isinstance(d1_op, D1AccessOp):
                field_val = value_map[d1_op.field]
                offsets = list(d1_op.offset)
                new_acc = StdAccessOp(field_val, offsets)
                apply_block.add_op(new_acc)
                value_map[d1_op.res] = new_acc.res

            elif isinstance(d1_op, D1YieldOp):
                ret_val = value_map[d1_op.value]
                apply_block.add_op(ReturnOp([ret_val]))

            elif isinstance(d1_op, arith.ConstantOp):
                new_op = d1_op.clone()
                apply_block.add_op(new_op)
                value_map[d1_op.results[0]] = new_op.results[0]

            elif isinstance(
                d1_op,
                (arith.AddfOp, arith.SubfOp, arith.MulfOp,
                 arith.DivfOp, arith.NegfOp),
            ):
                new_operands = [value_map[o] for o in d1_op.operands]
                new_op = type(d1_op)(*new_operands)
                apply_block.add_op(new_op)
                for old_r, new_r in zip(d1_op.results, new_op.results):
                    value_map[old_r] = new_r

            else:
                # Generic fallback -- clone and remap
                new_op = d1_op.clone()
                apply_block.add_op(new_op)
                for old_r, new_r in zip(d1_op.results, new_op.results):
                    value_map[old_r] = new_r

        # ---- stencil.apply ----
        result_temp_type = TempType(ndim, float_type)
        apply_op = ApplyOp([], apply_block, [result_temp_type])

        apply_op.attributes["ndim"] = IntAttr(ndim)
        apply_op.attributes["order"] = IntAttr(order)
        apply_op.attributes["dtype"] = StringAttr(dtype_str)
        apply_op.attributes["num_arrays"] = IntAttr(num_arrays)

        if d1_func.boundary is not None:
            apply_op.attributes["boundary"] = d1_func.boundary
        if d1_func.constants is not None:
            apply_op.attributes["constants"] = d1_func.constants

        # ---- func.func wrapper ----
        func_type = FunctionType.from_lists(
            [temp_type] * num_arrays, [result_temp_type],
        )
        func_block = Block()
        func_block.add_op(apply_op)

        std_func = StdFuncOp(func_name, func_type, Region(func_block))
        std_func.attributes["stencil_ndim"] = IntAttr(ndim)
        std_func.attributes["stencil_order"] = IntAttr(order)
        std_func.attributes["stencil_dtype"] = StringAttr(dtype_str)
        if d1_func.boundary is not None:
            std_func.attributes["boundary"] = d1_func.boundary

        # Replace Dialect 1 func with standard func in the module
        d1_func.parent_block().add_op(std_func)
        d1_func.detach()
        d1_func.erase()
