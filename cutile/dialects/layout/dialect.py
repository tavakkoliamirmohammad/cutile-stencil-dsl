"""Data layout dialect for row-major and bricked memory layouts.

Provides a layout attribute and ops for casting between layouts
and computing linear addresses within a layout.
"""

from xdsl.ir import Dialect, ParametrizedAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    var_operand_def,
    result_def,
    prop_def,
    param_def,
    AnyAttr,
    AnyOf,
)
from xdsl.dialects.builtin import IntAttr, NoneAttr, StringAttr


@irdl_attr_definition
class LayoutAttr(ParametrizedAttribute):
    """Describes a data layout: row_major or bricked.

    For bricked layouts, brick_size specifies the brick dimension.
    For row_major layouts, brick_size is NoneAttr.
    """

    name = "layout.type"

    kind: StringAttr = param_def(StringAttr)
    brick_size: IntAttr | NoneAttr = param_def(AnyOf([IntAttr, NoneAttr]))


@irdl_op_definition
class CastOp(IRDLOperation):
    """Convert a buffer between two data layouts."""

    name = "layout.cast"

    from_layout = prop_def(LayoutAttr)
    to_layout = prop_def(LayoutAttr)

    input = operand_def(AnyAttr())

    result = result_def(AnyAttr())


@irdl_op_definition
class AddressOp(IRDLOperation):
    """Compute the linear address for given indices under a layout."""

    name = "layout.address"

    layout = prop_def(LayoutAttr)

    indices = var_operand_def(AnyAttr())

    result = result_def(AnyAttr())


LayoutDialect = Dialect("layout", [CastOp, AddressOp], [LayoutAttr])
