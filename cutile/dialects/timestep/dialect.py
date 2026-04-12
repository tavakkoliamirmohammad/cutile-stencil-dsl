"""Time integration dialect for Runge-Kutta and related methods.

Provides ops to represent individual RK stages, stage combination,
and the enclosing time-step loop.
"""

from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    var_operand_def,
    result_def,
    prop_def,
    region_def,
    traits_def,
    AnyAttr,
)
from xdsl.traits import IsTerminator
from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntAttr, StringAttr


@irdl_op_definition
class StageOp(IRDLOperation):
    """Compute one RK stage: u_stage = u + coeff * dt * k."""

    name = "timestep.stage"

    coeff = prop_def(FloatAttr)
    dt_name = prop_def(StringAttr)

    base_state = operand_def(AnyAttr())
    stage_derivative = operand_def(AnyAttr())

    result = result_def(AnyAttr())


@irdl_op_definition
class CombineOp(IRDLOperation):
    """Combine all RK stages: u_new = u + dt * sum(weight_i * k_i)."""

    name = "timestep.combine"

    weights = prop_def(ArrayAttr)
    dt_name = prop_def(StringAttr)

    base_state = operand_def(AnyAttr())
    stage_derivatives = var_operand_def(AnyAttr())

    result = result_def(AnyAttr())


@irdl_op_definition
class RKLoopOp(IRDLOperation):
    """Enclosing op for a full Runge-Kutta time step."""

    name = "timestep.rk_loop"

    method = prop_def(StringAttr)
    num_stages = prop_def(IntAttr)

    body = region_def()

    inputs = var_operand_def(AnyAttr())


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """Terminator for the RK loop body."""

    name = "timestep.return"

    values = var_operand_def(AnyAttr())

    traits = traits_def(IsTerminator())


TimestepDialect = Dialect(
    "timestep", [StageOp, CombineOp, RKLoopOp, ReturnOp], []
)
