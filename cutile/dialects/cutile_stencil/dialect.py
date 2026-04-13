"""
cuTile Stencil Dialect (Dialect 1).

This dialect mirrors the Python ``@stencil`` decorator syntax and serves as
the first stage of a three-dialect lowering stack.  It captures stencil
metadata (dimensionality, order, boundary conditions) alongside the
computation body so that later passes can analyze access patterns and
generate tiled GPU kernels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    FloatAttr,
    IntAttr,
    StringAttr,
    f32,
    f64,
)
from xdsl.dialects.stencil import IndexAttr
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import IsTerminator

# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


@irdl_attr_definition
class BoundaryAttr(ParametrizedAttribute):
    """Boundary-condition specification for a stencil.

    Encodes the type of boundary (``dirichlet``, ``neumann``, ``periodic``,
    ``reflecting``) and an optional constant value (used, e.g., for Dirichlet
    conditions).
    """

    name = "cutile_stencil.boundary"

    bc_type: StringAttr
    value: FloatAttr | IntAttr  # IntAttr used as a sentinel for "no value"

    def __init__(
        self,
        bc_type: str | StringAttr,
        value: float | FloatAttr | None = None,
    ):
        if isinstance(bc_type, str):
            bc_type = StringAttr(bc_type)

        if value is None:
            # Sentinel: IntAttr(0) signals "no boundary value supplied".
            value_attr: FloatAttr | IntAttr = IntAttr(0)
        elif isinstance(value, (int, float)):
            value_attr = FloatAttr(float(value), f64)
        else:
            value_attr = value

        super().__init__(bc_type, value_attr)

    @property
    def has_value(self) -> bool:
        """Return *True* when an explicit boundary value was provided."""
        return isinstance(self.value, FloatAttr)

    @property
    def bc_type_str(self) -> str:
        return self.bc_type.data

    _VALID_BC_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"dirichlet", "neumann", "periodic", "reflecting"}
    )

    def verify(self) -> None:
        if self.bc_type.data not in self._VALID_BC_TYPES:
            from xdsl.utils.exceptions import VerifyException

            raise VerifyException(
                f"Invalid boundary type '{self.bc_type.data}'. "
                f"Expected one of {sorted(self._VALID_BC_TYPES)}."
            )


@irdl_attr_definition
class StencilSpecAttr(ParametrizedAttribute):
    """Compound attribute holding full stencil metadata.

    Parameters
    ----------
    ndim : IntAttr
        Number of spatial dimensions (1-3).
    order : IntAttr
        Finite-difference order of accuracy.
    dtype : StringAttr
        Element data-type name (``"float32"`` / ``"float64"``).
    boundary : BoundaryAttr | ArrayAttr | IntAttr
        Boundary condition specification.  A single ``BoundaryAttr`` applies
        uniformly; an ``ArrayAttr`` of ``BoundaryAttr`` gives per-dimension
        conditions.  ``IntAttr(0)`` is a sentinel for *none supplied*.
    """

    name = "cutile_stencil.spec"

    ndim: IntAttr
    order: IntAttr
    dtype: StringAttr
    boundary: BoundaryAttr | ArrayAttr | IntAttr

    def __init__(
        self,
        ndim: int | IntAttr,
        order: int | IntAttr,
        dtype: str | StringAttr = "float64",
        boundary: BoundaryAttr | ArrayAttr | None = None,
    ):
        if isinstance(ndim, int):
            ndim = IntAttr(ndim)
        if isinstance(order, int):
            order = IntAttr(order)
        if isinstance(dtype, str):
            dtype = StringAttr(dtype)
        if boundary is None:
            boundary_attr: BoundaryAttr | ArrayAttr | IntAttr = IntAttr(0)
        else:
            boundary_attr = boundary

        super().__init__(ndim, order, dtype, boundary_attr)

    @property
    def has_boundary(self) -> bool:
        return not isinstance(self.boundary, IntAttr)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


@irdl_op_definition
class FuncOp(IRDLOperation):
    """A complete stencil function.

    Mirrors the Python ``@stencil(ndim=…, order=…, dtype=…)`` decorator.
    The body region contains the stencil computation expressed via
    ``AccessOp``, ``ComputeOp`` and ``YieldOp``.
    """

    name = "cutile_stencil.func"

    # -- properties (appear in the op's printed form) --
    func_name = prop_def(StringAttr)
    ndim = prop_def(IntAttr)
    order = prop_def(IntAttr)
    dtype = opt_prop_def(StringAttr)
    boundary = opt_prop_def(BoundaryAttr)
    constants = opt_prop_def(DictionaryAttr)

    # -- operands / results --
    inputs = var_operand_def(AnyAttr())
    outputs = var_result_def(AnyAttr())

    # -- body --
    body = region_def()

    def __init__(
        self,
        func_name: str | StringAttr,
        ndim: int | IntAttr,
        order: int | IntAttr,
        body: Region | Block,
        input_types: Sequence[Attribute] = (),
        output_types: Sequence[Attribute] = (),
        dtype: str | StringAttr | None = None,
        boundary: BoundaryAttr | None = None,
        constants: DictionaryAttr | None = None,
        *,
        operands: Sequence[SSAValue | Operation] = (),
    ):
        if isinstance(func_name, str):
            func_name = StringAttr(func_name)
        if isinstance(ndim, int):
            ndim = IntAttr(ndim)
        if isinstance(order, int):
            order = IntAttr(order)
        if isinstance(dtype, str):
            dtype = StringAttr(dtype)
        if isinstance(body, Block):
            body = Region(body)

        properties: dict[str, Attribute] = {
            "func_name": func_name,
            "ndim": ndim,
            "order": order,
        }
        if dtype is not None:
            properties["dtype"] = dtype
        if boundary is not None:
            properties["boundary"] = boundary
        if constants is not None:
            properties["constants"] = constants

        super().__init__(
            operands=[list(operands)],
            result_types=[list(output_types)],
            regions=[body],
            properties=properties,
        )


@irdl_op_definition
class AccessOp(IRDLOperation):
    """Read from a field/temp at a relative offset.

    Example (2-D, centre access)::

        %v = cutile_stencil.access %field [0, 0] : f64

    Custom printer produces::

        cutile_stencil.access %field [-1, 0] {"i", "j"} : f64
    """

    name = "cutile_stencil.access"

    field = operand_def(AnyAttr())
    offset = attr_def(IndexAttr)
    index_names = opt_attr_def(ArrayAttr)
    res = result_def(AnyAttr())

    def __init__(
        self,
        field: SSAValue | Operation,
        offset: Sequence[int] | IndexAttr,
        result_type: Attribute,
        index_names: Sequence[str] | ArrayAttr | None = None,
    ):
        if isinstance(offset, Sequence):
            offset = IndexAttr.from_indices(*offset)

        attributes: dict[str, Attribute] = {"offset": offset}

        if index_names is not None:
            if isinstance(index_names, Sequence):
                index_names = ArrayAttr(
                    [StringAttr(n) for n in index_names]
                )
            attributes["index_names"] = index_names

        super().__init__(
            operands=[field],
            attributes=attributes,
            result_types=[result_type],
        )

    # -- Custom printer / parser ----------------------------------------

    def print(self, printer) -> None:  # type: ignore[override]
        """Print a compact form: ``cutile_stencil.access %field [-1, 0] : f64``."""
        printer.print(" ")
        printer.print_operand(self.field)
        printer.print(" ")

        # Offset as [d0, d1, …]
        offsets = [int(idx.data) for idx in self.offset.parameters[0].data]
        printer.print("[")
        printer.print(", ".join(str(o) for o in offsets))
        printer.print("]")

        # Optional index names
        if self.index_names is not None:
            names = [a.data for a in self.index_names]
            printer.print(" {")
            printer.print(", ".join(f'"{n}"' for n in names))
            printer.print("}")

        # Result type
        printer.print(" : ")
        printer.print_attribute(self.res.type)


@irdl_op_definition
class ComputeOp(IRDLOperation):
    """Contains arithmetic computation as a region.

    The body region should end with a ``YieldOp`` producing the scalar
    result of the stencil point evaluation.
    """

    name = "cutile_stencil.compute"

    body = region_def()
    res = result_def(AnyAttr())

    def __init__(
        self,
        body: Region | Block,
        result_type: Attribute,
    ):
        if isinstance(body, Block):
            body = Region(body)

        super().__init__(
            regions=[body],
            result_types=[result_type],
        )


@irdl_op_definition
class YieldOp(IRDLOperation):
    """Terminates a ``ComputeOp`` region, yielding the computed value."""

    name = "cutile_stencil.yield"

    value = operand_def(AnyAttr())

    assembly_format = "$value attr-dict `:` type($value)"

    traits = traits_def(IsTerminator())

    def __init__(self, value: SSAValue | Operation):
        super().__init__(operands=[value])


# ---------------------------------------------------------------------------
# Dialect registration
# ---------------------------------------------------------------------------

CutileStencilDialect = Dialect(
    "cutile_stencil",
    [FuncOp, AccessOp, ComputeOp, YieldOp],
    [BoundaryAttr, StencilSpecAttr],
)
