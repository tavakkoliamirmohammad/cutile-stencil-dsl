"""AST-based stencil footprint extraction.

Walks the stencil function body, finds all subscript accesses like ``u[i+1]``
or ``u[i-1, j+2]``, and returns them as ``OffsetAccess`` instances.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import List, Tuple

from cutile_stencil.dsl.types import OffsetAccess, StencilSpec


class _OffsetVisitor(ast.NodeVisitor):
    """Walks AST to extract array subscript accesses with integer offsets."""

    def __init__(self, array_names: Tuple[str, ...], index_names: Tuple[str, ...]):
        self.array_names = array_names
        self.index_names = index_names
        self.accesses: List[OffsetAccess] = []

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in self.array_names:
            arr_name = node.value.id
            slc = node.slice
            # Normalise to a list of index expressions
            if isinstance(slc, ast.Tuple):
                elts = slc.elts
            else:
                elts = [slc]
            offsets = []
            for elt in elts:
                offset = self._extract_offset(elt)
                if offset is not None:
                    offsets.append(offset)
                else:
                    # Could not determine offset — skip this access
                    offsets = None
                    break
            if offsets is not None:
                self.accesses.append(OffsetAccess(arr_name, tuple(offsets)))
        self.generic_visit(node)

    def _extract_offset(self, node: ast.expr) -> int | None:
        """Resolve an index expression to a constant integer offset."""
        # Plain index var: u[i] → offset 0
        if isinstance(node, ast.Name) and node.id in self.index_names:
            return 0
        # Constant integer: u[0] → 0
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        # BinOp: i + N or i - N
        if isinstance(node, ast.BinOp):
            left_off = self._extract_offset(node.left)
            right_off = self._extract_offset(node.right)
            if left_off is not None and right_off is not None:
                if isinstance(node.op, ast.Add):
                    return left_off + right_off
                if isinstance(node.op, ast.Sub):
                    return left_off - right_off
        # UnaryOp: -N
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val = self._extract_offset(node.operand)
            if val is not None:
                return -val
        return None


def extract_footprint(spec: StencilSpec) -> List[OffsetAccess]:
    """Extract all array offset accesses from a stencil function."""
    src = textwrap.dedent(inspect.getsource(spec.update_fn))
    tree = ast.parse(src)
    sig = inspect.signature(spec.update_fn)
    params = list(sig.parameters.keys())
    index_names = tuple(params[-spec.ndim:])
    visitor = _OffsetVisitor(spec.inputs, index_names)
    visitor.visit(tree)
    spec.accesses = visitor.accesses
    return visitor.accesses


def compute_halo(accesses: List[OffsetAccess], ndim: int) -> Tuple[int, ...]:
    """Compute the maximum absolute offset per dimension (the halo width)."""
    halos = [0] * ndim
    for acc in accesses:
        for d, off in enumerate(acc.offsets):
            halos[d] = max(halos[d], abs(off))
    return tuple(halos)


def compute_max_cfl(spec: StencilSpec) -> float:
    """Estimate the maximum stable CFL number for an explicit stencil.

    For an explicit time-stepping stencil of order p in d dimensions,
    the CFL constraint is approximately:
        CFL <= 1 / (d * p)

    This is a conservative heuristic based on Von Neumann stability analysis
    for standard finite-difference schemes.

    Parameters
    ----------
    spec : StencilSpec
        Stencil specification with ndim and order.

    Returns
    -------
    float
        Estimated maximum stable CFL number.
    """
    # Heuristic: for order-p stencil in d dimensions
    d = spec.ndim
    p = max(spec.order, 1)
    return 1.0 / (d * (p / 2))
