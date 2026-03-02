"""AST-level transformations for stencil code generation.

Replaces fragile string-based substitution with proper AST node transformers.
Fixes the wave_2d bug where intermediate variables (lap_x, lap_y) were left
undefined in generated code.
"""

from __future__ import annotations

import ast
import copy
import inspect
import textwrap
import types
from typing import Any, Dict, List, Tuple


class InlineTransformer(ast.NodeTransformer):
    """Inline all intermediate assignments into the return expression.

    Given::

        lap_x = expr1
        lap_y = expr2
        return 0.1 * (lap_x + lap_y)

    Produces a single return expression with lap_x and lap_y replaced
    by their defining expressions.
    """

    def __init__(self):
        self.bindings: Dict[str, ast.expr] = {}

    def visit_Assign(self, node: ast.Assign) -> ast.Assign | None:
        # Only handle simple name targets: x = expr
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            # Inline any previously bound variables in this expression
            value = self._substitute(node.value)
            self.bindings[name] = value
            return None  # Remove the assignment
        return node

    def visit_Return(self, node: ast.Return) -> ast.Return:
        if node.value is not None:
            node.value = self._substitute(node.value)
        return node

    def _substitute(self, node: ast.expr) -> ast.expr:
        """Replace all Name references with their bound expressions."""
        return _NameReplacer(self.bindings).visit(copy.deepcopy(node))


class _NameReplacer(ast.NodeTransformer):
    """Replace Name nodes with bound expressions."""

    def __init__(self, bindings: Dict[str, ast.expr]):
        self.bindings = bindings

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id in self.bindings:
            return copy.deepcopy(self.bindings[node.id])
        return node


class SubscriptReplacer(ast.NodeTransformer):
    """Replace array subscript AST nodes with tile variable Name nodes.

    Given a mapping from (array_name, offsets_tuple) -> replacement_name,
    transforms ``u[i + 1, j - 2]`` into ``Name(id='nb_p1_m2')``.
    """

    def __init__(
        self,
        replacements: Dict[Tuple[str, Tuple[int, ...]], str],
        index_names: Tuple[str, ...],
    ):
        self.replacements = replacements
        self.index_names = index_names

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        if not isinstance(node.value, ast.Name):
            return self.generic_visit(node)

        arr_name = node.value.id
        slc = node.slice
        if isinstance(slc, ast.Tuple):
            elts = slc.elts
        else:
            elts = [slc]

        offsets = []
        for elt in elts:
            off = self._extract_offset(elt)
            if off is None:
                return self.generic_visit(node)
            offsets.append(off)

        key = (arr_name, tuple(offsets))
        if key in self.replacements:
            return ast.Name(id=self.replacements[key], ctx=ast.Load())

        return self.generic_visit(node)

    def _extract_offset(self, node: ast.expr) -> int | None:
        """Resolve an index expression to a constant offset."""
        if isinstance(node, ast.Name) and node.id in self.index_names:
            return 0
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.BinOp):
            left = self._extract_offset(node.left)
            right = self._extract_offset(node.right)
            if left is not None and right is not None:
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val = self._extract_offset(node.operand)
            if val is not None:
                return -val
        return None


def get_inlined_return_expr(func, ndim: int) -> str:
    """Extract the return expression from a stencil function, inlining intermediates.

    Returns the expression as a source string with all intermediate variables
    expanded inline.
    """
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)

    # Find the function definition
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break

    if func_def is None:
        raise ValueError("Could not find function definition in source")

    # Apply inlining
    inliner = InlineTransformer()
    new_body = []
    for stmt in func_def.body:
        result = inliner.visit(stmt)
        if result is not None:
            new_body.append(result)

    # Find the return expression
    for stmt in new_body:
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            return ast.unparse(stmt.value)

    raise ValueError("No return expression found after inlining")


def transform_stencil_expr(
    func,
    accesses: list,
    ndim: int,
    term_names: List[str],
) -> str:
    """Full AST transform: inline intermediates, then replace subscripts with tile names.

    Parameters
    ----------
    func : callable
        The stencil update function.
    accesses : list of OffsetAccess
        The stencil accesses (array_name, offsets).
    ndim : int
        Number of spatial dimensions.
    term_names : list of str
        Replacement variable names, one per access.

    Returns
    -------
    str
        The transformed expression as a Python source string.
    """
    try:
        src = textwrap.dedent(inspect.getsource(func))
    except OSError:
        if hasattr(func, '_source'):
            src = func._source
        else:
            raise
    tree = ast.parse(src)

    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break

    if func_def is None:
        raise ValueError("Could not find function definition")

    # Get index parameter names
    params = list(func_def.args.args)
    index_names = tuple(p.arg for p in params[-ndim:])

    # Step 1: Inline intermediate assignments
    inliner = InlineTransformer()
    new_body = []
    for stmt in func_def.body:
        result = inliner.visit(stmt)
        if result is not None:
            new_body.append(result)

    # Find return expression
    return_expr = None
    for stmt in new_body:
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            return_expr = stmt.value
            break

    if return_expr is None:
        raise ValueError("No return expression found")

    # Step 2: Replace subscripts with tile variable names
    replacements = {}
    for acc, tname in zip(accesses, term_names):
        key = (acc.array_name, acc.offsets)
        if key not in replacements:
            replacements[key] = tname

    replacer = SubscriptReplacer(replacements, index_names)
    transformed = replacer.visit(copy.deepcopy(return_expr))
    ast.fix_missing_locations(transformed)

    return ast.unparse(transformed)


def extract_closure_constants(func) -> Dict[str, Any]:
    """Extract captured closure variables from a stencil function.

    When a stencil function references variables from its enclosing scope
    (e.g., ``dt``, ``eps``, ``dx``), these are captured in the function's
    closure. This function extracts them as a name->value mapping so the
    code generator can emit them as constants in the generated kernel.

    Only extracts int, float, bool, and str values (safe scalar constants).
    Skips non-scalar values (arrays, modules, functions, etc.).

    Parameters
    ----------
    func : callable
        The stencil update function.

    Returns
    -------
    dict
        Mapping from variable name to its constant value.
    """
    constants: Dict[str, Any] = {}

    # Method 1: Check __closure__ and __code__.co_freevars
    if hasattr(func, '__closure__') and func.__closure__ is not None:
        freevars = func.__code__.co_freevars
        for name, cell in zip(freevars, func.__closure__):
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if isinstance(val, (int, float, bool, str)):
                constants[name] = val

    # Method 2: Check __globals__ for names used in the function body
    # that aren't parameters, builtins, or array names
    if hasattr(func, '__code__'):
        try:
            src = textwrap.dedent(inspect.getsource(func))
        except OSError:
            if hasattr(func, '_source'):
                src = func._source
            else:
                return constants
        tree = ast.parse(src)
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_def = node
                break

        if func_def is not None:
            param_names = {arg.arg for arg in func_def.args.args}
            # Collect all Name references in the function body
            used_names = set()
            for node in ast.walk(func_def):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used_names.add(node.id)

            # Check globals for scalar constants
            func_globals = getattr(func, '__globals__', {})
            for name in used_names - param_names:
                if name in constants:
                    continue  # Already captured from closure
                if name in func_globals:
                    val = func_globals[name]
                    if isinstance(val, (int, float, bool, str)):
                        constants[name] = val

    return constants
