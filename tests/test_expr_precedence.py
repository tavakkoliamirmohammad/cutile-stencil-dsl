"""Operator precedence in emitted expressions.

Found by random-expression fuzzing: ``c * (-0.125 * abs(x) - y)`` was emitted
as ``_c0 * (-0.125) * (ct.abs(x)) - y`` because the parenthesization check
took a string that merely starts with ``(`` and ends with ``)`` for an
already-parenthesized operand.  The emitted expression must keep the source
tree's structure for every mix of products, sums, negations, calls and
constants.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cutile import stencil, where
from cutile.lowering.stencil_to_cutile import lower_stencil_to_python


def _result_expr(fn, ndim):
    code = lower_stencil_to_python(fn._ir, tile_sizes=(1024,) if ndim == 1 else (1, 256), halo_widths=(16,) if ndim == 1 else (2, 16))
    line = next(l for l in code.splitlines() if l.strip().startswith("result = "))
    return line.split("result = ", 1)[1].strip()


def _top_level_binary_ops(expr):
    """Binary operators at parenthesis depth 0 (a leading or post-operator '-' is unary)."""
    ops, depth, prev = [], 0, None
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and ch in "+-*/" and expr[i - 1:i] == " " and expr[i + 1:i + 2] == " ":
            ops.append(ch)
        i += 1
    return ops


class TestProductOfDifference:
    def test_negative_constant_times_difference_stays_a_product(self):
        @stencil(ndim=1, order=2)
        def f(u, i):
            return 0.961953 * (-0.125 * abs(u[i + 1] - u[i]) - u[i - 1])

        expr = _result_expr(f, 1)
        assert _top_level_binary_ops(expr) == ["*"], expr

    def test_where_inside_a_scaled_difference(self):
        @stencil(ndim=2, order=2)
        def g(a, b, i, j):
            return 0.3 * (-0.5 * abs(a[i + 1, j] - a[i, j - 1]) - where(b[i, j] > b[i + 1, j], a[i, j], 0.1 * b[i, j]))

        expr = _result_expr(g, 2)
        assert _top_level_binary_ops(expr) == ["*"], expr

    def test_division_by_a_sum_keeps_its_parentheses(self):
        @stencil(ndim=1, order=2)
        def h(u, i):
            return 0.8 * u[i - 1] / (u[i + 1] + 1.5) - 7 * (-u[i] - (2 * u[i + 1]))

        expr = _result_expr(h, 1)
        assert _top_level_binary_ops(expr) == ["*", "/", "-", "*"], expr


try:
    import cupy as cp
    cp.cuda.Device(0).compute_capability
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False


@pytest.mark.skipif(not _HAS_GPU, reason="No GPU available")
class TestGPU:
    def test_fuzz_case_matches_reference(self):
        from cutile import compile as stencil_compile

        @stencil(ndim=2, order=8)
        def s2(a, b, c, d, i, j):
            return (2.0 * b[i, j]
                    + 0.961953 * (-0.125 * abs(d[i + 4, j - 1] - d[i, j + 3])
                                  - where(c[i + 3, j + 3] > c[i + 4, j + 3], d[i - 1, j + 1], 0.1 * c[i + 3, j]))
                    + 0.8 * d[i, j - 1] / (d[i + 4, j + 1] + 1.5) - -a[i + 1, j - 4])

        result = stencil_compile(s2, temporal_blocking=False)
        assert result.inputs == ("a", "b", "c", "d")
        assert result.validate(atol=1e-11)
