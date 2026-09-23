"""Loads are inlined at their use site so they interleave with the arithmetic.

tileiras keeps every tile that has been loaded live until it is consumed.
With all ``ct.load`` statements hoisted above one big expression, a
125-point stencil holds 125 tiles at once (254 registers, spills), which
halves occupancy.  Emitting each single-use load inside the expression, in
left-to-right evaluation order, lets the compiler consume tiles as they
arrive: measured 15.0 ms -> 9.4 ms for the 125-point box stencil at two
elements per thread, and identical times where pressure was already low.
Accesses used more than once must stay as one hoisted load.
"""

import ast
import re

from cutile import stencil
from cutile.lowering.stencil_to_cutile import lower_stencil_to_python


def _kernel(code: str) -> str:
    return code[code.index("@ct.kernel"):code.index("\ndef launch_")]


def _result_line(code: str) -> str:
    return next(line for line in code.splitlines() if line.strip().startswith("result ="))


class TestInlineLoads:
    def test_single_use_loads_are_inlined_in_expression(self):
        @stencil(ndim=3, order=2)
        def lap(u, i, j, k):
            return (u[i - 1, j, k] + u[i + 1, j, k] + u[i, j - 1, k] + u[i, j + 1, k]
                    + u[i, j, k - 1] + u[i, j, k + 1] - 6.0 * u[i, j, k])

        code = lower_stencil_to_python(lap._ir, tile_sizes=(1, 1, 256), halo_widths=(1, 1, 16))
        ast.parse(code)
        k = _kernel(code)
        assert _result_line(code).count("ct.load(") == 7
        assert not re.search(r"^\s*t_u\w* = ct\.load", k, re.M)
        # order of evaluation follows the source: first term loads u[i-1]
        assert _result_line(code).index("ct.load(u_m1_0_0") < _result_line(code).index("ct.load(u_p1_0_0")

    def test_multi_use_access_stays_a_single_hoisted_load(self):
        @stencil(ndim=2, order=2)
        def gs(u, v, i, j):
            return u[i, j] + 0.5 * (u[i - 1, j] + u[i + 1, j] - 2.0 * u[i, j]) - u[i, j] * v[i, j] * v[i, j]

        code = lower_stencil_to_python(gs._ir, tile_sizes=(1, 256), halo_widths=(1, 16))
        ast.parse(code)
        k = _kernel(code)
        assert k.count("t_u_0_0 = ct.load(u_0_0") == 1
        assert k.count("t_v_0_0 = ct.load(v_0_0") == 1
        assert _result_line(code).count("t_u_0_0") == 3
        assert _result_line(code).count("t_v_0_0") == 2
        # the two single-use neighbours are inlined
        assert "ct.load(u_m1_0" in _result_line(code)
        assert "ct.load(u_p1_0" in _result_line(code)

    def test_view_definitions_precede_use(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        code = lower_stencil_to_python(heat._ir, tile_sizes=(1024,), halo_widths=(16,))
        k = _kernel(code)
        assert k.index("u_m1 = u.slice(") < k.index("result =")
        assert "shape=(TX,)" in _result_line(code)
