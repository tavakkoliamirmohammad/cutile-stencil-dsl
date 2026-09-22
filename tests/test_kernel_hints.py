"""Compiler hints on the generated ``@ct.kernel`` decorator.

cuTile lets a kernel declare the number of resident blocks per SM it expects
(``occupancy``); tileiras then budgets registers for that occupancy instead
of hoisting every tile load.  The DSL exposes this per compile and lets the
tiling pass pick a value for wide stencils.
"""

import ast

from cutile import compile as stencil_compile
from cutile import stencil
from cutile.lowering.stencil_to_cutile import lower_stencil_to_python


def _heat():
    @stencil(ndim=3, order=2)
    def heat(u, i, j, k):
        return (u[i - 1, j, k] + u[i + 1, j, k] + u[i, j - 1, k] + u[i, j + 1, k]
                + u[i, j, k - 1] + u[i, j, k + 1] - 6.0 * u[i, j, k])
    return heat


def _decorator(code: str) -> str:
    return next(line for line in code.splitlines() if line.startswith("@ct.kernel"))


class TestKernelHints:
    def test_default_has_bare_decorator(self):
        code = lower_stencil_to_python(_heat()._ir, tile_sizes=(1, 1, 256), halo_widths=(1, 1, 16))
        assert _decorator(code) == "@ct.kernel"

    def test_occupancy_hint_is_emitted(self):
        code = lower_stencil_to_python(
            _heat()._ir, tile_sizes=(1, 1, 256), halo_widths=(1, 1, 16),
            kernel_hints={"occupancy": 8},
        )
        ast.parse(code)
        assert _decorator(code) == "@ct.kernel(occupancy=8)"

    def test_hints_are_emitted_in_sorted_order(self):
        code = lower_stencil_to_python(
            _heat()._ir, tile_sizes=(1, 1, 256), halo_widths=(1, 1, 16),
            kernel_hints={"occupancy": 8, "num_worker_warps": 4},
        )
        assert _decorator(code) == "@ct.kernel(num_worker_warps=4, occupancy=8)"

    def test_compile_passes_occupancy_through(self):
        result = stencil_compile(_heat(), temporal_blocking=False, occupancy=6)
        assert "@ct.kernel(occupancy=6)" in result.code
        assert result.kernel_hints == {"occupancy": 6}

    def test_compile_default_has_no_hints_for_narrow_stencil(self):
        result = stencil_compile(_heat(), temporal_blocking=False)
        assert result.kernel_hints == {}
        assert "@ct.kernel\n" in result.code
