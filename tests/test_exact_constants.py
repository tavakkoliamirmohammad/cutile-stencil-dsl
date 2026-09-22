"""cuTile rounds Python float scalars to float32, even in float64 kernels.

Constants that are not exactly representable in float32 (0.1, 1/12, ...)
therefore lose ~8 digits if they are inlined as literals.  The emitter
must fold constant sub-expressions in Python (float64) and deliver any
remaining non-representable constant to the kernel through a small
float64 array that the kernel loads with one tile load.
"""

import ast

from cutile import stencil
from cutile.lowering.stencil_to_cutile import lower_stencil_to_python


def _kernel_source(code: str) -> str:
    """Return the text of the ``@ct.kernel`` function only."""
    start = code.index("@ct.kernel")
    end = code.index("\ndef launch_", start)
    return code[start:end]


def _result_line(code: str) -> str:
    return next(line for line in code.splitlines() if "result =" in line)


class TestExactConstants:
    def test_non_representable_constant_is_loaded_not_inlined(self):
        c = 0.1

        @stencil(ndim=1, order=2)
        def damp(u, i):
            return c * u[i] + 0.25 * u[i - 1]

        code = lower_stencil_to_python(damp._ir, tile_sizes=(1024,), halo_widths=(16,))
        ast.parse(code)
        kernel = _kernel_source(code)
        assert "_kernel_constants()" in code
        assert "0.1 *" not in kernel and "0.1*" not in kernel
        assert "_c0" in _result_line(code)
        assert "_KERNEL_CONSTANTS = [0.1]" in code

    def test_constant_subexpressions_are_folded_in_float64(self):
        c = 0.1

        @stencil(ndim=2, order=4)
        def wave(u, i, j):
            return c * (-1 / 12) * u[i - 2, j] + c * (4 / 3) * u[i - 1, j]

        code = lower_stencil_to_python(wave._ir, tile_sizes=(4, 256), halo_widths=(2, 16))
        ast.parse(code)
        kernel = _kernel_source(code)
        assert "/ 12.0" not in kernel and "/ 3.0" not in kernel
        assert f"_KERNEL_CONSTANTS = [{c * (-1 / 12)!r}, {c * (4 / 3)!r}]" in code

    def test_float32_exact_constants_stay_inline(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        code = lower_stencil_to_python(heat2d._ir, tile_sizes=(4, 256), halo_widths=(1, 16))
        assert "_kernel_constants" not in code
        assert "0.25 *" in _result_line(code)

    def test_float32_stencil_keeps_literals(self):
        c = 0.1

        @stencil(ndim=1, order=2, dtype="float32")
        def damp(u, i):
            return c * u[i]

        code = lower_stencil_to_python(damp._ir, tile_sizes=(1024,), halo_widths=(32,))
        assert "_kernel_constants" not in code

    def test_constants_are_passed_to_every_launch(self):
        c = 0.1

        @stencil(ndim=1, order=2)
        def damp(u, i):
            return c * u[i]

        code = lower_stencil_to_python(
            damp._ir, tile_sizes=(1024,), halo_widths=(16,), temporal_steps=3
        )
        ast.parse(code)
        assert "def damp_kernel(u, output, consts, TX: ConstInt, HX: ConstInt):" in code
        assert "(bufs[_step], bufs[_step + 1], _kernel_constants(), TX, HX)" in code

    def test_kernel_loads_constants_with_one_power_of_two_tile(self):
        a, b, d = 0.1, 0.2, 0.3

        @stencil(ndim=1, order=2)
        def mix(u, i):
            return a * u[i - 1] + b * u[i] + d * u[i + 1]

        code = lower_stencil_to_python(mix._ir, tile_sizes=(1024,), halo_widths=(16,))
        kernel = _kernel_source(code)
        assert kernel.count("ct.load(consts") == 1
        assert "shape=(4,)" in kernel  # 3 constants padded to 4
        assert "_c2 = " in kernel
