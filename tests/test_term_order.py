"""Very wide stencils sum their terms in spatial (x, y, z) order.

tileiras issues loads in the order the sum lists them.  For a 125-point box
that order decides whether the low-register schedule is found: terms sorted by
access offset run in 6.5 ms, the source order (grouped by coefficient) in
7.3 to 7.7 ms.  Narrow stencils are unaffected (13-, 27-point within noise),
so they keep the source order and stay bit-identical; re-association is
applied only above the very-wide threshold or on request.
"""

import ast
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cutile import stencil
from cutile.lowering.stencil_to_cutile import lower_stencil_to_python
from wide_stencils import box125


def _result_line(code: str) -> str:
    return next(line for line in code.splitlines() if line.strip().startswith("result ="))


def _load_order(code: str) -> list[str]:
    return re.findall(r"ct\.load\((u_[mp0-9_]+),", _result_line(code))


class TestSpatialTermOrder:
    def test_very_wide_stencil_terms_are_sorted_by_offset(self):
        code = lower_stencil_to_python(box125._ir, tile_sizes=(1, 1, 256), halo_widths=(2, 2, 16))
        ast.parse(code)
        order = _load_order(code)
        assert len(order) == 125
        assert order[0] == "u_m2_m2_m2"
        assert order[-1] == "u_p2_p2_p2"
        assert order[1] == "u_m2_m2_m1"          # z varies fastest

    def test_narrow_stencil_keeps_source_order(self):
        @stencil(ndim=3, order=2)
        def lap(u, i, j, k):
            return (u[i, j, k + 1] + u[i, j, k - 1] + u[i - 1, j, k] + u[i + 1, j, k]
                    + u[i, j - 1, k] + u[i, j + 1, k] - 6.0 * u[i, j, k])

        code = lower_stencil_to_python(lap._ir, tile_sizes=(1, 1, 256), halo_widths=(1, 1, 16))
        assert _load_order(code)[:2] == ["u_0_0_p1", "u_0_0_m1"]

    def test_reordering_can_be_forced_and_keeps_signs(self):
        @stencil(ndim=1, order=2)
        def diff(u, i):
            return 0.25 * u[i + 1] + 0.5 * u[i] - 0.25 * u[i - 1]

        code = lower_stencil_to_python(
            diff._ir, tile_sizes=(1024,), halo_widths=(16,), spatial_term_order=True
        )
        ast.parse(code)
        line = _result_line(code)
        assert _load_order(code) == ["u_m1", "u_0", "u_p1"]
        assert line.strip().startswith("result = -(0.25 * ct.load(u_m1")

    def test_reordering_can_be_disabled(self):
        code = lower_stencil_to_python(
            box125._ir, tile_sizes=(1, 1, 256), halo_widths=(2, 2, 16), spatial_term_order=False
        )
        assert _load_order(code)[0] == "u_m2_m2_m2" or _load_order(code) != sorted(_load_order(code))
        # source order of the fixture starts at the (-2,-2,-2) corner too, so
        # check the tell-tale: source order is not the fully sorted order
        assert _load_order(code) != sorted(_load_order(code), key=_offset_key)


def _offset_key(name: str):
    parts = name.split("_")[1:]
    return tuple(int(p[1:]) * (-1 if p[0] == "m" else 1) if p[0] in "mp" else 0 for p in parts)
