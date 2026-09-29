"""Property tests over random stencil expressions.

The emitted kernel expression must keep the source's operator structure for
any mix of products, sums, divisions, negations, calls, comparisons and
constants (the fuzz campaign that motivated this found a precedence bug in
the parenthesization of ``c * (-k * abs(x) - y)``), and the compiled kernel
must agree with the NumPy reference on the GPU.
"""

import ast
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from random_stencils import generate, make_stencil, used_fields
from cutile.lowering.stencil_to_cutile import lower_stencil_to_python

_SHAPES = [(1, 1, 1), (1, 2, 2), (2, 1, 3), (2, 2, 2), (2, 4, 4), (3, 1, 2), (3, 2, 1), (3, 3, 3), (3, 4, 2), (3, 1, 4)]


def _result_expr(code):
    line = next(l for l in code.splitlines() if l.strip().startswith("result = "))
    return line.split("result = ", 1)[1].strip()


def _offsets(view):
    """``u_p1_m2_0`` -> (input letter, (1, -2, 0))."""
    name, *toks = view.split("_")
    return name, tuple(0 if t == "0" else (int(t[1:]) if t[0] == "p" else -int(t[1:])) for t in toks)


def evaluate_emitted(code, inputs, arrays, halo):
    """Evaluate the emitted kernel expression with NumPy: every ``ct.load`` of a
    shifted view becomes the correspondingly shifted interior of the input
    array, ``_c<k>`` the constant it stands for, ``ct.*`` the NumPy function."""
    import re
    from types import SimpleNamespace

    import numpy as np

    from cutile.reference.stencil_ref import _ArrayProxy

    letters = list("uvwxyz")

    def view_array(view):
        letter, offs = _offsets(view)
        return _ArrayProxy(arrays[inputs[letters.index(letter)]], halo)[offs]

    ns = {"ct": SimpleNamespace(abs=np.abs, maximum=np.maximum, minimum=np.minimum, where=np.where)}
    for m in re.finditer(r"^\s*(_c\d+) = .*# (\S+)$", code, re.M):
        ns[m.group(1)] = float(m.group(2))
    for m in re.finditer(r"^\s*(t_\w+) = ct\.load\((\w+),", code, re.M):
        ns[m.group(1)] = view_array(m.group(2))
    expr = re.sub(r"ct\.load\((\w+), index=\([^)]*\), shape=\([^)]*\)\)", lambda m: f"__view('{m.group(1)}')", _result_expr(code))
    ns["__view"] = view_array
    return eval(expr, ns)


@pytest.mark.parametrize("seed", range(40))
def test_emitted_expression_evaluates_like_the_source(tmp_path, seed):
    """The kernel expression, read back with NumPy, must reproduce the Python
    source bit for bit: same operator structure, offsets, fields and
    constants (the fuzz campaign found ``c * (-k * abs(x) - y)`` emitted
    without its parentheses)."""
    import numpy as np

    from cutile.reference.elementwise import array_aware
    from cutile.reference.stencil_ref import _ArrayProxy

    ndim, nfields, radius = _SHAPES[seed % len(_SHAPES)]
    nterms = [3, 6, 12][seed % 3]
    fields, idx, body = generate(seed, ndim, nfields, radius, nterms)
    fn, _ = make_stencil(tmp_path, f"r{seed}", fields, idx, body)
    used = used_fields(body, fields)
    halo = tuple([radius] * (ndim - 1) + [16])
    tile = (1024,) if ndim == 1 else (1,) * (ndim - 1) + (256,)
    code = lower_stencil_to_python(fn._ir, tile_sizes=tile, halo_widths=halo)
    ast.parse(code)
    rs = np.random.RandomState(seed)
    shape = tuple(12 + 2 * h for h in halo)
    arrays = {f: rs.rand(*shape) + 0.5 for f in used}
    got = evaluate_emitted(code, used, arrays, halo)
    ref = array_aware(fn._fn)(*[_ArrayProxy(arrays.get(f, np.zeros(shape)), halo) for f in fields], *([0] * ndim))
    assert np.array_equal(got, ref), body


try:
    import cupy as cp
    cp.cuda.Device(0).compute_capability
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False


@pytest.mark.skipif(not _HAS_GPU, reason="No GPU available")
@pytest.mark.parametrize("seed", [2, 3, 10, 30, 41, 57, 73, 99])
def test_random_expression_matches_the_reference_on_gpu(tmp_path, seed):
    import numpy as np

    from cutile import compile as stencil_compile

    ndim, nfields, radius = _SHAPES[seed % len(_SHAPES)]
    fields, idx, body = generate(seed, ndim, nfields, radius, 12)
    fn, _ = make_stencil(tmp_path, f"g{seed}", fields, idx, body)
    result = stencil_compile(fn, temporal_blocking=False)
    assert result.inputs == tuple(used_fields(body, fields))
    assert result.validate(atol=1e-10)
