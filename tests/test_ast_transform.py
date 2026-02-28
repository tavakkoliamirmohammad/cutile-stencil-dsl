"""Tests for AST-level stencil expression transformation."""

import ast
import pytest
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import OffsetAccess
from cutile_stencil.dsl.registry import clear
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.codegen.ast_transform import (
    InlineTransformer,
    SubscriptReplacer,
    get_inlined_return_expr,
    transform_stencil_expr,
)


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


class TestInlineTransformer:
    def test_simple_inline(self):
        """Single intermediate variable gets inlined."""
        @stencil(ndim=1, order=2)
        def s(u, i):
            tmp = u[i - 1] + u[i + 1]
            return 0.5 * tmp

        expr = get_inlined_return_expr(s._stencil_spec.update_fn, 1)
        assert "tmp" not in expr
        assert "u[i - 1]" in expr or "u[i + 1]" in expr

    def test_nested_inline(self):
        """Chained intermediate variables get fully inlined."""
        @stencil(ndim=1, order=2)
        def s(u, i):
            a = u[i - 1]
            b = u[i + 1]
            c = a + b
            return 0.5 * c

        expr = get_inlined_return_expr(s._stencil_spec.update_fn, 1)
        assert "a" not in expr
        assert "b" not in expr
        assert "c" not in expr

    def test_wave_2d_inline(self):
        """The wave_2d pattern with lap_x, lap_y gets inlined."""
        @stencil(ndim=2, order=4)
        def wave(u, i, j):
            lap_x = u[i - 1, j] + u[i + 1, j] - 2 * u[i, j]
            lap_y = u[i, j - 1] + u[i, j + 1] - 2 * u[i, j]
            return 0.1 * (lap_x + lap_y)

        expr = get_inlined_return_expr(wave._stencil_spec.update_fn, 2)
        assert "lap_x" not in expr
        assert "lap_y" not in expr
        # The expression should reference array accesses
        assert "u[" in expr


class TestSubscriptReplacer:
    def test_1d_replacement(self):
        """1D subscript replacement works."""
        code = "u[i - 1] + u[i] + u[i + 1]"
        tree = ast.parse(code, mode="eval")
        replacements = {
            ("u", (-1,)): "t_left",
            ("u", (0,)): "t_center",
            ("u", (1,)): "t_right",
        }
        replacer = SubscriptReplacer(replacements, ("i",))
        result = replacer.visit(tree)
        result_str = ast.unparse(result)
        assert "t_left" in result_str
        assert "t_center" in result_str
        assert "t_right" in result_str
        assert "u[" not in result_str

    def test_2d_replacement(self):
        """2D subscript replacement works."""
        code = "u[i - 1, j] + u[i + 1, j] - 2 * u[i, j]"
        tree = ast.parse(code, mode="eval")
        replacements = {
            ("u", (-1, 0)): "nb_left",
            ("u", (1, 0)): "nb_right",
            ("u", (0, 0)): "nb_center",
        }
        replacer = SubscriptReplacer(replacements, ("i", "j"))
        result = replacer.visit(tree)
        result_str = ast.unparse(result)
        assert "nb_left" in result_str
        assert "nb_right" in result_str
        assert "nb_center" in result_str


class TestTransformStencilExpr:
    def test_heat_1d(self):
        """Simple 1D heat stencil transforms correctly."""
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        spec = heat._stencil_spec
        accesses = extract_footprint(spec)
        terms = ["t_left", "t_center", "t_right"]
        result = transform_stencil_expr(spec.update_fn, accesses, 1, terms)
        assert "t_left" in result
        assert "t_center" in result
        assert "t_right" in result
        assert "u[" not in result

    def test_wave_2d_multi_statement(self):
        """wave_2d-style multi-statement stencil produces valid expression."""
        @stencil(ndim=2, order=2)
        def wave(u, i, j):
            lap_x = u[i - 1, j] + u[i + 1, j] - 2 * u[i, j]
            lap_y = u[i, j - 1] + u[i, j + 1] - 2 * u[i, j]
            return 0.1 * (lap_x + lap_y)

        spec = wave._stencil_spec
        accesses = extract_footprint(spec)
        terms = [f"nb{i}" for i in range(len(accesses))]
        result = transform_stencil_expr(spec.update_fn, accesses, 2, terms)
        # Should not contain intermediate variable names
        assert "lap_x" not in result
        assert "lap_y" not in result
        # Should be valid Python
        ast.parse(f"result = {result}", mode="exec")

    def test_codegen_wave_2d_produces_valid_python(self):
        """End-to-end: wave_2d codegen produces valid Python (the original bug)."""
        from cutile_stencil.dsl.types import HardwareSpec
        from cutile_stencil.analysis.tiling import compute_tile_config
        from cutile_stencil.analysis.temporal import compute_temporal_config
        from cutile_stencil.codegen.stencil_codegen import StencilCodeGenerator

        @stencil(ndim=2, order=4)
        def wave_2d(u, i, j):
            lap_x = (-1/12) * u[i - 2, j] + (4/3) * u[i - 1, j] + (-5/2) * u[i, j] + (4/3) * u[i + 1, j] + (-1/12) * u[i + 2, j]
            lap_y = (-1/12) * u[i, j - 2] + (4/3) * u[i, j - 1] + (-5/2) * u[i, j] + (4/3) * u[i, j + 1] + (-1/12) * u[i, j + 2]
            return 0.1 * (lap_x + lap_y)

        spec = wave_2d._stencil_spec
        accesses = extract_footprint(spec)
        spec.halo_widths = compute_halo(accesses, 2)
        hw = HardwareSpec(shared_mem_bytes=49152, dtype_bytes=8)
        cfg = compute_tile_config(spec, (256, 256), hw)

        gen = StencilCodeGenerator(spec, cfg)
        code = gen.emit()

        # Must parse as valid Python
        ast.parse(code)

        # Must NOT contain undefined variables lap_x, lap_y
        assert "lap_x" not in code
        assert "lap_y" not in code

    def test_higher_order_3d(self):
        """3D stencil with intermediate variables."""
        @stencil(ndim=3, order=2)
        def s(u, i, j, k):
            diff = u[i + 1, j, k] - u[i - 1, j, k]
            return 0.5 * diff

        spec = s._stencil_spec
        accesses = extract_footprint(spec)
        terms = [f"t{i}" for i in range(len(accesses))]
        result = transform_stencil_expr(spec.update_fn, accesses, 3, terms)
        assert "diff" not in result
        assert "u[" not in result
