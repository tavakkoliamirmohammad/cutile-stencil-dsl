"""Tests for named stencil templates."""

import ast
import pytest
from cutile_stencil.dsl.templates import (
    compact_stencil, cross_stencil, diamond_stencil, box_stencil,
)
from cutile_stencil.dsl.types import HardwareSpec
from cutile_stencil.dsl.pipeline import compile


class TestCompactStencil:
    def test_1d_accesses(self):
        spec = compact_stencil(ndim=1, order=2)
        assert spec.ndim == 1
        assert spec.order == 2
        assert spec.halo_widths == (1,)
        assert len(spec.accesses) == 3  # center + 2 neighbors

    def test_2d_accesses(self):
        spec = compact_stencil(ndim=2, order=2)
        assert spec.ndim == 2
        assert len(spec.accesses) == 5  # center + 4 neighbors
        assert spec.halo_widths == (1, 1)

    def test_3d_accesses(self):
        spec = compact_stencil(ndim=3, order=2)
        assert len(spec.accesses) == 7  # center + 6 neighbors

    def test_compiles_to_valid_code(self):
        spec = compact_stencil(ndim=2, order=2)
        result = compile(spec, domain=(256, 256))
        ast.parse(result.code)


class TestCrossStencil:
    def test_1d_order4(self):
        spec = cross_stencil(ndim=1, order=4)
        assert spec.halo_widths == (2,)
        assert len(spec.accesses) == 5  # center + 4 neighbors (±1, ±2)

    def test_2d_order2(self):
        spec = cross_stencil(ndim=2, order=2)
        assert len(spec.accesses) == 5  # same as compact for order=2


class TestDiamondStencil:
    def test_2d_order2(self):
        spec = diamond_stencil(ndim=2, order=2)
        # L1 ball radius 1 in 2D: center + 4 axis-aligned = 5
        assert len(spec.accesses) == 5

    def test_2d_order4(self):
        spec = diamond_stencil(ndim=2, order=4)
        # L1 ball radius 2 in 2D: 13 points
        assert len(spec.accesses) == 13


class TestBoxStencil:
    def test_2d_order2(self):
        spec = box_stencil(ndim=2, order=2)
        # 3x3 box = 9 points
        assert len(spec.accesses) == 9

    def test_3d_order2(self):
        spec = box_stencil(ndim=3, order=2)
        # 3x3x3 box = 27 points
        assert len(spec.accesses) == 27

    def test_compiles_to_valid_code(self):
        spec = box_stencil(ndim=2, order=2)
        result = compile(spec, domain=(256, 256))
        ast.parse(result.code)
