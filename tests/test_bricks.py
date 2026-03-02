"""Tests for bricked memory layout: BrickLayout, to_bricks, from_bricks, codegen."""

import ast

import numpy as np
import pytest

from cutile_stencil.dsl.types import BrickLayout
from cutile_stencil.layout.bricks import to_bricks, from_bricks


# ── BrickLayout dataclass tests ─────────────────────────────────────


class TestBrickLayoutValidate:
    def test_mismatched_ndim(self):
        bl = BrickLayout(brick_sizes=(16,))
        with pytest.raises(ValueError, match="dims, expected 2"):
            bl.validate(ndim=2, halo_widths=(1, 1))

    def test_brick_too_small(self):
        bl = BrickLayout(brick_sizes=(2, 4))
        with pytest.raises(ValueError, match="brick_sizes\\[0\\]=2 < 2\\*halo=4"):
            bl.validate(ndim=2, halo_widths=(2, 1))

    def test_valid_passes(self):
        bl = BrickLayout(brick_sizes=(16, 16))
        bl.validate(ndim=2, halo_widths=(2, 2))  # should not raise

    def test_valid_1d(self):
        bl = BrickLayout(brick_sizes=(32,))
        bl.validate(ndim=1, halo_widths=(1,))

    def test_valid_3d(self):
        bl = BrickLayout(brick_sizes=(8, 8, 8))
        bl.validate(ndim=3, halo_widths=(1, 1, 1))


class TestBrickLayoutNumBricks:
    def test_divisible(self):
        bl = BrickLayout(brick_sizes=(16, 32))
        assert bl.num_bricks((64, 128)) == (4, 4)

    def test_not_divisible(self):
        bl = BrickLayout(brick_sizes=(16,))
        with pytest.raises(ValueError, match="not divisible"):
            bl.num_bricks((100,))

    def test_1d(self):
        bl = BrickLayout(brick_sizes=(8,))
        assert bl.num_bricks((64,)) == (8,)


class TestBrickLayoutShape:
    def test_bricked_shape_2d(self):
        bl = BrickLayout(brick_sizes=(16, 16))
        shape = bl.bricked_shape(domain=(64, 64), halo_widths=(2, 2))
        # (4, 4, 20, 20)
        assert shape == (4, 4, 20, 20)

    def test_bricked_shape_1d(self):
        bl = BrickLayout(brick_sizes=(32,))
        shape = bl.bricked_shape(domain=(128,), halo_widths=(1,))
        assert shape == (4, 34)


# ── to_bricks / from_bricks round-trip tests ────────────────────────


class TestToBricksFromBricks:
    def test_round_trip_1d(self):
        N, h, B = 64, 1, 16
        flat = np.random.randn(N + 2 * h)
        bricked = to_bricks(flat, (B,), (h,))
        assert bricked.shape == (4, B + 2 * h)
        recovered = from_bricks(bricked, (B,), (h,))
        assert recovered.shape == flat.shape
        # Interior should match exactly
        np.testing.assert_array_equal(recovered[h:-h], flat[h:-h])

    def test_round_trip_2d(self):
        Nx, Ny, h, Bx, By = 64, 64, 2, 16, 16
        flat = np.random.randn(Nx + 2 * h, Ny + 2 * h)
        bricked = to_bricks(flat, (Bx, By), (h, h))
        assert bricked.shape == (4, 4, Bx + 2 * h, By + 2 * h)
        recovered = from_bricks(bricked, (Bx, By), (h, h))
        assert recovered.shape == flat.shape
        np.testing.assert_array_equal(
            recovered[h:-h, h:-h], flat[h:-h, h:-h]
        )

    def test_round_trip_3d(self):
        Nx, Ny, Nz, h = 32, 32, 32, 1
        B = 8
        flat = np.random.randn(Nx + 2 * h, Ny + 2 * h, Nz + 2 * h)
        bricked = to_bricks(flat, (B, B, B), (h, h, h))
        assert bricked.shape == (4, 4, 4, B + 2 * h, B + 2 * h, B + 2 * h)
        recovered = from_bricks(bricked, (B, B, B), (h, h, h))
        assert recovered.shape == flat.shape
        np.testing.assert_array_equal(
            recovered[h:-h, h:-h, h:-h], flat[h:-h, h:-h, h:-h]
        )

    def test_round_trip_preserves_boundary_halo(self):
        """Boundary halos (from_bricks) should match original flat array."""
        N, h, B = 32, 2, 8
        flat = np.random.randn(N + 2 * h)
        bricked = to_bricks(flat, (B,), (h,))
        recovered = from_bricks(bricked, (B,), (h,))
        # Left halo
        np.testing.assert_array_equal(recovered[:h], flat[:h])
        # Right halo
        np.testing.assert_array_equal(recovered[-h:], flat[-h:])


class TestBrickHaloPadding:
    def test_brick_halo_matches_neighbors_1d(self):
        """Each brick's halo region should contain data from neighbours."""
        N, h, B = 32, 1, 8
        flat = np.arange(N + 2 * h, dtype=float)
        bricked = to_bricks(flat, (B,), (h,))
        nb = N // B  # 4 bricks

        for bi in range(nb):
            # Interior in flat starts at bi * B (flat already has halo offset)
            src_start = bi * B
            src_stop = bi * B + B + 2 * h
            expected = flat[src_start:src_stop]
            np.testing.assert_array_equal(bricked[bi], expected)

    def test_brick_halo_matches_neighbors_2d(self):
        """2D: each brick's padded region matches the correct flat subregion."""
        Nx, Ny, h = 16, 16, 1
        Bx, By = 8, 8
        flat = np.arange((Nx + 2 * h) * (Ny + 2 * h), dtype=float).reshape(
            Nx + 2 * h, Ny + 2 * h
        )
        bricked = to_bricks(flat, (Bx, By), (h, h))
        for bix in range(Nx // Bx):
            for biy in range(Ny // By):
                expected = flat[
                    bix * Bx : bix * Bx + Bx + 2 * h,
                    biy * By : biy * By + By + 2 * h,
                ]
                np.testing.assert_array_equal(bricked[bix, biy], expected)


class TestToBricksErrors:
    def test_mismatched_dims(self):
        flat = np.zeros((10, 10))
        with pytest.raises(ValueError, match="dims"):
            to_bricks(flat, (8,), (1,))

    def test_not_divisible(self):
        flat = np.zeros(15)  # interior = 13, not divisible by 8
        with pytest.raises(ValueError, match="not divisible"):
            to_bricks(flat, (8,), (1,))


class TestFromBricksErrors:
    def test_wrong_ndim(self):
        bricked = np.zeros((4, 10))  # 2D, but brick_sizes implies ndim=1
        # This is actually fine (2*1=2 dims)
        from_bricks(bricked, (8,), (1,))  # should work

        # Wrong: 3D array for ndim=1 (expects 2D)
        bricked_bad = np.zeros((4, 10, 5))
        with pytest.raises(ValueError, match="dims"):
            from_bricks(bricked_bad, (8,), (1,))


# ── Bricked codegen tests ───────────────────────────────────────────


class TestBrickedCodegen:
    def _make_bricked_gen(self, ndim):
        from cutile_stencil.dsl.decorator import stencil
        from cutile_stencil.dsl.types import HardwareSpec
        from cutile_stencil.dsl.registry import clear
        from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
        from cutile_stencil.analysis.tiling import compute_tile_config
        from cutile_stencil.codegen.stencil_codegen import StencilCodeGenerator

        clear()
        hw = HardwareSpec(shared_mem_bytes=49152, dtype_bytes=8)

        if ndim == 1:
            @stencil(ndim=1, order=2)
            def heat(u, i):
                return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
            spec = heat._stencil_spec
            domain = (128,)
            brick_sizes = (32,)
        elif ndim == 2:
            @stencil(ndim=2, order=2)
            def lap2d(u, i, j):
                return u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - 4*u[i, j]
            spec = lap2d._stencil_spec
            domain = (128, 128)
            brick_sizes = (32, 32)
        else:
            @stencil(ndim=3, order=2)
            def lap3d(u, i, j, k):
                return (u[i-1, j, k] + u[i+1, j, k]
                        + u[i, j-1, k] + u[i, j+1, k]
                        + u[i, j, k-1] + u[i, j, k+1]
                        - 6 * u[i, j, k])
            spec = lap3d._stencil_spec
            domain = (64, 64, 64)
            brick_sizes = (16, 16, 16)

        accesses = extract_footprint(spec)
        spec.halo_widths = compute_halo(accesses, ndim)
        cfg = compute_tile_config(spec, domain, hw)
        layout = BrickLayout(brick_sizes=brick_sizes)
        return StencilCodeGenerator(spec, cfg, layout=layout)

    def test_bricked_1d_valid_python(self):
        code = self._make_bricked_gen(1).emit()
        ast.parse(code)

    def test_bricked_2d_valid_python(self):
        code = self._make_bricked_gen(2).emit()
        ast.parse(code)

    def test_bricked_3d_valid_python(self):
        code = self._make_bricked_gen(3).emit()
        ast.parse(code)

    def test_bricked_2d_has_divmod(self):
        code = self._make_bricked_gen(2).emit()
        assert "// TPB_X" in code
        assert "% TPB_X" in code
        assert "// TPB_Y" in code
        assert "% TPB_Y" in code

    def test_bricked_2d_has_outer_inner_slices(self):
        code = self._make_bricked_gen(2).emit()
        # Outer slices for brick selection (axes 0, 1)
        assert ".slice(axis=0," in code
        assert ".slice(axis=1," in code
        # Inner slices for stencil offset (axes 2, 3)
        assert ".slice(axis=2," in code
        assert ".slice(axis=3," in code

    def test_bricked_1d_has_kernel_and_launcher(self):
        code = self._make_bricked_gen(1).emit()
        assert "@ct.kernel" in code
        assert "def heat_kernel(" in code
        assert "def launch_heat(" in code
        assert "ct.load" in code
        assert "ct.store" in code

    def test_bricked_2d_launcher_has_tpb(self):
        code = self._make_bricked_gen(2).emit()
        assert "TPB_X = ct.cdiv(BX, TX)" in code
        assert "TPB_Y = ct.cdiv(BY, TY)" in code
        assert "NBX * TPB_X" in code
        assert "NBY * TPB_Y" in code

    def test_bricked_kernel_params(self):
        code = self._make_bricked_gen(2).emit()
        # Should have brick size and tiles-per-brick params
        assert "BX: ConstInt" in code
        assert "BY: ConstInt" in code
        assert "TPB_X: ConstInt" in code
        assert "TPB_Y: ConstInt" in code


# ── Pipeline integration tests ───────────────────────────────────────


class TestCompileBricked:
    def test_compile_with_brick_layout_2d(self):
        from cutile_stencil.dsl.decorator import stencil
        from cutile_stencil.dsl.registry import clear
        from cutile_stencil.dsl.pipeline import compile, CompileResult

        clear()

        @stencil(ndim=2, order=2)
        def lap(u, i, j):
            return u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - 4*u[i, j]

        layout = BrickLayout(brick_sizes=(32, 32))
        result = compile(lap, domain=(128, 128), layout=layout)
        assert isinstance(result, CompileResult)
        ast.parse(result.code)
        assert "// TPB_X" in result.code
        assert ".slice(axis=2," in result.code

    def test_compile_bricked_rejects_bad_layout(self):
        from cutile_stencil.dsl.decorator import stencil
        from cutile_stencil.dsl.registry import clear
        from cutile_stencil.dsl.pipeline import compile

        clear()

        @stencil(ndim=2, order=2)
        def lap(u, i, j):
            return u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - 4*u[i, j]

        layout = BrickLayout(brick_sizes=(32,))  # wrong ndim
        with pytest.raises(ValueError, match="dims"):
            compile(lap, domain=(128, 128), layout=layout)

    def test_compile_bricked_rejects_indivisible(self):
        from cutile_stencil.dsl.decorator import stencil
        from cutile_stencil.dsl.registry import clear
        from cutile_stencil.dsl.pipeline import compile

        clear()

        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i-1] + 0.5 * u[i] + 0.25 * u[i+1]

        layout = BrickLayout(brick_sizes=(30,))  # 100 not divisible by 30
        with pytest.raises(ValueError, match="not divisible"):
            compile(heat, domain=(100,), layout=layout)

    def test_brick_layout_importable_from_top_level(self):
        from cutile_stencil import BrickLayout
        bl = BrickLayout(brick_sizes=(16, 16))
        assert bl.brick_sizes == (16, 16)


# ── GPU validation tests ────────────────────────────────────────────


class TestBrickedGPUValidation:
    """Validate bricked GPU kernels against NumPy reference."""

    def test_validate_bricked_2d_heat(self):
        from cutile_stencil.dsl.decorator import stencil
        from cutile_stencil.dsl.registry import clear
        from cutile_stencil.dsl.pipeline import compile

        clear()

        @stencil(ndim=2, order=2)
        def heat(u, i, j):
            return 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])

        domain = (64, 64)
        layout = BrickLayout(brick_sizes=(16, 16))
        result = compile(heat, domain=domain, layout=layout)

        halo = result.spec.halo_widths
        hx, hy = halo
        flat = np.random.randn(domain[0] + 2*hx, domain[1] + 2*hy)
        result.validate(flat)

    def test_validate_bricked_1d_heat(self):
        from cutile_stencil.dsl.decorator import stencil
        from cutile_stencil.dsl.registry import clear
        from cutile_stencil.dsl.pipeline import compile

        clear()

        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i-1] + 0.5 * u[i] + 0.25 * u[i+1]

        domain = (128,)
        layout = BrickLayout(brick_sizes=(32,))
        result = compile(heat, domain=domain, layout=layout)

        h = result.spec.halo_widths[0]
        flat = np.random.randn(domain[0] + 2*h)
        result.validate(flat)

    def test_validate_bricked_2d_laplacian(self):
        from cutile_stencil.dsl.decorator import stencil
        from cutile_stencil.dsl.registry import clear
        from cutile_stencil.dsl.pipeline import compile

        clear()

        @stencil(ndim=2, order=2)
        def lap(u, i, j):
            return u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - 4*u[i, j]

        domain = (128, 128)
        layout = BrickLayout(brick_sizes=(32, 32))
        result = compile(lap, domain=domain, layout=layout)

        halo = result.spec.halo_widths
        hx, hy = halo
        flat = np.random.randn(domain[0] + 2*hx, domain[1] + 2*hy)
        result.validate(flat)
