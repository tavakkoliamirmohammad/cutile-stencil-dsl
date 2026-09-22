"""Tests for the performance-oriented layout decisions of the compiler.

These pin down three codegen choices that together let the generated
7-point kernel match a hand-written CUDA kernel on a memory-bound GPU:

1. Tiles span whole rows (innermost dim = 256 or 128 elements, i.e. two or
   one elements per thread of a 128-thread block) instead of near-cubic tiles.
2. Thread blocks are ordered so that consecutive blocks walk consecutive
   rows of one plane (rows-fastest), not consecutive planes.
3. The innermost halo is padded to a 128-byte multiple so every row of the
   interior starts on an aligned address.

It also covers the temporal launcher no longer allocating scratch buffers
on every call.
"""

import ast
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xdsl.context import Context

from cutile import compile as stencil_compile
from cutile import stencil
from cutile.lowering.normalize import NormalizePass
from cutile.lowering.stencil_to_cutile import lower_stencil_to_python
from cutile.passes.analysis.footprint import AnalysisPass
from cutile.passes.tiling import TilingPass
from cutile.runtime.autotune import _generate_candidates


# ------------------------------------------------------------------ #
# Stencil fixtures
# ------------------------------------------------------------------ #


def _lap3d():
    @stencil(ndim=3, order=2)
    def lap3d(u, i, j, k):
        return (
            u[i - 1, j, k] + u[i + 1, j, k]
            + u[i, j - 1, k] + u[i, j + 1, k]
            + u[i, j, k - 1] + u[i, j, k + 1]
            - 6.0 * u[i, j, k]
        )
    return lap3d


def _heat2d():
    @stencil(ndim=2, order=2)
    def heat2d(u, i, j):
        return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])
    return heat2d


def _heat1d():
    @stencil(ndim=1, order=2)
    def heat1d(u, i):
        return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
    return heat1d


def _lap4_1d():
    @stencil(ndim=1, order=4)
    def lap4(u, i):
        return -u[i - 2] + 16 * u[i - 1] - 30 * u[i] + 16 * u[i + 1] - u[i + 2]
    return lap4


def _apply_attrs(stencil_fn, **kwargs):
    from xdsl.dialects.stencil import ApplyOp

    ctx = Context()
    module = stencil_fn._ir.clone()
    NormalizePass().apply(ctx, module)
    AnalysisPass().apply(ctx, module)
    TilingPass(**kwargs).apply(ctx, module)
    for op in module.walk():
        if isinstance(op, ApplyOp):
            return dict(op.attributes)
    pytest.fail("No stencil.ApplyOp found")


def _tile_sizes(stencil_fn, **kwargs):
    return tuple(a.data for a in _apply_attrs(stencil_fn, **kwargs)["tile_sizes"])


# ------------------------------------------------------------------ #
# 1. Row-shaped tiles
# ------------------------------------------------------------------ #


class TestRowTiles:
    """cuTile runs 128 threads per block, so a tile's element count is the
    number of elements each thread owns.  Every loaded tile stays live until
    consumed, so wide stencils (many loads) must keep one element per thread
    to avoid register spills; narrow, memory-bound ones benefit from two.
    The innermost dimension always spans the row (up to the row cap); the
    launcher clamps it to the actual row length."""

    def test_7pt_gets_two_elements_per_thread(self):
        assert _tile_sizes(_lap3d()) == (1, 1, 256)

    def test_wide_stencil_gets_one_element_per_thread(self):
        @stencil(ndim=3, order=2)
        def box27(u, i, j, k):
            return (u[i-1,j-1,k-1] + u[i,j-1,k-1] + u[i+1,j-1,k-1] + u[i-1,j,k-1] + u[i,j,k-1]
                    + u[i+1,j,k-1] + u[i-1,j+1,k-1] + u[i,j+1,k-1] + u[i+1,j+1,k-1]
                    + u[i-1,j-1,k] + u[i,j-1,k] + u[i+1,j-1,k] + u[i-1,j,k] + u[i,j,k]
                    + u[i+1,j,k] + u[i-1,j+1,k] + u[i,j+1,k] + u[i+1,j+1,k]
                    + u[i-1,j-1,k+1] + u[i,j-1,k+1] + u[i+1,j-1,k+1] + u[i-1,j,k+1] + u[i,j,k+1]
                    + u[i+1,j,k+1] + u[i-1,j+1,k+1] + u[i,j+1,k+1] + u[i+1,j+1,k+1])
        # 128 elements as two rows of 64: a y-shifted access then re-fetches
        # (2 + 2r) rows per two output rows instead of (1 + 2r) per one.
        # Measured: 27-point equal, 25-point -4%, 49-point in-plane box -20%.
        assert _tile_sizes(box27) == (1, 2, 64)

    def test_threshold_is_configurable(self):
        assert _tile_sizes(_lap3d(), wide_stencil_loads=4) == (1, 2, 64)

    def test_wide_row_width_is_configurable(self):
        assert _tile_sizes(_lap3d(), wide_stencil_loads=4, wide_inner=128) == (1, 1, 128)

    def test_2d_wide_tile_is_two_rows(self):
        assert _tile_sizes(_heat2d(), wide_stencil_loads=2) == (2, 64)

    def test_very_wide_stencil_gets_two_elements_and_occupancy_hint(self):
        # 125 loads at one element per thread need 254 registers (one block
        # per SM); at two elements per thread with occupancy=4 tileiras finds
        # a 62-register schedule: 9.4 ms -> 6.5 ms on the 125-point box.
        from wide_stencils import box125
        assert _tile_sizes(box125) == (1, 1, 256)
        assert _apply_attrs(box125)["occupancy"].data == 4

    def test_wide_but_not_very_wide_has_no_occupancy_hint(self):
        from wide_stencils import box125
        assert "occupancy" not in _apply_attrs(_lap3d())
        assert _tile_sizes(box125, very_wide_stencil_loads=200) == (1, 2, 64)
        assert "occupancy" not in _apply_attrs(box125, very_wide_stencil_loads=200)

    def test_compile_emits_occupancy_hint_for_very_wide_stencil(self):
        from wide_stencils import box125
        result = stencil_compile(box125, temporal_blocking=False)
        assert result.tile_sizes == (1, 1, 256)
        assert result.kernel_hints == {"occupancy": 4}
        assert "@ct.kernel(occupancy=4)" in result.code

    def test_2d_tile_is_a_single_row(self):
        assert _tile_sizes(_heat2d()) == (1, 256)

    def test_1d_tile_is_large_power_of_two(self):
        (t,) = _tile_sizes(_heat1d())
        assert t == 1024

    def test_tile_never_drops_below_one_element_per_thread(self):
        # 125 loads drove the old budget model down to 32 elements (5x slower)
        @stencil(ndim=1, order=2)
        def wide(u, i):
            return u[i - 1] + u[i] + u[i + 1]
        assert _tile_sizes(wide, wide_stencil_loads=1)[0] >= 128

    def test_access_count_is_unique_field_offset_pairs(self):
        # u[i]*u[i] + u[i+1] reads two distinct values, not three; a
        # multi-field stencil reading the same offset of two fields reads two
        @stencil(ndim=1, order=2)
        def sq(u, i):
            return u[i] * u[i] + u[i + 1]
        from cutile.passes.tiling import unique_access_count
        assert _tile_sizes(sq, wide_stencil_loads=2) == (1024,)      # 2 accesses: narrow
        assert _tile_sizes(sq, wide_stencil_loads=1)[0] == 512        # forced wide

    def test_shared_mem_budget_is_ignored(self):
        # kept only for API compatibility with older pipelines
        assert _tile_sizes(_lap3d(), shared_mem_bytes=4096) == _tile_sizes(_lap3d())

    def test_row_cap_applies_to_2d_rows(self):
        assert _tile_sizes(_heat2d(), narrow_elements_per_thread=8) == (1, 1024)
        assert _tile_sizes(_heat2d(), narrow_elements_per_thread=16) == (2, 1024)


class TestTileClamp:
    """The launcher shrinks tiles to the domain so a wide row tile never
    covers mostly padding on a small domain (compile time does not know
    the array shape)."""

    def _fit_tiles(self, code):
        start = code.index("def _fit_tiles(")
        end = code.index("\n\n", start)
        ns = {}
        exec(code[start:end], ns)
        return ns["_fit_tiles"]

    def test_launcher_clamps_each_tile_dim_to_interior_pow2(self):
        code = lower_stencil_to_python(
            _lap3d()._ir, tile_sizes=(1, 1, 512), halo_widths=(1, 1, 16)
        )
        ast.parse(code)
        assert "TX, TY, TZ = _fit_tiles((TX, TY, TZ), u_in.shape, (HX, HY, HZ))" in code
        fit = self._fit_tiles(code)
        assert fit((1, 1, 512), (66, 66, 96), (1, 1, 16)) == (1, 1, 64)
        assert fit((1, 1, 512), (258, 258, 544), (1, 1, 16)) == (1, 1, 512)
        assert fit((4, 256), (6, 132), (1, 16)) == (4, 128)  # 100 -> 128

    def test_temporal_launcher_clamps_too(self):
        code = lower_stencil_to_python(
            _heat2d()._ir, tile_sizes=(1, 1024), halo_widths=(1, 16), temporal_steps=2
        )
        ast.parse(code)
        assert "TX, TY = _fit_tiles((TX, TY), u_in.shape, (HX, HY))" in code

    def test_1d_launcher_clamp_is_valid_python(self):
        code = lower_stencil_to_python(
            _heat1d()._ir, tile_sizes=(1024,), halo_widths=(16,)
        )
        ast.parse(code)
        assert "TX, = _fit_tiles((TX,), u_in.shape, (HX,))" in code


# ------------------------------------------------------------------ #
# 2. Rows-fastest block ordering
# ------------------------------------------------------------------ #


class TestBlockOrder:
    def test_3d_kernel_maps_bid0_to_row_axis(self):
        code = lower_stencil_to_python(
            _lap3d()._ir, tile_sizes=(1, 4, 256), halo_widths=(1, 1, 16)
        )
        ast.parse(code)
        assert "by = ct.bid(0)" in code
        assert "bx = ct.bid(1)" in code
        assert "bz = ct.bid(2)" in code

    def test_3d_launcher_grid_matches_bid_order(self):
        code = lower_stencil_to_python(
            _lap3d()._ir, tile_sizes=(1, 4, 256), halo_widths=(1, 1, 16)
        )
        assert (
            "grid = (ct.cdiv(Ny - 2 * HY, TY), ct.cdiv(Nx - 2 * HX, TX), "
            "ct.cdiv(Nz - 2 * HZ, TZ))" in code
        )

    def test_3d_temporal_launcher_grid_matches_bid_order(self):
        code = lower_stencil_to_python(
            _lap3d()._ir, tile_sizes=(1, 4, 256), halo_widths=(1, 1, 16),
            temporal_steps=2,
        )
        assert (
            "grid = (ct.cdiv(u_in.shape[1] - 2, TY), "
            "ct.cdiv(u_in.shape[0] - 2, TX), "
            "ct.cdiv(u_in.shape[2] - 32, TZ))" in code
        )

    def test_2d_kernel_keeps_natural_order(self):
        code = lower_stencil_to_python(
            _heat2d()._ir, tile_sizes=(4, 256), halo_widths=(1, 16)
        )
        assert "bx = ct.bid(0)" in code
        assert "by = ct.bid(1)" in code
        assert "grid = (ct.cdiv(Nx - 2 * HX, TX), ct.cdiv(Ny - 2 * HY, TY))" in code


# ------------------------------------------------------------------ #
# 3. Aligned inner halo
# ------------------------------------------------------------------ #


class TestAlignedHalo:
    def test_compile_pads_innermost_halo_to_16_elements(self):
        result = stencil_compile(_lap3d(), temporal_blocking=False)
        assert result.halo_widths == (1, 1, 16)
        assert result.stencil_halo == (1, 1, 1)

    def test_compile_keeps_padding_a_multiple_of_16_for_wide_stencils(self):
        result = stencil_compile(_lap4_1d(), temporal_blocking=False)
        assert result.halo_widths == (16,)
        assert result.stencil_halo == (2,)

    def test_generated_code_uses_padded_halo_constants(self):
        result = stencil_compile(_lap3d(), temporal_blocking=False)
        assert "HX, HY, HZ = 1, 1, 16" in result.code

    def test_align_halo_can_be_disabled(self):
        result = stencil_compile(_lap3d(), temporal_blocking=False, align_halo=False)
        assert result.halo_widths == (1, 1, 1)
        assert result.stencil_halo == (1, 1, 1)

    def test_multigpu_path_keeps_unpadded_halo(self):
        result = stencil_compile(_lap3d(), num_gpus=2, temporal_blocking=False)
        assert result.halo_widths == (1, 1, 1)


# ------------------------------------------------------------------ #
# 4. Temporal launcher buffer reuse
# ------------------------------------------------------------------ #


class TestTemporalBuffers:
    def test_temporal_launcher_does_not_allocate_per_call(self):
        code = lower_stencil_to_python(
            _heat2d()._ir, tile_sizes=(4, 256), halo_widths=(1, 16),
            temporal_steps=3,
        )
        ast.parse(code)
        assert "bufs.append(cp.zeros_like" not in code
        assert "_temporal_buffers(u_in, 2)" in code

    def test_temporal_buffer_helper_is_emitted_once(self):
        code = lower_stencil_to_python(
            _heat2d()._ir, tile_sizes=(4, 256), halo_widths=(1, 16),
            temporal_steps=3,
        )
        assert code.count("def _temporal_buffers(") == 1

    def test_single_step_launcher_has_no_buffer_helper(self):
        code = lower_stencil_to_python(
            _heat2d()._ir, tile_sizes=(4, 256), halo_widths=(1, 16),
            temporal_steps=1,
        )
        assert "_temporal_buffers" not in code


# ------------------------------------------------------------------ #
# 5. Autotune searches row-shaped tiles
# ------------------------------------------------------------------ #


class TestAutotuneCandidates:
    def test_3d_candidates_include_row_tiles(self):
        cands = [t for t, _, _ in _generate_candidates(3, (1, 1, 1))]
        assert (1, 4, 256) in cands
        assert (1, 1, 256) in cands
        assert (1, 1, 1024) in cands

    def test_2d_candidates_include_row_tiles(self):
        cands = [t for t, _, _ in _generate_candidates(2, (1, 1))]
        assert (4, 256) in cands

    def test_two_element_tiles_also_get_an_occupancy_hinted_variant(self):
        cands = _generate_candidates(3, (1, 1, 1))
        assert ((1, 1, 256), 1, {}) in cands
        assert ((1, 1, 256), 1, {"occupancy": 4}) in cands
        # one element per thread never profits from the hint (it only spills)
        assert ((1, 1, 128), 1, {}) in cands
        assert ((1, 2, 64), 1, {}) in cands
        assert not any(t == (1, 1, 128) and h for t, _, h in cands)

    def test_candidate_pruning_keeps_tiles_near_the_default_size(self):
        # with a cap, the search must keep 128-512 element tiles (the winning
        # region on every stencil measured) rather than the largest tiles
        from cutile.runtime.autotune import _prioritize
        cands = _prioritize(_generate_candidates(3, (1, 1, 1)), 20)
        assert len(cands) == 20
        sizes = {__import__("math").prod(t) for t, _, _ in cands}
        assert sizes <= {128, 256, 512}
        assert ((1, 2, 64), 1, {}) in cands
        assert ((1, 1, 256), 1, {"occupancy": 4}) in cands

    def test_autotune_result_carries_hints(self):
        from cutile.runtime.autotune import AutotuneResult
        r = AutotuneResult(tile_sizes=(1, 1, 256), temporal_steps=1, throughput_gpoints=0.0, bandwidth_gbs=0.0)
        assert r.kernel_hints == {}


class TestTileCandidates:
    """The pass ranks (tile, hints) configurations per tier.  The runtime
    keeps the first one whose compiled kernel fits the register budget of the
    occupancy target without spilling (see regcheck): a 56-access stencil at
    one element per thread compiles to 210 registers (two blocks per SM,
    5.2 ms) where the hinted two-element tile takes 64 registers (3.4 ms),
    while a 27-access stencil fits in 84 registers and is fastest unhinted."""

    def test_narrow_stencil_has_a_single_candidate(self):
        assert TilingPass().candidates(3, 7) == [((1, 1, 256), {})]

    def test_wide_stencil_falls_back_to_the_hinted_ladder(self):
        # one element unhinted first; then the hinted two-element tile; then
        # one element per thread with decreasing occupancy targets (a
        # 144-load fused kernel spills under occupancy=4 but fits 164
        # registers under occupancy=3 and runs 2x faster than unhinted)
        assert TilingPass().candidates(3, 27) == [
            ((1, 2, 64), {}),
            ((1, 1, 256), {"occupancy": 4}),
            ((1, 2, 64), {"occupancy": 4}),
            ((1, 2, 64), {"occupancy": 3}),
            ((1, 2, 64), {"occupancy": 2}),
        ]

    def test_very_wide_stencil_tries_hints_first_and_unhinted_last(self):
        assert TilingPass().candidates(3, 125) == [
            ((1, 1, 256), {"occupancy": 4}),
            ((1, 2, 64), {"occupancy": 4}),
            ((1, 2, 64), {"occupancy": 3}),
            ((1, 2, 64), {"occupancy": 2}),
            ((1, 2, 64), {}),
        ]

    def test_occupancy_ladder_is_configurable(self):
        cands = TilingPass(very_wide_occupancy=8, min_occupancy=4).candidates(3, 125)
        assert [c[1].get("occupancy") for c in cands] == [8, 8, 7, 6, 5, 4, None]

    def test_pass_attaches_the_first_candidate(self):
        from wide_stencils import box27
        tile, hints = TilingPass().candidates(3, 27)[0]
        assert _tile_sizes(box27) == tile
        assert "occupancy" not in _apply_attrs(box27)
