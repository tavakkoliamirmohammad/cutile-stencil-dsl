"""Spill-aware configuration selection for very wide stencils.

An ``occupancy`` hint bounds the compiler's register budget; whether tileiras
meets it without spilling to local memory depends on the kernel (125-point
3D box: 62 registers, no spill, 6.5 ms; 81-point 2D box under occupancy=8:
spills, 31 ms).  Instead of guessing, the compiler probes each candidate's
resource usage from its cubin and keeps the first that does not spill.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cutile import compile as stencil_compile
from cutile.runtime import regcheck


class TestParseResUsage:
    def test_parses_cuobjdump_output(self):
        text = "Resource usage:\n Common:\n  GLOBAL:0\n Function k:\n  REG:46 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:1008 TEXTURE:0 SURFACE:0 SAMPLER:0\n"
        assert regcheck.parse_res_usage(text) == regcheck.Resources(registers=46, stack_bytes=0)

    def test_reports_spill_through_stack(self):
        text = "REG:64 STACK:1120 SHARED:0 LOCAL:0"
        r = regcheck.parse_res_usage(text)
        assert r.spills
        assert r.stack_bytes == 1120

    def test_unparseable_is_none(self):
        assert regcheck.parse_res_usage("nothing here") is None


class TestRegisterBudget:
    def test_four_blocks_of_128_threads_per_sm_allow_128_registers(self):
        assert regcheck.register_budget(occupancy=4) == 128

    def test_budget_scales_with_the_occupancy_target(self):
        assert regcheck.register_budget(occupancy=2) == 256
        assert regcheck.register_budget(occupancy=8) == 64


class TestSelectConfig:
    def _cfgs(self):
        return [
            regcheck.Config(tile_sizes=(1, 1, 256), kernel_hints={"occupancy": 4}),
            regcheck.Config(tile_sizes=(1, 1, 128), kernel_hints={}),
        ]

    def test_keeps_first_config_that_does_not_spill(self):
        seen = []
        def probe(cfg):
            seen.append(cfg)
            return regcheck.Resources(46, 0)
        assert regcheck.select_config(self._cfgs(), probe) == self._cfgs()[0]
        assert len(seen) == 1

    def test_skips_spilling_config(self):
        answers = iter([regcheck.Resources(64, 1120), regcheck.Resources(86, 0)])
        assert regcheck.select_config(self._cfgs(), lambda cfg: next(answers)) == self._cfgs()[1]

    def test_all_spill_returns_last(self):
        assert regcheck.select_config(self._cfgs(), lambda cfg: regcheck.Resources(64, 8)) == self._cfgs()[1]

    def test_probe_unavailable_keeps_first(self):
        assert regcheck.select_config(self._cfgs(), lambda cfg: None) == self._cfgs()[0]

    def test_skips_config_that_exceeds_the_register_budget(self):
        answers = iter([regcheck.Resources(210, 0), regcheck.Resources(64, 0)])
        assert regcheck.select_config(self._cfgs(), lambda cfg: next(answers), max_registers=128) == self._cfgs()[1]

    def test_keeps_config_within_the_register_budget(self):
        assert regcheck.select_config(self._cfgs(), lambda cfg: regcheck.Resources(96, 0), max_registers=128) == self._cfgs()[0]

    def test_a_config_may_carry_its_own_register_budget(self):
        # occupancy=3 allows 170 registers per thread: 164 fits it although it
        # exceeds the default budget of 128
        cfgs = [
            regcheck.Config((1, 2, 64), {"occupancy": 4}, max_registers=128),
            regcheck.Config((1, 2, 64), {"occupancy": 3}, max_registers=170),
        ]
        answers = iter([regcheck.Resources(128, 192), regcheck.Resources(164, 0)])
        assert regcheck.select_config(cfgs, lambda cfg: next(answers), max_registers=128) == cfgs[1]

    def test_over_budget_everywhere_prefers_the_non_spilling_config(self):
        answers = iter([regcheck.Resources(210, 0), regcheck.Resources(140, 1120)])
        assert regcheck.select_config(self._cfgs(), lambda cfg: next(answers), max_registers=128) == self._cfgs()[0]


class TestSelectFastest:
    """Very wide kernels are chosen by measurement: every candidate is
    compiled and timed on a slab, and the fastest one without a large spill
    wins (a 112-load kernel: 15.0 ms as a single 128-row at occupancy 3
    against 20.3 ms for the first spill-free rung of the ladder)."""

    def _cfgs(self):
        return [
            regcheck.Config((1, 1, 256), {"occupancy": 4}, max_registers=128),
            regcheck.Config((1, 2, 64), {"occupancy": 4}, max_registers=128),
            regcheck.Config((1, 2, 64), {"occupancy": 3}, max_registers=170),
            regcheck.Config((1, 1, 128), {"occupancy": 3}, max_registers=170),
            regcheck.Config((1, 2, 64), {}, max_registers=128),
        ]

    def _probe(self, table):
        cfgs = self._cfgs()
        return lambda cfg: table[cfgs.index(cfg)]

    def test_picks_the_fastest_measured_config(self):
        table = [regcheck.Resources(255, 648, ms=27.8), regcheck.Resources(128, 144, ms=15.8),
                 regcheck.Resources(168, 8, ms=16.8), regcheck.Resources(126, 0, ms=15.0), regcheck.Resources(219, 0, ms=18.3)]
        assert regcheck.select_fastest(self._cfgs(), self._probe(table)) == self._cfgs()[3]

    def test_a_large_spill_is_not_trusted_even_if_it_timed_well(self):
        table = [regcheck.Resources(255, 900, ms=10.0), regcheck.Resources(128, 144, ms=15.8),
                 regcheck.Resources(168, 8, ms=16.8), regcheck.Resources(126, 0, ms=15.0), regcheck.Resources(219, 0, ms=18.3)]
        assert regcheck.select_fastest(self._cfgs(), self._probe(table), max_spill_bytes=512) == self._cfgs()[3]

    def test_earlier_candidate_wins_within_the_margin(self):
        table = [regcheck.Resources(62, 0, ms=6.43), regcheck.Resources(56, 0, ms=6.30), regcheck.Resources(56, 0, ms=6.31),
                 regcheck.Resources(128, 824, ms=28.8), regcheck.Resources(248, 0, ms=9.4)]
        # 6.30 vs 6.43 is outside a 1% margin -> the two-row tile; inside a 5% margin -> the first candidate
        assert regcheck.select_fastest(self._cfgs(), self._probe(table), margin=0.01) == self._cfgs()[1]
        assert regcheck.select_fastest(self._cfgs(), self._probe(table), margin=0.05) == self._cfgs()[0]

    def test_without_timings_falls_back_to_the_budget_rule(self):
        table = [regcheck.Resources(64, 1120), regcheck.Resources(64, 1120), regcheck.Resources(164, 0),
                 regcheck.Resources(200, 0), regcheck.Resources(212, 0)]
        assert regcheck.select_fastest(self._cfgs(), self._probe(table)) == self._cfgs()[2]

    def test_probe_unavailable_keeps_the_first(self):
        assert regcheck.select_fastest(self._cfgs(), lambda cfg: None) == self._cfgs()[0]


class TestCompileUsesProbe:
    def test_very_wide_stencil_is_measured_and_takes_the_fastest_fit(self, monkeypatch):
        from wide_stencils import box125

        def probe(**kw):
            tile, code = tuple(kw["tile_sizes"]), kw["code"]
            assert kw["time_shape"] is not None          # very wide: every candidate is timed
            if tile == (1, 1, 256):
                return regcheck.Resources(255, 648, ms=27.8)
            if tile == (1, 1, 128):
                return regcheck.Resources(126, 0, ms=15.0)
            if "occupancy=4" in code:
                return regcheck.Resources(128, 144, ms=15.8)
            if "occupancy=3" in code:
                return regcheck.Resources(168, 8, ms=16.8)
            return regcheck.Resources(219, 0, ms=18.3)

        monkeypatch.setattr(regcheck, "probe_resources", probe)
        result = stencil_compile(box125, temporal_blocking=False)
        assert result.tile_sizes == (1, 1, 128)
        assert result.kernel_hints == {"occupancy": 3}
        assert result.resources == regcheck.Resources(126, 0, ms=15.0)

    def test_wide_stencil_is_not_timed(self, monkeypatch):
        from wide_stencils import box27
        calls = []
        monkeypatch.setattr(regcheck, "probe_resources", lambda **kw: calls.append(kw) or regcheck.Resources(96, 0))
        stencil_compile(box27, temporal_blocking=False)
        assert calls and all(kw["time_shape"] is None for kw in calls)

    def test_selection_mode_can_be_forced(self, monkeypatch):
        from wide_stencils import box27
        calls = []
        monkeypatch.setattr(regcheck, "probe_resources", lambda **kw: calls.append(kw) or regcheck.Resources(96, 0, ms=1.0))
        stencil_compile(box27, temporal_blocking=False, select="measure")
        assert len(calls) == 5 and all(kw["time_shape"] is not None for kw in calls)

    def test_very_wide_stencil_falls_back_when_hinted_kernel_spills(self, monkeypatch):
        from wide_stencils import box125
        monkeypatch.setattr(regcheck, "probe_resources", lambda **kw: regcheck.Resources(64, 1120))
        result = stencil_compile(box125, temporal_blocking=False)
        assert result.tile_sizes == (1, 2, 64)
        assert result.kernel_hints == {}
        assert "@ct.kernel\n" in result.code

    def test_very_wide_stencil_keeps_hint_when_it_does_not_spill(self, monkeypatch):
        from wide_stencils import box125
        monkeypatch.setattr(regcheck, "probe_resources", lambda **kw: regcheck.Resources(46, 0))
        result = stencil_compile(box125, temporal_blocking=False)
        assert result.tile_sizes == (1, 1, 256)
        assert result.kernel_hints == {"occupancy": 4}
        assert result.resources == regcheck.Resources(46, 0)

    def test_wide_stencil_moves_to_the_hinted_tile_when_registers_exceed_the_budget(self, monkeypatch):
        from wide_stencils import box27

        def probe(**kw):
            return regcheck.Resources(64, 0) if "occupancy=" in kw["code"] else regcheck.Resources(210, 0)

        monkeypatch.setattr(regcheck, "probe_resources", probe)
        result = stencil_compile(box27, temporal_blocking=False)
        assert result.tile_sizes == (1, 1, 256)
        assert result.kernel_hints == {"occupancy": 4}
        assert result.resources == regcheck.Resources(64, 0)
        assert "@ct.kernel(occupancy=4)" in result.code

    def test_wide_stencil_keeps_the_one_element_tile_when_registers_fit(self, monkeypatch):
        from wide_stencils import box27
        calls = []
        monkeypatch.setattr(regcheck, "probe_resources", lambda **kw: calls.append(kw) or regcheck.Resources(96, 0))
        result = stencil_compile(box27, temporal_blocking=False)
        assert result.tile_sizes == (1, 2, 64)
        assert result.kernel_hints == {}
        assert result.resources == regcheck.Resources(96, 0)
        assert len(calls) == 1

    def test_very_wide_stencil_settles_on_a_lower_occupancy_when_the_hinted_tiles_spill(self, monkeypatch):
        from wide_stencils import box125

        def probe(**kw):
            if "occupancy=4" in kw["code"]:
                return regcheck.Resources(128, 192)      # spills under the 128-register budget
            if "occupancy=3" in kw["code"]:
                return regcheck.Resources(164, 0)        # fits its own 170-register budget
            return regcheck.Resources(212, 0)

        monkeypatch.setattr(regcheck, "probe_resources", probe)
        result = stencil_compile(box125, temporal_blocking=False)
        assert result.tile_sizes == (1, 2, 64)
        assert result.kernel_hints == {"occupancy": 3}
        assert result.resources == regcheck.Resources(164, 0)

    def test_narrow_stencil_is_not_probed(self, monkeypatch):
        from cutile import stencil

        @stencil(ndim=3, order=2)
        def lap(u, i, j, k):
            return u[i - 1, j, k] + u[i + 1, j, k] - 2.0 * u[i, j, k]

        calls = []
        monkeypatch.setattr(regcheck, "probe_resources", lambda **kw: calls.append(kw) or None)
        stencil_compile(lap, temporal_blocking=False)
        assert calls == []


@pytest.mark.skipif(regcheck.cuobjdump_path() is None, reason="needs the CUDA toolkit's cuobjdump")
class TestRealProbe:
    def test_probe_can_time_the_kernel_on_a_slab(self):
        from cutile import stencil
        from cutile.lowering.stencil_to_cutile import lower_stencil_to_python

        @stencil(ndim=2, order=2)
        def heat(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        code = lower_stencil_to_python(heat._ir, tile_sizes=(1, 256), halo_widths=(1, 16))
        r = regcheck.probe_resources(
            code=code, kernel_name="heat_kernel", ndim=2, num_inputs=1, has_consts=False,
            tile_sizes=(1, 256), halo_widths=(1, 16), dtype="float64", time_shape=regcheck.timing_shape(2),
        )
        assert r is not None and r.ms is not None and 0 < r.ms < 100

    def test_probe_reads_registers_of_a_generated_kernel(self):
        from cutile import stencil
        from cutile.lowering.stencil_to_cutile import lower_stencil_to_python

        @stencil(ndim=3, order=2)
        def lap(u, i, j, k):
            return (u[i - 1, j, k] + u[i + 1, j, k] + u[i, j - 1, k] + u[i, j + 1, k]
                    + u[i, j, k - 1] + u[i, j, k + 1] - 6.0 * u[i, j, k])

        code = lower_stencil_to_python(lap._ir, tile_sizes=(1, 1, 256), halo_widths=(1, 1, 16))
        r = regcheck.probe_resources(
            code=code, kernel_name="lap_kernel", ndim=3, num_inputs=1, has_consts=False,
            tile_sizes=(1, 1, 256), halo_widths=(1, 1, 16), dtype="float64",
        )
        assert r is not None
        assert 20 < r.registers < 128
        assert not r.spills
