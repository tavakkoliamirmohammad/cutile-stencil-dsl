"""Tests for AMR hierarchy, prolongation, and restriction."""

import ast
import numpy as np
import pytest
from cutile_stencil.amr.levels import RefinementLevel, AMRHierarchy
from cutile_stencil.amr.operators import (
    prolongation_weights_1d,
    restriction_weights_1d,
    generate_prolongation_kernel,
    generate_restriction_kernel,
)


class TestRefinementLevel:
    def test_basic_creation(self):
        lvl = RefinementLevel(level=0, domain=(256,))
        assert lvl.level == 0
        assert lvl.spacing == 1.0
        assert lvl.ndim == 1

    def test_spacing_decreases(self):
        lvl0 = RefinementLevel(level=0, domain=(128,), refinement_ratio=2)
        lvl1 = RefinementLevel(level=1, domain=(256,), refinement_ratio=2)
        assert lvl0.spacing == 1.0
        assert lvl1.spacing == 0.5

    def test_2d_level(self):
        lvl = RefinementLevel(level=0, domain=(128, 256))
        assert lvl.ndim == 2


class TestAMRHierarchy:
    def test_create_1_level(self):
        h = AMRHierarchy.create(base_domain=(256,), num_levels=1, halo_widths=(1,))
        assert h.num_levels == 1
        assert h.coarsest.domain == (256,)

    def test_create_2_levels(self):
        h = AMRHierarchy.create(base_domain=(128,), num_levels=2, halo_widths=(1,))
        assert h.num_levels == 2
        assert h.coarsest.domain == (128,)
        assert h.finest.domain == (256,)  # 128 * 2

    def test_create_3_levels_2d(self):
        h = AMRHierarchy.create(base_domain=(64, 64), num_levels=3, halo_widths=(1, 1))
        assert h.num_levels == 3
        assert h.levels[0].domain == (64, 64)
        assert h.levels[1].domain == (128, 128)
        assert h.levels[2].domain == (256, 256)

    def test_subcycling_steps(self):
        h = AMRHierarchy.create(base_domain=(128,), num_levels=3, halo_widths=(1,))
        assert h.subcycling_steps(0) == 1
        assert h.subcycling_steps(1) == 2
        assert h.subcycling_steps(2) == 4

    def test_empty_hierarchy_raises(self):
        with pytest.raises(ValueError):
            AMRHierarchy(levels=[], halo_widths=(1,))

    def test_inconsistent_ndim_raises(self):
        with pytest.raises(ValueError):
            AMRHierarchy(
                levels=[
                    RefinementLevel(level=0, domain=(128,)),
                    RefinementLevel(level=1, domain=(128, 128)),
                ],
                halo_widths=(1,),
            )


class TestProlongationWeights:
    def test_ratio_2(self):
        w = prolongation_weights_1d(ratio=2)
        assert len(w) == 2
        assert abs(w.sum() - 1.0) < 1e-10  # Weights should sum ~1

    def test_ratio_4(self):
        w = prolongation_weights_1d(ratio=4)
        assert len(w) == 4


class TestRestrictionWeights:
    def test_ratio_2(self):
        w = restriction_weights_1d(ratio=2)
        assert len(w) == 2
        np.testing.assert_allclose(w, [0.5, 0.5])


class TestProlongationKernel:
    def test_1d_valid_python(self):
        code = generate_prolongation_kernel(ndim=1, ratio=2)
        ast.parse(code)

    def test_2d_valid_python(self):
        code = generate_prolongation_kernel(ndim=2, ratio=2)
        ast.parse(code)

    def test_1d_contains_function(self):
        code = generate_prolongation_kernel(ndim=1)
        assert "def prolongate_1d" in code


class TestRestrictionKernel:
    def test_1d_valid_python(self):
        code = generate_restriction_kernel(ndim=1, ratio=2)
        ast.parse(code)

    def test_2d_valid_python(self):
        code = generate_restriction_kernel(ndim=2, ratio=2)
        ast.parse(code)

    def test_1d_contains_function(self):
        code = generate_restriction_kernel(ndim=1)
        assert "def restrict_1d" in code


class TestProlongationRestrictionRoundTrip:
    def test_1d_restriction_then_prolongation(self):
        """Restrict then prolongate should approximately recover original."""
        # Create fine array
        fine = np.sin(np.linspace(0, 2 * np.pi, 64))
        coarse = np.zeros(32)

        # Manual restriction (average pairs)
        for i in range(32):
            coarse[i] = (fine[2*i] + fine[2*i + 1]) / 2

        # Manual prolongation (linear interp)
        fine_recovered = np.zeros(64)
        for i in range(31):
            fine_recovered[2*i] = 0.75 * coarse[i] + 0.25 * coarse[i+1]
            fine_recovered[2*i + 1] = 0.25 * coarse[i] + 0.75 * coarse[i+1]
        fine_recovered[62] = coarse[31]
        fine_recovered[63] = coarse[31]

        # Should approximately recover the original (with some smoothing)
        assert np.allclose(fine_recovered[:62], fine[:62], atol=0.2)
