"""Tests for AMR prolongation and restriction operators."""

import ast
import pytest
import numpy as np

from cutile_stencil.amr.operators import (
    generate_prolongation_kernel,
    generate_restriction_kernel,
)


class TestProlongationKernelGeneration:
    """Tests for generate_prolongation_kernel."""

    def test_1d_valid_python(self):
        """1D prolongation kernel is valid Python."""
        src = generate_prolongation_kernel(ndim=1)
        ast.parse(src)

    def test_2d_valid_python(self):
        """2D prolongation kernel is valid Python."""
        src = generate_prolongation_kernel(ndim=2)
        ast.parse(src)

    def test_3d_prolongation_valid_python(self):
        """3D trilinear prolongation kernel is valid Python."""
        src = generate_prolongation_kernel(ndim=3)
        ast.parse(src)

    def test_3d_prolongation_contains_trilinear(self):
        """3D prolongation source contains trilinear interpolation logic."""
        src = generate_prolongation_kernel(ndim=3)
        assert "u_coarse[i, j, k]" in src
        assert "u_coarse[i + 1, j, k]" in src
        assert "u_coarse[i, j + 1, k]" in src
        assert "u_coarse[i, j, k + 1]" in src
        assert "u_coarse[i + 1, j + 1, k]" in src
        assert "u_coarse[i + 1, j, k + 1]" in src
        assert "u_coarse[i, j + 1, k + 1]" in src
        assert "u_coarse[i + 1, j + 1, k + 1]" in src

    def test_3d_prolongation_has_function(self):
        """3D prolongation source defines prolongation_3d function."""
        src = generate_prolongation_kernel(ndim=3)
        assert "def prolongation_3d(" in src

    def test_3d_prolongation_has_triple_loops(self):
        """3D prolongation uses triple nested loops over fine grid offsets."""
        src = generate_prolongation_kernel(ndim=3)
        # Should have loops over di, dj, dk
        assert "for di in range" in src
        assert "for dj in range" in src
        assert "for dk in range" in src

    def test_3d_prolongation_alphas(self):
        """3D prolongation defines interpolation weights ai, aj, ak."""
        src = generate_prolongation_kernel(ndim=3)
        assert "ai" in src
        assert "aj" in src
        assert "ak" in src

    def test_invalid_ndim_raises(self):
        """ndim=4 raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            generate_prolongation_kernel(ndim=4)

    def test_zero_ndim_raises(self):
        """ndim=0 raises ValueError."""
        with pytest.raises(ValueError):
            generate_prolongation_kernel(ndim=0)

    def test_custom_refinement_factor(self):
        """Custom refinement factor is reflected in source."""
        src = generate_prolongation_kernel(ndim=3, refinement_factor=4)
        ast.parse(src)
        assert "4" in src

    def test_3d_prolongation_correctness(self):
        """3D prolongation produces correct trilinear interpolation values."""
        src = generate_prolongation_kernel(ndim=3, refinement_factor=2)
        ns = {}
        exec(compile(ast.parse(src), "<amr_prolongation_3d>", "exec"), ns)
        prolongation_3d = ns["prolongation_3d"]

        # Simple 2x2x2 coarse grid, corners filled with known values
        u_coarse = np.zeros((2, 2, 2))
        u_coarse[0, 0, 0] = 8.0
        u_coarse[1, 0, 0] = 0.0
        u_coarse[0, 1, 0] = 0.0
        u_coarse[0, 0, 1] = 0.0
        u_coarse[1, 1, 0] = 0.0
        u_coarse[1, 0, 1] = 0.0
        u_coarse[0, 1, 1] = 0.0
        u_coarse[1, 1, 1] = 0.0

        u_fine = np.zeros((3, 3, 3))  # (n-1)*r + 1 = 3 for n=2, r=2 but we init bigger
        u_fine = np.zeros((4, 4, 4))
        prolongation_3d(u_coarse, u_fine)

        # At fine[0,0,0]: di=dj=dk=0, ai=aj=ak=0 => all weight on u_coarse[0,0,0]=8
        assert abs(u_fine[0, 0, 0] - 8.0) < 1e-12

        # At fine[1,0,0]: di=1, ai=0.5 => 0.5*u_coarse[0,0,0] + 0.5*u_coarse[1,0,0] = 4
        assert abs(u_fine[1, 0, 0] - 4.0) < 1e-12


class TestRestrictionKernelGeneration:
    """Tests for generate_restriction_kernel."""

    def test_1d_valid_python(self):
        """1D restriction kernel is valid Python."""
        src = generate_restriction_kernel(ndim=1)
        ast.parse(src)

    def test_2d_valid_python(self):
        """2D restriction kernel is valid Python."""
        src = generate_restriction_kernel(ndim=2)
        ast.parse(src)

    def test_3d_restriction_valid_python(self):
        """3D volume-weighted restriction kernel is valid Python."""
        src = generate_restriction_kernel(ndim=3)
        ast.parse(src)

    def test_3d_restriction_has_function(self):
        """3D restriction source defines restriction_3d function."""
        src = generate_restriction_kernel(ndim=3)
        assert "def restriction_3d(" in src

    def test_3d_restriction_uses_mean(self):
        """3D restriction uses mean() for volume averaging."""
        src = generate_restriction_kernel(ndim=3)
        assert ".mean()" in src

    def test_3d_restriction_has_triple_loops(self):
        """3D restriction has loops over all three coarse dimensions."""
        src = generate_restriction_kernel(ndim=3)
        # Should have loops for i, j, k
        assert "for i in range" in src
        assert "for j in range" in src
        assert "for k in range" in src

    def test_3d_restriction_slice_syntax(self):
        """3D restriction uses 3D slice indexing into fine grid."""
        src = generate_restriction_kernel(ndim=3)
        assert "u_fine[" in src
        assert ".mean()" in src

    def test_invalid_ndim_raises(self):
        """ndim=4 raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            generate_restriction_kernel(ndim=4)

    def test_zero_ndim_raises(self):
        """ndim=0 raises ValueError."""
        with pytest.raises(ValueError):
            generate_restriction_kernel(ndim=0)

    def test_3d_restriction_correctness(self):
        """3D restriction correctly averages fine-grid cells into coarse cells."""
        src = generate_restriction_kernel(ndim=3, refinement_factor=2)
        ns = {}
        exec(compile(ast.parse(src), "<amr_restriction_3d>", "exec"), ns)
        restriction_3d = ns["restriction_3d"]

        r = 2
        n_coarse = 4
        n_fine = n_coarse * r

        # Fine grid filled with constant 1.0 — coarse should also be 1.0
        u_fine = np.ones((n_fine, n_fine, n_fine))
        u_coarse = np.zeros((n_coarse, n_coarse, n_coarse))
        restriction_3d(u_fine, u_coarse)
        np.testing.assert_allclose(u_coarse, 1.0, err_msg="Constant field should be preserved")

    def test_3d_restriction_mean_values(self):
        """3D restriction correctly averages non-uniform fine-grid values."""
        src = generate_restriction_kernel(ndim=3, refinement_factor=2)
        ns = {}
        exec(compile(ast.parse(src), "<amr_restriction_3d>", "exec"), ns)
        restriction_3d = ns["restriction_3d"]

        r = 2
        n_coarse = 2
        n_fine = n_coarse * r

        # Fine grid: block [0:2, 0:2, 0:2] has value 8.0, rest 0.0
        u_fine = np.zeros((n_fine, n_fine, n_fine))
        u_fine[0:r, 0:r, 0:r] = 8.0
        u_coarse = np.zeros((n_coarse, n_coarse, n_coarse))
        restriction_3d(u_fine, u_coarse)

        # Coarse cell [0,0,0] should be mean of u_fine[0:2, 0:2, 0:2] = 8.0
        assert abs(u_coarse[0, 0, 0] - 8.0) < 1e-12
        # All other coarse cells should be 0.0
        assert abs(u_coarse[0, 0, 1]) < 1e-12
        assert abs(u_coarse[1, 0, 0]) < 1e-12
