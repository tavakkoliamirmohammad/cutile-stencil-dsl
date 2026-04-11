"""Smoke tests: run each example's main() to ensure the full pipeline works.
"""

import shutil
import pytest

from conftest import gpu_required


@gpu_required
def test_heat_1d():
    from examples.heat_1d import main
    main()


@gpu_required
def test_wave_2d():
    from examples.wave_2d import main
    main()


@gpu_required
def test_laplacian_3d():
    from examples.laplacian_3d import main
    main()


def test_poisson_cg():
    from examples.poisson_cg import main
    main()


def test_mixed_precision_cg():
    from examples.mixed_precision_cg import main
    main()


@gpu_required
def test_heat_2d_bricked():
    from examples.heat_2d_bricked import main
    main()


def test_benchmark_suite():
    from examples.benchmark_suite import main
    main()


def test_variable_diffusion_example():
    """Smoke test for variable-coefficient diffusion example."""
    from examples.variable_diffusion import main
    main()
