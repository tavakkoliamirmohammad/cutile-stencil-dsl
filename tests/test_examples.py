"""Smoke tests: run each example's main() to ensure the full pipeline works."""

import pytest


def test_heat_1d():
    from examples.heat_1d import main
    main()


def test_wave_2d():
    from examples.wave_2d import main
    main()


def test_laplacian_3d():
    from examples.laplacian_3d import main
    main()


def test_poisson_cg():
    from examples.poisson_cg import main
    main()


def test_mixed_precision_cg():
    from examples.mixed_precision_cg import main
    main()
