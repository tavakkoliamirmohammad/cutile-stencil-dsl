"""Smoke tests: run each example's main() to ensure the full pipeline works.

Requires ``module load cuda/13.1.0`` so that the tileiras compiler is on PATH.
"""

import shutil
import pytest

needs_cuda = pytest.mark.skipif(
    shutil.which("tileiras") is None,
    reason="tileiras not found – run: module load cuda/13.1.0",
)


@needs_cuda
def test_heat_1d():
    from examples.heat_1d import main
    main()


@needs_cuda
def test_wave_2d():
    from examples.wave_2d import main
    main()


@needs_cuda
def test_laplacian_3d():
    from examples.laplacian_3d import main
    main()


@needs_cuda
def test_poisson_cg():
    from examples.poisson_cg import main
    main()


@needs_cuda
def test_mixed_precision_cg():
    from examples.mixed_precision_cg import main
    main()
