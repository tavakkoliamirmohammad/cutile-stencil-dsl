"""Tests for show_generated example and GPU timing backend."""

import pytest
from cutile_stencil.benchmark.gpu_backend import GPUTimingResult


class TestGPUTimingResult:
    def test_creation(self):
        result = GPUTimingResult(
            kernel_name="test",
            elapsed_ms=100.0,
            n_iterations=10,
            avg_ms=10.0,
            domain_size=1024,
        )
        assert result.avg_ms == 10.0
        assert result.kernel_name == "test"


def test_show_generated_runs():
    """Smoke test for show_generated example (without --show-code for brevity)."""
    import sys
    sys.argv = ["show_generated.py"]  # Reset argv for argparse
    from examples.show_generated import main
    main()
