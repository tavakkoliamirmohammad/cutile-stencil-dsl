"""Shared test fixtures and markers."""

import pytest

_HAS_GPU = False
try:
    import cupy as cp
    cp.cuda.Device(0).compute_capability
    _HAS_GPU = True
except Exception:
    pass

gpu_required = pytest.mark.skipif(not _HAS_GPU, reason="No GPU available")
