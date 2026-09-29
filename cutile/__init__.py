"""cuTile Stencil DSL -- top-level package."""

from cutile.runtime.cuda_env import prefer_pip_cuda_libs as _prefer_pip_cuda_libs

# The pip tileiras compiler must not pick up an older toolkit's libnvvm from a
# loaded CUDA module; do this before anything can spawn the compiler.
_prefer_pip_cuda_libs()

from cutile.frontend.decorator import stencil
from cutile.reference.elementwise import maximum, minimum, where
from cutile.runtime.launcher import compile, compile_fused
from cutile.runtime.pipeline import Pipeline
from cutile.runtime.rk_integrator import RKIntegrator
from cutile.frontend.types import BoundarySpec

__all__ = ["stencil", "compile", "compile_fused", "Pipeline", "RKIntegrator", "BoundarySpec",
           "maximum", "minimum", "where"]
