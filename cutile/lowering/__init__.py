"""cuTile lowering passes: Dialect 1/2 IR to cuTile target code.

The lowering pipeline follows the three-dialect stack:

    Dialect 1 (cutile_stencil) -> Dialect 3 (cutile_target) -> Python

Key entry points:

* ``lower_stencil_to_python`` -- full pipeline (Dialect 1 to Python source)
* ``lower_to_target_ir``      -- Dialect 1 to Dialect 3 IR
* ``emit_python``             -- Dialect 3 IR to Python source
"""

from cutile.lowering.stencil_to_cutile import lower_stencil_to_python
from cutile.lowering.stencil_to_target import lower_to_target_ir
from cutile.lowering.target_to_python import emit_python
from cutile.lowering.fusion_emitter import lower_fused_stencils_to_python
from cutile.lowering.emitter import CodeEmitter
from cutile.lowering.multigpu_emitter import (
    lower_stencil_to_multigpu_python,
    lower_stencil_to_bricked_python,
)

__all__ = [
    "lower_stencil_to_python",
    "lower_to_target_ir",
    "emit_python",
    "lower_fused_stencils_to_python",
    "lower_stencil_to_multigpu_python",
    "lower_stencil_to_bricked_python",
    "CodeEmitter",
]
