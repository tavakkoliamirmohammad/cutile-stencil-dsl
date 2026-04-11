"""Multi-GPU domain decomposition for cuTile stencil computations.

Uses cuTile gather() + symmetric memory for single-node multi-GPU
stencil execution. Each GPU owns a subdomain and reads halo data
directly from neighboring GPUs via peer-mapped buffers.
"""

from cutile_stencil.multigpu.decomposition import (
    Partition,
    DomainDecomposition,
    decompose,
)
from cutile_stencil.multigpu.codegen import MultiGPUStencilCodeGenerator
from cutile_stencil.multigpu.launcher import emit_multigpu_launcher

__all__ = [
    "Partition",
    "DomainDecomposition",
    "decompose",
    "MultiGPUStencilCodeGenerator",
    "emit_multigpu_launcher",
]
