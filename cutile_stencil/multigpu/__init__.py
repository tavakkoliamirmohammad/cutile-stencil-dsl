"""Multi-GPU domain decomposition for cuTile stencil computations.

Uses cuTile gather() + symmetric memory for single-node multi-GPU
stencil execution. Each GPU owns a subdomain and reads halo data
directly from neighboring GPUs via peer-mapped buffers.
"""

from cutile_stencil.multigpu.decomposition import (
    Partition,
    DomainDecomposition,
    decompose,
    CartesianPartition,
    CartesianDecomposition,
    decompose_cartesian,
    _factorize_gpus,
)
from cutile_stencil.multigpu.codegen import MultiGPUStencilCodeGenerator
from cutile_stencil.multigpu.launcher import emit_multigpu_launcher
from cutile_stencil.multigpu.halo_exchange import (
    HaloExchangePlan,
    create_exchange_plan,
    emit_halo_exchange_function,
)

__all__ = [
    "Partition",
    "DomainDecomposition",
    "decompose",
    "CartesianPartition",
    "CartesianDecomposition",
    "decompose_cartesian",
    "_factorize_gpus",
    "MultiGPUStencilCodeGenerator",
    "emit_multigpu_launcher",
    "HaloExchangePlan",
    "create_exchange_plan",
    "emit_halo_exchange_function",
]
