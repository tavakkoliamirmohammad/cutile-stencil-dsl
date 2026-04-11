"""Multi-GPU domain decomposition for cuTile stencil computations.

Uses cuTile gather() + symmetric memory for single-node multi-GPU
stencil execution. Each GPU owns a subdomain and reads halo data
directly from neighboring GPUs via peer-mapped buffers.

Two approaches are provided:

- **P2P launcher** (``emit_multigpu_p2p_launcher``): CuPy P2P halo
  exchange before launching the standard single-GPU kernel. Simple and
  proven -- works with any cuTile kernel unchanged.
- **Gather kernel** (``emit_multigpu_gather_kernel``): The kernel itself
  reads halo data from peer arrays via ``ct.gather()``/``ct.load()``.
  No pre-kernel copy needed, but the kernel is multi-GPU-aware.
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
from cutile_stencil.multigpu.codegen import (
    MultiGPUStencilCodeGenerator,
    emit_multigpu_p2p_launcher,
    emit_multigpu_gather_kernel,
)
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
    "emit_multigpu_p2p_launcher",
    "emit_multigpu_gather_kernel",
]
