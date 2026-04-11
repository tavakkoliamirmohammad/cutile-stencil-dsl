"""Multi-GPU launcher code generation.

Generates a launcher function that sets up symmetric memory,
creates peer buffer handles, and launches the multi-GPU stencil kernel
on each GPU. The generated code uses ``torch.distributed`` and
``torch.distributed._symmetric_memory`` for peer-to-peer buffer mapping.

This module generates Python *source code* as strings. It does NOT
import torch or cupy at module level.
"""

from __future__ import annotations

from cutile_stencil.codegen.emitter import CodeEmitter
from cutile_stencil.dsl.types import StencilSpec
from cutile_stencil.multigpu.decomposition import DomainDecomposition


def emit_multigpu_launcher(
    spec: StencilSpec,
    decomposition: DomainDecomposition,
) -> str:
    """Generate the multi-GPU launcher function as Python source.

    The generated launcher:

    1. Uses ``torch.distributed`` to set up a NCCL process group.
    2. Creates symmetric memory handles via ``rendezvous()`` so each
       GPU can read peer buffers.
    3. Builds a ``peer_arrays`` list (one entry per GPU).
    4. Launches the multi-GPU stencil kernel on the local GPU.

    Parameters
    ----------
    spec : StencilSpec
        Stencil specification.
    decomposition : DomainDecomposition
        Domain decomposition describing the split.

    Returns
    -------
    str
        Python source code for the launcher module.
    """
    e = CodeEmitter()
    ndim = spec.ndim
    split_axis = decomposition.split_axis
    num_gpus = decomposition.num_gpus
    halo_widths = decomposition.halo_widths

    # --- Header ---
    e.line(f'"""Multi-GPU launcher for {spec.name} stencil (auto-generated)."""')
    e.blank()
    e.line("import os")
    e.blank()
    e.line("import torch")
    e.line("import torch.distributed as dist")
    e.line("from torch.distributed._symmetric_memory import rendezvous")
    e.blank()

    # --- Setup function ---
    e.line("def setup_distributed(rank, world_size, backend='nccl'):")
    with e.indent():
        e.line('"""Initialize the distributed process group."""')
        e.line("os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')")
        e.line("os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')")
        e.line("dist.init_process_group(backend, rank=rank, world_size=world_size)")
        e.line("torch.cuda.set_device(rank)")
    e.blank()
    e.blank()

    # --- Symmetric memory setup ---
    e.line("def setup_symmetric_buffers(local_array, rank, world_size):")
    with e.indent():
        e.line('"""Map peer GPU buffers into local address space via symmetric memory.')
        e.blank()
        e.line("Parameters")
        e.line("----------")
        e.line("local_array : torch.Tensor")
        e.line("    This GPU's padded subdomain (including halo cells).")
        e.line("rank : int")
        e.line("    Local GPU rank.")
        e.line("world_size : int")
        e.line("    Total number of GPUs.")
        e.blank()
        e.line("Returns")
        e.line("-------")
        e.line("list of torch.Tensor")
        e.line("    One tensor per GPU. peer_arrays[rank] is the local array;")
        e.line("    peer_arrays[other] is a view into the other GPU's memory.")
        e.line('"""')
        e.line("# Rendezvous maps all peer buffers into local address space")
        e.line("handle = rendezvous(local_array)")
        e.line("peer_arrays = []")
        e.line("for r in range(world_size):")
        with e.indent():
            shape_parts = []
            for d in range(ndim):
                shape_parts.append(f"local_array.shape[{d}]")
            shape_str = ", ".join(shape_parts)
            e.line(
                f"buf = handle.get_buffer(r, ({shape_str},), "
                f"local_array.dtype)"
            )
            e.line("peer_arrays.append(buf)")
        e.line("return peer_arrays")
    e.blank()
    e.blank()

    # --- Decomposition helper ---
    e.line(f"def decompose_domain(global_domain, world_size):")
    with e.indent():
        e.line('"""Compute local domain parameters for each rank."""')
        e.line(f"split_axis = {split_axis}")
        e.line("extent = global_domain[split_axis]")
        e.line("assert extent % world_size == 0, (")
        with e.indent():
            e.line("f'Domain extent {extent} along axis {split_axis} not divisible by {world_size}'")
        e.line(")")
        e.line("local_extent = extent // world_size")
        halo_tuple = ", ".join(str(h) for h in halo_widths)
        e.line(f"halo_widths = ({halo_tuple},)" if ndim == 1 else f"halo_widths = ({halo_tuple})")
        e.line("return local_extent, halo_widths")
    e.blank()
    e.blank()

    # --- Main launcher ---
    e.line(
        f"def run_{spec.name}_multigpu("
        f"rank, world_size, global_domain, input_data, num_iterations=1):"
    )
    with e.indent():
        e.line(f'"""Run {spec.name} stencil on multiple GPUs.')
        e.blank()
        e.line("Parameters")
        e.line("----------")
        e.line("rank : int")
        e.line("    This GPU's rank.")
        e.line("world_size : int")
        e.line("    Total number of GPUs.")
        e.line("global_domain : tuple of int")
        e.line("    Global interior domain size.")
        e.line("input_data : torch.Tensor")
        e.line("    This GPU's local padded subdomain (with halo).")
        e.line("num_iterations : int")
        e.line("    Number of stencil iterations to run.")
        e.blank()
        e.line("Returns")
        e.line("-------")
        e.line("torch.Tensor")
        e.line("    The output array after all iterations.")
        e.line('"""')
        e.line("# Set up symmetric memory")
        e.line("peer_arrays = setup_symmetric_buffers(input_data, rank, world_size)")
        e.line("output = torch.zeros_like(input_data)")
        e.blank()
        e.line(f"from {spec.name}_multigpu_kernel import launch_{spec.name}_multigpu")
        e.blank()
        e.line("stream = torch.cuda.current_stream()")
        e.line("for _ in range(num_iterations):")
        with e.indent():
            e.line(
                f"launch_{spec.name}_multigpu("
                f"peer_arrays, output, rank, world_size, stream)"
            )
            e.line("# Swap buffers for next iteration")
            e.line("peer_arrays, output = setup_symmetric_buffers(output, rank, world_size), torch.zeros_like(output)")
        e.blank()
        e.line("return output")
    e.blank()
    e.blank()

    # --- Entry point ---
    e.line("if __name__ == '__main__':")
    with e.indent():
        e.line("import argparse")
        e.line("parser = argparse.ArgumentParser()")
        e.line("parser.add_argument('--rank', type=int, required=True)")
        e.line("parser.add_argument('--world-size', type=int, required=True)")
        domain_default = ", ".join(str(d) for d in decomposition.global_domain)
        e.line(f"parser.add_argument('--domain', type=int, nargs={ndim}, default=[{domain_default}])")
        e.line("args = parser.parse_args()")
        e.blank()
        e.line("setup_distributed(args.rank, args.world_size)")
        e.line("global_domain = tuple(args.domain)")
        e.line("local_extent, halo_widths = decompose_domain(global_domain, args.world_size)")
        e.blank()
        e.line("# Create local padded input (placeholder)")
        local_shape_parts = []
        for d in range(ndim):
            if d == split_axis:
                local_shape_parts.append(f"local_extent + 2 * halo_widths[{d}]")
            else:
                local_shape_parts.append(f"global_domain[{d}] + 2 * halo_widths[{d}]")
        local_shape = ", ".join(local_shape_parts)
        e.line(f"local_input = torch.randn({local_shape}, device=f'cuda:{{args.rank}}')")
        e.blank()
        e.line(
            f"result = run_{spec.name}_multigpu("
            f"args.rank, args.world_size, global_domain, local_input)"
        )
        e.line(f"print(f'Rank {{args.rank}}: {spec.name} complete, output shape={{result.shape}}')")
        e.line("dist.destroy_process_group()")

    return e.render()
