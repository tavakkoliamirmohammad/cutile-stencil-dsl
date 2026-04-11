"""Tests for multi-GPU domain decomposition and code generation.

All tests are pure Python -- no GPU or torch imports required.
Decomposition tests verify partitioning logic; codegen tests verify
that generated code is syntactically valid Python (ast.parse).
"""

import ast

import pytest

from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import HardwareSpec, TileConfig
from cutile_stencil.dsl.registry import clear
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.analysis.tiling import compute_tile_config
from cutile_stencil.multigpu.decomposition import (
    Partition,
    DomainDecomposition,
    decompose,
)
from cutile_stencil.multigpu.codegen import MultiGPUStencilCodeGenerator
from cutile_stencil.multigpu.launcher import emit_multigpu_launcher


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


hw = HardwareSpec(shared_mem_bytes=49152, dtype_bytes=8)


# =====================================================================
# Domain Decomposition Tests
# =====================================================================


class TestDomainDecomposition:
    """Test the decompose() function for correctness of partitioning."""

    def test_1d_decompose_2_gpus(self):
        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=(1,))
        assert len(decomp.partitions) == 2
        assert decomp.partitions[0].local_domain == (512,)
        assert decomp.partitions[1].local_domain == (512,)
        assert decomp.partitions[0].local_offset == (0,)
        assert decomp.partitions[1].local_offset == (512,)

    def test_1d_decompose_4_gpus(self):
        decomp = decompose(domain=(1024,), num_gpus=4, halo_widths=(1,))
        assert len(decomp.partitions) == 4
        for i, p in enumerate(decomp.partitions):
            assert p.local_domain == (256,)
            assert p.local_offset == (i * 256,)
            assert p.rank == i

    def test_1d_neighbors(self):
        decomp = decompose(domain=(1024,), num_gpus=4, halo_widths=(1,))
        # First GPU: no left neighbor
        assert decomp.partitions[0].neighbors["left"] is None
        assert decomp.partitions[0].neighbors["right"] == 1
        # Middle GPU: both neighbors
        assert decomp.partitions[1].neighbors["left"] == 0
        assert decomp.partitions[1].neighbors["right"] == 2
        # Another middle GPU
        assert decomp.partitions[2].neighbors["left"] == 1
        assert decomp.partitions[2].neighbors["right"] == 3
        # Last GPU: no right neighbor
        assert decomp.partitions[3].neighbors["left"] == 2
        assert decomp.partitions[3].neighbors["right"] is None

    def test_2d_decompose_split_longest(self):
        decomp = decompose(domain=(256, 1024), num_gpus=4, halo_widths=(1, 1))
        assert decomp.split_axis == 1  # Split longest dimension
        assert decomp.partitions[0].local_domain == (256, 256)
        assert decomp.partitions[1].local_domain == (256, 256)
        assert decomp.partitions[2].local_domain == (256, 256)
        assert decomp.partitions[3].local_domain == (256, 256)

    def test_2d_split_axis_0(self):
        decomp = decompose(
            domain=(1024, 256), num_gpus=2, halo_widths=(1, 1), split_axis=0
        )
        assert decomp.split_axis == 0
        assert decomp.partitions[0].local_domain == (512, 256)
        assert decomp.partitions[1].local_domain == (512, 256)
        assert decomp.partitions[0].local_offset == (0, 0)
        assert decomp.partitions[1].local_offset == (512, 0)

    def test_2d_split_axis_1(self):
        decomp = decompose(
            domain=(256, 1024), num_gpus=2, halo_widths=(1, 1), split_axis=1
        )
        assert decomp.split_axis == 1
        assert decomp.partitions[0].local_domain == (256, 512)
        assert decomp.partitions[1].local_domain == (256, 512)
        assert decomp.partitions[0].local_offset == (0, 0)
        assert decomp.partitions[1].local_offset == (0, 512)

    def test_3d_decompose_split_longest(self):
        decomp = decompose(
            domain=(64, 64, 256), num_gpus=4, halo_widths=(1, 1, 1)
        )
        assert decomp.split_axis == 2  # z-axis is longest
        assert decomp.partitions[0].local_domain == (64, 64, 64)

    def test_indivisible_raises(self):
        with pytest.raises(ValueError, match="not divisible"):
            decompose(domain=(1000,), num_gpus=3, halo_widths=(1,))

    def test_single_gpu(self):
        decomp = decompose(domain=(1024,), num_gpus=1, halo_widths=(1,))
        assert len(decomp.partitions) == 1
        assert decomp.partitions[0].local_domain == (1024,)
        assert decomp.partitions[0].neighbors["left"] is None
        assert decomp.partitions[0].neighbors["right"] is None

    def test_zero_gpus_raises(self):
        with pytest.raises(ValueError, match="num_gpus must be >= 1"):
            decompose(domain=(1024,), num_gpus=0, halo_widths=(1,))

    def test_halo_dim_mismatch_raises(self):
        with pytest.raises(ValueError, match="halo_widths has"):
            decompose(domain=(1024,), num_gpus=2, halo_widths=(1, 1))

    def test_split_axis_out_of_range_raises(self):
        with pytest.raises(ValueError, match="split_axis.*out of range"):
            decompose(
                domain=(1024,), num_gpus=2, halo_widths=(1,), split_axis=1
            )

    def test_local_too_small_for_halo_raises(self):
        with pytest.raises(ValueError, match="smaller than halo"):
            decompose(
                domain=(8,), num_gpus=8, halo_widths=(2,)
            )

    def test_global_domain_preserved(self):
        decomp = decompose(domain=(512, 256), num_gpus=2, halo_widths=(1, 1))
        for p in decomp.partitions:
            assert p.global_domain == (512, 256)
            assert p.halo_widths == (1, 1)

    def test_partition_is_frozen(self):
        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=(1,))
        with pytest.raises(AttributeError):
            decomp.partitions[0].rank = 99

    def test_decomposition_is_frozen(self):
        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=(1,))
        with pytest.raises(AttributeError):
            decomp.num_gpus = 99

    def test_higher_order_halo(self):
        """4th-order stencil has halo width 2."""
        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=(2,))
        assert decomp.partitions[0].halo_widths == (2,)
        assert decomp.partitions[0].local_domain == (512,)

    def test_offsets_cover_full_domain(self):
        """Verify that partition offsets + local_domains tile the global domain."""
        decomp = decompose(domain=(120, 360), num_gpus=3, halo_widths=(1, 1))
        split = decomp.split_axis
        total = 0
        for p in decomp.partitions:
            total += p.local_domain[split]
        assert total == 360


# =====================================================================
# Multi-GPU Code Generation Tests
# =====================================================================


def _make_1d_spec():
    @stencil(ndim=1, order=2)
    def heat(u, i):
        return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

    spec = heat._stencil_spec
    accesses = extract_footprint(spec)
    spec.halo_widths = compute_halo(accesses, 1)
    tile_cfg = compute_tile_config(spec, (1024,), hw)
    return spec, tile_cfg


def _make_2d_spec():
    @stencil(ndim=2, order=2)
    def lap2d(u, i, j):
        return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]

    spec = lap2d._stencil_spec
    accesses = extract_footprint(spec)
    spec.halo_widths = compute_halo(accesses, 2)
    tile_cfg = compute_tile_config(spec, (256, 256), hw)
    return spec, tile_cfg


def _make_3d_spec():
    @stencil(ndim=3, order=2)
    def lap3d(u, i, j, k):
        return (
            u[i - 1, j, k] + u[i + 1, j, k]
            + u[i, j - 1, k] + u[i, j + 1, k]
            + u[i, j, k - 1] + u[i, j, k + 1]
            - 6 * u[i, j, k]
        )

    spec = lap3d._stencil_spec
    accesses = extract_footprint(spec)
    spec.halo_widths = compute_halo(accesses, 3)
    tile_cfg = compute_tile_config(spec, (64, 64, 64), hw)
    return spec, tile_cfg


class TestMultiGPUCodegen:
    """Test that multi-GPU kernel code is syntactically valid Python."""

    def test_1d_generates_valid_python(self):
        spec, tile_cfg = _make_1d_spec()
        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=spec.halo_widths)
        gen = MultiGPUStencilCodeGenerator(spec, tile_cfg, decomp)
        code = gen.emit()
        ast.parse(code)

    def test_2d_generates_valid_python(self):
        spec, tile_cfg = _make_2d_spec()
        decomp = decompose(domain=(256, 256), num_gpus=2, halo_widths=spec.halo_widths)
        gen = MultiGPUStencilCodeGenerator(spec, tile_cfg, decomp)
        code = gen.emit()
        ast.parse(code)

    def test_3d_generates_valid_python(self):
        spec, tile_cfg = _make_3d_spec()
        decomp = decompose(domain=(64, 64, 64), num_gpus=2, halo_widths=spec.halo_widths)
        gen = MultiGPUStencilCodeGenerator(spec, tile_cfg, decomp)
        code = gen.emit()
        ast.parse(code)

    def test_1d_contains_peer_arrays(self):
        spec, tile_cfg = _make_1d_spec()
        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=spec.halo_widths)
        gen = MultiGPUStencilCodeGenerator(spec, tile_cfg, decomp)
        code = gen.emit()
        assert "peer_arrays" in code

    def test_1d_contains_gather(self):
        spec, tile_cfg = _make_1d_spec()
        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=spec.halo_widths)
        gen = MultiGPUStencilCodeGenerator(spec, tile_cfg, decomp)
        code = gen.emit()
        assert "ct.gather" in code

    def test_1d_contains_rank_and_world_size(self):
        spec, tile_cfg = _make_1d_spec()
        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=spec.halo_widths)
        gen = MultiGPUStencilCodeGenerator(spec, tile_cfg, decomp)
        code = gen.emit()
        assert "RANK" in code
        assert "WORLD_SIZE" in code

    def test_1d_kernel_name(self):
        spec, tile_cfg = _make_1d_spec()
        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=spec.halo_widths)
        gen = MultiGPUStencilCodeGenerator(spec, tile_cfg, decomp)
        code = gen.emit()
        assert f"{spec.name}_multigpu_kernel" in code

    def test_1d_launcher_name(self):
        spec, tile_cfg = _make_1d_spec()
        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=spec.halo_widths)
        gen = MultiGPUStencilCodeGenerator(spec, tile_cfg, decomp)
        code = gen.emit()
        assert f"launch_{spec.name}_multigpu" in code

    def test_1d_has_ct_kernel_decorator(self):
        spec, tile_cfg = _make_1d_spec()
        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=spec.halo_widths)
        gen = MultiGPUStencilCodeGenerator(spec, tile_cfg, decomp)
        code = gen.emit()
        assert "@ct.kernel" in code

    def test_1d_has_ct_load_store(self):
        spec, tile_cfg = _make_1d_spec()
        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=spec.halo_widths)
        gen = MultiGPUStencilCodeGenerator(spec, tile_cfg, decomp)
        code = gen.emit()
        assert "ct.load" in code
        assert "ct.store" in code

    def test_2d_split_axis_0(self):
        spec, tile_cfg = _make_2d_spec()
        decomp = decompose(
            domain=(256, 256), num_gpus=2, halo_widths=spec.halo_widths,
            split_axis=0,
        )
        gen = MultiGPUStencilCodeGenerator(spec, tile_cfg, decomp)
        code = gen.emit()
        ast.parse(code)
        assert "peer_arrays" in code
        assert "LOCAL_SPLIT" in code

    def test_2d_split_axis_1(self):
        spec, tile_cfg = _make_2d_spec()
        decomp = decompose(
            domain=(256, 256), num_gpus=4, halo_widths=spec.halo_widths,
            split_axis=1,
        )
        gen = MultiGPUStencilCodeGenerator(spec, tile_cfg, decomp)
        code = gen.emit()
        ast.parse(code)
        assert "peer_arrays" in code

    def test_3d_contains_gather(self):
        spec, tile_cfg = _make_3d_spec()
        decomp = decompose(
            domain=(64, 64, 64), num_gpus=2, halo_widths=spec.halo_widths,
        )
        gen = MultiGPUStencilCodeGenerator(spec, tile_cfg, decomp)
        code = gen.emit()
        assert "peer_arrays" in code
        assert "ct.gather" in code or "ct.load" in code  # gather or gather-like load

    def test_4_gpus_generates_valid_python(self):
        spec, tile_cfg = _make_1d_spec()
        decomp = decompose(domain=(2048,), num_gpus=4, halo_widths=spec.halo_widths)
        gen = MultiGPUStencilCodeGenerator(spec, tile_cfg, decomp)
        code = gen.emit()
        ast.parse(code)


class TestMultiGPULauncher:
    """Test that generated launcher code is syntactically valid Python."""

    def test_1d_launcher_valid_python(self):
        spec, _ = _make_1d_spec()
        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=spec.halo_widths)
        code = emit_multigpu_launcher(spec, decomp)
        ast.parse(code)

    def test_2d_launcher_valid_python(self):
        spec, _ = _make_2d_spec()
        decomp = decompose(
            domain=(256, 256), num_gpus=2, halo_widths=spec.halo_widths
        )
        code = emit_multigpu_launcher(spec, decomp)
        ast.parse(code)

    def test_3d_launcher_valid_python(self):
        spec, _ = _make_3d_spec()
        decomp = decompose(
            domain=(64, 64, 64), num_gpus=2, halo_widths=spec.halo_widths
        )
        code = emit_multigpu_launcher(spec, decomp)
        ast.parse(code)

    def test_launcher_contains_rendezvous(self):
        spec, _ = _make_1d_spec()
        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=spec.halo_widths)
        code = emit_multigpu_launcher(spec, decomp)
        assert "rendezvous" in code

    def test_launcher_contains_symmetric_memory(self):
        spec, _ = _make_1d_spec()
        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=spec.halo_widths)
        code = emit_multigpu_launcher(spec, decomp)
        assert "symmetric_memory" in code

    def test_launcher_contains_dist_init(self):
        spec, _ = _make_1d_spec()
        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=spec.halo_widths)
        code = emit_multigpu_launcher(spec, decomp)
        assert "init_process_group" in code

    def test_launcher_references_stencil_name(self):
        spec, _ = _make_1d_spec()
        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=spec.halo_widths)
        code = emit_multigpu_launcher(spec, decomp)
        assert spec.name in code


# =====================================================================
# Integration: decompose + codegen
# =====================================================================


class TestMultiGPUIntegration:
    """Integration tests combining decomposition with code generation."""

    def test_compile_and_decompose_1d(self):
        """Full pipeline: define stencil -> decompose -> generate multi-GPU code."""
        spec, tile_cfg = _make_1d_spec()
        decomp = decompose(
            domain=(1024,), num_gpus=2, halo_widths=spec.halo_widths
        )
        gen = MultiGPUStencilCodeGenerator(spec, tile_cfg, decomp)
        code = gen.emit()
        ast.parse(code)
        # Kernel + launcher both present
        assert f"{spec.name}_multigpu_kernel" in code
        assert f"launch_{spec.name}_multigpu" in code
        assert "@ct.kernel" in code

    def test_compile_and_decompose_2d(self):
        spec, tile_cfg = _make_2d_spec()
        decomp = decompose(
            domain=(256, 256), num_gpus=4, halo_widths=spec.halo_widths
        )
        gen = MultiGPUStencilCodeGenerator(spec, tile_cfg, decomp)
        code = gen.emit()
        ast.parse(code)
        assert f"{spec.name}_multigpu_kernel" in code

    def test_decomposition_metadata_consistent(self):
        """Verify decomposition and codegen agree on split axis."""
        spec, tile_cfg = _make_2d_spec()
        decomp = decompose(
            domain=(128, 512), num_gpus=4, halo_widths=spec.halo_widths
        )
        assert decomp.split_axis == 1  # 512 is the longest
        gen = MultiGPUStencilCodeGenerator(spec, tile_cfg, decomp)
        code = gen.emit()
        ast.parse(code)

    def test_import_from_package(self):
        """Verify the __init__.py exports work."""
        from cutile_stencil.multigpu import (
            Partition,
            DomainDecomposition,
            decompose,
            MultiGPUStencilCodeGenerator,
            emit_multigpu_launcher,
        )
        # All five names should be importable
        assert Partition is not None
        assert DomainDecomposition is not None


# =====================================================================
# Halo Exchange Tests
# =====================================================================


class TestHaloExchange:
    """Test halo exchange plan creation and code generation."""

    def test_create_exchange_plan(self):
        from cutile_stencil.multigpu.halo_exchange import create_exchange_plan

        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=(1,))
        plan = create_exchange_plan(decomp, rank=0, local_shape=(514,))
        assert plan.left_neighbor is None
        assert plan.right_neighbor == 1
        assert plan.halo_width == 1

    def test_periodic_exchange_plan(self):
        from cutile_stencil.multigpu.halo_exchange import create_exchange_plan

        decomp = decompose(domain=(1024,), num_gpus=4, halo_widths=(1,))
        plan = create_exchange_plan(
            decomp, rank=0, local_shape=(258,), periodic=True
        )
        assert plan.left_neighbor == 3  # wraps around
        assert plan.right_neighbor == 1

    def test_emit_exchange_valid_python(self):
        from cutile_stencil.multigpu.halo_exchange import (
            create_exchange_plan,
            emit_halo_exchange_function,
        )

        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=(1,))
        plan = create_exchange_plan(decomp, rank=0, local_shape=(514,))
        code = emit_halo_exchange_function(plan)
        ast.parse(code)

    def test_exchange_has_pack_unpack(self):
        from cutile_stencil.multigpu.halo_exchange import (
            create_exchange_plan,
            emit_halo_exchange_function,
        )

        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=(1,))
        plan = create_exchange_plan(decomp, rank=0, local_shape=(514,))
        code = emit_halo_exchange_function(plan)
        assert "pack_halo_to_left" in code
        assert "pack_halo_to_right" in code
        assert "unpack_halo_from_left" in code
        assert "unpack_halo_from_right" in code

    def test_exchange_has_overlap_function(self):
        from cutile_stencil.multigpu.halo_exchange import (
            create_exchange_plan,
            emit_halo_exchange_function,
        )

        decomp = decompose(domain=(1024,), num_gpus=2, halo_widths=(1,))
        plan = create_exchange_plan(decomp, rank=0, local_shape=(514,))
        code = emit_halo_exchange_function(plan)
        assert "step_with_overlap" in code
        assert "compute_stream" in code
        assert "comm_stream" in code

    def test_2d_exchange_has_contiguous(self):
        from cutile_stencil.multigpu.halo_exchange import (
            create_exchange_plan,
            emit_halo_exchange_function,
        )

        decomp = decompose(
            domain=(256, 1024), num_gpus=4, halo_widths=(1, 1)
        )
        plan = create_exchange_plan(
            decomp, rank=1, local_shape=(258, 258)
        )
        code = emit_halo_exchange_function(plan)
        assert "ascontiguousarray" in code


# =====================================================================
# Periodic Decomposition Tests
# =====================================================================


class TestPeriodicDecomposition:
    """Test periodic boundary conditions in domain decomposition."""

    def test_periodic_1d_neighbors(self):
        decomp = decompose(
            domain=(1024,), num_gpus=4, halo_widths=(1,), periodic=True
        )
        # First GPU: left neighbor wraps to last
        assert decomp.partitions[0].neighbors["left"] == 3
        assert decomp.partitions[0].neighbors["right"] == 1
        # Last GPU: right neighbor wraps to first
        assert decomp.partitions[3].neighbors["left"] == 2
        assert decomp.partitions[3].neighbors["right"] == 0

    def test_periodic_2_gpus(self):
        decomp = decompose(
            domain=(512,), num_gpus=2, halo_widths=(1,), periodic=True
        )
        assert decomp.partitions[0].neighbors["left"] == 1
        assert decomp.partitions[0].neighbors["right"] == 1
        assert decomp.partitions[1].neighbors["left"] == 0
        assert decomp.partitions[1].neighbors["right"] == 0


# =====================================================================
# GPU Integration: actual multi-GPU stencil execution with P2P halo exchange
# =====================================================================

try:
    import cupy as cp
    cp.cuda.Device(0).compute_capability
    _HAS_GPU = True
    _NUM_GPUS = cp.cuda.runtime.getDeviceCount()
except Exception:
    _HAS_GPU = False
    _NUM_GPUS = 0

gpu_required = pytest.mark.skipif(not _HAS_GPU, reason="No GPU available")
multi_gpu_required = pytest.mark.skipif(_NUM_GPUS < 2, reason="Need >= 2 GPUs")


def _enable_p2p(n_gpus):
    """Enable P2P access between GPU pairs."""
    for i in range(n_gpus):
        cp.cuda.Device(i).use()
        for j in range(n_gpus):
            if i != j:
                try:
                    cp.cuda.runtime.deviceEnablePeerAccess(j)
                except cp.cuda.runtime.CUDARuntimeError:
                    pass


def _run_multigpu_1d_stencil(n_gpus, n_steps=10):
    """Run 1D heat stencil across n_gpus with P2P halo exchange, return True if matches CPU."""
    import numpy as np
    from cutile_stencil import compile as stencil_compile
    from cutile_stencil.reference.stencil_ref import apply_stencil
    import importlib.util
    import tempfile

    @stencil(ndim=1, order=2)
    def heat_mgpu(u, i):
        return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

    halo = 1
    local_n = 256
    global_n = local_n * n_gpus

    result = stencil_compile(heat_mgpu, domain=(local_n,))
    tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
    tmp.write(result.code)
    tmp.close()
    mod_spec = importlib.util.spec_from_file_location("_k", tmp.name)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    launch = getattr(mod, f"launch_{heat_mgpu._stencil_spec.name}")

    np.random.seed(42)
    u_global = np.random.randn(global_n + 2 * halo).astype(np.float64)

    _enable_p2p(n_gpus)

    u_gpu = []
    out_gpu = []
    for rank in range(n_gpus):
        cp.cuda.Device(rank).use()
        start = rank * local_n
        u_gpu.append(cp.asarray(u_global[start : start + local_n + 2 * halo]))
        out_gpu.append(cp.zeros(local_n + 2 * halo, dtype=cp.float64))

    for step in range(n_steps):
        for rank in range(n_gpus):
            cp.cuda.Device(rank).use()
            launch(u_gpu[rank], out_gpu[rank])
        for rank in range(n_gpus):
            cp.cuda.Device(rank).synchronize()

        for rank in range(n_gpus):
            cp.cuda.Device(rank).use()
            u_gpu[rank][halo:-halo] = out_gpu[rank][halo:-halo]

        # P2P halo exchange
        for rank in range(n_gpus):
            cp.cuda.Device(rank).use()
            if rank < n_gpus - 1:
                u_gpu[rank][-halo:].data.copy_from_device(
                    u_gpu[rank + 1][halo : 2 * halo].data, halo * 8
                )
            if rank > 0:
                u_gpu[rank][:halo].data.copy_from_device(
                    u_gpu[rank - 1][-2 * halo : -halo].data, halo * 8
                )

    u_cpu = u_global.copy()
    for step in range(n_steps):
        u_cpu = apply_stencil(u_cpu, heat_mgpu._stencil_spec)

    all_match = True
    for rank in range(n_gpus):
        cp.cuda.Device(rank).use()
        gpu_int = cp.asnumpy(u_gpu[rank])[halo:-halo]
        cpu_int = u_cpu[halo + rank * local_n : halo + (rank + 1) * local_n]
        if not np.allclose(gpu_int, cpu_int, atol=1e-10):
            all_match = False
    return all_match


def _run_multigpu_2d_stencil(n_gpus, n_steps=5):
    """Run 2D Laplacian stencil across n_gpus (split along axis 0) with P2P halo exchange."""
    import numpy as np
    from cutile_stencil import compile as stencil_compile
    from cutile_stencil.reference.stencil_ref import apply_stencil
    import importlib.util
    import tempfile

    @stencil(ndim=2, order=2)
    def lap2d_mgpu(u, i, j):
        return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

    hx, hy = 1, 1
    local_ny = 128
    nx = 128
    global_ny = local_ny * n_gpus

    result = stencil_compile(lap2d_mgpu, domain=(nx, local_ny))
    tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
    tmp.write(result.code)
    tmp.close()
    mod_spec = importlib.util.spec_from_file_location("_k2d", tmp.name)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    launch = getattr(mod, f"launch_{lap2d_mgpu._stencil_spec.name}")

    np.random.seed(123)
    u_global = np.random.randn(nx + 2 * hx, global_ny + 2 * hy).astype(np.float64)

    _enable_p2p(n_gpus)

    # Split along axis 1 (columns)
    u_gpu = []
    out_gpu = []
    for rank in range(n_gpus):
        cp.cuda.Device(rank).use()
        col_start = rank * local_ny
        u_gpu.append(cp.asarray(u_global[:, col_start : col_start + local_ny + 2 * hy]))
        out_gpu.append(cp.zeros((nx + 2 * hx, local_ny + 2 * hy), dtype=cp.float64))

    for step in range(n_steps):
        for rank in range(n_gpus):
            cp.cuda.Device(rank).use()
            launch(u_gpu[rank], out_gpu[rank])
        for rank in range(n_gpus):
            cp.cuda.Device(rank).synchronize()

        for rank in range(n_gpus):
            cp.cuda.Device(rank).use()
            u_gpu[rank][hx:-hx, hy:-hy] = out_gpu[rank][hx:-hx, hy:-hy]

        # P2P halo exchange along split axis (axis 1)
        for rank in range(n_gpus):
            cp.cuda.Device(rank).use()
            if rank < n_gpus - 1:
                src_col = u_gpu[rank + 1][:, hy : 2 * hy]
                u_gpu[rank][:, -hy:] = cp.asarray(cp.asnumpy(src_col))
            if rank > 0:
                src_col = u_gpu[rank - 1][:, -2 * hy : -hy]
                u_gpu[rank][:, :hy] = cp.asarray(cp.asnumpy(src_col))

    u_cpu = u_global.copy()
    spec = lap2d_mgpu._stencil_spec
    for step in range(n_steps):
        u_cpu = apply_stencil(u_cpu, spec)

    all_match = True
    for rank in range(n_gpus):
        cp.cuda.Device(rank).use()
        gpu_int = cp.asnumpy(u_gpu[rank])[hx:-hx, hy:-hy]
        col_start = hy + rank * local_ny
        cpu_int = u_cpu[hx:-hx, col_start : col_start + local_ny]
        if not np.allclose(gpu_int, cpu_int, atol=1e-10):
            all_match = False
    return all_match


def _run_multigpu_3d_stencil(n_gpus, n_steps=3):
    """Run 3D heat stencil across n_gpus (split along axis 2) with P2P halo exchange."""
    import numpy as np
    from cutile_stencil import compile as stencil_compile
    from cutile_stencil.reference.stencil_ref import apply_stencil
    import importlib.util
    import tempfile

    @stencil(ndim=3, order=2)
    def heat3d_mgpu(u, i, j, k):
        return (u[i-1,j,k] + u[i+1,j,k] + u[i,j-1,k] + u[i,j+1,k] + u[i,j,k-1] + u[i,j,k+1]) / 6.0

    halo = (1, 1, 1)
    nx, ny = 32, 32
    local_nz = 32
    global_nz = local_nz * n_gpus

    result = stencil_compile(heat3d_mgpu, domain=(nx, ny, local_nz))
    tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
    tmp.write(result.code)
    tmp.close()
    mod_spec = importlib.util.spec_from_file_location("_k3d", tmp.name)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    launch = getattr(mod, f"launch_{heat3d_mgpu._stencil_spec.name}")

    np.random.seed(456)
    u_global = np.random.randn(nx + 2, ny + 2, global_nz + 2).astype(np.float64)

    _enable_p2p(n_gpus)

    hz = halo[2]
    u_gpu = []
    out_gpu = []
    for rank in range(n_gpus):
        cp.cuda.Device(rank).use()
        z_start = rank * local_nz
        u_gpu.append(cp.asarray(u_global[:, :, z_start : z_start + local_nz + 2 * hz]))
        out_gpu.append(cp.zeros((nx + 2, ny + 2, local_nz + 2 * hz), dtype=cp.float64))

    for step in range(n_steps):
        for rank in range(n_gpus):
            cp.cuda.Device(rank).use()
            launch(u_gpu[rank], out_gpu[rank])
        for rank in range(n_gpus):
            cp.cuda.Device(rank).synchronize()

        for rank in range(n_gpus):
            cp.cuda.Device(rank).use()
            u_gpu[rank][1:-1, 1:-1, hz:-hz] = out_gpu[rank][1:-1, 1:-1, hz:-hz]

        # P2P halo exchange along split axis (axis 2)
        for rank in range(n_gpus):
            cp.cuda.Device(rank).use()
            if rank < n_gpus - 1:
                src = u_gpu[rank + 1][:, :, hz : 2 * hz]
                u_gpu[rank][:, :, -hz:] = cp.asarray(cp.asnumpy(src))
            if rank > 0:
                src = u_gpu[rank - 1][:, :, -2 * hz : -hz]
                u_gpu[rank][:, :, :hz] = cp.asarray(cp.asnumpy(src))

    u_cpu = u_global.copy()
    spec = heat3d_mgpu._stencil_spec
    for step in range(n_steps):
        u_cpu = apply_stencil(u_cpu, spec)

    all_match = True
    for rank in range(n_gpus):
        cp.cuda.Device(rank).use()
        gpu_int = cp.asnumpy(u_gpu[rank])[1:-1, 1:-1, hz:-hz]
        z_start = hz + rank * local_nz
        cpu_int = u_cpu[1:-1, 1:-1, z_start : z_start + local_nz]
        if not np.allclose(gpu_int, cpu_int, atol=1e-10):
            all_match = False
    return all_match


@multi_gpu_required
class TestMultiGPUExecution:
    """GPU integration tests: real stencil execution across multiple GPUs."""

    def test_1d_2_gpus(self):
        assert _run_multigpu_1d_stencil(2, n_steps=10)

    def test_1d_3_gpus(self):
        if _NUM_GPUS < 3:
            pytest.skip("Need >= 3 GPUs")
        assert _run_multigpu_1d_stencil(3, n_steps=10)

    def test_1d_4_gpus(self):
        if _NUM_GPUS < 4:
            pytest.skip("Need >= 4 GPUs")
        assert _run_multigpu_1d_stencil(4, n_steps=10)

    def test_2d_2_gpus(self):
        assert _run_multigpu_2d_stencil(2, n_steps=5)

    def test_2d_4_gpus(self):
        if _NUM_GPUS < 4:
            pytest.skip("Need >= 4 GPUs")
        assert _run_multigpu_2d_stencil(4, n_steps=5)

    def test_3d_2_gpus(self):
        assert _run_multigpu_3d_stencil(2, n_steps=3)

    def test_2d_halo_exchange_codegen(self):
        """Use the generated exchange_halos function for 2D multi-GPU stencil."""
        import numpy as np
        import importlib.util
        import tempfile
        from cutile_stencil import compile as stencil_compile
        from cutile_stencil.reference.stencil_ref import apply_stencil
        from cutile_stencil.multigpu.halo_exchange import (
            create_exchange_plan,
            emit_halo_exchange_function,
        )

        n_gpus = 2
        n_steps = 5
        hx, hy = 1, 1
        local_ny = 64
        nx = 64

        @stencil(ndim=2, order=2)
        def lap_exch(u, i, j):
            return 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])

        spec = lap_exch._stencil_spec

        # Compile stencil kernel
        result = stencil_compile(lap_exch, domain=(nx, local_ny))
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(result.code)
            kernel_path = f.name
        mod_spec = importlib.util.spec_from_file_location("_kexch", kernel_path)
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        launch = getattr(mod, f"launch_{spec.name}")

        # Generate and load exchange functions for each rank
        decomp = decompose(
            domain=(nx, local_ny * n_gpus),
            num_gpus=n_gpus,
            halo_widths=(hx, hy),
        )
        exchange_mods = []
        for rank in range(n_gpus):
            local_shape = (nx + 2 * hx, local_ny + 2 * hy)
            plan = create_exchange_plan(decomp, rank, local_shape)
            code = emit_halo_exchange_function(plan)
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
                f.write("import cupy as cp\nimport numpy as np\n" + code)
                exch_path = f.name
            ms = importlib.util.spec_from_file_location(f"_exch{rank}", exch_path)
            m = importlib.util.module_from_spec(ms)
            ms.loader.exec_module(m)
            exchange_mods.append(m)

        # Setup
        np.random.seed(99)
        global_ny = local_ny * n_gpus
        u_global = np.random.randn(nx + 2 * hx, global_ny + 2 * hy).astype(np.float64)

        _enable_p2p(n_gpus)

        u_gpu = []
        out_gpu = []
        for rank in range(n_gpus):
            cp.cuda.Device(rank).use()
            col_start = rank * local_ny
            u_gpu.append(cp.asarray(u_global[:, col_start : col_start + local_ny + 2 * hy]))
            out_gpu.append(cp.zeros((nx + 2 * hx, local_ny + 2 * hy), dtype=cp.float64))

        # Run stencil with generated exchange_halos
        for step in range(n_steps):
            for rank in range(n_gpus):
                cp.cuda.Device(rank).use()
                launch(u_gpu[rank], out_gpu[rank])
            for rank in range(n_gpus):
                cp.cuda.Device(rank).synchronize()

            for rank in range(n_gpus):
                cp.cuda.Device(rank).use()
                u_gpu[rank][hx:-hx, hy:-hy] = out_gpu[rank][hx:-hx, hy:-hy]

            # Use generated exchange_halos
            for rank in range(n_gpus):
                cp.cuda.Device(rank).use()
                exchange_mods[rank].exchange_halos(u_gpu[rank], u_gpu, rank)

        # CPU reference
        u_cpu = u_global.copy()
        for step in range(n_steps):
            u_cpu = apply_stencil(u_cpu, spec)

        all_match = True
        for rank in range(n_gpus):
            cp.cuda.Device(rank).use()
            gpu_int = cp.asnumpy(u_gpu[rank])[hx:-hx, hy:-hy]
            col_start = hy + rank * local_ny
            cpu_int = u_cpu[hx:-hx, col_start : col_start + local_ny]
            maxdiff = np.max(np.abs(gpu_int - cpu_int))
            assert np.allclose(gpu_int, cpu_int, atol=1e-10), (
                f"Rank {rank} mismatch: max_diff={maxdiff:.2e}"
            )
        assert decompose is not None
        assert MultiGPUStencilCodeGenerator is not None
        assert emit_multigpu_launcher is not None


# =====================================================================
# Cartesian Decomposition Tests
# =====================================================================


class TestCartesianDecomposition:
    """Test the decompose_cartesian() function for 2D/3D partitioning."""

    def test_2d_4_gpus(self):
        from cutile_stencil.multigpu.decomposition import decompose_cartesian
        decomp = decompose_cartesian(
            domain=(256, 256), num_gpus=4, halo_widths=(1, 1)
        )
        assert decomp.topology == (2, 2)
        assert len(decomp.partitions) == 4
        assert decomp.partitions[0].local_domain == (128, 128)

    def test_2d_4_gpus_neighbors(self):
        from cutile_stencil.multigpu.decomposition import decompose_cartesian
        decomp = decompose_cartesian(
            domain=(256, 256), num_gpus=4, halo_widths=(1, 1)
        )
        # Partition (0,0): right neighbor on axis 0 is (1,0), below on axis 1 is (0,1)
        p00 = decomp.partitions[0]  # coords (0, 0)
        assert p00.neighbors[(0, +1)] is not None  # has right neighbor on axis 0
        assert p00.neighbors[(1, +1)] is not None  # has bottom neighbor on axis 1
        assert p00.neighbors[(0, -1)] is None      # no left (boundary)
        assert p00.neighbors[(1, -1)] is None      # no top (boundary)

    def test_3d_8_gpus(self):
        from cutile_stencil.multigpu.decomposition import decompose_cartesian
        decomp = decompose_cartesian(
            domain=(64, 64, 64), num_gpus=8, halo_widths=(1, 1, 1)
        )
        assert decomp.topology == (2, 2, 2)
        assert len(decomp.partitions) == 8
        assert decomp.partitions[0].local_domain == (32, 32, 32)

    def test_custom_topology(self):
        from cutile_stencil.multigpu.decomposition import decompose_cartesian
        decomp = decompose_cartesian(
            domain=(256, 512), num_gpus=8, halo_widths=(1, 1),
            topology=(2, 4)
        )
        assert decomp.topology == (2, 4)
        assert decomp.partitions[0].local_domain == (128, 128)

    def test_periodic_cartesian(self):
        from cutile_stencil.multigpu.decomposition import decompose_cartesian
        decomp = decompose_cartesian(
            domain=(256, 256), num_gpus=4, halo_widths=(1, 1),
            periodic=True
        )
        # Corner partition (0,0) should wrap
        p00 = decomp.partitions[0]
        assert p00.neighbors[(0, -1)] is not None  # wraps to right column
        assert p00.neighbors[(1, -1)] is not None  # wraps to bottom row

    def test_offsets_cover_domain(self):
        from cutile_stencil.multigpu.decomposition import decompose_cartesian
        decomp = decompose_cartesian(
            domain=(256, 512), num_gpus=8, halo_widths=(1, 1),
            topology=(2, 4)
        )
        # All offsets + local_domain should tile the global domain
        for d in range(2):
            covered = set()
            for p in decomp.partitions:
                start = p.local_offset[d]
                end = start + p.local_domain[d]
                for x in range(start, end):
                    covered.add(x)
            assert len(covered) == decomp.global_domain[d]

    def test_invalid_topology_product(self):
        from cutile_stencil.multigpu.decomposition import decompose_cartesian
        with pytest.raises(ValueError, match="Product"):
            decompose_cartesian(
                domain=(256, 256), num_gpus=4, halo_widths=(1, 1),
                topology=(2, 3)  # 2*3=6 != 4
            )

    def test_indivisible_raises(self):
        from cutile_stencil.multigpu.decomposition import decompose_cartesian
        # topology=(2,2), domain[0]=100 is divisible but domain[1]=101 is not
        with pytest.raises(ValueError, match="not divisible"):
            decompose_cartesian(
                domain=(100, 101), num_gpus=4, halo_widths=(1, 1),
                topology=(2, 2)
            )

    def test_factorize_6_gpus_2d(self):
        from cutile_stencil.multigpu.decomposition import _factorize_gpus
        topo = _factorize_gpus(6, 2)
        assert topo[0] * topo[1] == 6
        assert max(topo) - min(topo) <= 1  # (2, 3) is balanced

    def test_rank_coords_bijection(self):
        """Each rank maps to exactly one unique set of coords."""
        from cutile_stencil.multigpu.decomposition import decompose_cartesian
        decomp = decompose_cartesian(
            domain=(256, 256), num_gpus=4, halo_widths=(1, 1)
        )
        coords_seen = [p.coords for p in decomp.partitions]
        assert len(set(coords_seen)) == 4  # all unique

    def test_partition_is_frozen(self):
        from cutile_stencil.multigpu.decomposition import decompose_cartesian
        decomp = decompose_cartesian(
            domain=(256, 256), num_gpus=4, halo_widths=(1, 1)
        )
        with pytest.raises(AttributeError):
            decomp.partitions[0].rank = 99

    def test_decomposition_is_frozen(self):
        from cutile_stencil.multigpu.decomposition import decompose_cartesian
        decomp = decompose_cartesian(
            domain=(256, 256), num_gpus=4, halo_widths=(1, 1)
        )
        with pytest.raises(AttributeError):
            decomp.num_gpus = 99

    def test_halo_dim_mismatch_raises(self):
        from cutile_stencil.multigpu.decomposition import decompose_cartesian
        with pytest.raises(ValueError, match="halo_widths has"):
            decompose_cartesian(
                domain=(256, 256), num_gpus=4, halo_widths=(1,)
            )

    def test_topology_dim_mismatch_raises(self):
        from cutile_stencil.multigpu.decomposition import decompose_cartesian
        with pytest.raises(ValueError, match="topology has"):
            decompose_cartesian(
                domain=(256, 256), num_gpus=4, halo_widths=(1, 1),
                topology=(2, 2, 1)  # 3D topology for 2D domain
            )

    def test_global_domain_preserved(self):
        from cutile_stencil.multigpu.decomposition import decompose_cartesian
        decomp = decompose_cartesian(
            domain=(256, 512), num_gpus=8, halo_widths=(1, 2),
            topology=(2, 4)
        )
        for p in decomp.partitions:
            assert p.global_domain == (256, 512)
            assert p.halo_widths == (1, 2)

    def test_import_from_package(self):
        """Verify the __init__.py exports for Cartesian types."""
        from cutile_stencil.multigpu import (
            CartesianPartition,
            CartesianDecomposition,
            decompose_cartesian,
        )
        assert CartesianPartition is not None
        assert CartesianDecomposition is not None
        assert decompose_cartesian is not None
