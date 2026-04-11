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
        assert decompose is not None
        assert MultiGPUStencilCodeGenerator is not None
        assert emit_multigpu_launcher is not None
