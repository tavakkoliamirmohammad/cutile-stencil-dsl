"""Multi-GPU stencil kernel code generation using cuTile gather().

Generates a cuTile kernel that reads halo data directly from peer GPU
buffers via ``ct.gather()`` instead of explicit halo-exchange copies.
Each GPU receives a list of peer arrays (one per GPU in the symmetric
memory group) and determines at runtime which peer to read from based
on the global position of each tile.

This module generates Python *source code* as strings. It does NOT
import torch, cupy, or cuda.tile at module level.
"""

from __future__ import annotations

from typing import Optional

from cutile_stencil.codegen.emitter import CodeEmitter
from cutile_stencil.codegen.stencil_codegen import StencilCodeGenerator
from cutile_stencil.dsl.types import StencilSpec, TileConfig
from cutile_stencil.multigpu.decomposition import DomainDecomposition


class MultiGPUStencilCodeGenerator:
    """Generates a cuTile kernel that uses gather() for halo exchange.

    The generated kernel:

    1. Takes a **list of peer arrays** (one per GPU, from symmetric memory).
    2. Determines which peer to read from based on global block position.
    3. Uses ``ct.load()`` for interior tiles and ``ct.gather()`` for halo
       tiles that cross partition boundaries.
    4. Applies the stencil expression.
    5. Stores the result to the local output array.

    Parameters
    ----------
    spec : StencilSpec
        Stencil specification with accesses and halo widths populated.
    tile_config : TileConfig
        Tile sizes and halo widths for the kernel.
    decomposition : DomainDecomposition
        Domain decomposition describing how the domain is split.
    """

    def __init__(
        self,
        spec: StencilSpec,
        tile_config: TileConfig,
        decomposition: DomainDecomposition,
    ):
        self.spec = spec
        self.tile = tile_config
        self.decomp = decomposition

    def emit(self) -> str:
        """Generate the complete multi-GPU kernel + launcher as Python source."""
        ndim = self.spec.ndim
        if ndim == 1:
            return self._emit_1d()
        elif ndim == 2:
            return self._emit_2d()
        else:
            return self._emit_3d()

    # ------------------------------------------------------------------
    # 1D kernel
    # ------------------------------------------------------------------
    def _emit_1d(self) -> str:
        e = CodeEmitter()
        spec = self.spec
        tile_size = self.tile.tile_sizes[0]
        halo = self.tile.halo_widths[0]
        split_axis = self.decomp.split_axis

        self._emit_header(e)
        e.blank()
        e.line("ConstInt = ct.Constant[int]")
        e.blank()

        # --- Kernel ---
        e.line("@ct.kernel")
        e.line(
            f"def {spec.name}_multigpu_kernel("
            f"peer_arrays, output, "
            f"TILE: ConstInt, HALO: ConstInt, "
            f"LOCAL_N: ConstInt, RANK: ConstInt, WORLD_SIZE: ConstInt):"
        )
        with e.indent():
            e.line("pid = ct.bid(0)")
            e.blank()

            # Global position of this tile's start within the full domain
            e.line("# Global position of this tile in the full domain")
            e.line("local_start = pid * TILE")
            e.line("global_start = RANK * LOCAL_N + local_start")
            e.blank()

            # Interior: load from own array (which has HALO padding on each side)
            e.line("# Load center tile from own padded array")
            e.line("my_array = peer_arrays[RANK]")
            e.line(
                "center_view = my_array.slice("
                "axis=0, start=HALO + local_start, stop=HALO + local_start + TILE)"
            )
            e.line(
                "t_center = ct.load(center_view, index=(0,), shape=(TILE,))"
            )
            e.blank()

            # For each stencil access that has a nonzero offset, generate
            # the gather/load logic.
            access_names = StencilCodeGenerator._build_access_names(spec)

            for vname, offsets, arr_name in access_names:
                off = offsets[0]
                if off == 0:
                    # Center access — already loaded as t_center
                    e.line(f"# {vname}: center access (offset 0)")
                    e.line(f"t_{vname} = t_center")
                else:
                    e.line(f"# {vname}: offset {off} along axis 0")
                    self._emit_1d_halo_load(e, vname, off, arr_name)
                e.blank()

            # Stencil expression: reuse the AST transform from single-GPU codegen
            self._emit_stencil_expr(e, access_names)
            e.blank()

            # Store result
            e.line(
                "out_view = output.slice("
                "axis=0, start=HALO + local_start, stop=HALO + local_start + TILE)"
            )
            e.line("ct.store(out_view, index=(0,), tile=result)")

        # --- Launcher ---
        self._emit_launcher_1d(e, tile_size, halo)

        return e.render()

    def _emit_1d_halo_load(
        self, e: CodeEmitter, vname: str, offset: int, arr_name: str
    ):
        """Emit a 1D halo-aware load for a single stencil access.

        For interior tiles the shifted access stays within the local array.
        For boundary tiles that would read past the local domain, we use
        ``ct.gather()`` to read from the neighbor GPU's buffer.
        """
        abs_off = abs(offset)
        sign = "+" if offset > 0 else "-"
        off_expr = f"HALO + local_start {sign} {abs_off}"

        # The shifted position within the local padded array
        e.line(f"shifted_start_{vname} = {off_expr}")
        e.line(
            f"shifted_view_{vname} = my_array.slice("
            f"axis=0, start=shifted_start_{vname}, "
            f"stop=shifted_start_{vname} + TILE)"
        )

        if offset < 0:
            # Left halo: may need data from RANK - 1
            e.line(f"need_left_{vname} = (local_start + HALO + ({offset})) < HALO")
            e.line(f"if need_left_{vname} and RANK > 0:")
            with e.indent():
                e.line(f"left_peer = peer_arrays[RANK - 1]")
                e.line(
                    f"t_{vname} = ct.gather("
                    f"left_peer, index=(LOCAL_N + {off_expr},), shape=(TILE,))"
                )
            e.line("else:")
            with e.indent():
                e.line(
                    f"t_{vname} = ct.load("
                    f"shifted_view_{vname}, index=(0,), shape=(TILE,))"
                )
        else:
            # Right halo: may need data from RANK + 1
            e.line(
                f"need_right_{vname} = "
                f"(local_start + TILE - 1 + {offset}) >= LOCAL_N"
            )
            e.line(f"if need_right_{vname} and RANK < WORLD_SIZE - 1:")
            with e.indent():
                e.line(f"right_peer = peer_arrays[RANK + 1]")
                e.line(
                    f"t_{vname} = ct.gather("
                    f"right_peer, index=({off_expr} - LOCAL_N,), shape=(TILE,))"
                )
            e.line("else:")
            with e.indent():
                e.line(
                    f"t_{vname} = ct.load("
                    f"shifted_view_{vname}, index=(0,), shape=(TILE,))"
                )

    # ------------------------------------------------------------------
    # 2D kernel
    # ------------------------------------------------------------------
    def _emit_2d(self) -> str:
        e = CodeEmitter()
        spec = self.spec
        tx, ty = self.tile.tile_sizes
        hx, hy = self.tile.halo_widths
        split_axis = self.decomp.split_axis

        self._emit_header(e)
        e.blank()
        e.line("ConstInt = ct.Constant[int]")
        e.blank()

        access_names = StencilCodeGenerator._build_access_names(spec)

        e.line("@ct.kernel")
        e.line(
            f"def {spec.name}_multigpu_kernel("
            f"peer_arrays, output, "
            f"TX: ConstInt, TY: ConstInt, "
            f"HX: ConstInt, HY: ConstInt, "
            f"LOCAL_SPLIT: ConstInt, RANK: ConstInt, WORLD_SIZE: ConstInt):"
        )
        with e.indent():
            e.line("bx = ct.bid(0)")
            e.line("by = ct.bid(1)")
            e.blank()

            # Interior sizes from the first input peer array
            first_arr = spec.inputs[0]
            e.line("my_array = peer_arrays[RANK]")
            if split_axis == 0:
                e.line("nx = LOCAL_SPLIT")
                e.line(f"ny = my_array.shape[1] - 2 * HY")
                tile_var_split = "TX"
                bid_split = "bx"
                tile_var_other = "TY"
                bid_other = "by"
            else:
                e.line(f"nx = my_array.shape[0] - 2 * HX")
                e.line("ny = LOCAL_SPLIT")
                tile_var_split = "TY"
                bid_split = "by"
                tile_var_other = "TX"
                bid_other = "bx"

            e.line(f"local_start = {bid_split} * {tile_var_split}")
            e.blank()

            halo_vars = ["HX", "HY"]
            n_vars = ["nx", "ny"]
            bid_vars = ["bx", "by"]
            tile_vars = ["TX", "TY"]

            # Create sliced views for each access
            for vname, offsets, arr_name in access_names:
                self._emit_nd_access(
                    e, vname, offsets, arr_name, 2,
                    halo_vars, n_vars, bid_vars, tile_vars,
                    split_axis,
                )
            e.blank()

            # Load tiles
            for vname, _, _ in access_names:
                e.line(
                    f"t_{vname} = ct.load("
                    f"{vname}_view, index=(0, 0), shape=(TX, TY))"
                )
            e.blank()

            # Stencil expression
            self._emit_stencil_expr(e, access_names)
            e.blank()

            # Store
            out_slices = []
            for d in range(2):
                hv = halo_vars[d]
                nv = n_vars[d]
                bv = bid_vars[d]
                tv = tile_vars[d]
                out_slices.append(
                    f".slice(axis={d}, start={hv} + {bv} * {tv}, "
                    f"stop={hv} + {bv} * {tv} + {tv})"
                )
            out_chain = "output" + "".join(out_slices)
            e.line(f"out_view = {out_chain}")
            e.line("ct.store(out_view, index=(0, 0), tile=result)")

        self._emit_launcher_nd(e, 2)
        return e.render()

    # ------------------------------------------------------------------
    # 3D kernel
    # ------------------------------------------------------------------
    def _emit_3d(self) -> str:
        e = CodeEmitter()
        spec = self.spec
        split_axis = self.decomp.split_axis

        self._emit_header(e)
        e.blank()
        e.line("ConstInt = ct.Constant[int]")
        e.blank()

        access_names = StencilCodeGenerator._build_access_names(spec)

        halo_vars = ["HX", "HY", "HZ"]
        tile_vars = ["TX", "TY", "TZ"]
        bid_vars = ["bx", "by", "bz"]
        n_vars = ["nx", "ny", "nz"]

        halo_params = ", ".join(f"{hv}: ConstInt" for hv in halo_vars)
        tile_params = ", ".join(f"{tv}: ConstInt" for tv in tile_vars)

        e.line("@ct.kernel")
        e.line(
            f"def {spec.name}_multigpu_kernel("
            f"peer_arrays, output, "
            f"{tile_params}, {halo_params}, "
            f"LOCAL_SPLIT: ConstInt, RANK: ConstInt, WORLD_SIZE: ConstInt):"
        )
        with e.indent():
            for i, bv in enumerate(bid_vars):
                e.line(f"{bv} = ct.bid({i})")
            e.blank()

            e.line("my_array = peer_arrays[RANK]")
            for d in range(3):
                if d == split_axis:
                    e.line(f"{n_vars[d]} = LOCAL_SPLIT")
                else:
                    e.line(
                        f"{n_vars[d]} = my_array.shape[{d}] - 2 * {halo_vars[d]}"
                    )
            e.line(
                f"local_start = {bid_vars[split_axis]} * {tile_vars[split_axis]}"
            )
            e.blank()

            for vname, offsets, arr_name in access_names:
                self._emit_nd_access(
                    e, vname, offsets, arr_name, 3,
                    halo_vars, n_vars, bid_vars, tile_vars,
                    split_axis,
                )
            e.blank()

            idx_tuple = ", ".join(["0"] * 3)
            shape_tuple = ", ".join(tile_vars)
            for vname, _, _ in access_names:
                e.line(
                    f"t_{vname} = ct.load("
                    f"{vname}_view, index=({idx_tuple}), shape=({shape_tuple}))"
                )
            e.blank()

            self._emit_stencil_expr(e, access_names)
            e.blank()

            out_slices = []
            for d in range(3):
                hv = halo_vars[d]
                bv = bid_vars[d]
                tv = tile_vars[d]
                out_slices.append(
                    f".slice(axis={d}, start={hv} + {bv} * {tv}, "
                    f"stop={hv} + {bv} * {tv} + {tv})"
                )
            out_chain = "output" + "".join(out_slices)
            e.line(f"out_view = {out_chain}")
            e.line(f"ct.store(out_view, index=({idx_tuple}), tile=result)")

        self._emit_launcher_nd(e, 3)
        return e.render()

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _emit_nd_access(
        self,
        e: CodeEmitter,
        vname: str,
        offsets,
        arr_name: str,
        ndim: int,
        halo_vars,
        n_vars,
        bid_vars,
        tile_vars,
        split_axis: int,
    ):
        """Emit a sliced view for one stencil access in the nD multi-GPU kernel.

        For the split axis, if the offset is nonzero we emit a conditional
        ``ct.gather()`` from the neighbor peer; for non-split axes we emit
        a normal ``.slice()`` as in the single-GPU codegen.
        """
        split_off = offsets[split_axis]

        # Build slice chain for all dimensions
        slice_parts = []
        for d in range(ndim):
            off = offsets[d]
            hv = halo_vars[d]

            if d == split_axis:
                # Use bid-relative slicing on the split axis
                bv = bid_vars[d]
                tv = tile_vars[d]
                start_expr = StencilCodeGenerator._offset_expr(
                    hv, off
                ).replace(hv, f"{hv} + {bv} * {tv}")
                # Simplify: if off == 0, start = HX + bx * TX
                if off == 0:
                    start_expr = f"{hv} + {bv} * {tv}"
                elif off > 0:
                    start_expr = f"{hv} + {off} + {bv} * {tv}"
                else:
                    start_expr = f"{hv} - {abs(off)} + {bv} * {tv}"
                stop_expr = f"{start_expr} + {tv}"
                slice_parts.append(
                    f".slice(axis={d}, start={start_expr}, stop={stop_expr})"
                )
            else:
                # Non-split axis: standard slice covering full interior
                start = StencilCodeGenerator._offset_expr(hv, off)
                stop = StencilCodeGenerator._offset_expr(
                    hv, off, add_n=True, n_var=n_vars[d]
                )
                slice_parts.append(
                    f".slice(axis={d}, start={start}, stop={stop})"
                )

        # For the split axis with nonzero offset, add gather logic
        if split_off != 0:
            bv = bid_vars[split_axis]
            tv = tile_vars[split_axis]
            hv = halo_vars[split_axis]

            e.line(f"# {vname}: offset {split_off} on split axis {split_axis}")
            if split_off < 0:
                e.line(
                    f"if {bv} == 0 and RANK > 0:"
                )
                with e.indent():
                    # Build gather view from left neighbor
                    gather_parts = []
                    for d in range(ndim):
                        if d == split_axis:
                            # Read from end of left neighbor's padded array
                            off = offsets[d]
                            start_g = f"LOCAL_SPLIT + {hv} + ({off})"
                            stop_g = f"{start_g} + {tv}"
                            gather_parts.append(
                                f".slice(axis={d}, start={start_g}, stop={stop_g})"
                            )
                        else:
                            gather_parts.append(slice_parts[d])
                    gather_chain = "peer_arrays[RANK - 1]" + "".join(gather_parts)
                    idx = ", ".join(["0"] * ndim)
                    shape = ", ".join(
                        tv if d == split_axis else n_vars[d]
                        for d in range(ndim)
                    )
                    e.line(f"{vname}_view = {gather_chain}")
                e.line("else:")
                with e.indent():
                    normal_chain = "my_array" + "".join(slice_parts)
                    e.line(f"{vname}_view = {normal_chain}")
            else:
                num_tiles = f"ct.cdiv(LOCAL_SPLIT, {tv})"
                e.line(
                    f"if {bv} == {num_tiles} - 1 and RANK < WORLD_SIZE - 1:"
                )
                with e.indent():
                    gather_parts = []
                    for d in range(ndim):
                        if d == split_axis:
                            off = offsets[d]
                            # Read from start of right neighbor's padded array
                            start_g = f"{hv} + ({off}) - {tv}"
                            stop_g = f"{start_g} + {tv}"
                            gather_parts.append(
                                f".slice(axis={d}, start={start_g}, stop={stop_g})"
                            )
                        else:
                            gather_parts.append(slice_parts[d])
                    gather_chain = "peer_arrays[RANK + 1]" + "".join(gather_parts)
                    e.line(f"{vname}_view = {gather_chain}")
                e.line("else:")
                with e.indent():
                    normal_chain = "my_array" + "".join(slice_parts)
                    e.line(f"{vname}_view = {normal_chain}")
        else:
            # No offset on split axis -- always read from own array
            normal_chain = "my_array" + "".join(slice_parts)
            e.line(f"{vname}_view = {normal_chain}")

    def _emit_stencil_expr(self, e: CodeEmitter, access_names):
        """Emit the stencil arithmetic expression via AST transform."""
        spec = self.spec
        offset_to_tile = {}
        for vname, offsets, arr_name in access_names:
            offset_to_tile[(arr_name, offsets)] = f"t_{vname}"

        term_names = []
        for acc in spec.accesses:
            key = (acc.array_name, acc.offsets)
            term_names.append(offset_to_tile.get(key, "t_unknown"))

        try:
            from cutile_stencil.codegen.ast_transform import transform_stencil_expr
            expr_src = transform_stencil_expr(
                spec.update_fn, spec.accesses, spec.ndim, term_names
            )
            e.line(f"result = {expr_src}")
        except Exception:
            terms = [f"t_{vn}" for vn, _, _ in access_names]
            e.line(f"result = {' + '.join(terms)}")

    def _emit_launcher_1d(self, e: CodeEmitter, tile_size: int, halo: int):
        """Emit 1D multi-GPU launcher."""
        spec = self.spec
        e.blank()
        e.blank()
        e.line(
            f"def launch_{spec.name}_multigpu("
            f"peer_arrays, output, rank, world_size, stream=None):"
        )
        with e.indent():
            e.line(f"TILE = {tile_size}")
            e.line(f"HALO = {halo}")
            e.line("my_array = peer_arrays[rank]")
            e.line("N_padded = my_array.shape[0]")
            e.line("LOCAL_N = N_padded - 2 * HALO")
            e.line("if stream is None:")
            with e.indent():
                e.line("stream = cp.cuda.get_current_stream()")
            e.line("grid = (ct.cdiv(LOCAL_N, TILE),)")
            e.line(
                f"ct.launch(stream, grid, {spec.name}_multigpu_kernel, "
                f"(peer_arrays, output, TILE, HALO, LOCAL_N, rank, world_size))"
            )

    def _emit_launcher_nd(self, e: CodeEmitter, ndim: int):
        """Emit nD multi-GPU launcher."""
        spec = self.spec
        split_axis = self.decomp.split_axis
        tile_vars = ["TX", "TY", "TZ"][:ndim]
        halo_vars = ["HX", "HY", "HZ"][:ndim]
        tile_sizes = self.tile.tile_sizes
        halo_widths = self.tile.halo_widths

        e.blank()
        e.blank()
        e.line(
            f"def launch_{spec.name}_multigpu("
            f"peer_arrays, output, rank, world_size, stream=None):"
        )
        with e.indent():
            e.line(
                f"{', '.join(tile_vars)} = "
                f"{', '.join(str(t) for t in tile_sizes)}"
            )
            e.line(
                f"{', '.join(halo_vars)} = "
                f"{', '.join(str(h) for h in halo_widths)}"
            )
            e.line("my_array = peer_arrays[rank]")
            # Compute grid dims
            for d in range(ndim):
                if d == split_axis:
                    e.line(
                        f"LOCAL_SPLIT = my_array.shape[{d}] - 2 * {halo_vars[d]}"
                    )
                    e.line(
                        f"grid_{d} = ct.cdiv(LOCAL_SPLIT, {tile_vars[d]})"
                    )
                else:
                    e.line(
                        f"n_{d} = my_array.shape[{d}] - 2 * {halo_vars[d]}"
                    )
                    e.line(f"grid_{d} = ct.cdiv(n_{d}, {tile_vars[d]})")

            e.line("if stream is None:")
            with e.indent():
                e.line("stream = cp.cuda.get_current_stream()")

            grid_parts = ", ".join(f"grid_{d}" for d in range(ndim))
            e.line(f"grid = ({grid_parts})")

            t_args = ", ".join(tile_vars)
            h_args = ", ".join(halo_vars)
            e.line(
                f"ct.launch(stream, grid, {spec.name}_multigpu_kernel, "
                f"(peer_arrays, output, {t_args}, {h_args}, "
                f"LOCAL_SPLIT, rank, world_size))"
            )

    def _emit_header(self, e: CodeEmitter):
        """Emit file header with imports."""
        e.line(
            f'"""cuTile multi-GPU kernel for {self.spec.name} '
            f'stencil (auto-generated)."""'
        )
        e.blank()
        e.line("import cuda.tile as ct")
        e.line("import cupy as cp")

        # Closure constants
        consts = self.spec.closure_constants
        if consts:
            e.blank()
            e.line("# Captured constants from stencil definition scope")
            for name, value in sorted(consts.items()):
                e.line(f"{name} = {value!r}")


# ======================================================================
# Approach A: P2P halo exchange launcher (kernel unchanged from single-GPU)
# ======================================================================


def emit_multigpu_p2p_launcher(
    spec: StencilSpec,
    decomposition: DomainDecomposition,
) -> str:
    """Generate a multi-GPU launcher with P2P halo exchange.

    The kernel is the **standard single-GPU kernel** -- no modifications
    needed.  Halo exchange happens before each kernel launch via CuPy
    device-to-device P2P copies.

    The generated module provides:

    - ``setup_p2p(num_gpus)`` -- enable P2P access between all GPU pairs.
    - ``exchange_halos_p2p(u_gpus, halo, split_axis)`` -- copy halo cells
      from neighbor GPUs into each GPU's local padded array.
    - ``run_stencil_step(u_gpus, out_gpus, launch_fn, halo, split_axis)``
      -- one full timestep: exchange halos then launch the kernel on
      every GPU.
    - ``run_multigpu_p2p(launch_fn, global_input, num_gpus, halo,
      split_axis, dtype, n_steps)`` -- end-to-end driver that decomposes,
      exchanges, launches, and gathers the result.

    Parameters
    ----------
    spec : StencilSpec
        Stencil specification.
    decomposition : DomainDecomposition
        Domain decomposition describing how the domain is split.

    Returns
    -------
    str
        Python source code for the P2P launcher module.
    """
    e = CodeEmitter()
    ndim = spec.ndim
    split_axis = decomposition.split_axis
    halo_widths = decomposition.halo_widths

    # --- Header ---
    e.line(
        f'"""P2P halo-exchange launcher for {spec.name} '
        f'stencil (auto-generated).'
    )
    e.blank()
    e.line(
        "Uses CuPy P2P device-to-device copies for halo exchange, then"
    )
    e.line("launches the standard single-GPU cuTile kernel on each GPU.")
    e.line('"""')
    e.blank()
    e.line("import cupy as cp")
    e.line("import numpy as np")
    e.blank()

    # Closure constants
    consts = spec.closure_constants
    if consts:
        e.line("# Captured constants from stencil definition scope")
        for name, value in sorted(consts.items()):
            e.line(f"{name} = {value!r}")
        e.blank()

    # --- setup_p2p ---
    e.line("def setup_p2p(num_gpus):")
    with e.indent():
        e.line('"""Enable P2P access between all GPU pairs."""')
        e.line("for i in range(num_gpus):")
        with e.indent():
            e.line("cp.cuda.Device(i).use()")
            e.line("for j in range(num_gpus):")
            with e.indent():
                e.line("if i != j:")
                with e.indent():
                    e.line("try:")
                    with e.indent():
                        e.line("cp.cuda.runtime.deviceEnablePeerAccess(j)")
                    e.line("except cp.cuda.runtime.CUDARuntimeError:")
                    with e.indent():
                        e.line("pass")
    e.blank()
    e.blank()

    # --- exchange_halos_p2p ---
    # Generate for all ndim, but only exchange along split_axis
    _emit_exchange_halos_p2p(e, ndim, split_axis, halo_widths)
    e.blank()
    e.blank()

    # --- run_stencil_step ---
    e.line(
        "def run_stencil_step(u_gpus, out_gpus, launch_fn, "
        "halo, split_axis=0):"
    )
    with e.indent():
        e.line('"""One timestep: exchange halos, then launch on all GPUs."""')
        e.line("n_gpus = len(u_gpus)")
        e.line("exchange_halos_p2p(u_gpus, halo, split_axis)")
        e.line("for rank in range(n_gpus):")
        with e.indent():
            e.line("cp.cuda.Device(rank).use()")
            e.line("launch_fn(u_gpus[rank], out_gpus[rank])")
        e.line("for rank in range(n_gpus):")
        with e.indent():
            e.line("cp.cuda.Device(rank).synchronize()")
    e.blank()
    e.blank()

    # --- run_multigpu_p2p ---
    split_halo = halo_widths[split_axis]
    e.line(
        f"def run_multigpu_p2p(launch_fn, global_input, "
        f"num_gpus, n_steps=1, "
        f"split_axis={split_axis}, halo={split_halo}):"
    )
    with e.indent():
        e.line(
            f'"""Run {spec.name} stencil across GPUs with P2P halo exchange.'
        )
        e.blank()
        e.line("Parameters")
        e.line("----------")
        e.line("launch_fn : callable")
        e.line("    The compiled single-GPU kernel launch function.")
        e.line("global_input : numpy.ndarray")
        e.line("    Full-domain padded input array (including global halo).")
        e.line("num_gpus : int")
        e.line("    Number of GPUs to use.")
        e.line("n_steps : int")
        e.line("    Number of stencil iterations.")
        e.line("split_axis : int")
        e.line("    Axis along which the domain is decomposed.")
        e.line("halo : int")
        e.line("    Halo width along the split axis.")
        e.blank()
        e.line("Returns")
        e.line("-------")
        e.line("numpy.ndarray")
        e.line("    Gathered result as a numpy array (same shape as interior).")
        e.line('"""')
        e.line("setup_p2p(num_gpus)")
        e.blank()

        e.line("# Compute local extent along split axis")
        e.line(f"global_interior = global_input.shape[split_axis] - 2 * halo")
        e.line("assert global_interior % num_gpus == 0, (")
        with e.indent():
            e.line(
                "f'Interior size {global_interior} not divisible "
                "by {num_gpus}'"
            )
        e.line(")")
        e.line("local_n = global_interior // num_gpus")
        e.blank()

        e.line("# Decompose global input into per-GPU padded sub-arrays")
        e.line("u_gpus = []")
        e.line("out_gpus = []")
        e.line("for rank in range(num_gpus):")
        with e.indent():
            e.line("cp.cuda.Device(rank).use()")
            e.line("start = rank * local_n")
            e.line("# Extract local_n + 2*halo slice from global array")
            if ndim == 1:
                e.line(
                    "chunk = global_input[start : start + local_n + 2 * halo]"
                )
            else:
                slices = []
                for d in range(ndim):
                    if d == split_axis:
                        slices.append("start : start + local_n + 2 * halo")
                    else:
                        slices.append(":")
                e.line(f"chunk = global_input[{', '.join(slices)}]")
            e.line("u_gpus.append(cp.asarray(chunk))")
            e.line("out_gpus.append(cp.zeros_like(u_gpus[-1]))")
        e.blank()

        e.line("# Run stencil iterations")
        e.line("for step in range(n_steps):")
        with e.indent():
            e.line("run_stencil_step(u_gpus, out_gpus, launch_fn, halo, split_axis)")
            e.line("# Swap input/output for next step")
            e.line("for rank in range(num_gpus):")
            with e.indent():
                e.line("cp.cuda.Device(rank).use()")
                # Copy interior from output back to input
                if ndim == 1:
                    e.line("u_gpus[rank][halo:-halo] = out_gpus[rank][halo:-halo]")
                else:
                    interior_slices = []
                    for d in range(ndim):
                        if d == split_axis:
                            interior_slices.append("halo:-halo")
                        else:
                            interior_slices.append(":")
                    sl = ", ".join(interior_slices)
                    e.line(f"u_gpus[rank][{sl}] = out_gpus[rank][{sl}]")
        e.blank()

        e.line("# Gather results from all GPUs")
        e.line("parts = []")
        e.line("for rank in range(num_gpus):")
        with e.indent():
            e.line("cp.cuda.Device(rank).use()")
            if ndim == 1:
                e.line("parts.append(cp.asnumpy(u_gpus[rank][halo:-halo]))")
            else:
                interior_slices = []
                for d in range(ndim):
                    if d == split_axis:
                        interior_slices.append("halo:-halo")
                    else:
                        halo_d = halo_widths[d]
                        interior_slices.append(f"{halo_d}:-{halo_d}")
                sl = ", ".join(interior_slices)
                e.line(f"parts.append(cp.asnumpy(u_gpus[rank][{sl}]))")
        e.line(f"return np.concatenate(parts, axis={split_axis})")

    return e.render()


def _emit_exchange_halos_p2p(
    e: CodeEmitter,
    ndim: int,
    split_axis: int,
    halo_widths: tuple,
):
    """Emit the exchange_halos_p2p function.

    Copies halo cells from neighbor GPUs along the split axis using
    CuPy's ``MemoryPointer.copy_from_device()`` for P2P transfer.
    """
    split_halo = halo_widths[split_axis]

    e.line(f"def exchange_halos_p2p(u_gpus, halo={split_halo}, split_axis={split_axis}):")
    with e.indent():
        e.line('"""Exchange halo cells between neighbor GPUs via P2P copy."""')
        e.line("n_gpus = len(u_gpus)")
        e.line("dtype_bytes = u_gpus[0].dtype.itemsize")
        e.blank()

        e.line("for rank in range(n_gpus):")
        with e.indent():
            e.line("cp.cuda.Device(rank).use()")
            e.blank()

            # Build the slicing expressions based on ndim and split_axis.
            # For N-D arrays, we need to compute contiguous byte sizes
            # and slice along the split axis.
            if ndim == 1:
                # 1D: simple direct copy
                e.line("nbytes = halo * dtype_bytes")
                e.blank()
                e.line("# Copy from left neighbor's right interior")
                e.line("if rank > 0:")
                with e.indent():
                    e.line(
                        "u_gpus[rank][:halo].data.copy_from_device("
                    )
                    with e.indent():
                        e.line(
                            "u_gpus[rank - 1][-2 * halo : -halo].data, nbytes"
                        )
                    e.line(")")
                e.blank()
                e.line("# Copy from right neighbor's left interior")
                e.line("if rank < n_gpus - 1:")
                with e.indent():
                    e.line(
                        "u_gpus[rank][-halo:].data.copy_from_device("
                    )
                    with e.indent():
                        e.line(
                            "u_gpus[rank + 1][halo : 2 * halo].data, nbytes"
                        )
                    e.line(")")
            else:
                # N-D: use slicing and contiguous copy.
                # We need contiguous slabs along the split axis.
                e.line("local_shape = u_gpus[rank].shape")
                e.line(
                    "# Number of elements in one halo slab "
                    "along the split axis"
                )
                # Compute slab_size = product of all dims except split * halo
                slab_parts = []
                for d in range(ndim):
                    if d != split_axis:
                        slab_parts.append(f"local_shape[{d}]")
                if slab_parts:
                    e.line(
                        f"slab_elements = halo * "
                        f"{' * '.join(slab_parts)}"
                    )
                else:
                    e.line("slab_elements = halo")
                e.line("slab_bytes = slab_elements * dtype_bytes")
                e.blank()

                # Build slice tuples for source and destination
                def _make_slab_slice(axis_slice, as_str=True):
                    """Create a tuple of slice strings for an N-D array."""
                    parts = []
                    for d in range(ndim):
                        if d == split_axis:
                            parts.append(axis_slice)
                        else:
                            parts.append(":")
                    return ", ".join(parts)

                e.line("# Copy from left neighbor's right interior")
                e.line("if rank > 0:")
                with e.indent():
                    dst_sl = _make_slab_slice(":halo")
                    src_sl = _make_slab_slice("-2 * halo : -halo")
                    e.line(
                        f"dst = u_gpus[rank][{dst_sl}]"
                    )
                    e.line(
                        f"src = u_gpus[rank - 1][{src_sl}]"
                    )
                    e.line(
                        "dst.data.copy_from_device("
                        "src.data, slab_bytes)"
                    )
                e.blank()

                e.line("# Copy from right neighbor's left interior")
                e.line("if rank < n_gpus - 1:")
                with e.indent():
                    dst_sl = _make_slab_slice("-halo:")
                    src_sl = _make_slab_slice("halo : 2 * halo")
                    e.line(
                        f"dst = u_gpus[rank][{dst_sl}]"
                    )
                    e.line(
                        f"src = u_gpus[rank + 1][{src_sl}]"
                    )
                    e.line(
                        "dst.data.copy_from_device("
                        "src.data, slab_bytes)"
                    )


# ======================================================================
# Approach B: ct.gather-based in-kernel halo loading (convenience wrapper)
# ======================================================================


def emit_multigpu_gather_kernel(
    spec: StencilSpec,
    tile_config: TileConfig,
    decomposition: DomainDecomposition,
) -> str:
    """Generate a kernel that loads halo directly from peer GPU memory.

    The kernel receives a list of per-GPU arrays. For tiles at partition
    boundaries, it reads from the neighbor's array using ``ct.gather()``
    / ``ct.load()`` with a peer-array view.  No pre-kernel copy needed.

    This is a convenience wrapper around
    :class:`MultiGPUStencilCodeGenerator`.

    Parameters
    ----------
    spec : StencilSpec
        Stencil specification with accesses and halo widths populated.
    tile_config : TileConfig
        Tile sizes and halo widths for the kernel.
    decomposition : DomainDecomposition
        Domain decomposition describing how the domain is split.

    Returns
    -------
    str
        Python source code containing the cuTile kernel + launcher that
        does in-kernel halo reads from peer arrays via ``ct.gather()``.
    """
    gen = MultiGPUStencilCodeGenerator(spec, tile_config, decomposition)
    return gen.emit()
