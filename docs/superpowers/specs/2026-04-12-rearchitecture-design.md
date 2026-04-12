# cuTile Stencil DSL Re-Architecture Design

## Goal

Re-architect the cuTile Stencil DSL into a publication-quality stencil compiler with a clean xDSL-based IR, composable optimization passes, principled multi-GPU distribution, and competitive performance against Devito and PyStencils. The system targets GPU stencil computations via NVIDIA's cuTile (`cuda.tile`) backend.

## Publication Targets

The paper targets three audiences simultaneously:

- **Systems (SC, PPoPP, ICS):** Performance that beats Devito/PyStencils on GPU stencils, multi-GPU scaling results
- **DSL/Compiler (PLDI, CGO):** Clean compiler architecture with xDSL-based IR, composable pass pipeline
- **Domain (SIAM, JCP):** Framework for GPU PDE solving with minimal boilerplate, real applications (heat, wave, reaction-diffusion, Maxwell, shallow water)

## Current State (Problems)

The existing codebase has these structural issues that make it unpublishable:

1. **No intermediate representation.** The system goes directly from Python AST to string-based code generation. There is no structured representation of a stencil kernel that can be reasoned about, transformed, or targeted.

2. **Massive code duplication.** `multigpu/codegen.py` (1,006 lines) is largely copy-pasted from `stencil_codegen.py` (664 lines). Variable naming lists (`["bx","by","bz"][:ndim]`), slice building logic, and stencil expression emission appear 5+ times across files. Adding a 4D stencil requires edits in 8+ locations.

3. **Fragile string-based codegen.** Kernel code is built via string concatenation through `CodeEmitter`. No syntax validation beyond post-hoc `ast.parse()`. Import stripping uses string matching (`if stripped.startswith("import")`).

4. **Mutable shared state.** `StencilSpec` gets mutated during analysis (`spec.accesses`, `spec.halo_widths` set as side effects in different modules). No guarantee analysis ran before codegen.

5. **Feature paths don't compose.** Temporal blocking, bricked layout, multi-GPU, and boundary conditions each have separate code paths in the codegen. Combining temporal + multi-GPU requires duplicating temporal logic in the multi-GPU codegen.

6. **Tests validate syntax, not correctness.** ~48 codegen tests just call `ast.parse(code)`. Most never run the generated GPU kernel or verify numerical results.

7. **No competitive benchmarks.** Only NumPy (CPU) comparisons. No benchmarks against Devito, PyStencils, or hand-tuned GPU code.

---

## Architecture

### Four-Layer Design

```
Layer 1: FRONTEND          @stencil decorator -> xDSL stencil dialect IR
Layer 2: PASSES            Analysis + optimization + distribution passes on IR
Layer 3: LOWERING          Stencil IR -> cuTile dialect -> Python source
Layer 4: RUNTIME           Kernel launching, communication, autotuning
```

Each layer depends only on the one above it. Passes don't know about cuTile. The emitter doesn't know about stencil semantics. The frontend doesn't know about tiling.

### End-to-End Data Flow (Three-Dialect Stack)

The system uses three dialect levels, each with a simple, well-defined conversion:

```
User writes:
    @stencil(ndim=2, order=2, boundary=BoundarySpec.periodic())
    def heat(u, i, j):
        return 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])

         | Frontend: trivial 1:1 Python AST -> our dialect

Dialect 1 - cuTile Stencil Dialect (our high-level dialect):
    cutile_stencil.func @heat(
        %u: field<?x?xf64>,
        ndim=2, order=2,
        boundary=periodic,
        constants={}) {
        %left  = cutile_stencil.access %u [i-1, j]
        %right = cutile_stencil.access %u [i+1, j]
        %up    = cutile_stencil.access %u [i, j-1]
        %down  = cutile_stencil.access %u [i, j+1]
        cutile_stencil.compute {
            return 0.25 * (%left + %right + %up + %down)
        }
    }
    - Directly mirrors Python syntax (named params, boundary spec, closure constants)
    - Symbolic shapes (? dimensions) supported by default

         | xDSL pass: normalize our dialect -> standard xDSL stencil dialect

Dialect 2 - xDSL Stencil Dialect (standard MLIR stencil):
    builtin.module {
        func.func @heat(%u: !stencil.field<?x?xf64>)
                        -> !stencil.field<?x?xf64> {
            %out = stencil.apply(%u_val = %u) {
                %left  = stencil.access %u_val [-1, 0] : f64
                %right = stencil.access %u_val [+1, 0] : f64
                %up    = stencil.access %u_val [0, -1] : f64
                %down  = stencil.access %u_val [0, +1] : f64
                %sum   = arith.addf %left, %right : f64
                %sum2  = arith.addf %sum, %up : f64
                %sum3  = arith.addf %sum2, %down : f64
                %coeff = arith.constant 0.25 : f64
                %res   = arith.mulf %sum3, %coeff : f64
                stencil.return %res : f64
            }
            return %out
        }
    }
    - Normalized, canonical SSA form
    - ALL optimization passes operate on this dialect

         | Analysis passes (footprint, halo, roofline)
         | Optimization passes (tiling, temporal blocking, bricked layout)
         | Multi-GPU passes (decomposition, halo exchange insertion)

         | xDSL pass: lower to cuTile target dialect

Dialect 3 - cuTile Target Dialect (device + host IR):
    cutile.kernel { tile_shape=[TX,TY], halo=[HX,HY] }
        cutile.slice(axis=0, start=..., stop=...)
        cutile.load(...)
        cutile.store(...)
    cutile.host_program {
        cutile.alloc_buffers(...)
        cutile.stream()
        host.for_loop(T) {
            cutile.launch(kernel, grid, args)
            boundary.apply(...)
            cutile.swap_buffers(...)
        }
    }

         | Emission: walk cuTile target ops -> Python source string

    @ct.kernel
    def heat_kernel(u_in, u_out, TX, TY, HX, HY, nx, ny):
        bx, by = ct.bid(0), ct.bid(1)
        inp = u_in.slice(axis=0, ...).slice(axis=1, ...)
        ...
        ct.store(out_tile, result)

    def launch_heat(u_in, u_out):
        nx, ny = u_in.shape[0] - 2*HX, u_in.shape[1] - 2*HY
        grid = (ct.cdiv(nx, TX), ct.cdiv(ny, TY))
        ...
```

Each conversion step is simple:
- **Python -> Dialect 1:** Trivial, almost 1:1 mapping from Python AST to our dialect ops
- **Dialect 1 -> Dialect 2:** xDSL pass that normalizes to canonical stencil form (SSA, offset tuples, arith ops)
- **Dialect 2 -> Dialect 3:** xDSL lowering pass after all optimizations are applied
- **Dialect 3 -> Python source:** Mechanical string emission from structured IR ops

### Module Structure

```
cutile/
|-- frontend/               # Layer 1: @stencil -> IR
|   |-- decorator.py        # @stencil decorator, auto-inference of ndim/order
|   |-- parser.py           # Python AST -> xDSL stencil dialect operations
|   +-- types.py            # BoundarySpec, HardwareSpec, GPU presets, configs
|
|-- dialects/               # xDSL dialect definitions (three-dialect stack + supporting dialects)
|   |-- cutile_stencil/     # Dialect 1: our high-level stencil dialect (mirrors Python syntax)
|   |-- cutile_target/      # Dialect 3: cuTile target dialect (kernel, slice, load, store, launch, host program)
|   |-- comm/               # Communication dialect (halo send/recv, barrier - location-agnostic)
|   |-- timestep/           # Time integration dialect (RK stages, combine, Euler/RK2/RK4/SSPRK3)
|   +-- layout/             # Data layout types and conversion ops (row_major, bricked)
|
|-- passes/                 # Layer 2: IR transformations
|   |-- analysis/           # Read-only passes
|   |   |-- footprint.py    # Extract access patterns, compute halo widths
|   |   +-- roofline.py     # FLOP count, arithmetic intensity, bound classification
|   |-- tiling.py           # Tile size selection (shared mem budget, overhead minimization)
|   |-- temporal.py         # Temporal blocking (fuse time steps, expand halos, buffer chain)
|   |-- fusion.py           # Fuse multi-field stencils sharing inputs (optional)
|   |-- bricked.py          # Bricked data layout (type-level layout change + conversion ops)
|   |-- boundary.py         # Boundary condition insertion (periodic, Neumann, Dirichlet, reflecting)
|   |-- decompose.py        # Domain decomposition with interior/boundary separation (multi-GPU)
|   |-- halo.py             # Halo exchange insertion with computation-communication overlap
|   +-- verify.py           # IR well-formedness verification (runs after any pass)
|
|-- lowering/               # Layer 3: IR -> code
|   |-- normalize.py          # Dialect 1 (cutile_stencil) -> Dialect 2 (xDSL stencil)
|   |-- stencil_to_cutile.py  # Dialect 2 (xDSL stencil) -> Dialect 3 (cutile_target, device + host IR)
|   +-- emitter.py            # Dialect 3 (cutile_target) -> Python source text
|
|-- runtime/                # Layer 4: Execution
|   |-- launcher.py         # Compile + launch generated kernels (importlib, CuPy)
|   |-- communicator.py     # Abstract communication protocol
|   |-- p2p.py              # CuPy P2P via NVLink (single-node)
|   |-- nccl.py             # NCCL + mpi4py (multi-node)
|   +-- autotune.py         # Model-guided autotuning with empirical validation
|
|-- reference/              # CPU reference for testing
|   +-- stencil_ref.py      # NumPy stencil executor (_ArrayProxy pattern)
|
+-- config.py               # GPU presets (H100, A100, RTX PRO 6000, etc.), auto-detection
```

---

## Layer 1: Frontend

### Decorator

The `@stencil` decorator is the user-facing entry point:

```python
@stencil(ndim=2, order=2, boundary=BoundarySpec.periodic())
def heat(u, i, j):
    return 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])
```

The decorator performs two steps:

1. **Parse** the function's AST and build our **cuTile Stencil Dialect** (Dialect 1) IR. This is a trivial, almost 1:1 mapping from Python syntax to IR ops. Named parameters, boundary specs, closure constants, and array accesses all have direct Dialect 1 counterparts.
2. **Preserve** the raw Python function alongside the IR for CPU reference execution in tests.

The decorator does NOT build xDSL's stencil dialect directly. The conversion from Dialect 1 to Dialect 2 (xDSL stencil) is a separate xDSL pass (`normalize.py`), keeping each step simple.

### Auto-inference

The current auto-inference of `ndim` (from subscript dimensions) and `order` (from max offset magnitudes) is retained. The parser walks the AST once to determine both, extract all offset patterns, capture closure constants, and build the Dialect 1 IR.

### Symbolic Shapes

The frontend produces IR with **symbolic (dynamic) dimensions** by default:

```
cutile_stencil.func @heat(%u: field<?x?xf64>, ...)
```

Domain sizes are not baked into the IR. Tile sizes and halo widths are compile-time constants determined by the stencil shape and hardware constraints. Grid dimensions are computed at runtime from the actual array shape. This means a kernel compiled once can run on any domain size.

When a concrete domain is provided to `compile(heat, domain=(256, 256))`, it is used for autotuning and validation but does not change the generated kernel code.

### Immutability

The decorator produces an **immutable xDSL module**. Once built, the IR is the source of truth. There is no mutable `StencilSpec` being modified in multiple places. Halo widths, tile sizes, and temporal steps are computed by passes and stored as attributes on IR operations.

### Multi-input stencils

For stencils with multiple input arrays (Gray-Scott with `u` and `v`, shallow water with `h`, `hu`, `hv`), each array becomes a separate field operand in Dialect 1. The normalization pass maps these to multiple operands in `stencil.apply`. xDSL's stencil dialect natively supports multiple input fields.

### Dialect 1 IR Example

For the 2D heat stencil, the frontend produces our high-level dialect:

```
cutile_stencil.func @heat(
    %u: field<?x?xf64>,
    ndim=2, order=2,
    boundary=periodic,
    constants={}) {
    %left  = cutile_stencil.access %u [i-1, j]
    %right = cutile_stencil.access %u [i+1, j]
    %up    = cutile_stencil.access %u [i, j-1]
    %down  = cutile_stencil.access %u [i, j+1]
    cutile_stencil.compute {
        return 0.25 * (%left + %right + %up + %down)
    }
}
```

After the normalization pass (`normalize.py`), this becomes standard xDSL stencil dialect:

```
builtin.module {
    func.func @heat(%u: !stencil.field<?x?xf64>)
                    -> !stencil.field<?x?xf64> {
        %out = stencil.apply(%u_val = %u) {
            %left  = stencil.access %u_val [-1, 0] : f64
            %right = stencil.access %u_val [+1, 0] : f64
            %up    = stencil.access %u_val [0, -1] : f64
            %down  = stencil.access %u_val [0, +1] : f64
            %sum   = arith.addf %left, %right : f64
            %sum2  = arith.addf %sum, %up : f64
            %sum3  = arith.addf %sum2, %down : f64
            %coeff = arith.constant 0.25 : f64
            %res   = arith.mulf %sum3, %coeff : f64
            stencil.return %res : f64
        }
        return %out
    }
}
```

---

## Layer 2: Passes

Each pass is an xDSL `RewritePattern` or `ModulePass` that transforms the IR. Passes are composable, independently testable, and can be enabled or disabled.

### Pipeline Architecture

The pass pipeline is exposed as an xDSL `PassManager` that users can configure. Instead of a hardcoded pass order, passes are composable building blocks:

```python
pipeline = Pipeline(hw=HardwareSpec.auto_detect())
pipeline.add(AnalysisPass())
pipeline.add(RooflinePass())
pipeline.add(TilingPass())
pipeline.add(BoundaryPass())
pipeline.add(TemporalBlockingPass(max_T=4))
pipeline.add(FusionPass())                          # fuse multi-field stencils
# pipeline.add(BrickedLayoutPass())                 # optional
pipeline.add(DecompositionPass(num_gpus=4, topology=(2,2)))
pipeline.add(HaloExchangePass(overlap=True))
pipeline.add(VerifyPass())                          # check IR well-formedness
result = pipeline.run(stencil_ir)
```

Default pipelines are provided for common configurations:
- `Pipeline.single_gpu(hw)` — passes 1-5
- `Pipeline.multi_gpu(hw, num_gpus, topology)` — passes 1-9
- `Pipeline.full(hw, num_gpus, topology)` — all passes including fusion and bricked layout

### Pass Pipeline (default order)

```
 1. AnalysisPass            Read-only: extract footprint, compute halo widths
 2. RooflinePass            Read-only: count FLOPs, classify memory/compute bound
 3. TilingPass              Attach tile sizes based on hardware + halo + shared mem
 4. BoundaryPass            Insert boundary condition ops into IR
 5. TemporalBlockingPass    Fuse N time steps, expand halos, insert buffer chain
 6. FusionPass              Fuse multi-field stencils sharing inputs (optional)
 7. BrickedLayoutPass       Apply bricked data layout (optional, type-level)
 8. DecompositionPass       Split domain across GPUs, separate interior/boundary (multi-GPU)
 9. HaloExchangePass        Insert communication ops with overlap support (multi-GPU)
10. VerifyPass              Validate IR well-formedness after all transformations
```

Passes 1-5 are the minimal single-GPU pipeline. Passes 6-7 are optional optimizations. Passes 8-9 add multi-GPU. Pass 10 runs after any transformation.

### Pass Details

**AnalysisPass (read-only):**
Walks `stencil.access` ops, collects offset patterns, computes halo width per dimension as `max(|offset|)` per axis. Attaches `halo_widths` as an attribute on the `stencil.apply` op. Replaces the current scattered logic across `footprint.py` (AST walk) and `pipeline.py` (side-effecting `spec.halo_widths = ...`).

**RooflinePass (read-only):**
Counts arithmetic ops by walking `arith.*` ops in the IR. Counts unique loads from deduplicated `stencil.access` ops. Computes arithmetic intensity and classifies memory vs compute bound. This is cleaner than the current `_FlopCounter` AST visitor because the IR already separates index arithmetic from stencil arithmetic -- no need for the `visit_Subscript` hack to skip index math.

**TilingPass:**
Reads halo widths and hardware spec (shared memory budget). Enumerates candidate tile sizes within cuTile's power-of-2 constraint, picks the minimum-overhead configuration. Attaches `tile_sizes` attribute to the operation. The principled autotuner (Layer 4) can override this with empirically-measured optimal sizes.

**BoundaryPass:**
Reads `BoundarySpec` from IR metadata. Inserts boundary-handling ops: periodic wrapping, Neumann reflection, Dirichlet clamping. These become explicit IR nodes that the lowering pass translates to either `ct.PaddingMode.ZERO` (Dirichlet) or host-side array slicing (periodic, Neumann, reflecting). Replaces the current 75-line `_emit_boundary_fill_kernel` method with dimension-specific branches.

**TemporalBlockingPass:**
Takes single-step stencil IR and produces multi-step IR. Determines max temporal depth T that fits in shared memory (expanded tile = `tile + 2*T*halo` per dimension). Wraps the stencil body in a temporal loop, inserts buffer swap ops between steps. The IR now contains a `host.for_loop(T)` with `cutile.launch` + `cutile.swap_buffers` inside. Replaces the tangled temporal logic in `stencil_codegen.py:_emit_temporal()`.

**FusionPass (optional):**
Detects multiple `stencil.apply` ops within the same module that read from overlapping input fields. For example, Gray-Scott updates both `u` and `v`, and both reads from `u` and `v`. The FusionPass fuses these into a single `stencil.apply` that loads the shared inputs once and computes both outputs. This eliminates redundant memory traffic for multi-field problems. The fused kernel returns multiple output fields. If stencil applications have incompatible access patterns or data dependencies that prevent fusion, the pass leaves them separate.

**BrickedLayoutPass (optional, type-level):**
Data layout is a **type-level concept** in the IR, not an access transformation. Fields carry their layout:

```
!stencil.field<?x?xf64, layout=row_major>       # standard
!stencil.field<?x?xf64, layout=bricked{32}>      # bricked with brick_size=32
```

The BrickedLayoutPass changes the layout attribute on field types and inserts layout conversion ops (`layout.cast`) at I/O boundaries (flat input → bricked for computation → flat output). The lowering pass reads the layout from the type and emits the appropriate addressing (standard indexing for row_major, divmod for bricked). This is more MLIR-idiomatic than transforming individual access patterns, and it composes cleanly: the emitter doesn't branch on layout — it reads the type.

**DecompositionPass (multi-GPU):**
Takes domain shape, number of GPUs, and topology (1D split along longest axis or Cartesian grid). Splits the `stencil.apply` domain into sub-domains. Creates per-GPU IR with sub-domain bounds. Attaches `gpu_id` and `neighbors` metadata.

Critically, this pass also **separates each sub-domain into interior and boundary regions.** Interior tiles do not depend on halo data from neighboring GPUs. Boundary tiles do. This separation enables computation-communication overlap in the HaloExchangePass.

**HaloExchangePass (multi-GPU, with overlap):**
Reads sub-domain boundaries and halo widths. Inserts `comm.send_halo` / `comm.recv_halo` ops at domain edges. Communication ops are **location-agnostic** — they can be placed in either host IR or device IR (see Communication Architecture below).

When `overlap=True` (default), the pass produces an overlapped schedule:

```
host.async {
    cutile.launch(interior_kernel, ...)     # compute interior (no halo dependency)
    comm.exchange_halos(...)                # simultaneously exchange halos
}
host.sync()
cutile.launch(boundary_kernel, ...)         # compute boundary (needs received halos)
```

This overlaps computation with communication, nearly hiding transfer latency for large domains where interior >> boundary. For temporal blocking with multi-GPU, exchanges are inserted between time steps within the temporal loop.

**VerifyPass:**
Runs after any transformation (or after all transformations). Uses xDSL's verification infrastructure to check IR invariants:
- All `stencil.access` offsets are within declared halo bounds
- Tile sizes are power-of-2 (cuTile constraint)
- Buffer chains have balanced alloc/free
- Multi-GPU sub-domain bounds are contiguous and cover the full domain
- Communication ops have matching send/recv pairs
- Layout types are consistent across operations

This catches bugs in pass implementations early, rather than at emission or GPU runtime.

### Communication Architecture

Communication ops (`comm.send_halo`, `comm.recv_halo`, `comm.exchange_halos`, `comm.barrier`) are defined in the `comm` dialect and are **location-agnostic** — they carry semantics but not placement. The lowering pass decides where they go based on backend capabilities:

**Today (host-side communication):**
cuTile does not support GPU-initiated communication. The lowering pass places comm ops in the host IR:

```
cutile.host_program {
    cutile.launch(interior_kernel, ...)
    comm.exchange_halos(...)            # host orchestrates P2P/NCCL transfers
    cutile.launch(boundary_kernel, ...)
}
```

**Future (fused communication kernels):**
When cuTile adds GPU-initiated communication (e.g., `ct.nccl_send()`), the lowering pass can place comm ops inside the device IR:

```
cutile.kernel {
    cutile.compute_interior(...)
    comm.send_halo(...)                 # GPU-initiated, inside kernel
    comm.recv_halo(...)
    cutile.compute_boundary(...)
}
```

The IR does not change — only the lowering pass changes. This means the optimization passes (decomposition, overlap, temporal blocking) work identically regardless of whether communication is host-side or fused. The architecture is ready for future cuTile capabilities without redesign.

### Time Stepping as a First-Class IR Concept

Time integration methods (Euler, RK2, RK4, SSPRK3) are represented in the IR, not as external wrappers:

```
timestep.rk4 @heat_rk4(%u, dt) {
    %k1 = stencil.apply @heat(%u)
    %u1 = timestep.stage %u, %k1, dt, 0.5        # u + 0.5*dt*k1
    %k2 = stencil.apply @heat(%u1)
    %u2 = timestep.stage %u, %k2, dt, 0.5        # u + 0.5*dt*k2
    %k3 = stencil.apply @heat(%u2)
    %u3 = timestep.stage %u, %k3, dt, 1.0        # u + dt*k3
    %k4 = stencil.apply @heat(%u3)
    %out = timestep.combine %u, (%k1,%k2,%k3,%k4), dt, (1/6,1/3,1/3,1/6)
    timestep.return %out
}
```

This enables the TemporalBlockingPass to reason about multi-stage methods. For example, blocking across 2 RK4 steps means 8 kernel launches with specific data dependencies — the pass can determine expanded halo requirements and buffer chains for the full multi-stage sequence, not just single Euler steps. The FusionPass can also fuse the stage computations where data dependencies allow.

### Composability

The key architectural win: passes compose without new code paths. Today, adding temporal blocking to multi-GPU requires duplicating temporal logic in `multigpu/codegen.py`. With passes, running `TemporalBlockingPass` and `DecompositionPass` on the same IR produces a temporally-blocked multi-GPU kernel automatically. Same for bricked + multi-GPU, or bricked + temporal + multi-GPU, or fused + temporal + multi-GPU with overlap.

Any combination of passes produces a valid IR that the lowering and emitter can handle, because each pass operates on the same standard dialect and the VerifyPass confirms well-formedness.

---

## Layer 3: Lowering

Three stages matching the three-dialect stack:

### Stage 0: Normalization (Dialect 1 -> Dialect 2)

An xDSL pass (`normalize.py`) that converts our high-level cuTile Stencil Dialect into the standard xDSL stencil dialect. This pass:
- Converts named index accesses (`[i-1, j]`) to offset tuples (`[-1, 0]`)
- Decomposes `cutile_stencil.compute` expressions into `arith.*` SSA operations
- Maps `cutile_stencil.func` metadata (boundary, constants) to IR attributes on `func.func`
- Converts `cutile_stencil.access` to `stencil.access`

This is a straightforward structural conversion. All semantic analysis and optimization happens after this step, on the standard xDSL stencil dialect.

### Stage 1: Stencil-to-cuTile Lowering (Dialect 2 -> Dialect 3)

An xDSL pass that converts optimized xDSL stencil operations into cuTile target dialect operations, producing both device IR (the kernel) and host IR (the launcher):

**Device IR lowering:**

| Stencil op | cuTile op |
|---|---|
| `stencil.access %u [+1, 0]` | `cutile.slice(axis=0, ...) + cutile.load` |
| `stencil.apply { body }` | `cutile.kernel { params, body }` |
| `stencil.return %result` | `cutile.store(out_tile, result)` |
| `arith.addf, arith.mulf` | Kept as Python arithmetic |
| `bricked.address` | `cutile.divmod_index` |

**Host IR lowering:**

| Source op | Host IR op |
|---|---|
| Single kernel | `cutile.alloc -> cutile.launch -> cutile.sync` |
| Temporal loop | `host.for_loop(T) { cutile.launch -> cutile.swap_buffers }` |
| Boundary application | `boundary.apply(buf, spec)` after each launch |
| Bricked layout | `host.flat_to_bricked` before, `host.bricked_to_flat` after |
| Multi-GPU | `host.for_each_gpu { cutile.launch } -> comm.exchange_halos` |
| Multi-GPU + temporal | `host.for_loop(T) { for_each_gpu { launch } -> exchange -> swap }` |

The host IR captures the complete orchestration for any feature combination. Every combination is a different IR graph -- the emitter does not have conditional branches for different feature combinations.

**Slice chain construction:** There is exactly ONE function that builds a `.slice()` chain from an axis, start offset, and stop offset. Every `stencil.access` lowers through this same path. This eliminates the 5+ copies of slice-building logic in the current codebase.

**Temporal blocking lowers to multi-launch:** The temporal loop in the IR becomes a Python `for _step in range(T):` loop with `ct.launch()` calls and buffer swapping. This is the validated multi-launch approach, not fused single-kernel (which hits cuTile's lack of grid sync).

**Boundary lowering by type:**
- Dirichlet: `ct.PaddingMode.ZERO` in the load (handled by cuTile natively)
- Periodic/Neumann/Reflecting: host-side Python function applied after each kernel launch

### Stage 2: cuTile Emitter

Walks the cuTile dialect ops (both device and host) and emits Python source text. This is the **only place** in the entire codebase that produces strings. Everything upstream is structured IR.

The emitter is simple and mechanical:

| cuTile op | Python output |
|---|---|
| `cutile.kernel { name, params, tile, halo }` | `@ct.kernel` function definition |
| `cutile.slice(var, axis, start, stop)` | `var.slice(axis=N, start=..., stop=...)` |
| `cutile.load(var)` | `ct.load(var)` |
| `cutile.store(var, expr)` | `ct.store(var, expr)` |
| `cutile.launch(kernel, grid, args)` | `ct.launch(stream, grid, kernel, args)` |
| `host.for_loop(N)` | `for _step in range(N):` |
| `host.for_each_gpu(n)` | `for gpu_id in range(n):` with `cp.cuda.Device(gpu_id)` |
| `comm.exchange_halos(...)` | `communicator.exchange_halos(partitions, ...)` |

### Generated Output

The emitted Python file is self-contained, matching the current output format:

```python
import cuda.tile as ct
import cupy as cp

ConstInt = ct.Constant[int]

@ct.kernel
def heat_kernel(u_in: ct.Tensor, u_out: ct.Tensor, TX: ConstInt, ...):
    bx, by = ct.bid(0), ct.bid(1)
    inp = u_in.slice(axis=0, ...).slice(axis=1, ...)
    out_tile = u_out.slice(axis=0, ...).slice(axis=1, ...)
    val = ct.load(inp)
    result = 0.25 * (val_left + val_right + val_up + val_down)
    ct.store(out_tile, result)

def launch_heat(u_in, u_out):
    stream = cp.cuda.get_current_stream()
    grid = (ct.cdiv(nx, TX), ct.cdiv(ny, TY))
    ct.launch(stream, grid, heat_kernel, (u_in, u_out, TX, TY, HX, HY, nx, ny))
```

### User-Facing API

The compilation API stays familiar:

```python
result = compile(heat, domain=(256, 256))
result.code              # Python source string
result.emit_to_file("heat_kernel.py")
result.validate(u0)      # GPU vs CPU reference check
result.benchmark(u0)     # timed execution with metrics

# Multi-GPU
result = compile(heat, domain=(1024, 1024), num_gpus=4, topology=(2, 2))
result.run(u_in, u_out)
```

---

## Layer 4: Runtime

### Launcher

Compiles and executes generated Python source. Handles `importlib` loading, CuPy array allocation, and stream management. No codegen logic lives in the launcher.

```python
result = compile(heat, domain=(256, 256), hw=HardwareSpec.auto_detect())
result.run(u_in, u_out)      # compile + launch on GPU
result.validate(u_in)        # GPU vs CPU reference comparison
result.benchmark(u_in)       # timed execution with throughput metrics
```

### Communication

Abstract `Communicator` protocol with two backends:

```python
class Communicator(Protocol):
    def send_halo(self, buf, dst_gpu, stream) -> None: ...
    def recv_halo(self, buf, src_gpu, stream) -> None: ...
    def exchange_halos(self, partitions, halo_width) -> None: ...
    def barrier(self) -> None: ...
```

**P2PCommunicator:** CuPy `data.copy_from_device()` via NVLink. For single-node multi-GPU (2-8 GPUs).

**NCCLCommunicator:** NCCL `send`/`recv` via `cupy.cuda.nccl` or PyTorch `ProcessGroupNCCL`. For multi-node scaling. Uses `mpi4py` for process management (rank assignment, topology setup).

**Backend selection:** Automatic based on topology. Detection logic: if all GPUs share the same `cupy.cuda.runtime.getDeviceCount()` and `cupy.cuda.Device(i).canAccessPeer(j)` returns True for all pairs, use P2P. Otherwise, use NCCL + MPI. User can override with explicit `communicator="p2p"` or `communicator="nccl"` parameter.

The generated host IR contains abstract communication ops. The launcher binds these to the appropriate communicator at runtime.

### Principled Autotuning

Three-phase approach replacing the current ad-hoc Optuna search:

**Phase 1: Analytical model.** Use roofline analysis to estimate performance for each configuration (tile size, temporal steps T, bricked vs flat). Narrows the search space from thousands of candidates to ~20-50 promising configurations.

**Phase 2: Empirical measurement.** Run the top candidates on GPU, measure actual throughput. Pick the best.

**Phase 3: Cache results.** Store winning configurations keyed by `(stencil_name, domain_shape, gpu_model)`. Reuse on subsequent runs.

```python
result = compile(heat, domain=(1024, 1024), autotune=True)
# Runs ~20 configurations, picks best, caches result
# Subsequent compilations with same key: instant lookup
```

The autotuner searches across: tile sizes (within cuTile's power-of-2 constraint), temporal blocking depth (T=1,2,3,4), bricked vs flat layout, and for multi-GPU, decomposition axis.

### CPU Reference

The existing `stencil_ref.py` (NumPy executor with `_ArrayProxy` pattern) is retained for validation. It operates on the original Python function to provide a ground-truth reference. Used in tests and `result.validate()`.

---

## Testing Strategy

### Testing Pyramid

**Level 1: IR Unit Tests (fast, no GPU)**
- Frontend parser produces correct xDSL IR from `@stencil` functions
- Each pass transforms IR correctly (input IR -> expected output IR)
- Lowering produces expected cuTile dialect ops from stencil ops
- Emitter produces valid Python source from cuTile ops (`ast.parse` + structure checks)

**Level 2: Convergence Tests (GPU required)**
- Generated kernel output matches CPU reference (NumPy) within `atol=1e-10` for float64
- Every feature path tested: plain, temporal, bricked, boundary, multi-GPU
- All dimensionalities: 1D, 2D, 3D stencils

**Level 3: Integration Tests (GPU required)**
- Full pipeline: `@stencil -> compile -> run -> validate`
- All applications run end-to-end (heat, wave, Gray-Scott, Maxwell, shallow water)
- Multi-step time marching: verify stability and correctness over 100+ steps
- Multi-GPU: verify results match single-GPU reference

**Level 4: Performance Tests (GPU required, benchmark suite)**
- Throughput: GPoints/s, GBytes/s, GFLOPS
- Compare against Devito and PyStencils on same hardware, same stencils
- Scaling: 1, 2, 4, 5 GPUs on single node; multi-node if available

### Key difference from current tests

The current codebase has ~48 codegen tests that only call `ast.parse(code)` -- they verify syntax, not correctness. In the new design, Level 1 tests the IR transformations directly (fast, deterministic, no GPU). Level 2 is where GPU correctness lives -- every feature path has a convergence test that actually runs the kernel.

---

## Benchmarking

### Application Suite

All existing applications are preserved and benchmarked:

| Application | Type | Key Feature |
|---|---|---|
| 1D Heat equation | Single-field, 3-point | Baseline |
| 2D Wave equation | Single-field, 4th-order | Higher-order stencils |
| 3D Laplacian | Single-field, 7-point | Memory-bound 3D |
| 2D Heat bricked | Single-field, bricked layout | Data locality optimization |
| 1D Advection upwind | Single-field, asymmetric | Non-symmetric offsets |
| Gray-Scott | 2-field reaction-diffusion | Multi-input stencils |
| FDTD Maxwell 1D | 2-field, staggered grid | Coupled E-H updates |
| Shallow water | 3-field flux stencils | Multi-field, conservation |

### Comparison Matrix

Each application benchmarked across frameworks:

| Stencil | cuTile-DSL | Devito | PyStencils | cuTile-raw |
|---|---|---|---|---|
| heat_1d | Y | Y | Y | Y |
| wave_2d | Y | Y | Y | Y |
| laplacian_3d | Y | Y | Y | Y |
| heat_2d_bricked | Y | - | - | Y |
| advection_1d | Y | Y | Y | Y |
| gray_scott | Y | Y | Y | Y |
| fdtd_maxwell | Y | Y | - | Y |
| shallow_water | Y | Y | - | Y |

("cuTile-raw" = hand-tuned cuTile kernel without the DSL, serving as upper bound.)

### Metrics

- Effective bandwidth (GB/s) and % of hardware peak
- Throughput (GPoints/s)
- Time to solution (ms)
- Roofline efficiency (% of achievable peak)

### Multi-GPU Scaling

| Stencil | 1 GPU | 2 GPU | 4 GPU | 5 GPU |
|---|---|---|---|---|
| wave_2d | Y | Y | Y | Y |
| laplacian_3d | Y | Y | Y | Y |
| gray_scott | Y | Y | Y | Y |
| shallow_water | Y | Y | Y | Y |

Results produce per-kernel performance tables and scaling efficiency plots.

---

## Time Integration

Time integration is a first-class feature, not a wrapper. The user API:

```python
@stencil(ndim=2, order=2)
def heat(u, i, j):
    return 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])

integrator = RKIntegrator(heat, method="RK4")
result = integrator.compile(domain=(256, 256))
```

The frontend builds `timestep.rk4` IR that contains `stencil.apply` ops for each RK stage. This IR flows through the full pass pipeline — temporal blocking can reason about multi-stage dependencies, fusion can merge RK stages where possible, and multi-GPU decomposition applies uniformly. The lowering pass emits the appropriate kernel launches with coefficient scaling per stage.

---

## Diagnostics

Each IR operation carries a source location pointing back to the original Python line in the user's `@stencil` function. When a pass produces an error or warning, the diagnostic references the user's code, not internal IR nodes.

Examples:
- TilingPass: "heat.py:3 — tile size 64x64 exceeds shared memory budget (requires 48KB, available 32KB), falling back to 32x32"
- AnalysisPass: "heat.py:4 — asymmetric halo detected: left=2, right=1 on axis 0"
- VerifyPass: "heat.py:3 — access offset [-3, 0] exceeds declared halo width of 1 on axis 0"
- DecompositionPass: "heat.py:1 — domain 128x128 with 4 GPUs yields sub-domains of 64x64, halo overhead is 12%"

The frontend parser attaches `loc` attributes (xDSL's source location infrastructure) when building Dialect 1 from the AST. The normalization pass propagates locations to Dialect 2. Passes that create new ops (e.g., boundary insertion, fusion) attach synthetic locations referencing the relevant source.

---

## Scope: What Is Cut

The following features from the current codebase are removed to keep the architecture focused:

- **CG solvers** (standard CG, preconditioned CG, persistent CG, mixed-precision CG, stencil CG) -- the solver framework is orthogonal to the stencil compilation contribution
- **AMR** (adaptive mesh refinement) -- incomplete in current code, a research area of its own. Mentioned as future work in the paper.
- **Triton backend** -- cuTile is the sole target
- **Numba backend** -- cuTile is the sole target

---

## Future Work

The following are architecturally sound extensions but are out of scope for the initial implementation:

- **IR serialization and compilation caching** — Serialize Dialect 1 IR for ahead-of-time compilation, cache generated code keyed by IR hash to skip recompilation.
- **Cost model as a pass output** — A queryable/updatable cost model that passes modify (e.g., temporal blocking updates memory traffic estimate, fusion updates FLOP count). Enables smarter autotuning without empirical measurement.
- **Stencil composition** — Explicit piping of one stencil's output into another's input (`compose(smooth, laplacian)`), enabling cross-stencil temporal blocking and fusion.
- **Additional backends** — The three-dialect architecture makes adding new targets (Triton, Numba CUDA, raw CUDA) possible by implementing only a new Dialect 3 and emitter. The optimization passes on Dialect 2 are reused.

---

## Dependencies

- **xDSL** (pip install xdsl) -- Pure Python MLIR framework. Provides dialect infrastructure, pass pipeline, stencil dialect.
- **cuTile / cuda.tile** (NVIDIA) -- GPU kernel API. The compilation target.
- **CuPy** -- GPU array operations, kernel launching, P2P transfers.
- **NumPy** -- CPU reference implementation.
- **mpi4py** -- MPI process management for multi-node (optional, only for distributed).
- **NCCL** (nvidia-nccl-cu13) -- GPU-to-GPU communication for multi-node (optional).
- **Devito** -- Benchmark comparison only (not a runtime dependency).
- **PyStencils** -- Benchmark comparison only (not a runtime dependency).

---

## Hardware

- Development and benchmarking on CHPC cluster
- 5x RTX PRO 6000 Blackwell GPUs (96 GB each, P2P enabled via NVLink)
- CUDA 13.1.0 or 13.2.0
- Python 3.12
