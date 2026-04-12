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

### End-to-End Data Flow

```
User writes:
    @stencil(ndim=2, order=2)
    def heat(u, i, j):
        return 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])

         | Frontend (Python AST -> xDSL IR)

xDSL stencil dialect:
    stencil.apply(%u) {
        %left  = stencil.access %u [-1, 0]
        %right = stencil.access %u [+1, 0]
        %up    = stencil.access %u [0, -1]
        %down  = stencil.access %u [0, +1]
        %sum   = arith.addf %left, %right, %up, %down
        %out   = arith.mulf %sum, 0.25
        stencil.return %out
    }

         | Analysis passes (footprint, halo, roofline)
         | Optimization passes (tiling, temporal blocking, bricked layout)
         | Multi-GPU passes (decomposition, halo exchange insertion)

         | Lowering pass (stencil -> cuTile dialect)

cuTile dialect (device IR):
    cutile.kernel { tile_shape=[TX,TY], halo=[HX,HY] }
        cutile.slice(axis=0, start=..., stop=...)
        cutile.load(...)
        cutile.store(...)

cuTile dialect (host IR):
    cutile.alloc_buffers(...)
    cutile.stream()
    host.for_loop(T) {
        cutile.launch(kernel, grid, args)
        boundary.apply(...)
        cutile.swap_buffers(...)
    }

         | Emission (cuTile dialect -> Python source string)

    @ct.kernel
    def heat_kernel(u_in, u_out, TX, TY, HX, HY, nx, ny):
        bx, by = ct.bid(0), ct.bid(1)
        inp = u_in.slice(axis=0, ...).slice(axis=1, ...)
        ...
        ct.store(out_tile, result)

    def launch_heat(u_in, u_out):
        stream = cp.cuda.get_current_stream()
        ...
```

### Module Structure

```
cutile/
|-- frontend/               # Layer 1: @stencil -> IR
|   |-- decorator.py        # @stencil decorator, auto-inference of ndim/order
|   |-- parser.py           # Python AST -> xDSL stencil dialect operations
|   +-- types.py            # BoundarySpec, HardwareSpec, GPU presets, configs
|
|-- dialects/               # xDSL dialect definitions
|   |-- stencil_ext/        # Additional ops/attributes extending (not forking) xDSL's stencil dialect
|   |-- cutile_dialect/     # Custom cuTile target dialect (kernel, slice, load, store, launch)
|   +-- comm/               # Communication dialect (halo send/recv, barrier, sync)
|
|-- passes/                 # Layer 2: IR transformations
|   |-- analysis/           # Read-only passes
|   |   |-- footprint.py    # Extract access patterns, compute halo widths
|   |   +-- roofline.py     # FLOP count, arithmetic intensity, bound classification
|   |-- tiling.py           # Tile size selection (shared mem budget, overhead minimization)
|   |-- temporal.py         # Temporal blocking (fuse time steps, expand halos, buffer chain)
|   |-- bricked.py          # Bricked data layout (transform accesses to divmod addressing)
|   |-- boundary.py         # Boundary condition insertion (periodic, Neumann, Dirichlet, reflecting)
|   |-- decompose.py        # Domain decomposition (1D split or Cartesian grid)
|   +-- halo.py             # Halo exchange insertion (abstract comm ops)
|
|-- lowering/               # Layer 3: IR -> code
|   |-- stencil_to_cutile.py  # Stencil dialect -> cuTile dialect (device + host IR)
|   +-- emitter.py            # cuTile dialect -> Python source text
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

The decorator performs three steps:

1. **Parse** the function's AST to extract array accesses and the computation expression.
2. **Build** an xDSL `stencil.apply` operation with proper `stencil.access` ops for each array read and `arith.*` ops for the computation.
3. **Attach** metadata as IR attributes: boundary spec, closure constants (dt, dx, etc.), dtype.

### Auto-inference

The current auto-inference of `ndim` (from subscript dimensions) and `order` (from max offset magnitudes) is retained. The parser walks the AST once to determine both, extract all offset patterns, capture closure constants, and build the IR.

### Immutability

The decorator produces an **immutable xDSL module**. Once built, the IR is the source of truth. There is no mutable `StencilSpec` being modified in multiple places. Halo widths, tile sizes, and temporal steps are computed by passes and stored as attributes on IR operations.

The raw Python function is preserved alongside the IR for CPU reference execution in tests, but it is never used for code generation.

### Multi-input stencils

For stencils with multiple input arrays (Gray-Scott with `u` and `v`, shallow water with `h`, `hu`, `hv`), each array becomes a separate operand to `stencil.apply`. xDSL's stencil dialect natively supports multiple input fields.

### xDSL IR Example

For the 2D heat stencil, the frontend produces:

```
builtin.module {
    func.func @heat(%u: !stencil.field<[-1,257]x[-1,257]xf64>)
                    -> !stencil.field<[0,256]x[0,256]xf64> {
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
        } : !stencil.field<[0,256]x[0,256]xf64>
        return %out
    }
}
```

---

## Layer 2: Passes

Each pass is an xDSL `RewritePattern` or `ModulePass` that transforms the IR. Passes are composable, independently testable, and can be enabled or disabled.

### Pass Pipeline (execution order)

```
1. AnalysisPass            Read-only: extract footprint, compute halo widths
2. RooflinePass            Read-only: count FLOPs, classify memory/compute bound
3. TilingPass              Attach tile sizes based on hardware + halo + shared mem
4. BoundaryPass            Insert boundary condition ops into IR
5. TemporalBlockingPass    Fuse N time steps, expand halos, insert buffer chain
6. BrickedLayoutPass       Transform accesses to bricked addressing (optional)
7. DecompositionPass       Split domain across GPUs (multi-GPU)
8. HaloExchangePass        Insert communication ops at sub-domain edges (multi-GPU)
```

Passes 1-6 are single-GPU. Passes 7-8 add multi-GPU support. A minimal compilation uses passes 1-4. A full multi-GPU temporally-blocked bricked-layout compilation uses all 8.

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

**BrickedLayoutPass:**
Transforms memory access patterns from row-major to bricked. Inserts `divmod` addressing: `brick_id = flat_index // brick_size`, `offset = flat_index % brick_size`. Adds layout conversion ops (flat-to-bricked, bricked-to-flat) for I/O boundaries. Replaces the current separate 180-line code path (`_emit_bricked`, `_emit_kernel_bricked_nd`).

**DecompositionPass (multi-GPU):**
Takes domain shape, number of GPUs, and topology (1D split along longest axis or Cartesian grid). Splits the `stencil.apply` domain into sub-domains. Creates per-GPU IR with sub-domain bounds. Attaches `gpu_id` and `neighbors` metadata.

**HaloExchangePass (multi-GPU):**
Reads sub-domain boundaries and halo widths. Inserts `comm.send_halo` / `comm.recv_halo` ops at domain edges. For temporal blocking with multi-GPU, inserts exchanges between time steps. Communication ops are abstract -- the lowering layer decides whether to use P2P or NCCL.

### Composability

The key architectural win: passes compose without new code paths. Today, adding temporal blocking to multi-GPU requires duplicating temporal logic in `multigpu/codegen.py`. With passes, running `TemporalBlockingPass` and `DecompositionPass` on the same IR produces a temporally-blocked multi-GPU kernel automatically. Same for bricked + multi-GPU, or bricked + temporal + multi-GPU.

---

## Layer 3: Lowering

Two stages: structural lowering (stencil dialect -> cuTile dialect) and emission (cuTile dialect -> Python source text).

### Stage 1: Stencil-to-cuTile Lowering

An xDSL pass that converts high-level stencil operations into cuTile-specific operations, producing both device IR (the kernel) and host IR (the launcher):

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

## RK Time Integrator (Demonstration)

The Runge-Kutta integrator is retained as a demonstration of composability:

```python
@stencil(ndim=2, order=2)
def heat(u, i, j):
    return 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])

integrator = RKIntegrator(heat, method="RK4")
result = integrator.compile(domain=(256, 256))
```

The stencil compiles through the normal pipeline. The RK4 wrapper generates host IR with four kernel launches per time step (one per RK stage) with coefficient scaling. This shows that compiled stencils compose into higher-level numerical methods.

---

## Scope: What Is Cut

The following features from the current codebase are removed to keep the architecture focused:

- **CG solvers** (standard CG, preconditioned CG, persistent CG, mixed-precision CG, stencil CG) -- the solver framework is orthogonal to the stencil compilation contribution
- **AMR** (adaptive mesh refinement) -- incomplete in current code, a research area of its own. Mentioned as future work in the paper.
- **Triton backend** -- cuTile is the sole target
- **Numba backend** -- cuTile is the sole target

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
