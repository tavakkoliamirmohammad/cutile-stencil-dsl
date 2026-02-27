# cutile-stencil-dsl Implementation Plan

## Overview

PhD research framework implementing two directions:
- **Direction 8**: High-order stencil compilation via tile abstractions
- **Direction 9**: Tile-based iterative solvers for sparse linear systems

Two layers:
1. **Pure Python** (DSL, analysis, reference implementations, tests) — runs on any machine
2. **Generated cuTile code** — syntactically correct Python files ready to run when cuTile is available

Location: `/Users/amirmohammadtavakkoli/project/cutile-stencil-dsl/`

---

## Current State

All files have been written. **Tests have NOT been run yet.** The next step is:

```bash
cd /Users/amirmohammadtavakkoli/project/cutile-stencil-dsl && python -m pytest tests/ -v
```

Then fix any failures. After that, run examples:
```bash
python examples/heat_1d.py
python examples/poisson_cg.py
```

---

## Project Structure

```
cutile-stencil-dsl/
├── pyproject.toml
├── cutile_stencil/
│   ├── __init__.py
│   ├── dsl/
│   │   ├── __init__.py
│   │   ├── types.py            # StencilSpec, OffsetAccess, HardwareSpec dataclasses
│   │   ├── decorator.py        # @stencil decorator
│   │   └── registry.py         # Global stencil registry
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── footprint.py        # AST-based stencil footprint extraction
│   │   ├── tiling.py           # Tile decomposition + halo optimization
│   │   ├── temporal.py         # Temporal blocking analysis
│   │   └── roofline.py         # Analytical roofline model
│   ├── codegen/
│   │   ├── __init__.py
│   │   ├── emitter.py          # CodeEmitter helper (indented code builder)
│   │   └── stencil_codegen.py  # StencilSpec → cuTile kernel .py file
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── formats.py          # DIAMatrix, BSRMatrix dataclasses + constructors
│   │   ├── kernels.py          # Generated cuTile kernels: SpMV, axpy, dot, norm
│   │   ├── cg.py               # Standard multi-launch CG (cuTile driver code)
│   │   ├── persistent_cg.py    # Single-kernel persistent CG (cuTile)
│   │   └── mixed_precision.py  # Mixed-precision iterative refinement (cuTile)
│   └── reference/
│       ├── __init__.py
│       ├── stencil_ref.py      # NumPy stencil executor (runs without GPU)
│       ├── spmv_ref.py         # NumPy DIA/BSR SpMV
│       └── cg_ref.py           # NumPy CG, mixed-precision CG
├── examples/
│   ├── heat_1d.py              # Full pipeline: DSL → analysis → codegen → numpy validation
│   ├── wave_2d.py              # 2D acoustic wave, 4th-order stencil
│   ├── laplacian_3d.py         # 3D 7-point Laplacian stencil
│   ├── poisson_cg.py           # Poisson equation solved with CG
│   └── mixed_precision_cg.py   # Mixed-precision CG demo
└── tests/
    ├── test_dsl.py             # Stencil spec + footprint extraction
    ├── test_tiling.py          # Tile config + temporal blocking
    ├── test_codegen.py         # Generated code is valid Python (ast.parse)
    ├── test_reference.py       # NumPy stencil correctness
    └── test_solvers.py         # NumPy CG convergence + SpMV correctness
```

---

## Phase Details

### Phase 1: DSL Core + Types (DONE — files written)

- `types.py`: Dataclasses `OffsetAccess(array_name, offsets)`, `StencilSpec(name, ndim, order, inputs, output, update_fn, accesses, dtype, tile_sizes, halo_widths, temporal_steps)`, `HardwareSpec(peak_bandwidth_gbs, shared_mem_bytes, sm_count, dtype_bytes)`, `TileConfig(tile_sizes, halo_widths, num_tiles, overhead_fraction)`, `TemporalConfig(steps, expanded_halo, bandwidth_reduction_factor)`, `RooflineResult(flops_per_point, bytes_per_point, arithmetic_intensity, bound, peak_gpoints_s)`
- `decorator.py`: `@stencil(ndim=, order=, dtype=)` captures the function, creates a `StencilSpec`, stores on `fn._stencil_spec`
- `registry.py`: Simple dict `_REGISTRY: Dict[str, StencilSpec]` with `register()`, `lookup()`, `all_stencils()`, `clear()`

### Phase 2: Analysis Pipeline (DONE — files written)

- `footprint.py`: AST visitor `_OffsetVisitor` that walks the stencil function body, finds all `name[offset]` subscripts, returns `List[OffsetAccess]`. `compute_halo()` returns max absolute offset per dimension.
- `tiling.py`: `compute_tile_config()` — given StencilSpec + domain size + HardwareSpec, enumerates power-of-2 tile sizes (32..1024), picks the one minimizing halo overhead while fitting in shared memory.
- `temporal.py`: `compute_temporal_config()` — finds max T_block such that expanded tile `(tile + 2*T*halo)` still fits in shared memory.
- `roofline.py`: `roofline_analysis()` — counts FLOPs from AST, counts unique array loads, computes arithmetic intensity, classifies as memory-bound or compute-bound.

### Phase 3: Code Generation (DONE — files written)

- `emitter.py`: `CodeEmitter` class with `line()`, `indent()` context manager, `blank()`, `render()`.
- `stencil_codegen.py`: `StencilCodeGenerator(spec, tile_config, temporal_config=None)`:
  - 1D → shifted-view approach (like `stencil1d.py` from cutile-exp): loads shifted views of the array, computes stencil on tiles
  - 2D → offset-tile-load with `padding_mode=ZERO`: loads expanded tile including halos, extracts neighbor slices
  - 3D → uses `ct.bid(0), ct.bid(1), ct.bid(2)` grid pattern (like `batch_matmul.py`)
  - 2D temporal blocking → fused multi-step kernel with shrinking halos per step
  - Each generated file includes: imports, `@ct.kernel` function, launcher wrapper, benchmark boilerplate
  - AST-based expression reconstruction: parses the stencil update_fn, extracts the return expression, substitutes array subscripts with tile variable names

### Phase 4: Solver Framework (DONE — files written)

- `formats.py`: `DIAMatrix(data, offsets, shape)` with `to_dense()`, `BSRMatrix(data, indices, indptr, blocksize, shape)` with `to_dense()`. Constructors: `laplacian_1d_dia(N)`, `laplacian_2d_dia(Nx, Ny)`, `laplacian_3d_dia(Nx, Ny, Nz)`.
- `kernels.py`: Code generators returning cuTile kernel strings:
  - `generate_dia_spmv()`: DIA SpMV using shifted-view + gather pattern
  - `generate_bsr_spmv()`: BSR SpMV using gather + mma
  - `generate_axpy()`: y = alpha*x + y
  - `generate_dot()`: atomic reduction dot product
  - `generate_norm2()`: sum-of-squares with atomic reduction
- `cg.py`: `generate_cg_driver()` — standard CG with 7 kernel launches per iteration (SpMV, dots, axpys)
- `persistent_cg.py`: `generate_persistent_cg()` — single `@ct.kernel` with `ct.num_blocks(0)` persistent loop, `ct.atomic_cas`/`ct.atomic_xchg` spinlock for barrier synchronization between phases
- `mixed_precision.py`: `generate_mixed_precision_cg()` — outer FP64 residual loop + inner FP32 CG with `ct.astype` casts

### Phase 5: NumPy Reference Implementations (DONE — files written)

- `stencil_ref.py`: `_ArrayProxy` intercepts `u[offset]` subscripts and maps to numpy slices. `apply_stencil(u, spec)` → single step. `time_march(u0, spec, steps)` → full simulation.
- `spmv_ref.py`: `dia_spmv(A, x)` and `bsr_spmv(A, x)` in pure numpy.
- `cg_ref.py`: `cg_solve(A_fn, b, x0, tol, max_iter)` → (x, iters, residuals). `mixed_precision_cg(A_fn, b, x0, inner_dtype, ...)` — outer FP64 + inner lower-precision CG.

### Phase 6: Examples (DONE — files written)

Each example demonstrates the full pipeline:
1. Define stencil with `@stencil` decorator
2. Run analysis (footprint → tiling → temporal → roofline)
3. Generate cuTile kernel code to `generated/` subfolder
4. Run NumPy reference simulation for validation
5. Print analysis results

- `heat_1d.py`: 1D heat equation, Gaussian initial condition, energy dissipation check
- `wave_2d.py`: 2D acoustic wave with 4th-order stencil, leapfrog time integration
- `laplacian_3d.py`: 3D 7-point Laplacian, point source test
- `poisson_cg.py`: 1D and 2D Poisson solved with CG, exact solution comparison
- `mixed_precision_cg.py`: FP64 outer + FP32 inner, compares with standard FP64 CG

### Phase 7: Tests (DONE — files written)

All tests use only numpy (no GPU). Key assertions:
- `test_dsl.py`: decorator creates spec, footprint extraction matches expected offsets (1D/2D/3D), registry works, callable preserved
- `test_tiling.py`: tile configs fit within hardware budgets, temporal blocking produces valid configs, roofline gives positive values
- `test_codegen.py`: all generated code (1D/2D/3D stencils, solver kernels, CG drivers) passes `ast.parse()`, contains expected cuTile API calls
- `test_reference.py`: heat equation single step matches manual calc, energy dissipation, constant preserved, 2D/3D Laplacian point source
- `test_solvers.py`: DIA/BSR SpMV matches dense matmul, CG converges on 1D/2D Poisson, mixed-precision CG converges

---

## What Needs To Be Done

1. **Run tests**: `cd /Users/amirmohammadtavakkoli/project/cutile-stencil-dsl && python -m pytest tests/ -v`
2. **Fix any test failures** — the most likely issues are:
   - AST expression reconstruction in `stencil_codegen.py` may not perfectly substitute all offset patterns
   - Edge cases in DIA SpMV boundary handling
   - Temporal blocking codegen may have indexing issues
3. **Run examples** to verify end-to-end:
   - `python examples/heat_1d.py`
   - `python examples/laplacian_3d.py`
   - `python examples/poisson_cg.py`
   - `python examples/mixed_precision_cg.py`
4. **Verify generated code** passes `ast.parse()` — this is critical since the user cannot run cuTile

---

## cuTile API Patterns Reference

All generated code follows patterns from `/Users/amirmohammadtavakkoli/project/cutile-exp/`:

| Pattern | Source File | Usage |
|---|---|---|
| 1D shifted-view stencil | `stencil1d.py` | 1D stencil codegen |
| 2D tile load with padding | `matmul.py`, `layernorm.py` | 2D/3D stencil codegen |
| Persistent kernel loop | `matmul.py:49-72` | Persistent CG |
| Atomic reduction | `dot_product.py`, `norm2.py` | CG dot products/norms |
| Spinlock sync | `layernorm.py:74-79` | Persistent CG phase sync |
| Code generation | `tensor_contraction_gen.py` | CodeEmitter pattern |

Key cuTile API:
- `import cuda.tile as ct`
- `ConstInt = ct.Constant[int]`
- `@ct.kernel` decorator
- `ct.bid(0)`, `ct.bid(1)`, `ct.bid(2)` — block IDs
- `ct.load(tensor, index=tuple, shape=tuple, padding_mode=ct.PaddingMode.ZERO)`
- `ct.store(tensor, index=tuple, tile=value)`
- `ct.gather(tensor, indices)`, `ct.scatter(tensor, indices, values)`
- `ct.full(shape, value, dtype=...)`, `ct.arange(size, dtype=...)`
- `ct.sum(tile, axis=None)`, `ct.maximum(a, b)`
- `ct.mma(a, b, accumulator)`, `ct.matmul(a, b)`
- `ct.astype(tile, dtype)`, `ct.reshape(tile, shape)`
- `ct.atomic_add(tensor, index, value)`
- `ct.atomic_cas(tensor, index, expected, new, memory_order=ct.MemoryOrder.ACQUIRE)`
- `ct.atomic_xchg(tensor, index, value, memory_order=ct.MemoryOrder.RELEASE)`
- `ct.num_blocks(dim)`, `ct.num_tiles(tensor, axis, shape)`, `ct.cdiv(a, b)`
- `ct.launch(stream, grid, kernel, args_tuple)`
