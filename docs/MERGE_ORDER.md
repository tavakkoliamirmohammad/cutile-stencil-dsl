# PR Merge & Review Order

**Total: 35 PRs**

## Merge Order (strict — each group must merge before the next)

### Group 1: Bug fixes (merge in any order within group)
| PR | Branch | Title |
|----|--------|-------|
| #1 | `fix/codegen-fallback-warning` | Fix codegen fallback warning |
| #2 | `fix/boundary-shared-references` | Fix boundary shared refs |
| #3 | `fix/roofline-flop-scope` | Fix roofline FLOP scope |
| #4 | `fix/closure-capture-scope` | Fix closure capture scope |
| #5 | `feat/auto-detect-gpu-pipeline` | Auto-detect GPU |
| #6 | `feat/add-all-exports` | Add `__all__` |

> **Note:** #3 and #4 both add classes to `test_bugfixes.py` — you'll get a conflict on the second merge. Just keep both classes.

### Group 2: Structural (merge in this order)
| PR | Branch | Title | Depends on |
|----|--------|-------|-----------|
| #20 | `fix/spec-mutation` | Fix spec mutation | Group 1 |
| #21 | `refactor/unify-codegen` | Unify 1D/2D/3D codegen | Group 1 |

### Group 3: Codegen features (merge in this order)
| PR | Branch | Title | Depends on |
|----|--------|-------|-----------|
| #22 | `feat/temporal-blocking-codegen` | Temporal blocking codegen | #21 |
| #23 | `feat/boundary-codegen` | Boundary condition codegen | #21 |

### Group 4: Independent features (merge in any order)
| PR | Branch | Title |
|----|--------|-------|
| #7 | `feat/named-stencil-templates` | Named stencil templates |
| #8 | `feat/non-pow2-tile-sizes` | Non-pow2 tile sizes |
| #9 | `feat/tiling-visualization` | Tiling visualization |
| #10 | `feat/4th-order-example` | 4th-order example |
| #11 | `feat/variable-diffusion-example` | Variable-coefficient example |
| #12 | `feat/rk-integrator` | RK integrator |
| #13 | `feat/bayesian-autotune` | Bayesian autotune |
| #14 | `feat/jacobi-preconditioner` | Jacobi preconditioner |
| #15 | `feat/convergence-history` | Convergence history |
| #16 | `feat/implicit-stencils` | Implicit stencils |
| #17 | `feat/benchmarks-gpu-timing` | Benchmarks + GPU timing |
| #18 | `feat/triton-backend` | Triton backend |

### Group 5: Multi-GPU (merge in this order)
| PR | Branch | Title | Depends on |
|----|--------|-------|-----------|
| #19 | `feat/multigpu-domain-decomposition` | Multi-GPU domain decomposition | independent |
| #24 | `feat/multigpu-improvements` | Halo packing + overlap + periodic | #19 |
| #27 | `feat/2d-cartesian-decomposition` | 2D Cartesian decomposition | #19 |
| #32 | `feat/multigpu-gather-codegen` | P2P + gather codegen | #19 |
| #31 | `test/ct-gather-multigpu` | ct.gather validation tests | independent |
| #35 | `bench/multigpu-halo-exchange` | Halo exchange benchmark | independent |

### Group 6: Advanced (merge in any order)
| PR | Branch | Title |
|----|--------|-------|
| #25 | `test/gpu-validation-temporal-boundary` | GPU validation tests |
| #26 | `feat/preconditioned-cg` | Preconditioned CG integration |
| #28 | `feat/amr-support` | AMR support |
| #29 | `feat/cg-convergence-integration` | CG convergence history integration |
| #30 | `feat/gpu-benchmarks` | GPU benchmarks (147-3195x speedup) |
| #33 | `feat/codegen-amr-presets` | 3D AMR + Numba backend + presets |
| #34 | `test/gpu-validate-cg-features` | GPU CG feature validation |

---

## Review Order (suggested — by impact)

1. **#1-6** — quick small diffs, obvious fixes
2. **#21** — structural backbone (unify codegen, 25% file reduction)
3. **#22, #23** — high-impact (temporal blocking, boundary codegen)
4. **#19, #32** — headline feature (multi-GPU)
5. **#30** — proof of value (GPU benchmarks: 147-3195x speedup)
6. Everything else

---

## Key Findings from This Session

- **Single GPU:** up to 184 Gpts/s on RTX PRO 6000 Blackwell (3,195x over NumPy)
- **Multi-GPU:** exact match 1-4 GPUs, 0.07-0.17ms per timestep
- **ct.gather cross-GPU:** confirmed working on NVLink P2P (PR #31)
- **P2P vs direct halo:** essentially tied, ~1% difference (PR #35)
- **Roofline model:** underestimates Blackwell by 4-6x due to L2 cache effects
