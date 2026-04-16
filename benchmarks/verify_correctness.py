"""End-to-end correctness check for the cuTile stencil compiler.

Compiles each stencil three ways:

1. Single-GPU (reference)
2. Multi-GPU on 2 GPUs
3. Multi-GPU on 4 GPUs (skipped if fewer than 4 GPUs are present)

Runs ``N_STEPS`` timesteps on each path starting from the same seeded
input, then compares the multi-GPU outputs against the single-GPU
reference bit-exactly (atol = 1e-10) on the interior.

Why multi-step matters: the event-chained async halo exchange in the
multi-GPU step body only exercises its cross-iteration event waits on
the second iteration onward. A single-step run would silently miss any
race between the previous step's halo arrival and the next step's
kernel.

Usage::

    python -m benchmarks.verify_correctness               # default N_STEPS=25
    python -m benchmarks.verify_correctness --n-steps 100 # stress
    python -m benchmarks.verify_correctness --stencils heat_2d laplacian_3d_7pt

Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import cupy as cp

from benchmarks.stencils import STENCIL_META, full_shape
from cutile import compile as stencil_compile


# Small domains chosen so the full check finishes in seconds while still
# exercising every dimensionality. Multi-GPU split is along axis 0; pick
# domain shapes whose first axis divides cleanly by 4.
_VERIFY_DOMAINS = {
    1: (4096,),
    2: (256, 256),
    3: (32, 32, 32),
}


@dataclass
class CheckResult:
    stencil: str
    domain: tuple
    num_gpus: int
    n_steps: int
    max_diff: float
    ok: bool


def _interior_slice(halo):
    return tuple(slice(h, -h) if h else slice(None) for h in halo)


def _run_single_gpu_reference(sname: str, domain: tuple, n_steps: int,
                              seed: int = 0):
    """Compile single-GPU, run N steps, return final state."""
    meta = STENCIL_META[sname]
    sfn = meta["cutile_fn"]
    halo = meta["halo"]
    shape = full_shape(domain, halo)

    res = stencil_compile(sfn, domain=domain, num_gpus=1,
                          temporal_blocking=False)
    launch = getattr(res.load_module(), f"launch_{sname}")

    cp.random.seed(seed)
    u_seed = cp.random.randn(*shape).astype(cp.float64)
    u_in = u_seed.copy()
    u_out = cp.zeros_like(u_in)
    for _ in range(n_steps):
        launch(u_in, u_out)
        u_in, u_out = u_out, u_in
    cp.cuda.Device(0).synchronize()
    return u_seed, u_in


def _run_multigpu(sname: str, domain: tuple, num_gpus: int, n_steps: int,
                  u_seed) -> cp.ndarray:
    """Compile multi-GPU, run N steps, gather result."""
    meta = STENCIL_META[sname]
    sfn = meta["cutile_fn"]

    res = stencil_compile(sfn, domain=domain, num_gpus=num_gpus,
                          temporal_blocking=False)
    mod = res.load_module()
    setup = getattr(mod, f"setup_multigpu_{sname}")
    step = getattr(mod, f"step_multigpu_{sname}")
    gather = getattr(mod, f"gather_multigpu_{sname}")

    p_in, p_out = setup(u_seed.copy(), num_gpus=num_gpus)
    for _ in range(n_steps):
        step(p_in, p_out, num_gpus=num_gpus)
        p_in, p_out = p_out, p_in
    u_test = cp.zeros_like(u_seed)
    gather(u_test, p_in, num_gpus=num_gpus)
    return u_test


def verify_stencil(sname: str, n_steps: int, gpu_counts: list[int],
                   atol: float) -> list[CheckResult]:
    """Verify one stencil across all requested GPU counts."""
    meta = STENCIL_META[sname]
    domain = _VERIFY_DOMAINS[meta["ndim"]]
    halo = meta["halo"]
    interior = _interior_slice(halo)

    u_seed, ref_final = _run_single_gpu_reference(sname, domain, n_steps)

    results: list[CheckResult] = []
    for ng in gpu_counts:
        if meta["ndim"] < 2 and ng > 1:
            continue  # multi-GPU lowering only handles ndim>=2
        if ng == 1:
            continue  # reference path; nothing to compare against itself
        u_test = _run_multigpu(sname, domain, ng, n_steps, u_seed)
        diff = float(cp.abs(u_test[interior] - ref_final[interior]).max())
        results.append(CheckResult(
            stencil=sname,
            domain=domain,
            num_gpus=ng,
            n_steps=n_steps,
            max_diff=diff,
            ok=diff < atol,
        ))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stencils", nargs="+", default=None,
                        help="Subset of stencils to verify (default: all)")
    parser.add_argument("--n-steps", type=int, default=25,
                        help="Timesteps per check (default: 25)")
    parser.add_argument("--atol", type=float, default=1e-10,
                        help="Absolute tolerance for bit-exact compare")
    parser.add_argument("--max-gpus", type=int, default=None,
                        help="Cap GPU count (default: all available)")
    args = parser.parse_args()

    n_avail = cp.cuda.runtime.getDeviceCount()
    max_gpus = min(args.max_gpus or n_avail, n_avail)
    gpu_counts = [g for g in (2, 4) if g <= max_gpus]

    stencils = args.stencils or list(STENCIL_META.keys())
    print(f"GPUs available: {n_avail} (using up to {max_gpus})")
    print(f"Stencils: {', '.join(stencils)}")
    print(f"Steps per check: {args.n_steps}")
    print(f"Tolerance: {args.atol:.0e}")
    print()
    print(f"{'stencil':24s} {'domain':18s} {'GPUs':>5s} {'steps':>6s} "
          f"{'max_diff':>12s}  result")
    print("-" * 75)

    all_results: list[CheckResult] = []
    for sname in stencils:
        if sname not in STENCIL_META:
            print(f"  [skip] unknown stencil: {sname}")
            continue
        results = verify_stencil(sname, args.n_steps, gpu_counts, args.atol)
        for r in results:
            mark = "PASS" if r.ok else "FAIL"
            print(f"{r.stencil:24s} {str(r.domain):18s} {r.num_gpus:>5d} "
                  f"{r.n_steps:>6d} {r.max_diff:>12.2e}  {mark}")
            all_results.append(r)

    print()
    failed = [r for r in all_results if not r.ok]
    if failed:
        print(f"FAILED: {len(failed)} of {len(all_results)} checks")
        for r in failed:
            print(f"  {r.stencil} {r.num_gpus}G: max_diff={r.max_diff:.2e}")
        return 1
    print(f"OK: {len(all_results)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
