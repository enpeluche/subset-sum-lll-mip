"""
solve_cpsat_greedy.py
---------------------
CP-SAT with greedy-derived cardinality bounds.

    solve_cpsat_greedy_bound   — [k_lo, k_hi] from greedy
    solve_cpsat_smart_window   — tighter window via k_estimate ± tolerance
    solve_cpsat_smart_tightened — smart window + bound tightening (fix variables)

All build on solve_cpsat_bounded. The greedy bounds are O(n log n)
and never worsen CP-SAT performance (empirically validated).
"""

import time

from SubsetSumInstance import SubsetSumInstance
from results import SolveResult
from solvers.cpsat.solve_cpsat import solve_cpsat_bounded, _cpsat_guard, WORKERS
from constraints.range_constraints import compute_greedy_bounds
from heuristics.windowing import compute_smart_window
from constraints.tightening import bound_tightening


# =====================================================================
# Greedy bound
# =====================================================================

def solve_cpsat_greedy_bound(
    instance: SubsetSumInstance,
    hint: list[int] | None = None,
    tolerance: int = 0,
    timeout: float = 10.0,
    workers: int = WORKERS,
) -> SolveResult:
    """CP-SAT with greedy bounds [k_lo - tol, k_hi + tol]."""
    start = time.perf_counter()

    guard = _cpsat_guard(instance, start)
    if guard:
        return guard

    k_lo, k_hi = compute_greedy_bounds(instance)
    k_lo_c = max(0, k_lo - tolerance)
    k_hi_c = min(instance.n, k_hi + tolerance)

    result = solve_cpsat_bounded(
        instance, k_lo_c, k_hi_c,
        hint=hint, timeout=timeout, workers=workers,
    )
    result.elapsed = time.perf_counter() - start
    status = "Solved" if result.solution else "Timeout"
    result.label = f"Greedy_{status}[{k_lo_c},{k_hi_c}]"
    return result


# =====================================================================
# Smart window
# =====================================================================

def solve_cpsat_smart_window(
    instance: SubsetSumInstance,
    hint: list[int] | None = None,
    tolerance: int = 3,
    timeout: float = 10.0,
    workers: int = WORKERS,
) -> SolveResult:
    """CP-SAT with smart window: greedy ∩ [k_est - tol, k_est + tol]."""
    start = time.perf_counter()

    guard = _cpsat_guard(instance, start)
    if guard:
        return guard

    k_lo, k_hi = compute_smart_window(instance, tolerance)

    result = solve_cpsat_bounded(
        instance, k_lo, k_hi,
        hint=hint, timeout=timeout, workers=workers,
    )
    result.elapsed = time.perf_counter() - start
    status = "Solved" if result.solution else "Timeout"
    result.label = f"Smart_{status}[{k_lo},{k_hi}]"
    return result


# =====================================================================
# Smart + tightening
# =====================================================================

def solve_cpsat_smart_tightened(
    instance: SubsetSumInstance,
    hint: list[int] | None = None,
    tolerance: int = 3,
    timeout: float = 10.0,
    workers: int = WORKERS,
) -> SolveResult:
    """
    CP-SAT with smart window + bound tightening.

    1. Compute smart window [k_lo, k_hi]
    2. Fix variables forced by the window (bound tightening)
    3. Solve the reduced problem
    """
    start = time.perf_counter()

    guard = _cpsat_guard(instance, start)
    if guard:
        return guard

    k_lo, k_hi = compute_smart_window(instance, tolerance)
    fixed_zeros, fixed_ones = bound_tightening(instance, k_lo, k_hi)
    n_fixed = len(fixed_zeros) + len(fixed_ones)

    result = solve_cpsat_bounded(
        instance, k_lo, k_hi,
        hint=hint,
        fixed_zeros=fixed_zeros,
        fixed_ones=fixed_ones,
        timeout=timeout, workers=workers,
    )
    result.elapsed = time.perf_counter() - start
    status = "Solved" if result.solution else "Timeout"
    result.label = f"SmartBT_{status}[{k_lo},{k_hi}]_fixed={n_fixed}"
    return result