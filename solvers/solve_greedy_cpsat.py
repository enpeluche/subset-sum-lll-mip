from SubsetSumInstance import SubsetSumInstance
import time
from results import SolveResult
from solvers.solve_bounded_cpsat import solve_bounded_cpsat
from heuristics.windowing import compute_smart_window
from constraints.tightening import bound_tightening
from constraints.range_constraints import compute_greedy_bounds
from utils import search_space_window

def solve_cpsat_greedy_bound(
    instance: SubsetSumInstance,
    tolerance: int = 0,
    timeout: float = 100.0,
    workers: int = 8
) -> SolveResult:
    """CP-SAT with greedy bounds [k_lo, k_hi] (tolerance=extra slack)."""

    start = time.perf_counter()

    # 0. Early exit: no subset can reach T if the total sum is insufficient.
    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)
    
    # 0.1 Early exit if instance cannot be computed with cp sat solver.
    if not instance.fits_int64:
        return SolveResult.skipped("CPSAT_Overflow_Skip")
    
    k_lo, k_hi = compute_greedy_bounds(instance)
    
    n = instance.n
    
    k_lo_c = max(0, k_lo - tolerance)
    k_hi_c = min(n, k_hi + tolerance)

    result = solve_bounded_cpsat(instance, k_lo_c, k_hi_c, timeout=timeout, workers=workers)

    result.elapsed = time.perf_counter() - start
    status_str = "Solved" if result.solution is not None else "Timeout"
    result.label = f"CPSAT_Greedy_{status_str} [{k_lo_c},{k_hi_c}]"
    
    return result


def solve_cpsat_smart_window(
    instance: SubsetSumInstance,
    tolerance: int = 3,
    timeout: float = 100.0,
    workers: int = 8
) -> SolveResult:
    """
    CP-SAT with smart window (greedy ∩ [k_est-tol, k_est+tol]).
    Covers k* in ~97% of cases with tol=3.
    """
    start = time.perf_counter()

    # 0. Early exit: no subset can reach T if the total sum is insufficient.
    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)
    
    # 0.1 Early exit if instance cannot be computed with cp sat solver.
    if not instance.fits_int64:
        return SolveResult.skipped("CPSAT_Overflow_Skip")
    
    k_lo_s, k_hi_s = compute_smart_window(instance, tolerance)

    result= solve_bounded_cpsat(instance, k_lo_s, k_hi_s, timeout=timeout, workers=workers)

    # Space reduction
    total  = 2 ** instance.n
    window = search_space_window(instance.n, k_lo_s, k_hi_s)
    red    = 1 - window / total

    result.elapsed = time.perf_counter() - start
    status_str = "Solved" if result.solution is not None else "Timeout"
    result.label = f"CPSAT_Smart_{status_str} [{k_lo_s},{k_hi_s}] red={red:.0%}"
    
    return result


def solve_cpsat_smart_tightened(
    instance: SubsetSumInstance,
    tolerance: int = 3,
    timeout: float = 100.0,
    workers: int = 8
) -> SolveResult:
    """
    CP-SAT with smart window + bound tightening.

    Before solving:
        1. Compute smart window [k_lo_s, k_hi_s]
        2. Run bound tightening → fix some xi=0 or xi=1
        3. CP-SAT solves the reduced problem

    Bound tightening fixes variables that are forced by [k_lo_s, k_hi_s],
    giving CP-SAT a head start.
    """
    start = time.perf_counter()

    # 0. Early exit: no subset can reach T if the total sum is insufficient.
    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)
    
    # 0.1 Early exit if instance cannot be computed with cp sat solver.
    if not instance.fits_int64:
        return SolveResult.skipped("CPSAT_Overflow_Skip")

    k_lo_s, k_hi_s = compute_smart_window(instance, tolerance)
    fixed_zeros, fixed_ones = bound_tightening(instance, k_lo_s, k_hi_s)

    n_fixed = len(fixed_zeros) + len(fixed_ones)

    result = solve_bounded_cpsat(instance, k_lo_s, k_hi_s, fixed_zeros=fixed_zeros, fixed_ones=fixed_ones, timeout=timeout, workers=workers)

    result.elapsed = time.perf_counter() - start
    status_str = "Solved" if result.solution is not None else "Timeout"
    result.label = f"CPSAT_SmartBT_{status_str} [{k_lo_s},{k_hi_s}] fixed={n_fixed}"
    
    return result

