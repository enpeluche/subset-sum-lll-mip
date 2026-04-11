from solvers.solve_greedy_cpsat import solve_cpsat_greedy_bound, solve_cpsat_smart_window, solve_cpsat_smart_tightened
from solvers.solve_greedy_extreme import solve_greedy_extreme

from SubsetSumInstance import SubsetSumInstance
from results import SolveResult
import time

# voir pour utiliser solve_cpsat_greedy_bound

def solve_full_greedy(
    instance: SubsetSumInstance,
    max_subsets_extreme: int = 2_000_000,
    tolerance_smart: int = 3,
    timeout: float = 100.0,
    workers: int = 8
) -> SolveResult:
    """
    Full greedy pipeline:
        1. Greedy extreme (symmetric) — if C(n,k_lo/k_hi) ≤ max_subsets
        2. CP-SAT smart + bound tightening — fallback

    This is the complete approach combining all techniques.
    """
    start = time.perf_counter()
    t_lim = timeout if timeout is not None else timeout

    # Step 1: Extreme enumeration (symmetric)
    remaining = t_lim - (time.perf_counter() - start)
    if remaining > 0:
        r_extreme = solve_greedy_extreme(
            instance,
            max_subsets=max_subsets_extreme,
            n_k_near_extremes=3,
            timeout=remaining,
            workers = workers
        )
        if r_extreme.solution is not None:
            return r_extreme

    # Step 2: CP-SAT smart + bound tightening
    remaining = t_lim - (time.perf_counter() - start)
    if remaining > 0.1:
        r_smart = solve_cpsat_smart_tightened(
            instance,
            tolerance=tolerance_smart,
            timeout=remaining,
        )
        if r_smart.solution is not None:
            return r_smart

    return SolveResult(
        elapsed=time.perf_counter() - start,
        branches=0, conflicts=0, status=3,
        solution=None, label="Full_Greedy_NotFound",
        best_res=None, best_ham=None,
    )