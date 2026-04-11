
from results import SolveResult
from SubsetSumInstance import SubsetSumInstance
import time
from constraints.range_constraints import compute_greedy_bounds
from utils import enum_cost

from solvers.enumeration import _enum_symmetric

def solve_greedy_extreme(
    instance: SubsetSumInstance,
    max_subsets: int = 2_000_000,
    n_k_near_extremes: int = 3,
    workers: int = 8,
    timeout: float = 100.0,
) -> SolveResult:
    """
    Enumerate subsets near k_lo and k_hi using symmetric enumeration.

    Strategy:
        1. Compute [k_lo, k_hi] in O(n log n)
        2. Collect k values near k_lo and k_hi
        3. Sort by enum_cost(n,k) = min(C(n,k), C(n,n-k))
        4. Enumerate cheapest first, skip if > max_subsets

    Uses symmetric enumeration: always takes the cheaper side
    (enumerate ones if k ≤ n/2, zeros if k > n/2).

    Args:
        instance:          A SubsetSumInstance.
        max_subsets:       Skip k values with cost > max_subsets.
        n_k_near_extremes: Number of k values to try near each extreme.
        timeout:           Max solve time.

    Returns:
        SolveResult with label 'Greedy_Extreme_k{k}' or
        'Greedy_Extreme_Skip' or 'Greedy_Extreme_NotFound'.
    """
    start   = time.perf_counter()
    n       = instance.n
    t_limit = timeout if timeout is not None else TIMEOUT

    k_lo, k_hi = compute_greedy_bounds(instance)
    total_branches = 0

    # Build candidate k values sorted by symmetric cost
    k_candidates = []

    for offset in range(n_k_near_extremes):
        for k in [k_lo + offset, k_hi - offset]:
            if 0 < k <= n:
                cost = enum_cost(n, k)
                k_candidates.append((cost, k))

    # Deduplicate and sort by cost
    seen = set()
    k_candidates_unique = []
    for cost, k in sorted(k_candidates):
        if k not in seen:
            seen.add(k)
            k_candidates_unique.append((cost, k))

    # Filter by max_subsets
    feasible = [(cost, k) for cost, k in k_candidates_unique
                if cost <= max_subsets]

    if not feasible:
        min_cost = min(cost for cost, _ in k_candidates_unique) if k_candidates_unique else 0
        return SolveResult(
            elapsed=time.perf_counter() - start,
            branches=0, conflicts=0, status=3,
            solution=None,
            label=f"Greedy_Extreme_Skip "
                  f"(k_lo={k_lo}, k_hi={k_hi}, min_cost={min_cost:,})",
            best_res=None, best_ham=None,
        )

    for cost, k in feasible:
        total_branches += cost
        sol = _enum_symmetric(instance, k, start, t_limit)
        if sol is not None and instance.is_solution(sol):
            return SolveResult(
                elapsed=time.perf_counter() - start,
                branches=total_branches,
                conflicts=0, status=0,
                solution=sol,
                label=f"Greedy_Extreme_k{k}_cost{cost:,}",
                best_res=0,
                best_ham=instance.hamming_to_solution(sol),
            )

    return SolveResult(
        elapsed=time.perf_counter() - start,
        branches=total_branches,
        conflicts=0, status=3,
        solution=None,
        label="Greedy_Extreme_NotFound",
        best_res=None, best_ham=None,
    )