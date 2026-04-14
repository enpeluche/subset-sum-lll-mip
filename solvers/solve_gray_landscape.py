"""
solve_gray_landscape.py
-----------------------
Gray code walk with landscape analysis → CP-SAT with informed hints.

Two-phase solver:
    1. Gray walk: systematic exploration of {0,1}^n from LLL warm start.
       Collects per-bit residual statistics (mean-field approximation).
    2. CP-SAT probe: uses landscape stats as hints + optional variable fixing.

The walk visits x_init ⊕ gray(1), x_init ⊕ gray(2), ...
This is a shifted Hamiltonian path — bijection on the hypercube.
For n ≤ 21 with 2M iterations, the walk is exhaustive (exact solver).

Key insight: for each bit j, we track the average |residual| when j=0 vs j=1
across all visited configurations. A large gap means the landscape "prefers"
one value — this is a marginal signal we can pass to CP-SAT as a hint.
"""

import time
import math
import random

from ortools.sat.python import cp_model
from fpylll import LLL

from SubsetSumInstance import SubsetSumInstance
from results import SolveResult
from utils import extract_vectors_from_basis, filter_binary_vectors


# =====================================================================
# Gray code helpers
# =====================================================================

def _gray_bit(i: int) -> int:
    """Bit index that flips between Gray(i-1) and Gray(i)."""
    return (i & -i).bit_length() - 1


def _gray_budget(n: int, max_iter: int = 2_000_000) -> int:
    """Full enumeration if feasible, capped otherwise."""
    return min((1 << n) - 1, max_iter)


# =====================================================================
# LLL warm start
# =====================================================================

def _lll_warm_start(instance: SubsetSumInstance) -> list[int]:
    """LLL reduction → clip best vector to {0,1}."""
    n = instance.n
    scaling = 1 << n
    B = instance.to_knapsack_matrix(M=scaling)
    LLL.reduction(B)
    vectors = extract_vectors_from_basis(B)

    if not vectors:
        return [random.randint(0, 1) for _ in range(n)]

    # Direct binary solution?
    for v in filter_binary_vectors(vectors):
        if instance.is_solution(v):
            return v

    # Clip best by residual
    best_x, best_res = None, float("inf")
    for v in vectors:
        x = [max(0, min(1, round(float(c)))) for c in v]
        res = abs(instance.residual(x))
        if res < best_res:
            best_res, best_x = res, x

    return best_x or [random.randint(0, 1) for _ in range(n)]


# =====================================================================
# Phase 1: Gray walk with landscape stats
# =====================================================================

def _gray_walk_landscape(
    instance: SubsetSumInstance,
    x_init: list[int],
    n_iter: int,
    deadline: float,
) -> tuple[list[int] | None, int, int, dict]:
    """
    Gray code walk collecting per-bit residual statistics.

    For each bit j, tracks:
        - sum of |residual| when x[j] == 0, and count
        - sum of |residual| when x[j] == 1, and count
        - the residual at the best configuration seen

    Returns:
        (solution_or_None, branches, best_residual, landscape_stats)
    """
    n = instance.n
    w = instance.weights

    x = x_init.copy()
    residual = instance.residual(x)

    best_x = x.copy()
    best_res = abs(residual)
    branches = 0

    # Per-bit landscape accumulators
    sum_res_as_0 = [0.0] * n
    sum_res_as_1 = [0.0] * n
    count_as_0 = [0] * n
    count_as_1 = [0] * n

    # Min residual seen with each bit value
    min_res_as_0 = [float("inf")] * n
    min_res_as_1 = [float("inf")] * n

    limit = min(n_iter, (1 << n) - 1)

    for it in range(1, limit + 1):
        if time.perf_counter() > deadline:
            break

        j = _gray_bit(it)
        if j >= n:
            break

        branches += 1

        # Flip
        if x[j] == 0:
            x[j] = 1
            residual += w[j]
        else:
            x[j] = 0
            residual -= w[j]

        # Track landscape for the flipped bit — O(1) per step
        ar = abs(residual)
        if x[j] == 0:
            sum_res_as_0[j] += ar
            count_as_0[j] += 1
            min_res_as_0[j] = min(min_res_as_0[j], ar)
        else:
            sum_res_as_1[j] += ar
            count_as_1[j] += 1
            min_res_as_1[j] = min(min_res_as_1[j], ar)

        # Update best
        if ar < best_res:
            best_res = ar
            best_x = x.copy()

        if residual == 0:
            return x, branches, 0, {}

    # Compile landscape stats
    stats = {
        "sum_res_as_0": sum_res_as_0,
        "sum_res_as_1": sum_res_as_1,
        "count_as_0": count_as_0,
        "count_as_1": count_as_1,
        "min_res_as_0": min_res_as_0,
        "min_res_as_1": min_res_as_1,
        "best_x": best_x,
    }

    return (best_x if best_res == 0 else None), branches, best_res, stats


# =====================================================================
# Phase 2: CP-SAT with landscape hints
# =====================================================================

def _extract_hints(stats: dict, n: int, confidence_threshold: float = 0.3):
    """
    Extract per-bit hints from landscape statistics.

    For each bit j, compare avg |residual| when j=0 vs j=1.
    If one is significantly lower (ratio < threshold), hint that value.

    Also returns high-confidence bits that could be fixed (not just hinted).

    Returns:
        hints:  list of (bit_index, preferred_value)
        fixes:  list of (bit_index, value) for very high confidence bits
    """
    hints = []
    fixes = []

    for j in range(n):
        c0 = stats["count_as_0"][j]
        c1 = stats["count_as_1"][j]

        if c0 < 3 or c1 < 3:
            # Not enough data — use best_x as fallback hint
            hints.append((j, stats["best_x"][j]))
            continue

        avg_0 = stats["sum_res_as_0"][j] / c0
        avg_1 = stats["sum_res_as_1"][j] / c1

        # Ratio: how much better is the preferred value?
        worse = max(avg_0, avg_1)
        if worse < 1e-10:
            continue

        ratio = min(avg_0, avg_1) / worse
        preferred = 0 if avg_0 < avg_1 else 1

        if ratio < 0.1:
            # Very high confidence — candidate for fixing
            fixes.append((j, preferred))
            hints.append((j, preferred))
        elif ratio < confidence_threshold:
            hints.append((j, preferred))
        else:
            # Ambiguous — fall back to best_x value
            hints.append((j, stats["best_x"][j]))

    return hints, fixes


def _cpsat_with_landscape(
    instance: SubsetSumInstance,
    hints: list[tuple[int, int]],
    fixes: list[tuple[int, int]],
    best_x: list[int],
    timeout: float,
    workers: int,
    fix_variables: bool = False,
) -> tuple[list[int] | None, int, int]:
    """
    CP-SAT probe with landscape-informed hints and optional fixing.

    Returns: (solution, branches, conflicts)
    """
    n = instance.n

    

    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_search_workers = workers

    x = [model.new_bool_var(f"x{i}") for i in range(n)]
    model.add(sum(instance.weights[i] * x[i] for i in range(n)) == instance.target)

    # Hints from landscape analysis
    for j, val in hints:
        model.add_hint(x[j], val)

    # Fix high-confidence variables (optional, aggressive)
    if fix_variables and fixes:
        for j, val in fixes:
            model.add(x[j] == val)

    status = solver.solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sol = [solver.value(x[i]) for i in range(n)]
        return sol, solver.num_branches, solver.num_conflicts

    return None, solver.num_branches, solver.num_conflicts


# =====================================================================
# Main solver
# =====================================================================

def solve_gray_landscape(
    instance: SubsetSumInstance,
    max_iter: int = 2_000_000,
    confidence: float = 0.3,
    fix_variables: bool = False,
    timeout: float = 10.0,
    workers: int = 8,
) -> SolveResult:
    """
    Two-phase solver: Gray walk landscape → CP-SAT with hints.

    Phase 1: Gray code walk from LLL warm start. Collects per-bit
             mean-field statistics over all visited configurations.
             Exact for n ≤ 21 (~2M iterations).

    Phase 2: CP-SAT with landscape hints. High-confidence bits
             from the walk are hinted (or optionally fixed).

    Args:
        instance:       SubsetSumInstance to solve.
        max_iter:       Gray walk iteration cap (default 2M).
        confidence:     Threshold for landscape hints (default 0.3).
        fix_variables:  If True, fix very-high-confidence bits in CP-SAT.
        timeout:        Total time budget in seconds.
        workers:        CP-SAT worker threads.
    """
    start = time.perf_counter()
    n = instance.n

    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)
    

    # --- Phase 0: LLL warm start ---
    x_init = _lll_warm_start(instance)

    if instance.is_solution(x_init):
        return SolveResult(
            elapsed=time.perf_counter() - start,
            branches=0, conflicts=0, status=0,
            solution=x_init,
            label="GrayLandscape_LLL_Direct",
            best_res=0,
            best_ham=instance.hamming_to_solution(x_init) if hasattr(instance, 'hamming_to_ground_truth') else instance.hamming_to_solution(x_init),
        )

    # --- Phase 1: Gray walk ---
    budget_walk = min(timeout * 0.4, 3.0)  # max 40% of budget or 3s
    walk_deadline = time.perf_counter() + budget_walk
    n_iter = _gray_budget(n, max_iter)

    solution, branches, best_res, stats = _gray_walk_landscape(
        instance, x_init, n_iter, walk_deadline,
    )

    if solution is not None:
        is_exhaustive = n_iter >= (1 << n) - 1
        return SolveResult(
            elapsed=time.perf_counter() - start,
            branches=branches, conflicts=0, status=0,
            solution=solution,
            label=f"GrayLandscape_Walk_{'Exact' if is_exhaustive else 'Lucky'}",
            best_res=0,
            best_ham=instance.hamming_to_solution(solution),
        )

    # Phase 2: CP-SAT — seulement si les entiers tiennent
    remaining = timeout - (time.perf_counter() - start)
    if remaining <= 0.1 or not instance.fits_int64:
        return SolveResult(
            elapsed=time.perf_counter() - start,
            branches=branches, conflicts=0, status=3,
            solution=None,
            label="GrayLandscape_Walk_Only" if not instance.fits_int64 else "GrayLandscape_Walk_Timeout",
            best_res=best_res, best_ham=None,
        )
    
    # --- Phase 2: CP-SAT with landscape hints ---
    remaining = timeout - (time.perf_counter() - start)
    if remaining <= 0.1:
        return SolveResult(
            elapsed=time.perf_counter() - start,
            branches=branches, conflicts=0, status=3,
            solution=None,
            label="GrayLandscape_Walk_Timeout",
            best_res=best_res, best_ham=None,
        )

    hints, fixes = _extract_hints(stats, n, confidence)
    n_fixes = len(fixes) if fix_variables else 0
    best_x = stats["best_x"]

    sol, cpsat_branches, cpsat_conflicts = _cpsat_with_landscape(
        instance, hints, fixes, best_x,
        timeout=remaining,
        workers=workers,
        fix_variables=fix_variables,
    )

    total_branches = branches + cpsat_branches
    label_suffix = f"_fixed{n_fixes}" if fix_variables and n_fixes > 0 else ""

    if sol is not None and instance.is_solution(sol):
        return SolveResult(
            elapsed=time.perf_counter() - start,
            branches=total_branches,
            conflicts=cpsat_conflicts,
            status=0,
            solution=sol,
            label=f"GrayLandscape_CPSAT{label_suffix}",
            best_res=0,
            best_ham=instance.hamming_to_solution(sol),
        )

    return SolveResult(
        elapsed=time.perf_counter() - start,
        branches=total_branches,
        conflicts=cpsat_conflicts,
        status=3,
        solution=None,
        label=f"GrayLandscape_NotFound{label_suffix}",
        best_res=best_res, best_ham=None,
    )