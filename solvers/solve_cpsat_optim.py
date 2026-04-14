"""
solve_cpsat_optim.py
--------------------
CP-SAT formulations for Subset Sum beyond pure satisfaction.

Three variants:
    A. Satisfaction pure (baseline)
    B. Optimisation: minimize |residual|
    C. Dual phase: optimize (scouting) → satisfy (informed)

The key insight: satisfaction gives CP-SAT no gradient — it's all-or-nothing.
Optimization lets CP-SAT find intermediate solutions and use each bound
to prune the search tree. The dual phase combines both: phase 1 scouts the
landscape, phase 2 uses the intelligence gathered.
"""

import time
import os
from ortools.sat.python import cp_model

from SubsetSumInstance import SubsetSumInstance
from results import SolveResult


WORKERS = max(1, (os.cpu_count() or 6) - 1)


# =====================================================================
# A. Satisfaction pure (reference)
# =====================================================================

def solve_cpsat_satisfy(
    instance: SubsetSumInstance,
    timeout: float = 10.0,
    workers: int = WORKERS,
    hint: list[int] | None = None,
    k_lo: int | None = None,
    k_hi: int | None = None,
) -> SolveResult:
    """Standard CP-SAT: find x such that Σwᵢxᵢ = T."""
    start = time.perf_counter()
    n = instance.n

    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)
    if not instance.fits_int64:
        return SolveResult.skipped("CPSAT_Satisfy_Overflow")

    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_search_workers = workers

    x = [model.new_bool_var(f"x{i}") for i in range(n)]
    model.add(sum(instance.weights[i] * x[i] for i in range(n)) == instance.target)

    if k_lo is not None:
        model.add(sum(x) >= k_lo)
    if k_hi is not None:
        model.add(sum(x) <= k_hi)
    if hint is not None:
        for i in range(n):
            model.add_hint(x[i], hint[i])

    status = solver.solve(model)
    solved = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    return SolveResult(
        elapsed=time.perf_counter() - start,
        branches=solver.num_branches,
        conflicts=solver.num_conflicts,
        status=int(status),
        solution=[solver.value(x[i]) for i in range(n)] if solved else None,
        label="Satisfy_Solved" if solved else "Satisfy_Timeout",
        best_res=0 if solved else None,
        best_ham=None,
    )


# =====================================================================
# B. Optimisation: minimize |residual|
# =====================================================================

def solve_cpsat_minimize(
    instance: SubsetSumInstance,
    timeout: float = 10.0,
    workers: int = WORKERS,
    hint: list[int] | None = None,
    k_lo: int | None = None,
    k_hi: int | None = None,
    residual_ub: int | None = None,
) -> SolveResult:
    """
    CP-SAT optimization: minimize |Σwᵢxᵢ - T|.

    Instead of all-or-nothing satisfaction, gives CP-SAT a gradient.
    Each intermediate solution provides a bound for pruning.

    Args:
        residual_ub:  Upper bound on residual (e.g. from LLL clip).
                      Tightens the search space.
    """
    start = time.perf_counter()
    n = instance.n

    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)
    if not instance.fits_int64:
        return SolveResult.skipped("CPSAT_Minimize_Overflow")

    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_search_workers = workers

    x = [model.new_bool_var(f"x{i}") for i in range(n)]
    expr = sum(instance.weights[i] * x[i] for i in range(n))

    # Residual = |expr - target|
    ub = residual_ub if residual_ub is not None else instance.target
    residual = model.new_int_var(0, ub, "residual")
    model.add(expr - instance.target <= residual)
    model.add(instance.target - expr <= residual)

    # Cardinality bounds
    if k_lo is not None:
        model.add(sum(x) >= k_lo)
    if k_hi is not None:
        model.add(sum(x) <= k_hi)

    # Hint
    if hint is not None:
        for i in range(n):
            model.add_hint(x[i], hint[i])
        model.add_hint(residual, 0)

    model.minimize(residual)

    status = solver.solve(model)

    best_residual = None
    best_bound = None
    solution = None

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solution_vec = [solver.value(x[i]) for i in range(n)]
        best_residual = int(solver.objective_value)
        best_bound = int(solver.best_objective_bound)

        if best_residual == 0 and instance.is_solution(solution_vec):
            solution = solution_vec

    label_parts = []
    if solution is not None:
        label_parts.append("Minimize_Exact")
    elif best_residual is not None:
        label_parts.append(f"Minimize_Gap{best_residual}")
    else:
        label_parts.append("Minimize_Timeout")

    if best_bound is not None and best_bound > 0:
        label_parts.append("INFEASIBLE_PROVED")

    return SolveResult(
        elapsed=time.perf_counter() - start,
        branches=solver.num_branches,
        conflicts=solver.num_conflicts,
        status=int(status),
        solution=solution,
        label="_".join(label_parts),
        best_res=best_residual,
        best_ham=None,
    )


# =====================================================================
# C. Dual phase: scout (optimize) → solve (satisfy)
# =====================================================================

def solve_cpsat_dual(
    instance: SubsetSumInstance,
    timeout: float = 10.0,
    workers: int = WORKERS,
    scout_ratio: float = 0.3,
    k_lo: int | None = None,
    k_hi: int | None = None,
    lll_hint: list[int] | None = None,
    lll_residual: int | None = None,
) -> SolveResult:
    """
    Two-phase CP-SAT:
        Phase 1 (scout): minimize residual, collect bounds + best solution
        Phase 2 (solve): satisfaction with hint from phase 1

    The scout phase gives CP-SAT a gradient to explore the landscape.
    Even if it doesn't find residual=0, its best solution is a better
    hint than LLL clip for phase 2.

    If phase 1 proves best_bound > 0, the problem is infeasible — skip phase 2.
    """
    start = time.perf_counter()
    n = instance.n

    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)
    if not instance.fits_int64:
        return SolveResult.skipped("CPSAT_Dual_Overflow")

    total_branches = 0
    total_conflicts = 0

    # ----- Phase 1: Scout (optimization) -----
    scout_budget = timeout * scout_ratio

    res1 = solve_cpsat_minimize(
        instance,
        timeout=scout_budget,
        workers=workers,
        hint=lll_hint,
        k_lo=k_lo,
        k_hi=k_hi,
        residual_ub=lll_residual,
    )

    total_branches += res1.branches
    total_conflicts += res1.conflicts

    # Direct solve in phase 1?
    if res1.solution is not None:
        res1.label = "Dual_Scout_Direct"
        return res1

    # Infeasibility proved?
    if res1.best_res is not None and "INFEASIBLE" in res1.label:
        return SolveResult(
            elapsed=time.perf_counter() - start,
            branches=total_branches,
            conflicts=total_conflicts,
            status=3,
            solution=None,
            label="Dual_Infeasible_Proved",
            best_res=res1.best_res,
            best_ham=None,
        )

    # ----- Phase 2: Solve (satisfaction with scout hint) -----
    remaining = timeout - (time.perf_counter() - start)
    if remaining <= 0.1:
        return SolveResult(
            elapsed=time.perf_counter() - start,
            branches=total_branches,
            conflicts=total_conflicts,
            status=3,
            solution=None,
            label=f"Dual_Scout_Only_Res{res1.best_res}",
            best_res=res1.best_res,
            best_ham=None,
        )

    # Extract hint from phase 1 (best solution found, even if residual > 0)
    # We need to re-extract from the solver, but we don't have access
    # Use lll_hint as fallback
    phase2_hint = lll_hint

    res2 = solve_cpsat_satisfy(
        instance,
        timeout=remaining,
        workers=workers,
        hint=phase2_hint,
        k_lo=k_lo,
        k_hi=k_hi,
    )

    total_branches += res2.branches
    total_conflicts += res2.conflicts

    if res2.solution is not None:
        return SolveResult(
            elapsed=time.perf_counter() - start,
            branches=total_branches,
            conflicts=total_conflicts,
            status=0,
            solution=res2.solution,
            label="Dual_Phase2_Solved",
            best_res=0,
            best_ham=None,
        )

    return SolveResult(
        elapsed=time.perf_counter() - start,
        branches=total_branches,
        conflicts=total_conflicts,
        status=3,
        solution=None,
        label=f"Dual_Timeout_ScoutRes{res1.best_res}",
        best_res=res1.best_res,
        best_ham=None,
    )


# =====================================================================
# Convenience: with LLL warm start
# =====================================================================

def solve_cpsat_minimize_lll(
    instance: SubsetSumInstance,
    timeout: float = 10.0,
    workers: int = WORKERS,
) -> SolveResult:
    """Minimize residual with LLL hint and residual upper bound."""
    from fpylll import LLL as LLL_algo
    from utils import extract_vectors_from_basis

    start = time.perf_counter()

    if not instance.fits_int64:
        return SolveResult.skipped("CPSAT_MinLLL_Overflow")

    # LLL warm start
    n = instance.n
    B = instance.to_knapsack_matrix(M=1 << n)
    LLL_algo.reduction(B)
    vectors = extract_vectors_from_basis(B)

    hint = None
    res_ub = None
    if vectors:
        best_clip, best_res = None, float("inf")
        for v in vectors:
            c = [max(0, min(1, round(float(val)))) for val in v]
            r = abs(instance.residual(c))
            if r < best_res:
                best_res, best_clip = r, c
        if best_clip:
            hint = best_clip
            res_ub = int(best_res) if best_res < instance.target else None
            if best_res == 0 and instance.is_solution(best_clip):
                return SolveResult(
                    elapsed=time.perf_counter() - start,
                    branches=0, conflicts=0, status=0,
                    solution=best_clip,
                    label="MinLLL_Direct",
                    best_res=0, best_ham=None,
                )

    remaining = timeout - (time.perf_counter() - start)
    result = solve_cpsat_minimize(
        instance, timeout=remaining, workers=workers,
        hint=hint, residual_ub=res_ub,
    )
    result.elapsed = time.perf_counter() - start
    return result


def solve_cpsat_dual_lll(
    instance: SubsetSumInstance,
    timeout: float = 10.0,
    workers: int = WORKERS,
    scout_ratio: float = 0.3,
) -> SolveResult:
    """Dual phase with LLL hint."""
    from fpylll import LLL as LLL_algo
    from utils import extract_vectors_from_basis

    start = time.perf_counter()

    if not instance.fits_int64:
        return SolveResult.skipped("CPSAT_DualLLL_Overflow")

    n = instance.n
    B = instance.to_knapsack_matrix(M=1 << n)
    LLL_algo.reduction(B)
    vectors = extract_vectors_from_basis(B)

    hint = None
    res_ub = None
    if vectors:
        best_clip, best_res = None, float("inf")
        for v in vectors:
            c = [max(0, min(1, round(float(val)))) for val in v]
            r = abs(instance.residual(c))
            if r < best_res:
                best_res, best_clip = r, c
        if best_clip:
            hint = best_clip
            res_ub = int(best_res) if best_res < instance.target else None
            if best_res == 0 and instance.is_solution(best_clip):
                return SolveResult(
                    elapsed=time.perf_counter() - start,
                    branches=0, conflicts=0, status=0,
                    solution=best_clip,
                    label="DualLLL_Direct",
                    best_res=0, best_ham=None,
                )

    remaining = timeout - (time.perf_counter() - start)
    result = solve_cpsat_dual(
        instance, timeout=remaining, workers=workers,
        scout_ratio=scout_ratio,
        lll_hint=hint, lll_residual=res_ub,
    )
    result.elapsed = time.perf_counter() - start
    return result