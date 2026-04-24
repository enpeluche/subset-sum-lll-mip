"""
solve_cpsat_minimize.py
-----------------------
CP-SAT optimization formulation: minimize |Σwᵢxᵢ - T|.

Gives CP-SAT a gradient instead of all-or-nothing satisfaction.
Best when weight bits are 13-18 (empirically validated).

    solve_cpsat_minimize  — pure optimization
    solve_cpsat_dual      — scout (optimize 30%) → solve (satisfy 70%)

Both accept hint and residual_ub from lattice reduction.
No internal LLL — the caller provides hints.
"""

import os
import time

from ortools.sat.python import cp_model

from SubsetSumInstance import SubsetSumInstance
from results import SolveResult
from solvers.cpsat.solve_cpsat import _cpsat_guard, solve_cpsat, WORKERS


# =====================================================================
# Minimize |residual|
# =====================================================================

def solve_cpsat_minimize(
    instance: SubsetSumInstance,
    hint: list[int] | None = None,
    residual_ub: int | None = None,
    k_lo: int | None = None,
    k_hi: int | None = None,
    timeout: float = 10.0,
    workers: int = WORKERS,
) -> SolveResult:
    """
    Minimize |Σwᵢxᵢ - T|.

    Args:
        hint:         Starting point (e.g. lattice clip).
        residual_ub:  Upper bound on residual (from lattice). Tightens search.
        k_lo, k_hi:   Optional cardinality bounds.
    """
    start = time.perf_counter()

    guard = _cpsat_guard(instance, start)
    if guard:
        return guard

    n = instance.n
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_search_workers = workers

    x = [model.new_bool_var(f"x{i}") for i in range(n)]
    expr = sum(instance.weights[i] * x[i] for i in range(n))

    # Residual variable with optional upper bound from lattice
    ub = residual_ub if residual_ub is not None else instance.target
    residual = model.new_int_var(0, ub, "residual")
    model.add(expr - instance.target <= residual)
    model.add(instance.target - expr <= residual)

    if k_lo is not None:
        model.add(sum(x) >= k_lo)
    if k_hi is not None:
        model.add(sum(x) <= k_hi)

    if hint:
        for i in range(n):
            model.add_hint(x[i], hint[i])
        model.add_hint(residual, 0)

    model.minimize(residual)
    status = solver.solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solution_vec = [solver.value(x[i]) for i in range(n)]
        obj_val = int(solver.objective_value)
        obj_bound = int(solver.best_objective_bound)

        # Exact solution found
        if obj_val == 0 and instance.is_solution(solution_vec):
            return SolveResult.found(
                elapsed=time.perf_counter() - start,
                solution=solution_vec,
                label="Minimize_Exact",
                branches=solver.num_branches,
                conflicts=solver.num_conflicts,
                hamming_to_ground_solution=instance.hamming_to_solution(solution_vec),
            )

        # Infeasibility proved (best possible residual > 0)
        label = f"Minimize_Gap{obj_val}"
        if obj_bound > 0:
            label += "_InfeasibleProved"

        return SolveResult(
            elapsed=time.perf_counter() - start,
            status=int(status),
            label=label,
            solution=None,
            branches=solver.num_branches,
            conflicts=solver.num_conflicts,
            smallest_residual=obj_val,
            hint=solution_vec,  # best candidate even if residual > 0
        )

    return SolveResult(
        elapsed=time.perf_counter() - start,
        status=int(status),
        label="Minimize_Timeout",
        solution=None,
        branches=solver.num_branches,
        conflicts=solver.num_conflicts,
    )


# =====================================================================
# Dual phase: scout (minimize) → solve (satisfy)
# =====================================================================

def solve_cpsat_dual(
    instance: SubsetSumInstance,
    hint: list[int] | None = None,
    residual_ub: int | None = None,
    k_lo: int | None = None,
    k_hi: int | None = None,
    scout_ratio: float = 0.3,
    timeout: float = 10.0,
    workers: int = WORKERS,
) -> SolveResult:
    """
    Two-phase CP-SAT:
        Phase 1 (30% budget): minimize residual → collect bounds + best hint
        Phase 2 (70% budget): satisfy with hint from phase 1

    The scout phase gives CP-SAT a gradient. Even if it doesn't reach
    residual=0, its best solution is a better hint for phase 2.
    """
    start = time.perf_counter()

    guard = _cpsat_guard(instance, start)
    if guard:
        return guard

    total_branches = 0
    total_conflicts = 0

    # --- Phase 1: Scout ---
    scout_budget = timeout * scout_ratio
    scout = solve_cpsat_minimize(
        instance,
        hint=hint,
        residual_ub=residual_ub,
        k_lo=k_lo, k_hi=k_hi,
        timeout=scout_budget,
        workers=workers,
    )
    total_branches += scout.branches
    total_conflicts += scout.conflicts

    if scout.solution is not None:
        scout.label = "Dual_Scout_Direct"
        return scout

    # Use scout's best candidate as hint for phase 2
    phase2_hint = scout.hint or hint

    # --- Phase 2: Satisfy ---
    remaining = timeout - (time.perf_counter() - start)
    if remaining <= 0.1:
        return SolveResult(
            elapsed=time.perf_counter() - start,
            status=3,
            label=f"Dual_ScoutOnly_Res{scout.smallest_residual}",
            solution=None,
            branches=total_branches,
            conflicts=total_conflicts,
            smallest_residual=scout.smallest_residual,
            hint=phase2_hint,
        )

    satisfy = solve_cpsat(
        instance,
        hint=phase2_hint,
        timeout=remaining,
        workers=workers,
    )
    total_branches += satisfy.branches
    total_conflicts += satisfy.conflicts

    if satisfy.solution is not None:
        return SolveResult.found(
            elapsed=time.perf_counter() - start,
            solution=satisfy.solution,
            label="Dual_Phase2_Solved",
            branches=total_branches,
            conflicts=total_conflicts,
            hamming_to_ground_solution=instance.hamming_to_solution(satisfy.solution),
        )

    return SolveResult(
        elapsed=time.perf_counter() - start,
        status=3,
        label=f"Dual_Timeout_Res{scout.smallest_residual}",
        solution=None,
        branches=total_branches,
        conflicts=total_conflicts,
        smallest_residual=scout.smallest_residual,
        hint=phase2_hint,
    )