"""
solve_cpsat.py
--------------
Core CP-SAT solvers for Subset Sum.

    solve_cpsat          — vanilla: just Σwᵢxᵢ = T
    solve_cpsat_bounded  — with cardinality bounds [k_lo, k_hi] and optional fixing

Both accept an optional hint (e.g. from lattice reduction).
All CP-SAT solvers share the same guards (trivially infeasible, int64 overflow).
"""

import os
import time

from ortools.sat.python import cp_model

from SubsetSumInstance import SubsetSumInstance
from results import SolveResult

WORKERS = max(1, (os.cpu_count() or 6) // 2)


# =====================================================================
# Guards (shared by all CP-SAT solvers)
# =====================================================================

def _cpsat_guard(instance: SubsetSumInstance, start: float) -> SolveResult | None:
    """Return a SolveResult if we should skip, None if we can proceed."""
    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)
    if not instance.fits_int64:
        return SolveResult.skipped("CPSAT_Overflow")
    return None


# =====================================================================
# Vanilla CP-SAT
# =====================================================================

def solve_cpsat(
    instance: SubsetSumInstance,
    hint: list[int] | None = None,
    timeout: float = 10.0,
    workers: int = WORKERS,
) -> SolveResult:
    """Vanilla CP-SAT: find x such that Σwᵢxᵢ = T."""
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
    model.add(sum(instance.weights[i] * x[i] for i in range(n)) == instance.target)

    if hint:
        for i in range(n):
            model.add_hint(x[i], hint[i])

    status = solver.solve(model)
    solved = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    if solved:
        solution = [solver.value(x[i]) for i in range(n)]
        return SolveResult.found(
            elapsed=time.perf_counter() - start,
            solution=solution,
            label="CPSAT_Solved",
            branches=solver.num_branches,
            conflicts=solver.num_conflicts,
            hamming_to_ground_solution=instance.hamming_to_solution(solution),
        )

    return SolveResult(
        elapsed=time.perf_counter() - start,
        status=int(status),
        label="CPSAT_Infeasible" if status == cp_model.INFEASIBLE else "CPSAT_Timeout",
        solution=None,
        branches=solver.num_branches,
        conflicts=solver.num_conflicts,
    )


# =====================================================================
# Bounded CP-SAT (with cardinality constraints + optional fixing)
# =====================================================================

def solve_cpsat_bounded(
    instance: SubsetSumInstance,
    k_lo: int,
    k_hi: int,
    hint: list[int] | None = None,
    fixed_zeros: list[int] | None = None,
    fixed_ones: list[int] | None = None,
    timeout: float = 10.0,
    workers: int = WORKERS,
) -> SolveResult:
    """CP-SAT with cardinality bounds k_lo ≤ Σxᵢ ≤ k_hi and optional variable fixing."""
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
    model.add(sum(instance.weights[i] * x[i] for i in range(n)) == instance.target)
    model.add(sum(x) >= k_lo)
    model.add(sum(x) <= k_hi)

    if fixed_zeros:
        for i in fixed_zeros:
            model.add(x[i] == 0)
    if fixed_ones:
        for i in fixed_ones:
            model.add(x[i] == 1)
    if hint:
        for i in range(n):
            model.add_hint(x[i], hint[i])

    status = solver.solve(model)
    solved = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    if solved:
        solution = [solver.value(x[i]) for i in range(n)]
        return SolveResult.found(
            elapsed=time.perf_counter() - start,
            solution=solution,
            label=f"CPSAT_Bounded[{k_lo},{k_hi}]",
            branches=solver.num_branches,
            conflicts=solver.num_conflicts,
            hamming_to_ground_solution=instance.hamming_to_solution(solution),
        )

    return SolveResult(
        elapsed=time.perf_counter() - start,
        status=int(status),
        label=f"CPSAT_Bounded_Timeout[{k_lo},{k_hi}]",
        solution=None,
        branches=solver.num_branches,
        conflicts=solver.num_conflicts,
    )