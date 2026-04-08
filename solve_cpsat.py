# solve_cpsat.py
"""
Vanilla CP-SAT solver for the Subset Sum problem.

Serves as the baseline against which LLL and BKZ hybrid solvers are benchmarked.
Uses no hint or warm-start — pure constraint propagation and branch-and-bound.
"""

import time
from ortools.sat.python import cp_model
from SubsetSumInstance import SubsetSumInstance
from results import SolveResult
from constants import TIMEOUT


def solve_cpsat(instance: SubsetSumInstance) -> SolveResult | None:
    """
    Vanilla CP-SAT solver for the Subset Sum problem.

    Args:
        instance: A SubsetSumInstance (weights, target).

    Returns:
        None if trivially infeasible (sum(weights) < target).
        Otherwise a SolveResult with timing, search stats, and solution.
    """
    # Early exit: no subset can reach T if the total sum is insufficient.
    if sum(instance.weights) < instance.target:
        return None

    n = instance.n
    start = time.perf_counter()

    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIMEOUT
    solver.parameters.num_search_workers = 8

    x = [model.new_bool_var(f"x{i}") for i in range(n)]
    model.add(sum(instance.weights[i] * x[i] for i in range(n)) == instance.target)

    status = solver.solve(model)
    solved = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    solution = None
    if solved:
        solution = [solver.value(x[i]) for i in range(n)]
        assert instance.is_solution(solution), (
            f"CP-SAT returned an invalid solution: "
            f"sum={instance.residual(solution) + instance.target} != T={instance.target}"
        )

    return SolveResult(
        elapsed=time.perf_counter() - start,
        branches=solver.num_branches,
        conflicts=solver.num_conflicts,
        status=int(status),
        solution=solution,
        label="Solved" if solved else "Timeout",
        best_res=None,
        best_ham=None,
    )
