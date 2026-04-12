"""
Vanilla CP-SAT solver for the Subset Sum problem.
"""

import time
from ortools.sat.python import cp_model
from SubsetSumInstance import SubsetSumInstance
from results import SolveResult

def solve_cpsat(instance: SubsetSumInstance, workers: int = 8, timeout: float = 100.0) -> SolveResult:
    """
    Vanilla CP-SAT solver for the Subset Sum problem.

    Args:
        instance: A SubsetSumInstance (weights, target).

    Returns:
        A SolveResult containing timing, search statistics, and the solution.
        If trivially infeasible (sum(weights) < target), returns early with Infeasible status.
    """
    
    start = time.perf_counter()
    n = instance.n

    # 0. Early exit: no subset can reach T if the total sum is insufficient.
    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)
    
    # 0.1 Early exit if instance cannot be computed with cp sat solver.
    if not instance.fits_int64:
        return SolveResult.skipped("CPSAT_Overflow_Skip")

    # 1. Model initialization

    model = cp_model.CpModel()
    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_search_workers = workers

    # 2. Variables and constraints

    x = [model.new_bool_var(f"x{i}") for i in range(n)]

    model.add(sum(instance.weights[i] * x[i] for i in range(n)) == instance.target)

    # 3. Resolution and verification

    status = solver.solve(model)
    solved = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    solution = None
    if solved:
        label = "Solved"
        solution = [solver.value(x[i]) for i in range(n)]
        
        assert instance.is_solution(solution), (
            f"CP-SAT invalid solution! "
            f"Target={instance.target}, Actual={sum(instance.weights[i] * solution[i] for i in range(n))}"
        )
    elif status == cp_model.INFEASIBLE:
        label = "Infeasible"
    else:
        label = "Timeout"

    # 4. Result formatting

    return SolveResult(
        elapsed=time.perf_counter() - start,
        branches=solver.num_branches,
        conflicts=solver.num_conflicts,
        status=int(status),
        solution=solution,
        label=label,
        best_res=None,
        best_ham=None,
    )
