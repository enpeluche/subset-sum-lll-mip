import time
from ortools.sat.python import cp_model
from SubsetSumInstance import SubsetSumInstance
from results import SolveResult


def solve_bounded_cpsat(
    instance: SubsetSumInstance,
    k_lo: int,
    k_hi: int,
    fixed_zeros: list[int] | None = None,
    fixed_ones:  list[int] | None = None,
    timeout: float = 100.0,
    workers: int = 8,
    label="Bounded_CPSAT"
) -> SolveResult:
    """Build and solve a CP-SAT model with weight constraints."""
    
    start = time.perf_counter()
    n = instance.n
    weights = instance.weights
    T       = instance.target

    # 0. Early exit: no subset can reach T if the total sum is insufficient.
    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)
    
    # 1. Model initialization

    model  = cp_model.CpModel()
    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_search_workers  = workers

    # 2. Variables and constraints
    x = [model.new_bool_var(f"x{i}") for i in range(n)]

    model.add(sum(weights[i] * x[i] for i in range(n)) == T)

    model.add(sum(x) >= k_lo)
    model.add(sum(x) <= k_hi)

    # Tightening

    if fixed_zeros:
        for i in fixed_zeros:
            model.add(x[i] == 0)
    if fixed_ones:
        for i in fixed_ones:
            model.add(x[i] == 1)

    status = solver.solve(model)
    elapsed = time.perf_counter() - start
    solved = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    return SolveResult(
        elapsed=elapsed,
        branches=solver.num_branches,
        conflicts=solver.num_conflicts,
        status=int(status),
        solution=[solver.value(x[i]) for i in range(instance.n)] if solved else None,
        label=label,
        best_res=0 if solved else None,
        best_ham=0 if solved else None
    )