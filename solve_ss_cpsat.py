from ortools.sat.python import cp_model
from constants import TIMEOUT


def solve_subset_sum_cpsat(a: list[int], T: int) -> tuple | None:
    """
    Vanilla CP-SAT solver for the Subset Sum problem.

    Serves as the baseline against which the hybrid LLL+CP-SAT solver
    is benchmarked. Uses no hint or warm-start — pure constraint propagation
    and branch-and-bound.

    Args:
        a: List of positive integers a_1, ..., a_n.
        T: Target sum.

    Returns:
        None if the problem is trivially infeasible (sum(a) < T).
        Otherwise a tuple (time, branches, conflicts, status, solution):
            time      (float):      Wall-clock solve time in seconds.
            branches  (int):        Number of branches explored.
            conflicts (int):        Number of conflicts (backtracks) generated.
            status    (int):        CP-SAT status code (OPTIMAL, FEASIBLE, UNKNOWN).
            solution  (list|None):  The {0,1} solution found, or None if unsolved.
    """
    # Early exit: if the sum of all elements is less than T,
    # no subset can reach T and the problem is trivially infeasible.
    if sum(a) < T:
        return None

    n = len(a)

    model = cp_model.CpModel()
    solver = cp_model.CpSolver()

    # Use multiple workers to leverage parallel search.
    # This is the standard CP-SAT configuration for competitive performance.
    solver.parameters.max_time_in_seconds = TIMEOUT
    solver.parameters.num_search_workers = 8

    # Decision variables: x[i] = 1 if a[i] is included in the subset, 0 otherwise.
    x = [model.new_bool_var(f"x{i}") for i in range(n)]

    # Core constraint: the selected elements must sum exactly to T.
    model.add(sum(a[i] * x[i] for i in range(n)) == T)

    status = solver.solve(model)

    solution = None
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        solution = [solver.value(x[i]) for i in range(n)]
        # Sanity check: verify the solution satisfies the constraint.
        assert sum(a[i] * solution[i] for i in range(n)) == T, (
            f"CP-SAT returned an invalid solution: "
            f"sum={sum(a[i] * solution[i] for i in range(n))} != T={T}"
        )

    return (
        solver.wall_time(),
        solver.num_branches(),
        solver.num_conflicts(),
        status,
        solution,
    )
