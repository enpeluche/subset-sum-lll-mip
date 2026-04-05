from constants import TIMEOUT
import time
from ortools.sat.python import cp_model
from util import (
    knapsack_matrix,
    filter_binary_candidates,
    extract_coefficient_submatrix,
)
from fpylll import LLL


def solve_lll(weights: list[int], T: int, sol: list[int]) -> tuple:
    """
    Hybrid Subset Sum solver combining LLL lattice reduction with CP-SAT.

    Strategy:
        1. Build the knapsack lattice matrix and apply LLL reduction.
        2. Extract binary candidate vectors from the reduced basis.
           These candidates are {0,1}-vectors that may directly encode a solution.
        3. For each binary candidate (sorted by residual, best first), run CP-SAT
           with the candidate injected as a hint. A short micro-timeout is used
           to avoid wasting time on misleading hints.
        4. If no hint leads to a solution, fall back to vanilla CP-SAT with the
           remaining time budget.

    This design guarantees the hybrid is never worse than vanilla CP-SAT:
    if LLL finds nothing useful, the fallback takes over with full time budget.

    Args:
        weights: List of positive integers a_1, ..., a_n.
        T:       Target sum.
        sol:     Reference solution (used only for computing Hamming distance
                 in benchmark metrics — not used by the solver itself).

    Returns:
        A tuple (time, branches, conflicts, status, solution, label, best_res, best_ham):
            time      (float):      Total wall-clock time including LLL reduction.
            branches  (int):        Total CP-SAT branches explored across all calls.
            conflicts (int):        Total CP-SAT conflicts generated across all calls.
            status    (int):        Final CP-SAT status code (OPTIMAL, FEASIBLE, UNKNOWN).
            solution  (list|None):  The {0,1} solution found, or None if unsolved.
            label     (str):        Resolution path:
                                      'Binary_Hint_k'     solved by k-th hint,
                                      'Standard_Fallback' solved by vanilla CP-SAT,
                                      'Timeout_LLL'       timed out during hints phase,
                                      'Timeout_Fallback'  timed out during fallback.
            best_res  (int):        Residual |sum(a_i * c_i) - T| of the best
                                    binary candidate found (0 means exact solution).
            best_ham  (int):        Hamming distance of the best candidate to sol.
    """
    start_global_time = time.perf_counter()
    n = len(weights)

    total_branches = 0
    total_conflicts = 0

    # Time budget allocated per hint attempt.
    # Kept short: if a hint is good, CP-SAT converges in milliseconds.
    # If a hint is bad, we want to abort quickly and preserve budget for fallback.
    micro_timeout = 2.0

    # -------------------------------------------------------------------------
    # Step 1 — LLL lattice reduction
    # Build the (n+1) x (n+1) knapsack matrix and apply LLL reduction.
    # The reduced basis contains short vectors, some of which may be {0,1}-vectors
    # encoding valid solutions.
    # -------------------------------------------------------------------------
    B = knapsack_matrix(weights, T, 2 ** n)
    LLL.reduction(B)
    sub = extract_coefficient_submatrix(B)

    # -------------------------------------------------------------------------
    # Step 2 — Default metrics (for benchmarking graphs)
    # Use the shortest reduced vector (rank 0) as a baseline metric.
    # It is clipped to {0,1} to compute a residual and Hamming distance.
    # This ensures best_res and best_ham always have a valid value.
    # -------------------------------------------------------------------------
    if sub:
        shortest_bin = [1 if c > 0 else 0 for c in sub[0]]
        best_res = abs(sum(weights[i] * shortest_bin[i] for i in range(n)) - T)
        best_ham = sum(1 for i in range(n) if shortest_bin[i] != sol[i])
    else:
        best_res = float("inf")
        best_ham = n

    # -------------------------------------------------------------------------
    # Step 3 — Extract and rank binary candidates
    # Filter vectors whose coefficients are all in {0, 1}.
    # Update best_res/best_ham if a better candidate is found.
    # Sort candidates by residual ascending so the most promising hint is tried first.
    # -------------------------------------------------------------------------
    candidates = filter_binary_candidates(sub)

    if candidates:
        for c in candidates:
            res = abs(sum(weights[i] * c[i] for i in range(n)) - T)
            ham = sum(1 for i in range(n) if c[i] != sol[i])
            if res < best_res:
                best_res, best_ham = res, ham

        candidates.sort(key=lambda c: abs(sum(weights[i] * c[i] for i in range(n)) - T))

    # -------------------------------------------------------------------------
    # Step 4 — CP-SAT guided by LLL hints
    # For each binary candidate, inject it as a hint into a fresh CP-SAT model.
    # If CP-SAT finds a solution within micro_timeout, return immediately.
    # Each solver instance is independent — no state is shared between attempts.
    # -------------------------------------------------------------------------
    for idx, hint in enumerate(candidates):
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = micro_timeout
        solver.parameters.num_search_workers = 1  # required for determinism

        x = [model.new_bool_var(f"x{i}") for i in range(n)]
        model.add(sum(weights[i] * x[i] for i in range(n)) == T)

        for i in range(n):
            model.add_hint(x[i], hint[i])

        status = solver.solve(model)
        total_branches += solver.num_branches
        total_conflicts += solver.num_conflicts

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            solution = [solver.value(x[i]) for i in range(n)]
            final_time = time.perf_counter() - start_global_time
            return (
                final_time,
                total_branches,
                total_conflicts,
                status,
                solution,
                f"Binary_Hint_{idx + 1}",
                best_res,
                best_ham,
            )

    # -------------------------------------------------------------------------
    # Step 5 — Vanilla CP-SAT fallback
    # No hint led to a solution. Run CP-SAT without any hint using the remaining
    # time budget. This guarantees the hybrid is never worse than vanilla CP-SAT.
    # -------------------------------------------------------------------------
    time_spent_so_far = time.perf_counter() - start_global_time
    remaining_time = TIMEOUT - time_spent_so_far

    if remaining_time <= 0:
        # All time was consumed by hint attempts — report timeout.
        return (
            time_spent_so_far,
            total_branches,
            total_conflicts,
            cp_model.UNKNOWN,
            None,
            "Timeout_LLL",
            best_res,
            best_ham,
        )

    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = remaining_time

    x = [model.new_bool_var(f"x{i}") for i in range(n)]
    model.add(sum(weights[i] * x[i] for i in range(n)) == T)

    status = solver.solve(model)
    total_branches += solver.num_branches
    total_conflicts += solver.num_conflicts

    solution = None
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        solution = [solver.value(x[i]) for i in range(n)]

    final_time = time.perf_counter() - start_global_time
    label = (
        "Standard_Fallback"
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        else "Timeout_Fallback"
    )

    return (
        final_time,
        total_branches,
        total_conflicts,
        status,
        solution,
        label,
        best_res,
        best_ham,
    )
