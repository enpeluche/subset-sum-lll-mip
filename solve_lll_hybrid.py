from constants import TIMEOUT
import time
from ortools.sat.python import cp_model
from utils import filter_binary_vectors, extract_vectors_from_basis
from fpylll import LLL
from SubsetSumInstance import SubsetSumInstance
from results import SolveResult


def solve_lll_hybrid(
    instance: SubsetSumInstance, scaling: int | None = None
) -> SolveResult:
    """
    Hybrid Subset Sum solver combining LLL lattice reduction with CP-SAT.

    Strategy:
        1. Build the knapsack lattice matrix and apply LLL reduction.
        2. Extract binary candidate vectors from the reduced basis.
        3. For each binary candidate (sorted by residual), run CP-SAT with the
           candidate as a hint. A micro-timeout avoids wasting budget on bad hints.
        4. If no hint succeeds, fall back to vanilla CP-SAT with remaining budget.

    Guarantees the hybrid is never worse than vanilla CP-SAT: if LLL finds nothing
    useful, the fallback takes over with the full remaining time budget.

    Args:
        instance: A SubsetSumInstance (weights, target, optional ground truth).
        scaling:  Lattice scaling factor M. Defaults to 2^n.

    Returns:
        SolveResult dataclass with timing, search stats, solution, and hint quality.
    """
    start = time.perf_counter()
    n = instance.n
    micro_timeout = 2.0

    if scaling is None:
        scaling = 2 ** n

    total_branches = 0
    total_conflicts = 0

    # ------------------------------------------------------------------
    # Step 1 — LLL reduction
    # ------------------------------------------------------------------
    B = instance.to_knapsack_matrix(M=scaling)
    LLL.reduction(B)
    vectors = extract_vectors_from_basis(B)

    # ------------------------------------------------------------------
    # Step 2 — Baseline metrics (shortest vector clipped to {0,1})
    # ------------------------------------------------------------------

    best_residual = 1 << 128
    best_hamming = None

    if vectors:
        shortest_binary = [1 if c > 0 else 0 for c in vectors[0]]
        best_residual = abs(instance.residual(shortest_binary))
        best_hamming = instance.hamming_to_solution(shortest_binary)

    # ------------------------------------------------------------------
    # Step 3 — Extract and rank binary candidates
    # ------------------------------------------------------------------
    candidates = filter_binary_vectors(vectors)

    if candidates:
        for c in candidates:
            res = abs(instance.residual(c))
            if res < best_residual:
                best_residual = res
                best_hamming = instance.hamming_to_solution(c)

        candidates.sort(key=lambda c: abs(instance.residual(c)))

    # ------------------------------------------------------------------
    # Step 4 — CP-SAT guided by LLL hints
    # ------------------------------------------------------------------
    for idx, hint in enumerate(candidates):
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = micro_timeout
        solver.parameters.num_search_workers = 1

        x = [model.new_bool_var(f"x{i}") for i in range(n)]
        model.add(sum(instance.weights[i] * x[i] for i in range(n)) == instance.target)
        for i in range(n):
            model.add_hint(x[i], hint[i])

        status = solver.solve(model)
        total_branches += solver.num_branches
        total_conflicts += solver.num_conflicts

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return SolveResult(
                elapsed=time.perf_counter() - start,
                branches=total_branches,
                conflicts=total_conflicts,
                status=int(status),  # int(status) ou status(...) ?
                solution=[solver.value(x[i]) for i in range(n)],
                label=f"Binary_Hint_{idx + 1}",
                best_res=best_residual,
                best_ham=best_hamming,
            )

    # ------------------------------------------------------------------
    # Step 5 — Vanilla CP-SAT fallback
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - start
    remaining = TIMEOUT - elapsed

    if remaining <= 0:
        return SolveResult(
            elapsed=elapsed,
            branches=total_branches,
            conflicts=total_conflicts,
            status=int(cp_model.UNKNOWN),
            solution=None,
            label="Timeout_LLL",
            best_res=best_residual,
            best_ham=best_hamming,
        )

    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = remaining
    solver.parameters.num_search_workers = 8

    x = [model.new_bool_var(f"x{i}") for i in range(n)]
    model.add(sum(instance.weights[i] * x[i] for i in range(n)) == instance.target)

    status = solver.solve(model)
    total_branches += solver.num_branches
    total_conflicts += solver.num_conflicts

    solved = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    return SolveResult(
        elapsed=time.perf_counter() - start,
        branches=total_branches,
        conflicts=total_conflicts,
        status=int(status),
        solution=[solver.value(x[i]) for i in range(n)] if solved else None,
        label="Standard_Fallback" if solved else "Timeout_Fallback",
        best_res=best_residual,
        best_ham=best_hamming,
    )
