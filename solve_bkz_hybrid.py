"""
Hybrid Subset Sum solver combining BKZ lattice reduction with CP-SAT.

Identical to solve_lll_hybrid.py but uses BKZ reduction instead of LLL.
BKZ produces stronger reductions — more binary candidates in the hard
density regime (d ≈ 0.64-0.94) at the cost of ~3ms extra reduction time.

    LLL : fast reduction, lower binary candidate rate in hard regime
    BKZ : slower reduction (~3ms extra), higher binary candidate rate
"""

import time
from ortools.sat.python import cp_model
from fpylll import BKZ
from utils import filter_binary_vectors, extract_vectors_from_basis
from SubsetSumInstance import SubsetSumInstance
from results import SolveResult
from constants import TIMEOUT


def solve_bkz_hybrid(
    instance: SubsetSumInstance,
    scaling: int | None = None,
    block_size: int = 30,
) -> SolveResult:
    """
    Hybrid Subset Sum solver using BKZ lattice reduction with CP-SAT.

    Strategy identical to solve_lll_hybrid, replacing LLL with BKZ.
    BKZ with block_size=30 significantly improves the binary candidate
    rate in the hard density regime (d ≈ 0.64-0.94), at the cost of
    ~3ms extra reduction time — negligible against a 30s timeout.

    Args:
        instance:   A SubsetSumInstance (weights, target, optional ground truth).
        scaling:    Lattice scaling factor M. Defaults to 2^n.
        block_size: BKZ block size. Higher = stronger but slower.
                    Recommended: 10 (fast), 20 (balanced), 30 (strong).

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
    # Step 1 — BKZ reduction
    # ------------------------------------------------------------------
    B = instance.to_knapsack_matrix(M=scaling)
    BKZ.reduction(B, BKZ.Param(block_size=block_size))
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
    # Step 4 — CP-SAT guided by BKZ hints
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
                status=int(status),
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
            label="Timeout_BKZ",
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
