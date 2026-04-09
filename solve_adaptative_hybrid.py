# solve_adaptive_hybrid.py
"""
Adaptive hybrid solver: LLL → BKZ → CP-SAT.

Strategy:
    1. Try LLL first — fast, ~0ms overhead.
       If binary candidates found → use them as CP-SAT hints.
    2. If LLL finds no binary candidate → retry with BKZ(30).
       BKZ is stronger but costs ~3ms extra.
       If binary candidates found → use them as CP-SAT hints.
    3. If neither finds candidates → vanilla CP-SAT with full budget.

This captures the complementary strengths of LLL and BKZ observed
empirically: some instances LLL solves that BKZ misses, and vice versa.
Avoids BKZ overhead (~3ms) when LLL already succeeds.
"""

import time
from ortools.sat.python import cp_model
from fpylll import LLL, BKZ
from utils import filter_binary_vectors, extract_vectors_from_basis
from SubsetSumInstance import SubsetSumInstance
from results import SolveResult
from constants import TIMEOUT


def _run_cpsat_with_hints(
    instance: SubsetSumInstance,
    candidates: list[list[int]],
    micro_timeout: float,
    start: float,
    best_residual: int,
    best_hamming: int | None,
    total_branches: int,
    total_conflicts: int,
) -> SolveResult | None:
    """
    Try each binary candidate as a CP-SAT hint.
    Returns SolveResult if solved, None if no hint succeeded.
    """
    n = instance.n

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

    return None


def _extract_candidates(vectors, instance):
    """Extract and rank binary candidates from reduced basis vectors."""
    best_residual = 1 << 128
    best_hamming = None

    if vectors:
        shortest_binary = [1 if c > 0 else 0 for c in vectors[0]]
        best_residual = abs(instance.residual(shortest_binary))
        best_hamming = instance.hamming_to_solution(shortest_binary)

    candidates = filter_binary_vectors(vectors)

    if candidates:
        for c in candidates:
            res = abs(instance.residual(c))
            if res < best_residual:
                best_residual = res
                best_hamming = instance.hamming_to_solution(c)
        candidates.sort(key=lambda c: abs(instance.residual(c)))

    return candidates, best_residual, best_hamming


def solve_adaptive_hybrid(
    instance: SubsetSumInstance,
    scaling: int | None = None,
    block_size: int = 30,
    micro_timeout: float = 2.0,
) -> SolveResult:
    """
    Adaptive hybrid solver: LLL → BKZ → CP-SAT fallback.

    Args:
        instance:      A SubsetSumInstance.
        scaling:       Lattice scaling factor M. Defaults to 2^n.
        block_size:    BKZ block size (default 30).
        micro_timeout: Per-hint CP-SAT timeout in seconds.

    Returns:
        SolveResult with label in {
            'LLL_Hint_k'        : solved by k-th LLL hint,
            'BKZ_Hint_k'        : solved by k-th BKZ hint,
            'Standard_Fallback' : solved by vanilla CP-SAT,
            'Timeout_LLL'       : timeout during LLL hints phase,
            'Timeout_BKZ'       : timeout during BKZ hints phase,
            'Timeout_Fallback'  : timeout during CP-SAT fallback,
        }
    """
    start = time.perf_counter()
    n = instance.n
    total_branches = 0
    total_conflicts = 0

    if scaling is None:
        scaling = 2 ** n

    # ------------------------------------------------------------------
    # Step 1 — LLL reduction
    # ------------------------------------------------------------------
    B = instance.to_knapsack_matrix(M=scaling)
    LLL.reduction(B)
    vectors = extract_vectors_from_basis(B)
    candidates, best_residual, best_hamming = _extract_candidates(vectors, instance)

    # ------------------------------------------------------------------
    # Step 2 — CP-SAT guided by LLL hints
    # ------------------------------------------------------------------
    if candidates:
        result = _run_cpsat_with_hints(
            instance,
            candidates,
            micro_timeout,
            start,
            best_residual,
            best_hamming,
            total_branches,
            total_conflicts,
        )
        if result is not None:
            result.label = result.label.replace("Binary", "LLL")
            return result

    # ------------------------------------------------------------------
    # Step 3 — BKZ reduction (only if LLL found no candidates)
    # ------------------------------------------------------------------
    if not candidates:
        elapsed = time.perf_counter() - start
        if elapsed >= TIMEOUT:
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

        B_bkz = instance.to_knapsack_matrix(M=scaling)
        BKZ.reduction(B_bkz, BKZ.Param(block_size=block_size))
        vectors_bkz = extract_vectors_from_basis(B_bkz)
        candidates_bkz, best_residual, best_hamming = _extract_candidates(
            vectors_bkz, instance
        )

        if candidates_bkz:
            result = _run_cpsat_with_hints(
                instance,
                candidates_bkz,
                micro_timeout,
                start,
                best_residual,
                best_hamming,
                total_branches,
                total_conflicts,
            )
            if result is not None:
                result.label = result.label.replace("Binary", "BKZ")
                return result

    # ------------------------------------------------------------------
    # Step 4 — Vanilla CP-SAT fallback
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - start
    remaining = TIMEOUT - elapsed

    if remaining <= 0:
        label = "Timeout_BKZ" if not candidates else "Timeout_LLL"
        return SolveResult(
            elapsed=elapsed,
            branches=total_branches,
            conflicts=total_conflicts,
            status=int(cp_model.UNKNOWN),
            solution=None,
            label=label,
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
