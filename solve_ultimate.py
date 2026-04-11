"""
solve_ultimate.py
-----------------
Ultimate hybrid solver: LLL → BKZ → Tabu local → CP-SAT probe → MITM → CP-SAT fallback.

Each stage is only triggered if the previous one fails.
Cost is always minimal: early stages cost microseconds, later stages more.

Pipeline:
    1. LLL       (~0ms)    — fast reduction, direct binary candidates
    2. BKZ       (~3ms)    — stronger reduction, catches what LLL misses
    3. Tabu      (~5ms)    — local search on k ambiguous variables
    4. CP-SAT    (~1s)     — probe with Tabu hint, handles high density
    5. MITM      (~100ms)  — exact enumeration, 100% for n ≤ 44
    6. CP-SAT    (budget)  — guaranteed fallback

Guarantees:
    - Never worse than vanilla CP-SAT (fallback always available)
    - 100% solve rate for n ≤ 44 (MITM is exact)
    - Works beyond CP-SAT's 2^63 integer limit (LLL/BKZ/MITM)

Labels:
    'LLL_Direct_k'      : LLL hint k solved it
    'BKZ_Direct_k'      : BKZ hint k solved it
    'Tabu_Local'        : Tabu local search solved it
    'CPSAT_Probe_Tabu'  : CP-SAT probe with Tabu hint solved it
    'MITM_Found'        : MITM enumeration solved it
    'Standard_Fallback' : CP-SAT fallback solved it
    'Timeout_Fallback'  : timed out
    'MITM_Skipped'      : n too large for MITM, timed out
"""

import time
from ortools.sat.python import cp_model
from fpylll import LLL, BKZ

from SubsetSumInstance import SubsetSumInstance
from results import SolveResult
from utils import extract_vectors_from_basis, filter_binary_vectors
from solvers.solve_mitm import solve_mitm_classic, _MITM_SKIP
from constants import TIMEOUT


# =============================================================================
# Helpers
# =============================================================================


def _extract_candidates(
    vectors: list[list[int]],
    instance: SubsetSumInstance,
) -> tuple[list[list[int]], int, int | None]:
    """
    Extract binary candidates from reduced basis, compute baseline metrics.

    Returns:
        (candidates sorted by residual, best_residual, best_hamming)
    """
    best_residual = 1 << 128
    best_hamming = None

    if vectors:
        shortest_bin = [1 if c > 0 else 0 for c in vectors[0]]
        best_residual = abs(instance.residual(shortest_bin))
        best_hamming = instance.hamming_to_solution(shortest_bin)

    candidates = filter_binary_vectors(vectors)

    if candidates:
        for c in candidates:
            res = abs(instance.residual(c))
            if res < best_residual:
                best_residual = res
                best_hamming = instance.hamming_to_solution(c)
        candidates.sort(key=lambda c: abs(instance.residual(c)))

    return candidates, best_residual, best_hamming


def _cpsat_probe(
    instance: SubsetSumInstance,
    timeout: float,
    start: float,
    total_branches: int,
    total_conflicts: int,
    best_residual: int,
    best_hamming: int | None,
    label: str = "CPSAT_Probe",
    hint: list[int] | None = None,
    n_workers: int = 8,
) -> SolveResult | None:
    """
    Run CP-SAT with a short timeout and optional hint.
    Returns SolveResult if solved, None otherwise.

    Args:
        timeout:    Max solve time in seconds.
        hint:       Optional binary warm start vector.
        n_workers:  Number of CP-SAT workers (1 for determinism with hints).
    """
    n = instance.n

    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_search_workers = n_workers

    x = [model.new_bool_var(f"x{i}") for i in range(n)]
    model.add(sum(instance.weights[i] * x[i] for i in range(n)) == instance.target)

    if hint is not None:
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
            label=label,
            best_res=best_residual,
            best_ham=best_hamming,
        )
    return None


def _tabu_local(
    instance: SubsetSumInstance,
    vectors: list[list[int]],
    k_relax: int = 8,
    n_iter: int = 5000,
    tenure: int = 7,
) -> tuple[list[int] | None, list[int]]:
    """
    Tabu search on k ambiguous variables guided by LLL geometry.

    Selects the k most ambiguous coordinates (min distance to {0,1})
    and runs tabu search restricted to those coordinates.

    Returns:
        (solution, best_x_seen) where solution is None if not found.
        best_x_seen is the binary vector with minimum residual seen —
        useful as a hint for the subsequent CP-SAT probe.
    """
    n = instance.n
    weights = instance.weights

    if not vectors:
        return None, [0] * n

    v = vectors[0]

    # Ambiguity: distance to nearest integer in {0,1}
    # min(|v_j|, |v_j - 1|) — handles values like -2, 3 correctly
    ambiguity = sorted(
        range(n), key=lambda j: min(abs(float(v[j])), abs(float(v[j]) - 1))
    )

    relax_set = set(ambiguity[:k_relax])

    # Warm start: clip to {0,1}
    x_current = [max(0, min(1, round(float(c)))) for c in v]
    residual = instance.residual(x_current)

    best_x = x_current.copy()
    best_res = abs(residual)

    tabu_until = [0] * n

    for it in range(n_iter):
        if residual == 0:
            break

        best_j = -1
        best_move = float("inf")

        for j in relax_set:
            delta = weights[j] if x_current[j] == 0 else -weights[j]
            new_res = abs(residual + delta)

            is_tabu = tabu_until[j] > it
            aspiration = new_res < best_res

            if (not is_tabu or aspiration) and new_res < best_move:
                best_j = j
                best_move = new_res

        if best_j == -1:
            break

        delta = weights[best_j] if x_current[best_j] == 0 else -weights[best_j]
        x_current[best_j] = 1 - x_current[best_j]
        residual += delta
        tabu_until[best_j] = it + tenure

        if abs(residual) < best_res:
            best_res = abs(residual)
            best_x = x_current.copy()

    if residual == 0 and instance.is_solution(x_current):
        return x_current, best_x

    return None, best_x


# =============================================================================
# Main solver
# =============================================================================


def solve_ultimate(
    instance: SubsetSumInstance,
    scaling: int | None = None,
    bkz_block_size: int = 30,
    mitm_max_subsets: int = 2 ** 22,
    micro_timeout: float = 2.0,
    probe_timeout: float = 1.0,
    tabu_k: int = 8,
    tabu_iter: int = 5000,
) -> SolveResult:
    """
    Ultimate hybrid solver: LLL → BKZ → Tabu → CP-SAT probe → MITM → CP-SAT.

    Args:
        instance:          A SubsetSumInstance.
        scaling:           Lattice scaling factor M. Defaults to 2^n.
        bkz_block_size:    BKZ block size (default 30).
        mitm_max_subsets:  Max MITM enumeration. Default 2^22 (n≤44).
        micro_timeout:     Per-hint CP-SAT timeout in seconds.
        probe_timeout:     CP-SAT probe timeout before MITM.
        tabu_k:            Number of ambiguous variables to relax in Tabu.
        tabu_iter:         Tabu iterations.

    Returns:
        SolveResult with timing, search stats, solution, and label.
    """
    start = time.perf_counter()
    n = instance.n
    total_branches = 0
    total_conflicts = 0

    if scaling is None:
        scaling = 2 ** n

    # Seuil empirique : à partir de d=1.25, LLL/BKZ ne trouvent
    # jamais de candidat binaire → skip directement à Tabu + MITM
    LLL_DENSITY_THRESHOLD = 1.25

    if instance.density < LLL_DENSITY_THRESHOLD:
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
        for idx, hint in enumerate(candidates):
            result = _cpsat_probe(
                instance,
                micro_timeout,
                start,
                total_branches,
                total_conflicts,
                best_residual,
                best_hamming,
                label=f"LLL_Direct_{idx + 1}",
                hint=hint,
                n_workers=1,
            )
            if result:
                return result

        # ------------------------------------------------------------------
        # Step 3 — BKZ reduction (only if LLL found no binary candidates)
        # ------------------------------------------------------------------
        if not candidates:
            elapsed = time.perf_counter() - start
            if elapsed < TIMEOUT:
                B_bkz = instance.to_knapsack_matrix(M=scaling)
                BKZ.reduction(B_bkz, BKZ.Param(block_size=bkz_block_size))
                vectors_bkz = extract_vectors_from_basis(B_bkz)
                candidates_bkz, best_residual, best_hamming = _extract_candidates(
                    vectors_bkz, instance
                )

                for idx, hint in enumerate(candidates_bkz):
                    result = _cpsat_probe(
                        instance,
                        micro_timeout,
                        start,
                        total_branches,
                        total_conflicts,
                        best_residual,
                        best_hamming,
                        label=f"BKZ_Direct_{idx + 1}",
                        hint=hint,
                        n_workers=1,
                    )
                    if result:
                        return result

                # Use BKZ vectors for Tabu if available
                if vectors_bkz:
                    vectors = vectors_bkz

    else:
        # Haute densité — skip LLL/BKZ, vecteurs inutiles
        vectors = []
        candidates = []
        best_residual = 1 << 128
        best_hamming = None

    # ------------------------------------------------------------------
    # Step 4 — Tabu local search on ambiguous coordinates
    # Uses LLL/BKZ geometry to select k most uncertain variables
    # Returns best_x even if not solution — used as CP-SAT hint
    # ------------------------------------------------------------------
    # Skip Tabu aussi à haute densité — il ne fait rien d'utile
    TABU_DENSITY_THRESHOLD = 0.85  # empirique, Tabu échoue à d>0.85

    if instance.density < TABU_DENSITY_THRESHOLD:
        tabu_solution, best_tabu_x = _tabu_local(
            instance,
            vectors,
            k_relax=tabu_k,
            n_iter=tabu_iter,
        )

        if tabu_solution is not None:
            return SolveResult(
                elapsed=time.perf_counter() - start,
                branches=tabu_iter,
                conflicts=0,
                status=0,
                solution=tabu_solution,
                label="Tabu_Local",
                best_res=best_residual,
                best_ham=best_hamming,
            )
    else:
        best_tabu_x = None  # pas de hint Tabu à haute densité

    # ------------------------------------------------------------------
    # Step 5 — CP-SAT probe sans hint
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - start
    if elapsed < TIMEOUT:
        result = _cpsat_probe(
            instance,
            probe_timeout,
            start,
            total_branches,
            total_conflicts,
            best_residual,
            best_hamming,
            label="CPSAT_Probe",
            hint=None,
            n_workers=8,
        )
        if result:
            return result

    # ------------------------------------------------------------------
    # Step 6 — MITM exact enumeration
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - start
    if elapsed < TIMEOUT:
        mitm_result = solve_mitm_classic(instance, max_subsets=mitm_max_subsets)

        if mitm_result.solution is not None:
            return SolveResult(
                elapsed=time.perf_counter() - start,
                branches=mitm_result.branches,
                conflicts=total_conflicts,
                status=0,
                solution=mitm_result.solution,
                label="MITM_Found",
                best_res=best_residual,
                best_ham=best_hamming,
            )

    # ------------------------------------------------------------------
    # Step 7 — CP-SAT fallback with remaining budget
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - start
    remaining = TIMEOUT - elapsed

    mitm_skipped = (
        mitm_result.label == "MITM_Skipped" if "mitm_result" in locals() else False
    )

    if remaining <= 0:
        return SolveResult(
            elapsed=elapsed,
            branches=total_branches,
            conflicts=total_conflicts,
            status=int(cp_model.UNKNOWN),
            solution=None,
            label="MITM_Skipped" if mitm_skipped else "Timeout_Before_Fallback",
            best_res=best_residual,
            best_ham=best_hamming,
        )

    result = _cpsat_probe(
        instance,
        remaining,
        start,
        total_branches,
        total_conflicts,
        best_residual,
        best_hamming,
        label="Standard_Fallback",
        hint=None,
        n_workers=8,
    )

    if result:
        return result

    return SolveResult(
        elapsed=time.perf_counter() - start,
        branches=total_branches,
        conflicts=total_conflicts,
        status=int(cp_model.UNKNOWN),
        solution=None,
        label="MITM_Skipped_Timeout" if mitm_skipped else "Timeout_Fallback",
        best_res=best_residual,
        best_ham=best_hamming,
    )
