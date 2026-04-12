import time
from ortools.sat.python import cp_model
from utils import filter_binary_vectors, extract_vectors_from_basis
from fpylll import LLL, BKZ
from SubsetSumInstance import SubsetSumInstance
from results import SolveResult
from solvers.solve_cpsat import solve_cpsat
from typing import Optional, Tuple

def _evaluate_basis(
    instance: SubsetSumInstance, 
    B, 
    start_time: float, 
    label: str
) -> Tuple[Optional[SolveResult], int, int | None]:
    """
    Méthode privée : Extrait la base, calcule les métriques et cherche une solution exacte.
    Retourne un tuple : (SolveResult si solution trouvée sinon None, best_residual, best_hamming)
    """
    vectors = extract_vectors_from_basis(B)
    
    # 1. Calcul des métriques de base
    shortest_binary = [1 if c > 0 else 0 for c in vectors[0]]
    best_residual = abs(instance.residual(shortest_binary))
    best_hamming = instance.hamming_to_solution(shortest_binary)

    # 2. Vérification des candidats binaires
    candidates = filter_binary_vectors(vectors)
    if candidates:
        for c in candidates:
            if instance.is_solution(c):
                result = SolveResult(
                    elapsed=time.perf_counter() - start_time,
                    branches=0,
                    conflicts=0,
                    status=int(cp_model.OPTIMAL),
                    solution=c,
                    label=f"{label}_Direct_Exact",
                    best_res=0,
                    best_ham=0,
                )
                return result, best_residual, best_hamming
                
    return None, best_residual, best_hamming

import math

def _resolve_scaling(instance, scaling):
    n = instance.n
    strategies = {
        "sqrt_n": int(math.ceil(math.sqrt(n))),
        "n":      n,
        "sum_w":  sum(instance.weights),
        "2n":     1 << n,
        "2n2":    1 << (n // 2),
    }
    if isinstance(scaling, int):
        return scaling
    return strategies.get(scaling, 1 << n)

def solve_lattice_hybrid(
    instance: SubsetSumInstance,
    strategy: str = "SEQ_LLL_BKZ",  # "LLL_ONLY" ,"BKZ_ONLY", "SEQ_LLL_BKZ", ou "INDEP_LLL_BKZ"
    scaling: int | str | None = None,
    block_size: int = 30,
    delta: float = 0.99,
    eta: float = 0.51,
    workers: int = 8,
    timeout: float = 100.0
) -> SolveResult:
    """
    Hybrid Subset Sum solver combining LLL lattice reduction with CP-SAT fallback.

    Strategy:
        1. Trivial Check: Early exit if the target is mathematically unreachable.
        2. Lattice Reduction: Build a knapsack matrix and apply LLL.
        3. Direct Verification: Extract and sort candidates by norm. If a binary 
           vector solves the instance exactly, return it immediately.
        4. Fallback: If no direct solution is found, use the remaining time budget
           to run a full CP-SAT search.
    """
    start = time.perf_counter()
    n = instance.n
    best_residual = 0
    best_hamming = 0

    # 0. Early exit: no subset can reach T if the total sum is insufficient.
    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)

    # 1. Instance to knapsack matrix conversion

    #if scaling is None:
    #    scaling = 2 ** n

    scaling = _resolve_scaling(instance, scaling)

    B = instance.to_knapsack_matrix(M=scaling)

    if strategy in ["LLL_ONLY", "SEQ_LLL_BKZ", "INDEP_LLL_BKZ"]:
        LLL.reduction(B, delta=delta, eta=eta)

        result, best_residual, best_hamming = _evaluate_basis(instance, B, start, f"{strategy}")
        if result: 
            return result

    if strategy in ["BKZ_ONLY", "SEQ_LLL_BKZ"]:
        params = BKZ.Param(block_size=block_size)
        BKZ.reduction(B, params)

        result, best_residual, best_hamming = _evaluate_basis(instance, B, start, f"{strategy}")
        if result: 
            return result
    
    if strategy in ["INDEP_LLL_BKZ"]:
        B = instance.to_knapsack_matrix(M=scaling)
        params = BKZ.Param(block_size=block_size)
        BKZ.reduction(B, params)

        result, best_residual, best_hamming = _evaluate_basis(instance, B, start, f"{strategy}")
        if result: 
            return result

    # 3. We check if there is remaining time

    elapsed = time.perf_counter() - start
    remaining = timeout - elapsed

    remaining = 0

    if remaining <= 0:
        return SolveResult.timeout(
            elapsed, 
            label=f"Timeout_During_{strategy}", 
            res=best_residual, 
            ham=best_hamming
        )

    # 4. We start a vanilla cp sat fallback
    fallback_result = solve_cpsat(instance, workers=workers, timeout=remaining)
    final_label = f"{strategy}_Fallback_Success" if fallback_result.solution else f"{strategy}_Fallback_Timeout"

    return SolveResult(
        elapsed=time.perf_counter() - start,
        branches=fallback_result.branches,
        conflicts=fallback_result.conflicts,
        status=fallback_result.status,
        solution=fallback_result.solution,
        label=final_label,
        best_res=best_residual,
        best_ham=best_hamming,
    )
