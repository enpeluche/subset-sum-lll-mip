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

def _adaptive_scaling(instance):
    n = instance.n
    # On regarde le nombre de bits du plus gros poids
    bits_w = max(instance.weights).bit_length()
    
    if bits_w > 2 * n:
        return 1 << n       # M = 2^n (Poids lourds)
    elif bits_w > n:
        return n            # M = n (Zone de combat)
    else:
        return int(n**0.5)  # M = sqrt(n) (Haute densité)

def solve_lattice_hybrid(
    instance: SubsetSumInstance,
    strategy: str = "SEQ_LLL_BKZ", 
    scaling: int | str | None = None,
    block_size: int = 30, # Sera utilisé comme plafond (max)
    delta: float = 0.99,
    eta: float = 0.51,
    workers: int = 8,
    timeout: float = 100.0
) -> SolveResult:
    # Suppresion de INDEP_LLL_BKZ 14-04
    # BKZ Adaptif : 10 jusqu'a 20 puis n jusqu'à 30, 30 au dela
    # scaling à n a l'air pas mal pi sqert n    
    # Fusion probable de indep et seq au profit de la plus rapide
    # profit de adaptive bkz si il se démarque
    # scaling adaptatif

    start = time.perf_counter()
    n = instance.n

    best_res, best_ham = None, None

    # 0. Early exit
    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)

    # 1. Scaling et Préparation de la matrice
    scaling_val = _resolve_scaling(instance, scaling)

    if strategy == "SMART":
        scaling_val = _adaptive_scaling(instance)
        delta = 0.99

    B = instance.to_knapsack_matrix(M=scaling_val)

    # --- PHASE LLL ---
    if strategy in ["LLL_ONLY", "SEQ_LLL_BKZ", "SMART"]:
        LLL.reduction(B, delta=delta, eta=eta)
        res, best_res, best_ham = _evaluate_basis(instance, B, start, f"{strategy}_LLL")
        if res: return res

    # --- PHASE BKZ ---

    if strategy in ["BKZ_ONLY", "SEQ_LLL_BKZ", "ADAPTATIVE_BKZ", "SMART"]:
        params = BKZ.Param(block_size=block_size)
        if strategy in ["ADAPTATIVE_BKZ", "SMART"]:

            if n < 20:
                block_size = 10
            else:
                block_size =  max(2, min(n, block_size))

            params = BKZ.Param(block_size= block_size)
        BKZ.reduction(B, params)
        res, best_res, best_ham = _evaluate_basis(instance, B, start, f"{strategy}_BKZ")
        if res: return res

    # --- PHASE FALLBACK (CP-SAT) ---
    elapsed = time.perf_counter() - start
    remaining = timeout - elapsed

    if remaining > 0.1:
        fallback_result = solve_cpsat(instance, workers=workers, timeout=remaining)
        status_label = "Success" if fallback_result.solution else "Timeout"
        
        return SolveResult(
            elapsed=time.perf_counter() - start,
            branches=fallback_result.branches,
            conflicts=fallback_result.conflicts,
            status=fallback_result.status,
            solution=fallback_result.solution,
            label=f"{strategy}_Fallback_{status_label}",
            best_res=best_res, #type: ignore
            best_ham=best_ham, #type: ignore
        )

    return SolveResult.timeout(elapsed, label=f"{strategy}_Final_Timeout", res=best_res, ham=best_ham) #type: ignore