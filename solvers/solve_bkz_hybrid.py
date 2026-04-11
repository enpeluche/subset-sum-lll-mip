import time
from ortools.sat.python import cp_model
from fpylll import BKZ, load_strategies_json
from utils import filter_binary_vectors, extract_vectors_from_basis
from SubsetSumInstance import SubsetSumInstance
from results import SolveResult
from solvers.solve_cpsat import solve_cpsat

def solve_bkz_hybrid(
    instance: SubsetSumInstance,
    timeout: float = 100.0,
    scaling: int | None = None,
    block_size: int = 30,
    workers: int = 8
) -> SolveResult:
    """
    Hybrid Subset Sum solver combining BKZ lattice reduction with CP-SAT fallback.

    Strategy:
        1. Trivial Check: Early exit if target is unreachable.
        2. BKZ Reduction: Stronger than LLL, better for hard densities.
        3. Direct Verification: If a binary candidate solves the instance, exit.
        4. Fallback: Full CP-SAT search with remaining budget.
    """
    start = time.perf_counter()
    n = instance.n

    # 0. Early exit
    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)

    # 1. Scaling budget
    if scaling is None:
        scaling = 2 ** n

    # 2. BKZ Reduction
    B = instance.to_knapsack_matrix(M=scaling)

    # BKZ.Param control power of reduction


    params = BKZ.Param(block_size=block_size)
    BKZ.reduction(B, params)
    
    vectors = extract_vectors_from_basis(B)

    # Metrics for benchmark only

    shortest_binary = [1 if c > 0 else 0 for c in vectors[0]]
    best_residual = abs(instance.residual(shortest_binary))
    best_hamming = instance.hamming_to_solution(shortest_binary)

    # 3. Direct Binary Check
    
    candidates = filter_binary_vectors(vectors)
    if candidates:
        for c in candidates:
            if instance.is_solution(c):
                return SolveResult(
                    elapsed=time.perf_counter() - start,
                    branches=0,
                    conflicts=0,
                    status=int(cp_model.OPTIMAL),
                    solution=c,
                    label=f"BKZ_{block_size}_Direct_Exact",
                    best_res=0,
                    best_ham=0,
                )

    # 4. Check remaining time

    elapsed = time.perf_counter() - start
    remaining = timeout - elapsed

    if remaining <= 0:
        return SolveResult.timeout(
            elapsed, 
            label=f"Timeout_During_BKZ_{block_size}", 
            res=best_residual, 
            ham=best_hamming
        )

    # 5. Fallback to CP-SAT
    
    fallback_result = solve_cpsat(instance, workers=workers, timeout=remaining)

    return SolveResult(
        elapsed=time.perf_counter() - start,
        branches=fallback_result.branches,
        conflicts=fallback_result.conflicts,
        status=fallback_result.status,
        solution=fallback_result.solution,
        label=f"BKZ_{block_size}_Fallback",
        best_res=best_residual,
        best_ham=best_hamming,
    )