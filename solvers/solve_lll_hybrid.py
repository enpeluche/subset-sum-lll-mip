import time
from ortools.sat.python import cp_model
from utils import filter_binary_vectors, extract_vectors_from_basis
from fpylll import LLL
from SubsetSumInstance import SubsetSumInstance
from results import SolveResult
from solvers.solve_cpsat import solve_cpsat

def solve_lll_hybrid(
    instance: SubsetSumInstance, scaling: int | None = None, delta: float = 0.99, eta: float = 0.51, workers: int = 8, timeout: float = 100.0
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

    # 0. Early exit: no subset can reach T if the total sum is insufficient.
    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)

    # 1. Instance to knapsack matrix conversion

    if scaling is None:
        scaling = 2 ** n

    B = instance.to_knapsack_matrix(M=scaling)
    
    LLL.reduction(B, delta=delta, eta=eta)
    
    vectors = extract_vectors_from_basis(B)

    # for benchmark only, vectors[0] is the shortest vector thnaks to extract_vectors_from_basis
    
    shortest_binary = [1 if c > 0 else 0 for c in vectors[0]]
    best_residual = abs(instance.residual(shortest_binary))
    best_hamming = instance.hamming_to_solution(shortest_binary)

    # 2. We check binary candidates

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
                    label=f"LLL_{delta}_{eta}_Direct_Exact",
                    best_res=0,
                    best_ham=0,
                )

    # 3. We check if there is remaining time

    elapsed = time.perf_counter() - start
    remaining = timeout - elapsed

    if remaining <= 0:
        return SolveResult.timeout(
            elapsed, 
            label="Timeout_During_LLL", 
            res=best_residual, 
            ham=best_hamming
        )

    # 4. We start a vanilla cp sat fallback
    fallback_result = solve_cpsat(instance, workers=workers, timeout=remaining)

    return SolveResult(
        elapsed=time.perf_counter() - start,
        branches=fallback_result.branches,
        conflicts=fallback_result.conflicts,
        status=fallback_result.status,
        solution=fallback_result.solution,
        label="Standard_Fallback" if fallback_result.solution else "Timeout_Fallback", # à vérfier
        best_res=best_residual,
        best_ham=best_hamming,
    )
