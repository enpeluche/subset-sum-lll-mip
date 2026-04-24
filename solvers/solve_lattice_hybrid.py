"""
Hybrid lattice solver: LLL → BKZ (adaptive) → return best candidate.

Design choices validated by benchmarks:
    - δ = 0.99 (δ study: 0.999 gives no SR gain, marginal speed cost)
    - M = adaptive (scaling study: M=n best at high d, M=2^n at low d)
    - SEQ architecture (arch study: INDEP identical SR, slightly slower)
    - BKZ adaptive blocks
"""

import time

from fpylll import LLL, BKZ

from utils import filter_binary_vectors, extract_vectors_from_basis

from SubsetSumInstance import SubsetSumInstance
from results import SolveResult

def _evaluate_basis(instance: SubsetSumInstance, B, start: float, label: str) -> SolveResult:
    """
    Analyze the reduced basis to extract solutions or the best available candidates.

    Args:
        instance: The SubsetSumInstance against which to validate vectors.
        B: The reduced lattice basis (fpylll IntegerMatrix).
        start: The performance counter timestamp from the beginning of the solver.
        label: A string identifier (e.g., "LLL", "BKZ") for result tracking.

    Returns:
        A SolveResult object containing either an exact solution or metrics for 
        the best-fit candidate found through vector clipping.
    """
    vectors = extract_vectors_from_basis(B)
 
    if not vectors:
        return SolveResult(
            elapsed=time.perf_counter() - start,
            status=3, label=f"{label}_NoVectors",
            solution=None,
        )
 
    # 1. Check for exact binary solutions
    for c in filter_binary_vectors(vectors):
        if instance.is_solution(c):
            return SolveResult.found(
                elapsed=time.perf_counter() - start,
                solution=c,
                label=f"{label}_Direct",
                hamming_to_ground_solution=instance.hamming_to_solution(c),
            )
 
    # 2. Clip all vectors to {0,1}, keep best by residual

    best_clip = None
    smallest_residual = float("inf")
 
    for v in vectors:
        clipped = [max(0, min(1, c)) for c in v]
        residual = abs(instance.residual(clipped))
        if residual < smallest_residual:
            smallest_residual = residual
            best_clip = clipped
 
    return SolveResult(
        elapsed=time.perf_counter() - start,
        status=3, label=f"{label}_NoExact",
        solution=None,
        smallest_residual=int(smallest_residual),
        hamming_to_ground_solution=instance.hamming_to_solution(best_clip) if best_clip else None,
        hint=best_clip,
    )

#
#   Adaptive
#

def _adaptive_scaling(instance: SubsetSumInstance) -> int:
    """
    Determine the optimal scaling factor M based on the instance's bit-load.

    Args:
        instance: The SubsetSumInstance to analyze.

    Returns:
        An integer scaling factor M tailored to the weight distribution.
    """
    
    n = instance.n
    max_w = max(instance.weights) if instance.weights else 1
    bits_w = max_w.bit_length()

    if bits_w >= 2 * n:
        return 1 << n
    elif bits_w >= n // 2:
        return n
    else:
        return int(n ** 0.5)

def _adaptive_block_size(n: int) -> int:
    """
    Select the BKZ block size based on problem dimension to optimize the 
    efficiency-time tradeoff.

    Args:
        n: The dimension of the lattice (number of weights).

    Returns:
        An integer block size optimized for the given dimension.
    """
    if n <= 15:
        return min(n, 10)
    elif n <= 25:
        return 20
    else:
        return min(n, 30)
    
#
#   Solver
#

def solve_lattice_hybrid(
    instance: SubsetSumInstance,
    timeout: float = 10.0,
) -> SolveResult:
    """
    Execute a multi-stage lattice reduction strategy for the Subset Sum Problem.

    Args:
        instance: The SubsetSumInstance containing weights and target sum.
        timeout: Maximum execution time in seconds for the entire process.

    Returns:
        A SolveResult containing either an exact binary solution or the best 
        candidate vector found through basis reduction and clipping.
    """
    start = time.perf_counter()
 
    # Early exit
    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)
 
    # Build lattice matrix
    B = instance.to_knapsack_matrix(M=_adaptive_scaling(instance))
 
    # --- LLL reduction ---
    LLL.reduction(B, delta=0.99, eta=0.51)
    lll_result = _evaluate_basis(instance, B, start, "LLL")

    if lll_result.solution is not None:
        return lll_result
 
    # --- BKZ reduction (on the already-LLL-reduced basis) ---
    if time.perf_counter() - start < timeout:
        BKZ.reduction(B, BKZ.Param(block_size=_adaptive_block_size(instance.n)))

        bkz_result = _evaluate_basis(instance, B, start, "BKZ")

        if bkz_result.solution is not None:
            return bkz_result
 
        # Keep the better of LLL and BKZ clips
        if (bkz_result.smallest_residual is not None
                and (lll_result.smallest_residual is None
                     or bkz_result.smallest_residual < lll_result.smallest_residual)):
            return bkz_result
    
    lll_result.elapsed = time.perf_counter() - start
    return lll_result