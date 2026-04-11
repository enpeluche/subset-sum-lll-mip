import time
import numpy as np
from SubsetSumInstance import SubsetSumInstance
from results import SolveResult

# check if I can do Gray enumeration
def _enumerate_numpy(weights: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Fast enumeration using numpy bit-shifting and matrix multiplication."""
    m = len(weights)

    masks = np.arange(1 << m, dtype=np.int64) # all integers from 0 to 2^m - 1
    
    bits = ((masks[:, None] >> np.arange(m, dtype=np.int64)) & 1).astype(np.int8)
    
    w = np.array(weights, dtype=np.int64)
    
    sums = bits @ w
    
    return sums, bits

def _build_table(weights: list[int]) -> dict[int, list[int]]:
    """Builds the lookup table for the left half."""

    sums, bits = _enumerate_numpy(weights)
    
    table = {}
    
    for i in range(len(sums)):
        s = int(sums[i])
        if s not in table:
            table[s] = bits[i].tolist()
    
    return table


def solve_mitm_classic(
    instance: SubsetSumInstance,
    max_subsets: int = 2 ** 22,
) -> SolveResult:
    """
    Meet-in-the-Middle with NumPy acceleration.
    """

    start = time.perf_counter()
    n = instance.n
    mid = n // 2

    # 0. Early Exit: Trivially Infeasible

    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)

    # 1. Guard : Skip if n/2 is too large for memory/time budget

    if (1 << mid) > max_subsets:
        return SolveResult.skipped("MITM_Skipped_Size_Limit")
    
    weights_L = instance.weights[:mid]
    weights_R = instance.weights[mid:]

    # 2. Build Left Table

    table_L = _build_table(weights_L)

    # 3. Scan Right Half

    sums_R, bits_R = _enumerate_numpy(weights_R)

    solution = None
    for i in range(len(sums_R)):
        complement = instance.target - int(sums_R[i])
        if complement in table_L:
            candidate = table_L[complement] + bits_R[i].tolist()

            # Double check for safety
            if instance.is_solution(candidate):
                solution = candidate
                break

    elapsed = time.perf_counter() - start
    solved = solution is not None

    return SolveResult(
        elapsed=elapsed,
        branches=1 << mid,
        conflicts=0,
        status=0 if solved else 3,
        solution=solution,
        label="MITM_Found" if solved else "MITM_NotFound",
        best_res=None,
        best_ham=None,
    )
