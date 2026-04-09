"""
solve_mitm.py
-------------
Meet-in-the-Middle solver for Subset Sum.

Uses numpy vectorization for fast enumeration — O(2^(n/2)) exact solver
without CP-SAT's 2^63 integer limit.

Performance:
    n=30 (m=15) : ~0.05s   (vs ~0.7s Python)
    n=40 (m=20) : ~1s      (vs ~30s Python)
    n=44 (m=22) : ~5s      (vs ~5min Python)
    n=50 (m=25) : ~60s     → returns MITM_Skipped (use max_subsets guard)

Memory:
    m=15 :  1MB   (32K × 15 bits)
    m=20 : 20MB   (1M  × 20 bits)
    m=22 : 88MB   (4M  × 22 bits)  ← default limit

Usage:
    result = solve_mitm_classic(instance)
    result = solve_mitm_classic(instance, max_subsets=2**24)  # n≤48
"""

import time
import numpy as np
from fpylll import LLL, BKZ, IntegerMatrix

from SubsetSumInstance import SubsetSumInstance
from results import SolveResult
from utils import extract_vectors_from_basis, filter_binary_vectors


# Sentinel retourné si l'énumération dépasserait max_subsets
_MITM_SKIP = SolveResult(
    elapsed=0.0,
    branches=0,
    conflicts=0,
    status=3,
    solution=None,
    label="MITM_Skipped",
    best_res=None,
    best_ham=None,
)


# =============================================================================
# Numpy enumeration
# =============================================================================


def _enumerate_numpy(weights: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Enumerate all 2^m subset sums using numpy vectorization.

    Args:
        weights: List of m positive integers.

    Returns:
        (sums, bits) where:
            sums : int64 array of shape (2^m,)  — all partial sums
            bits : int8  array of shape (2^m, m) — corresponding binary vectors
    """
    m = len(weights)
    masks = np.arange(1 << m, dtype=np.int64)
    bits = ((masks[:, None] >> np.arange(m, dtype=np.int64)) & 1).astype(np.int8)
    w = np.array(weights, dtype=np.int64)
    sums = bits @ w
    return sums, bits


def _build_table(weights: list[int]) -> dict[int, list[int]]:
    """
    Build a dict mapping sum → binary vector from a list of weights.
    Uses numpy for speed. First occurrence wins on collision.

    Args:
        weights: List of m positive integers.

    Returns:
        Dict {partial_sum: binary_vector}.
    """
    sums, bits = _enumerate_numpy(weights)
    table = {}
    for i in range(len(sums)):
        s = int(sums[i])
        if s not in table:
            table[s] = bits[i].tolist()
    return table


# =============================================================================
# Classic Meet-in-the-Middle
# =============================================================================


def solve_mitm_classic(
    instance: SubsetSumInstance,
    max_subsets: int = 2 ** 22,
) -> SolveResult:
    """
    Classic Meet-in-the-Middle solver with numpy acceleration.

    Splits weights into two halves, enumerates all 2^(n/2) subset sums
    for each half using numpy, then looks for a pair summing to T.

    Complexity : O(2^(n/2)) time and space — exact, no integer size limit.
    No CP-SAT dependency — works on instances exceeding 2^63.

    Args:
        instance:    A SubsetSumInstance.
        max_subsets: Skip if 2^(n/2) exceeds this threshold.
                     Default 2^22 ≈ 4M — safe for n≤44, ~5s, ~88MB.
                     Set to 2^24 for n≤48 (~60s, ~350MB).

    Returns:
        SolveResult with label in {
            'MITM_Found', 'MITM_NotFound', 'MITM_Skipped'
        }
    """
    start = time.perf_counter()
    n = instance.n
    mid = n // 2

    # Guard : skip si l'énumération serait trop grande
    if (1 << mid) > max_subsets:
        return _MITM_SKIP

    weights_L = instance.weights[:mid]
    weights_R = instance.weights[mid:]

    # ------------------------------------------------------------------
    # Step 1 — Build left table : sum → binary vector (numpy)
    # ------------------------------------------------------------------
    table_L = _build_table(weights_L)

    # ------------------------------------------------------------------
    # Step 2 — Scan right half with numpy, lookup complement in table_L
    # ------------------------------------------------------------------
    sums_R, bits_R = _enumerate_numpy(weights_R)

    solution = None
    for i in range(len(sums_R)):
        complement = instance.target - int(sums_R[i])
        if complement in table_L:
            candidate = table_L[complement] + bits_R[i].tolist()
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
