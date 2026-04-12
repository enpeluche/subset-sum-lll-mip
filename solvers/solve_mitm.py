"""
Meet-in-the-Middle solver for Subset Sum.

Uses Gray code enumeration: O(1) per subset (single addition/subtraction),
zero allocation beyond the left-half lookup table.

Complexity: O(2^(n/2)) time and space.
"""

import time
from SubsetSumInstance import SubsetSumInstance
from results import SolveResult


# ---------------------------------------------------------------------------
# Gray code helpers (I made a thesis on this subject)
# ---------------------------------------------------------------------------

def _gray_bit(i: int) -> int:
    """Return the index of the bit that flips between Gray(i-1) and Gray(i)."""
    return (i & -i).bit_length() - 1


def _build_table_gray(weights: list[int]) -> dict[int, int]:
    """
    Build {sum: gray_code_mask} for all 2^m subsets using Gray code.

    Each step flips one bit → one add or subtract → O(1) per subset.
    We store the Gray mask (not the index) so we can recover bits directly.
    Only keeps the first occurrence per sum (sufficient for feasibility).
    """
    m = len(weights)
    size = 1 << m
    table: dict[int, int] = {0: 0}  # empty subset → sum 0
    s = 0
    gray = 0

    for i in range(1, size):
        bit = _gray_bit(i)
        gray ^= (1 << bit)

        # Gray code flips one bit: add if it turned ON, subtract if OFF
        if gray & (1 << bit):
            s += weights[bit]
        else:
            s -= weights[bit]

        if s not in table:
            table[s] = gray

    return table


def _mask_to_bits(mask: int, length: int) -> list[int]:
    """Convert a bitmask to a list of 0/1 of given length."""
    return [(mask >> j) & 1 for j in range(length)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_mitm_classic(
    instance: SubsetSumInstance,
    max_subsets: int = 1 << 22,
    workers: int = 6
) -> SolveResult:
    """
    Meet-in-the-Middle with Gray code enumeration.

    1. Build lookup table for the left half  (2^⌊n/2⌋ entries)
    2. Scan the right half with Gray code, checking complements

    Args:
        instance:    SubsetSumInstance to solve.
        max_subsets:  Skip if 2^(n/2) exceeds this (memory guard).

    Returns:
        SolveResult with the solution if found.
    """
    start = time.perf_counter()
    n = instance.n
    mid = n // 2

    # 0. Trivially infeasible
    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)

    # 1. Size guard
    if (1 << mid) > max_subsets:
        return SolveResult.skipped("MITM_Skipped_Size_Limit")

    weights_L = instance.weights[:mid]
    weights_R = instance.weights[mid:]
    target = instance.target

    # 2. Build left table — O(2^mid) time and space
    table_L = _build_table_gray(weights_L)

    # 3. Scan right half with Gray code — O(2^(n-mid)), O(1) space
    size_R = 1 << len(weights_R)
    s = 0
    gray = 0
    branches = (1 << mid) + size_R

    # Check empty right subset first
    complement = target - s
    solution = None
    if complement in table_L:
        left_bits = _mask_to_bits(table_L[complement], mid)
        right_bits = _mask_to_bits(0, n - mid)
        candidate = left_bits + right_bits
        if instance.is_solution(candidate):
            solution = candidate

    if solution is None:
        for i in range(1, size_R):
            bit = _gray_bit(i)
            gray ^= (1 << bit)

            if gray & (1 << bit):
                s += weights_R[bit]
            else:
                s -= weights_R[bit]

            complement = target - s
            if complement in table_L:
                left_bits = _mask_to_bits(table_L[complement], mid)
                right_bits = _mask_to_bits(gray, n - mid)
                candidate = left_bits + right_bits
                if instance.is_solution(candidate):
                    solution = candidate
                    break

    elapsed = time.perf_counter() - start
    return SolveResult(
        elapsed=elapsed,
        branches=branches,
        conflicts=0,
        status=0 if solution else 3,
        solution=solution,
        label="MITM_Found" if solution else "MITM_NotFound",
        best_res=0 if solution else None,
        best_ham=instance.hamming_to_solution(solution) if solution else None,
    )