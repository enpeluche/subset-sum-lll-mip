from SubsetSumInstance import SubsetSumInstance

def bound_tightening(
    instance: SubsetSumInstance,
    k_lo: int,
    k_hi: int,
) -> tuple[list[int], list[int]]:
    """
    Deduce forced variable values from [k_lo, k_hi] bounds.

    Rule 1 — force xi=0:
        If xi=1, remaining sum = T-wi must be achievable
        with k*-1 ∈ [k_lo-1, k_hi-1] elements from others.
        If max_sum(k_hi-1, others) < T-wi  → xi=0 forced
        If min_sum(k_lo-1, others) > T-wi  → xi=0 forced

    Rule 2 — force xi=1:
        If xi=0, sum T must be achievable
        with k* ∈ [k_lo, k_hi] elements from others.
        If max_sum(k_hi, others) < T  → xi=1 forced
        (cannot reach T without xi)

    Complexity: O(n log n) — one sort, then O(n) per variable.

    Args:
        instance: A SubsetSumInstance.
        k_lo:     Lower bound on k*.
        k_hi:     Upper bound on k*.

    Returns:
        (fixed_zeros, fixed_ones) — lists of variable indices.
    """
    n       = instance.n
    weights = instance.weights
    T       = instance.target

    # Precompute sorted arrays for fast prefix sums
    sorted_asc  = sorted(weights)
    sorted_desc = sorted(weights, reverse=True)

    # Max sum with k elements from ALL weights (upper bound)
    def max_sum_k(k: int) -> int:
        return sum(sorted_desc[:k]) if k > 0 else 0

    # Min sum with k elements from ALL weights (lower bound)
    def min_sum_k(k: int) -> int:
        return sum(sorted_asc[:k]) if k > 0 else 0

    fixed_zeros = []
    fixed_ones  = []

    for i, wi in enumerate(weights):

        # ── Rule 1: can xi=1 be part of a valid solution? ──────────────────
        remaining = T - wi

        if remaining < 0:
            # wi > T → can never take xi alone, but maybe with negative...
            # Since all weights > 0, xi=1 impossible if remaining < 0
            fixed_zeros.append(i)
            continue

        # Need to achieve `remaining` with k*-1 ∈ [k_lo-1, k_hi-1] elements
        # from the OTHER n-1 weights.
        # Conservative: use all n weights (slightly looser but O(1))
        k_rest_lo = max(0, k_lo - 1)
        k_rest_hi = min(n - 1, k_hi - 1)

        if k_rest_hi < 0:
            fixed_zeros.append(i)
            continue

        # Max achievable with k_rest_hi elements (from all — conservative)
        max_rest = max_sum_k(k_rest_hi)
        # Min achievable with k_rest_lo elements
        min_rest = min_sum_k(k_rest_lo)

        if remaining > max_rest or (k_rest_lo > 0 and remaining < min_rest):
            fixed_zeros.append(i)
            continue

        # ── Rule 2: can xi=0 lead to a valid solution? ─────────────────────
        # Need to achieve T with k* ∈ [k_lo, k_hi] elements from others.
        # Conservative check: can we reach T with k_hi elements total?
        max_total = max_sum_k(k_hi)

        if max_total < T:
            # Cannot reach T even with k_hi largest weights → xi=1 forced
            fixed_ones.append(i)

    # Remove conflicts (can't be both 0 and 1)
    ones_set  = set(fixed_ones)
    zeros_set = set(fixed_zeros)
    conflict  = ones_set & zeros_set

    # If conflict → instance might be infeasible in this window
    # Remove conflicted indices from both lists (let CP-SAT handle it)
    fixed_zeros = [i for i in fixed_zeros if i not in conflict]
    fixed_ones  = [i for i in fixed_ones  if i not in conflict]

    return fixed_zeros, fixed_ones