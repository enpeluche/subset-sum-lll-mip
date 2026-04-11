from SubsetSumInstance import SubsetSumInstance

def compute_greedy_bounds(instance: SubsetSumInstance) -> tuple[int, int]:
    """
    Compute [k_lo, k_hi] — guaranteed bounds on k* = Σxᵢ.

    k_lo: min elements to reach T (take largest first)
    k_hi: max elements to reach T (take smallest first)

    Guarantee: k* ∈ [k_lo, k_hi] ALWAYS.
    Complexity: O(n log n).
    """
    n       = instance.n
    weights = instance.weights
    T       = instance.target

    sorted_desc = sorted(weights, reverse=True)
    s, k_lo = 0, 0
    for w in sorted_desc:
        s += w; k_lo += 1
        if s >= T: break
    else:
        k_lo = n

    sorted_asc = sorted(weights)
    s, k_hi = 0, 0
    for w in sorted_asc:
        s += w; k_hi += 1
        if s >= T: break
    else:
        k_hi = n

    return k_lo, min(k_hi, n)