from SubsetSumInstance import SubsetSumInstance
from itertools import combinations
import time

def _enum_symmetric(
    instance: SubsetSumInstance,
    k: int,
    start_time: float,
    timeout: float,
) -> list[int] | None:
    """
    Enumerate subsets of size k using symmetry.

    If k > n/2: enumerate n-k zeros instead of k ones.
    → min(C(n,k), C(n,n-k)) operations.

    Returns solution vector or None.
    """
    n       = instance.n
    weights = instance.weights
    T       = instance.target

    if k <= n - k:
        # Enumerate k ones directly
        for combo in combinations(range(n), k):
            if time.perf_counter() - start_time > timeout:
                return None
            if sum(weights[i] for i in combo) == T:
                x = [0] * n
                for i in combo:
                    x[i] = 1
                return x
    else:
        # Enumerate n-k zeros — fewer combinations!
        n_zeros = n - k
        total_w = sum(weights)
        for combo_zeros in combinations(range(n), n_zeros):
            if time.perf_counter() - start_time > timeout:
                return None
            sum_zeros = sum(weights[i] for i in combo_zeros)
            if total_w - sum_zeros == T:
                x = [1] * n
                for i in combo_zeros:
                    x[i] = 0
                return x

    return None
