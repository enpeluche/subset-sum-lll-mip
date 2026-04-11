"""
solve_mitm_hgj.py
-----------------
Howgrave-Graham & Joux (2010) inspired MITM with multiple representations.

Key idea vs classic MITM:
    Classic: x* = xL + xR, xL/xR ∈ {0,1}^(n/2)
             → 1 representation → matching rate ≈ 0

    HGJ:     x* = y + z, y ⊂ A, z ⊂ B (A∩B=∅, |A|=|B|=n/2)
             with |y| = |z| = n/4
             → C(n/2, n/4) ≈ 2^(n/2) representations → matching O(1)

Core design choices:
    1. Random partition A/B: disjointness guaranteed by construction
       → x = y + z always ∈ {0,1}^n, no overlap check needed

    2. Direct enumeration on C(n/2, k) subsets per list
       → avoids inner MITM which produced near-empty lists

    3. M ≈ √C(n/2, k): calibrated on actual list size (not C(n,k))
       → list size ≈ √C(n/2,k), birthday paradox optimum

    4. Modular matching then exact check:
       → filter candidates by sum mod M, then verify sum_y + sum_z = T

Complexity per trial:
    List construction: O(C(n/2, n/4)) = O(C(15,7)) = O(6435) for n=30
    Matching:          O(list_size²/M) ≈ O(√C(n/2,k))
    Total per trial:   O(C(n/2, n/4))

References:
    Howgrave-Graham & Joux (2010) EUROCRYPT
        "A new generic algorithm for hard knapsacks"
    Becker, Coron & Joux (2011) EUROCRYPT
        "Improved generic algorithms for hard knapsacks"
"""

import time
import random
import math
from itertools import combinations

from SubsetSumInstance import SubsetSumInstance
from results import SolveResult


# =============================================================================
# Helpers
# =============================================================================

def _binom(n: int, k: int) -> int:
    return math.comb(n, k)


def _choose_modulus(n: int, k: int) -> int:
    """
    M ≈ √C(n/2, k) — calibrated on actual list size.

    Each list enumerates C(n/2, k) subsets.
    M = √C(n/2, k) gives list_size ≈ √C(n/2,k) after modular filtering
    → birthday paradox optimum for this list size.

    Args:
        n: Instance dimension.
        k: Hamming weight per sub-vector (= n//4).

    Returns:
        Modulus M.
    """
    list_total = _binom(n // 2, k)  # actual enumeration size per list
    M = max(1, int(math.isqrt(list_total)))
    return min(M, 2**25)


# =============================================================================
# List construction — direct enumeration
# =============================================================================

def _build_list(
    weights: list[int],
    full_indices: list[int],
    k: int,
    target_mod: int,
    M: int,
) -> dict[int, list[int]]:
    """
    Enumerate all weight-k subsets of full_indices with sum ≡ target_mod (mod M).

    Uses direct enumeration of C(|full_indices|, k) subsets.
    Disjointness from the other list is guaranteed by the caller
    (full_indices = A or B, disjoint partitions of {0..n-1}).

    Args:
        weights:      Full weight vector of the instance.
        full_indices: Partition half to enumerate over (A or B).
        k:            Desired Hamming weight.
        target_mod:   Target sum modulo M.
        M:            Modulus.

    Returns:
        Dict mapping exact_sum → binary vector of length n.
        First occurrence wins on collision.
    """
    n_full = len(weights)
    result: dict[int, list[int]] = {}

    for combo in combinations(range(len(full_indices)), k):
        s = sum(weights[full_indices[i]] for i in combo)
        if s % M == target_mod:
            if s not in result:
                vec = [0] * n_full
                for i in combo:
                    vec[full_indices[i]] = 1
                result[s] = vec

    return result


def _build_list_modular(
    weights: list[int],
    full_indices: list[int],
    k: int,
    target_mod: int,
    M: int,
) -> dict[int, list[tuple[int, list[int]]]]:
    """
    Same as _build_list but groups by (sum mod M) → list of (exact_sum, vec).
    Used for the right list in modular matching.

    Returns:
        Dict mapping (sum mod M) → list of (exact_sum, binary_vector).
    """
    n_full = len(weights)
    result: dict[int, list[tuple[int, list[int]]]] = {}

    for combo in combinations(range(len(full_indices)), k):
        s = sum(weights[full_indices[i]] for i in combo)
        key = s % M
        if key not in result:
            result[key] = []
        vec = [0] * n_full
        for i in combo:
            vec[full_indices[i]] = 1
        result[key].append((s, vec))

    return result


# =============================================================================
# Main solver
# =============================================================================

def solve_mitm_hgj(
    instance: SubsetSumInstance,
    k_trials: int = 200,
    seed: int | None = None,
) -> SolveResult:
    """
    HGJ-inspired MITM solver with multiple representations.

    For each trial:
        1. Random partition of {0..n-1} into A and B (|A|=|B|=n/2)
        2. Random modular target R
        3. Left  list: y ⊂ A, |y|=n/4, Σaᵢyᵢ ≡ R     (mod M)
        4. Right list: z ⊂ B, |z|=n/4, grouped by sum mod M
        5. Modular matching: find (y,z) with (sum_y + sum_z) % M == T % M
           then verify sum_y + sum_z == T exactly
           → x = y + z always binary (A∩B=∅)

    The C(n/2, n/4) representations of x* across random partitions
    ensure high probability of finding a valid pair within k_trials.

    Args:
        instance:  A SubsetSumInstance.
        k_trials:  Number of random partitions to try.
        seed:      Random seed for reproducibility.

    Returns:
        SolveResult with label in {
            'HGJ_Found_trial_k' : solution found at trial k,
            'HGJ_NotFound'      : exhausted k_trials,
            'HGJ_TooSmall'      : n < 4,
        }
    """
    start   = time.perf_counter()
    n       = instance.n
    weights = instance.weights
    T       = instance.target

    if seed is not None:
        random.seed(seed)

    k = n // 4
    if k == 0:
        return SolveResult(
            elapsed=time.perf_counter() - start,
            branches=0, conflicts=0, status=3,
            solution=None, label="HGJ_TooSmall",
            best_res=None, best_ham=None,
        )

    M              = _choose_modulus(n, k)
    T_mod          = T % M
    total_branches = 0
    best_res       = 1 << 128

    for trial in range(k_trials):

        # ------------------------------------------------------------------
        # Step 1 — Random partition A/B
        # y ⊂ A, z ⊂ B → x = y+z always ∈ {0,1}^n
        # ------------------------------------------------------------------
        perm = list(range(n))
        random.shuffle(perm)
        A = perm[:n // 2]
        B = perm[n // 2:]

        R = random.randint(0, M - 1)

        # ------------------------------------------------------------------
        # Step 2 — Build left list: exact_sum → vec, filtered by sum%M == R
        # ------------------------------------------------------------------
        list_L = _build_list(weights, A, k, R, M)

        # ------------------------------------------------------------------
        # Step 3 — Build right list: grouped by sum%M
        # No target_mod filter — we keep all and lookup by complement
        # ------------------------------------------------------------------
        list_R_mod: dict[int, list[tuple[int, list[int]]]] = {}
        for combo in combinations(range(len(B)), k):
            s = sum(weights[B[i]] for i in combo)
            key = s % M
            if key not in list_R_mod:
                list_R_mod[key] = []
            vec = [0] * n
            for i in combo:
                vec[B[i]] = 1
            list_R_mod[key].append((s, vec))

        total_branches += len(list_L) + sum(
            len(v) for v in list_R_mod.values()
        )

        # ------------------------------------------------------------------
        # Step 4 — Modular matching then exact check
        # sum_y ≡ R (mod M) and sum_z ≡ T-R (mod M) → sum_y+sum_z ≡ T (mod M)
        # Then verify exact integer equality
        # ------------------------------------------------------------------
        need_mod = (T_mod - R) % M

        if need_mod not in list_R_mod:
            continue

        for sum_y, y_vec in list_L.items():
            for sum_z, z_vec in list_R_mod[need_mod]:
                if sum_y + sum_z == T:
                    # Exact match — reconstruct x = y + z
                    x = [y_vec[i] + z_vec[i] for i in range(n)]
                    if instance.is_solution(x):
                        return SolveResult(
                            elapsed=time.perf_counter() - start,
                            branches=total_branches,
                            conflicts=trial + 1,
                            status=0,
                            solution=x,
                            label=f"HGJ_Found_trial_{trial + 1}",
                            best_res=0,
                            best_ham=instance.hamming_to_solution(x),
                        )
                # Track best residual
                res = abs(sum_y + sum_z - T)
                if res < best_res:
                    best_res = res

    return SolveResult(
        elapsed=time.perf_counter() - start,
        branches=total_branches,
        conflicts=k_trials,
        status=3,
        solution=None,
        label="HGJ_NotFound",
        best_res=best_res,
        best_ham=None,
    )


# =============================================================================
# Diagnostic
# =============================================================================

def diagnose_hgj(
    instance: SubsetSumInstance,
    n_trials: int = 20,
) -> None:
    """
    Measure actual list sizes, matching rate and success rate.
    """
    import numpy as np

    n = instance.n
    k = n // 4
    M = _choose_modulus(n, k)

    reps       = _binom(n // 2, n // 4)
    list_total = _binom(n // 2, k)

    print(f"\n{'='*55}")
    print(f"HGJ Diagnostics — n={n}, k={k}")
    print(f"{'='*55}")
    print(f"  C(n/2, k)    = {list_total:,}   (subsets enumerated per list)")
    print(f"  C(n/2, n/4)  = {reps:,}   (representations of x*)")
    print(f"  M            = {M:,}   (≈ √C(n/2,k))")
    print(f"  Expected list size ≈ {list_total // M:,}")
    print(f"  Expected reps/trial ≈ {reps / M:.2f}")
    print()

    sizes_L, sizes_R = [], []
    found = 0

    for _ in range(n_trials):
        perm = list(range(n))
        random.shuffle(perm)
        A, B = perm[:n // 2], perm[n // 2:]
        R    = random.randint(0, M - 1)

        lL = _build_list(instance.weights, A, k, R, M)

        lR_mod: dict[int, list] = {}
        for combo in combinations(range(len(B)), k):
            s = sum(instance.weights[B[i]] for i in combo)
            key = s % M
            if key not in lR_mod:
                lR_mod[key] = []
            vec = [0] * n
            for i in combo:
                vec[B[i]] = 1
            lR_mod[key].append((s, vec))

        sizes_L.append(len(lL))
        sizes_R.append(sum(len(v) for v in lR_mod.values()))

        need_mod = (instance.target % M - R) % M
        if need_mod in lR_mod:
            for sy, yv in lL.items():
                for sz, zv in lR_mod[need_mod]:
                    if sy + sz == instance.target:
                        x = [yv[i] + zv[i] for i in range(n)]
                        if instance.is_solution(x):
                            found += 1
                            break

    print(f"  Left  list: mean={np.mean(sizes_L):.1f}, "
          f"min={min(sizes_L)}, max={max(sizes_L)}")
    print(f"  Right list: mean={np.mean(sizes_R):.1f}, "
          f"min={min(sizes_R)}, max={max(sizes_R)}")
    print(f"  Success   : {found}/{n_trials} ({found/n_trials*100:.1f}%)")
    print(f"{'='*55}")