"""
solve_mitm_hgj.py
-----------------
HGJ-inspired Meet-in-the-Middle with LLL.

Core insight (from Howgrave-Graham & Joux 2010):
    Instead of one fixed split of indices into (L, R), try many random
    partitions. For each partition:
        - LLL without target on LEFT  → few binary candidates (fast)
        - LLL WITH target on RIGHT    → solves right half exactly (d_eff ≈ d/2)

    Why this works:
        At density d_eff = d/2 ≈ 0.40, LLL WITH a fixed target finds the
        solution with ~100% probability. The left half gives us a "probe"
        sum s_L, and we ask: "does T - s_L happen to be achievable by the
        right half?" With enough random splits, yes.

    This avoids exhaustive enumeration entirely — complexity is
    O(k × LLL_time) where k is the number of trials (~50-200).

    Connection to HGJ: their "representation technique" creates many ways
    to write x* = xL + xR, increasing matching probability. Our random
    split achieves a similar effect empirically.
"""

import time
import random
from fpylll import LLL, BKZ, IntegerMatrix

from SubsetSumInstance import SubsetSumInstance
from results import SolveResult
from utils import extract_vectors_from_basis, filter_binary_vectors


# =============================================================================
# Half-lattice helpers
# =============================================================================


def _build_half_lattice_no_target(weights: list[int], M: int) -> IntegerMatrix:
    """
    Build a (m+1)×(m+1) knapsack lattice for a subset of weights
    WITHOUT fixing a target sum.

    Short binary vectors in this lattice correspond to subsets with
    small partial sums. Used for the LEFT half (probe side).

    Structure:
        [a₁*M   1  0 ... 0   0]
        ...
        [aₘ*M   0  0 ... 1   0]
        [0      0  0 ... 0   Q]   ← neutral padding

    Args:
        weights: List of m positive integers.
        M:       Scaling factor.

    Returns:
        (m+1) × (m+1) IntegerMatrix.
    """
    m = len(weights)
    Q = M * sum(weights) + 1

    matrix = [[0] * (m + 1) for _ in range(m + 1)]
    for i in range(m):
        matrix[i][0] = weights[i] * M
        matrix[i][i + 1] = 1
    matrix[m][m] = Q

    return IntegerMatrix.from_matrix(matrix)


def _build_half_lattice_with_target(
    weights: list[int], target: int, M: int
) -> IntegerMatrix:
    """
    Build a (m+1)×(m+1) standard knapsack lattice WITH a fixed target.

    The unique binary vector encoding the solution has a zero in the
    weight column → it is short → LLL finds it with high probability
    at low density.

    Structure:
        [a₁*M   1  0 ... 0]
        ...
        [aₘ*M   0  0 ... 1]
        [-T*M   0  0 ... 0]   ← target row

    Args:
        weights: List of m positive integers.
        target:  Target partial sum for this half.
        M:       Scaling factor.

    Returns:
        (m+1) × (m+1) IntegerMatrix.
    """
    m = len(weights)

    matrix = [[0] * (m + 1) for _ in range(m + 1)]
    for i in range(m):
        matrix[i][0] = weights[i] * M
        matrix[i][i + 1] = 1
    matrix[m][0] = -target * M

    return IntegerMatrix.from_matrix(matrix)


def _lll_candidates_no_target(
    weights: list[int],
    M: int,
    use_bkz: bool = False,
    block_size: int = 20,
) -> dict[int, list[int]]:
    """
    LLL without target: find binary vectors with small partial sums.
    Returns dict mapping partial_sum → binary_vector.
    """
    B = _build_half_lattice_no_target(weights, M)

    if use_bkz:
        BKZ.reduction(B, BKZ.Param(block_size=block_size))
    else:
        LLL.reduction(B)

    m = len(weights)
    candidates = {}

    for row in range(B.nrows):
        coeffs = [B[row][i + 1] for i in range(m)]
        if not all(c in (0, 1) for c in coeffs):
            continue
        s = sum(weights[i] * coeffs[i] for i in range(m))
        if s not in candidates:
            candidates[s] = coeffs

    return candidates

    # Remplace _lll_candidates_no_target par :


def _lll_candidates_centered(weights_L, target_total, M, n_probes=5):
    """
    Try several LLL runs with targets near T/2 on the left half.
    """
    candidates = {}
    T_half = target_total // 2
    spread = sum(weights_L) // 4  # fourchette autour de T/2

    for _ in range(n_probes):
        # Cible aléatoire proche de T/2
        probe = random.randint(
            max(0, T_half - spread), min(sum(weights_L), T_half + spread)
        )
        result = _lll_solve_with_target(weights_L, probe, M)
        if result is not None:
            s = sum(weights_L[i] * result[i] for i in range(len(weights_L)))
            candidates[s] = result

    return candidates


def _lll_solve_with_target(
    weights: list[int],
    target: int,
    M: int,
    use_bkz: bool = False,
    block_size: int = 20,
) -> list[int] | None:
    """
    LLL with target: try to find the binary vector summing to target.
    Returns binary solution if found, None otherwise.

    At d_eff = d/2 ≈ 0.40, succeeds with ~100% probability.
    """
    if target < 0 or target > sum(weights):
        return None

    B = _build_half_lattice_with_target(weights, target, M)

    if use_bkz:
        BKZ.reduction(B, BKZ.Param(block_size=block_size))
    else:
        LLL.reduction(B)

    vectors = extract_vectors_from_basis(B)

    for v in filter_binary_vectors(vectors):
        if sum(weights[i] * v[i] for i in range(len(weights))) == target:
            return v

    return None


# =============================================================================
# Main solver
# =============================================================================


def solve_mitm_hgj(
    instance: SubsetSumInstance,
    k_trials: int = 200,
    scaling: int | None = None,
    use_bkz: bool = False,
    block_size: int = 20,
    seed: int | None = None,
) -> SolveResult:
    """
    HGJ-inspired MITM with LLL and random multi-splits.

    For each of k_trials random partitions of indices into (L, R):
        1. LLL without target on LEFT  → binary candidates with small sums
        2. For each left candidate xL:
               target_R = T - sum(xL)
               LLL with target_R on RIGHT → solves right half if feasible
        3. If both halves found, reconstruct and verify solution.

    This approach avoids exhaustive enumeration entirely.
    Complexity: O(k_trials × LLL_time), typically k_trials = 50-200.

    Connection to HGJ: random splits explore many "views" of the problem,
    analogous to HGJ's representation technique which creates many ways
    to write x* = xL + xR.

    Args:
        instance:   A SubsetSumInstance.
        k_trials:   Number of random splits to try.
        scaling:    Scaling factor M. Defaults to 2^(n/2).
        use_bkz:    Use BKZ instead of LLL (stronger, ~3ms extra per half).
        block_size: BKZ block size.
        seed:       Random seed for reproducibility.

    Returns:
        SolveResult with label in {
            'HGJ_Found_trial_k' : solved at trial k,
            'HGJ_NotFound'      : no solution found after k_trials,
        }
    """
    start = time.perf_counter()
    n = instance.n
    mid = n // 2
    total_branches = 0

    if scaling is None:
        scaling = 2 ** mid

    if seed is not None:
        random.seed(seed)

    for trial in range(k_trials):

        # ------------------------------------------------------------------
        # Step 1 — Random partition of indices
        # ------------------------------------------------------------------
        perm = list(range(n))
        random.shuffle(perm)
        idx_L = sorted(perm[:mid])
        idx_R = sorted(perm[mid:])

        w_L = [instance.weights[i] for i in idx_L]
        w_R = [instance.weights[i] for i in idx_R]

        # ------------------------------------------------------------------
        # Step 2 — LLL without target on LEFT half
        # Finds binary vectors with small partial sums (not necessarily x*_L)
        # ------------------------------------------------------------------
        cands_L = _lll_candidates_centered(w_L, instance.target, scaling, n_probes=5)
        total_branches += len(cands_L)

        # ------------------------------------------------------------------
        # Step 3 — For each left candidate: LLL WITH target on RIGHT half
        # At d_eff ≈ d/2, LLL with fixed target succeeds ~100%
        # ------------------------------------------------------------------
        for s_L, xL_local in cands_L.items():
            target_R = instance.target - s_L

            xR_local = _lll_solve_with_target(
                w_R, target_R, scaling, use_bkz=use_bkz, block_size=block_size
            )

            if xR_local is None:
                continue

            # ------------------------------------------------------------------
            # Step 4 — Reconstruct solution in original index order
            # ------------------------------------------------------------------
            x = [0] * n
            for j, orig_i in enumerate(idx_L):
                x[orig_i] = xL_local[j]
            for j, orig_i in enumerate(idx_R):
                x[orig_i] = xR_local[j]

            if instance.is_solution(x):
                return SolveResult(
                    elapsed=time.perf_counter() - start,
                    branches=total_branches,
                    conflicts=trial + 1,  # conflicts = trials used
                    status=0,
                    solution=x,
                    label=f"HGJ_Found_trial_{trial + 1}",
                    best_res=None,
                    best_ham=None,
                )

    return SolveResult(
        elapsed=time.perf_counter() - start,
        branches=total_branches,
        conflicts=k_trials,
        status=3,
        solution=None,
        label="HGJ_NotFound",
        best_res=None,
        best_ham=None,
    )
