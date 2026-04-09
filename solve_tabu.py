"""
solve_tabu.py
-------------
Tabu Search solver for Subset Sum with LLL warm start.

Strategy:
    1. LLL reduction → clip to {0,1} as warm start (works even with
       non-binary LLL vectors — we just clip them)
    2. Tabu search on the residual |Σ aᵢxᵢ - T|:
         - Neighbourhood: 1-flip (flip one bit)
         - Tabu list: forbid re-flipping j for `tenure` iterations
         - Aspiration: accept tabu move if it beats best residual seen
    3. Random restart if stuck

Key insight: LLL vectors contain values in {-3,-2,-1,0,1,2,3}.
We clip them to {0,1} for initialization — this gives a starting point
that is geometrically close to the solution, even if not binary.
The tabu search then corrects the remaining bits.

This is novel: LLL warm start + tabu search. Existing tabu approaches
use random initialization. The LLL clip gives a residual-minimizing
start, reducing the work for tabu.

Complexity per iteration: O(n) — evaluates all n 1-flip neighbours.
"""

import time
import random
import math

from SubsetSumInstance import SubsetSumInstance
from results import SolveResult
from utils import extract_vectors_from_basis, filter_binary_vectors
from fpylll import LLL, BKZ


# =============================================================================
# Initialization strategies
# =============================================================================


def _lll_warm_start(instance: SubsetSumInstance, scaling: int) -> list[int]:
    """
    Use LLL to generate a warm start binary vector.

    Even if LLL vectors are non-binary ({-3,...,3}), we clip them to {0,1}.
    This gives a starting point geometrically close to the solution.

    Strategy: among all LLL basis vectors, pick the one with minimum
    residual after clipping. Falls back to random if LLL fails.

    Args:
        instance: A SubsetSumInstance.
        scaling:  Lattice scaling factor M.

    Returns:
        Binary vector of length n.
    """
    n = instance.n
    B = instance.to_knapsack_matrix(M=scaling)
    LLL.reduction(B)
    vectors = extract_vectors_from_basis(B)

    if not vectors:
        return [random.randint(0, 1) for _ in range(n)]

    # Check for direct binary candidates first
    candidates = filter_binary_vectors(vectors)
    if candidates:
        candidates.sort(key=lambda c: abs(instance.residual(c)))
        if instance.is_solution(candidates[0]):
            return candidates[0]

    # Clip all vectors and pick best by residual
    best_x = None
    best_res = float("inf")

    for v in vectors:
        # Clip to {0,1} — handles values like -2, 3 etc.
        x = [max(0, min(1, round(float(c)))) for c in v]
        res = abs(instance.residual(x))
        if res < best_res:
            best_res = res
            best_x = x

    return best_x if best_x is not None else [random.randint(0, 1) for _ in range(n)]


def _sign_warm_start(instance: SubsetSumInstance, scaling: int) -> list[int]:
    """
    Alternative warm start: use sign of shortest LLL vector.
    1 if v_j > 0, else 0. Different from clip for negative/large values.
    """
    n = instance.n
    B = instance.to_knapsack_matrix(M=scaling)
    LLL.reduction(B)
    vectors = extract_vectors_from_basis(B)

    if not vectors:
        return [random.randint(0, 1) for _ in range(n)]

    v = vectors[0]
    return [1 if c > 0 else 0 for c in v]


# =============================================================================
# Core Tabu Search
# =============================================================================


def _tabu_search(
    instance: SubsetSumInstance,
    x_init: list[int],
    n_iter: int,
    tenure: int,
    start_time: float,
    iter_budget: float,
) -> tuple[list[int] | None, int]:
    """
    Core tabu search loop.

    Args:
        instance:     SubsetSumInstance.
        x_init:       Initial binary vector.
        n_iter:       Max iterations.
        tenure:       Tabu tenure (how long a flip is forbidden).
        start_time:   Global start time (for timeout check).
        iter_budget:  Max wall time for this tabu run.

    Returns:
        (solution, branches) where solution is None if not found.
    """
    n = instance.n
    weights = instance.weights
    T = instance.target

    x = x_init.copy()
    residual = instance.residual(x)  # signed residual Σaᵢxᵢ - T

    best_x = x.copy()
    best_res = abs(residual)

    # Tabu list: tabu_list[j] = iteration until which j is tabu
    tabu_until = [0] * n
    branches = 0
    deadline = start_time + iter_budget

    for it in range(n_iter):

        if time.perf_counter() > deadline:
            break

        # ------------------------------------------------------------------
        # Evaluate all 1-flip neighbours
        # ------------------------------------------------------------------
        best_move = -1
        best_move_res = float("inf")

        for j in range(n):
            branches += 1

            # Delta residual if we flip j
            # x[j]=0 → flip to 1 : residual += w[j]
            # x[j]=1 → flip to 0 : residual -= w[j]
            delta = weights[j] if x[j] == 0 else -weights[j]
            new_res = abs(residual + delta)

            is_tabu = tabu_until[j] > it
            aspiration = new_res < best_res  # aspiration criterion

            if (not is_tabu or aspiration) and new_res < best_move_res:
                best_move = j
                best_move_res = new_res

        if best_move == -1:
            break  # all moves tabu and no aspiration

        # ------------------------------------------------------------------
        # Apply best move
        # ------------------------------------------------------------------
        delta = weights[best_move] if x[best_move] == 0 else -weights[best_move]
        x[best_move] = 1 - x[best_move]
        residual += delta
        tabu_until[best_move] = it + tenure

        # ------------------------------------------------------------------
        # Update best
        # ------------------------------------------------------------------
        if abs(residual) < best_res:
            best_res = abs(residual)
            best_x = x.copy()

        if residual == 0:
            return x, branches  # solution found

    # Return best found (may not be solution)
    if best_res == 0:
        return best_x, branches
    return None, branches


# =============================================================================
# Main solver
# =============================================================================


def solve_tabu(
    instance: SubsetSumInstance,
    scaling: int | None = None,
    n_iter: int = 10_000,
    tenure: int = 7,
    n_restarts: int = 5,
    warm_start: str = "lll",  # "lll", "sign", "random"
) -> SolveResult:
    """
    Tabu Search solver for Subset Sum with LLL warm start.

    Neighbourhood: 1-flip — flip one bit at a time.
    Objective: minimize |Σ aᵢxᵢ - T| toward 0.
    Tabu list: forbid re-flipping j for `tenure` iterations.
    Aspiration: accept tabu move if it beats best residual seen.

    Three warm start strategies:
        'lll'    : clip of best LLL vector (min residual after clipping)
        'sign'   : sign projection of shortest LLL vector
        'random' : random binary vector

    Args:
        instance:   A SubsetSumInstance.
        scaling:    Scaling factor M. Defaults to 2^n.
        n_iter:     Max iterations per restart.
        tenure:     Tabu tenure.
        n_restarts: Number of random restarts.
        warm_start: Initialization strategy.

    Returns:
        SolveResult with label in {
            'Tabu_Direct'     : LLL found solution directly (clip = solution),
            'Tabu_restart_k'  : solved at restart k,
            'Tabu_NotFound'   : exhausted all restarts,
        }
    """
    start = time.perf_counter()
    n = instance.n
    total_branches = 0
    total_conflicts = 0

    if scaling is None:
        scaling = 2 ** n

    budget_per_restart = (10.0 - 0.1) / n_restarts  # reserve 100ms for LLL

    # ------------------------------------------------------------------
    # Step 1 — LLL warm start
    # ------------------------------------------------------------------
    if warm_start == "lll":
        x_init = _lll_warm_start(instance, scaling)
    elif warm_start == "sign":
        x_init = _sign_warm_start(instance, scaling)
    else:
        x_init = [random.randint(0, 1) for _ in range(n)]

    # Check if LLL clip is already a solution
    if instance.is_solution(x_init):
        return SolveResult(
            elapsed=time.perf_counter() - start,
            branches=0,
            conflicts=0,
            status=0,
            solution=x_init,
            label="Tabu_Direct",
            best_res=0,
            best_ham=instance.hamming_to_solution(x_init),
        )

    best_residual = abs(instance.residual(x_init))
    best_hamming = instance.hamming_to_solution(x_init)

    # ------------------------------------------------------------------
    # Step 2 — Tabu search with restarts
    # ------------------------------------------------------------------
    for restart in range(n_restarts):
        elapsed = time.perf_counter() - start
        if elapsed >= 10.0:
            break

        # First restart: LLL warm start
        # Subsequent restarts: random perturbation of best known start
        if restart == 0:
            x_start = x_init.copy()
        else:
            # Perturb: flip k random bits of the warm start
            x_start = x_init.copy()
            k_perturb = random.randint(1, max(1, n // 5))
            for j in random.sample(range(n), k_perturb):
                x_start[j] = 1 - x_start[j]

        solution, branches = _tabu_search(
            instance=instance,
            x_init=x_start,
            n_iter=n_iter,
            tenure=tenure,
            start_time=start,
            iter_budget=budget_per_restart,
        )
        total_branches += branches

        if solution is not None and instance.is_solution(solution):
            return SolveResult(
                elapsed=time.perf_counter() - start,
                branches=total_branches,
                conflicts=total_conflicts,
                status=0,
                solution=solution,
                label=f"Tabu_restart_{restart + 1}",
                best_res=best_residual,
                best_ham=best_hamming,
            )

    return SolveResult(
        elapsed=time.perf_counter() - start,
        branches=total_branches,
        conflicts=total_conflicts,
        status=3,
        solution=None,
        label="Tabu_NotFound",
        best_res=best_residual,
        best_ham=best_hamming,
    )
