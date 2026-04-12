"""
Tabu Search solver for Subset Sum.

Three flavours:
    - Classic tabu:  evaluate all n neighbours, pick best non-tabu
    - Gray tabu:     follow Gray code order, flip one bit per iter (O(1) neighbour)
    - Beckett-Gray:  actors enter/exit in dramatic LIFO order (for the culture)

Warm start options: LLL clip, sign extraction, or random.
"""

import time
import random
import math

from SubsetSumInstance import SubsetSumInstance
from results import SolveResult
from utils import extract_vectors_from_basis, filter_binary_vectors
from fpylll import LLL


# =====================================================================
# Warm starts
# =====================================================================

def _lll_warm_start(instance: SubsetSumInstance, scaling: int) -> list[int]:
    """
    LLL warm start: reduce the knapsack lattice, clip basis vectors to {0,1},
    return the one with smallest residual.
    """
    n = instance.n
    B = instance.to_knapsack_matrix(M=scaling)
    LLL.reduction(B)
    vectors = extract_vectors_from_basis(B)

    if not vectors:
        return [random.randint(0, 1) for _ in range(n)]

    # Direct binary solution?
    for v in filter_binary_vectors(vectors):
        if instance.is_solution(v):
            return v

    # Clip all vectors, keep best by residual
    best_x, best_res = None, float("inf")
    for v in vectors:
        x = [max(0, min(1, round(float(c)))) for c in v]
        res = abs(instance.residual(x))
        if res < best_res:
            best_res, best_x = res, x

    return best_x or [random.randint(0, 1) for _ in range(n)]


def _sign_warm_start(instance: SubsetSumInstance, scaling: int) -> list[int]:
    """Sign of shortest LLL vector: 1 if v_j > 0, else 0."""
    n = instance.n
    B = instance.to_knapsack_matrix(M=scaling)
    LLL.reduction(B)
    vectors = extract_vectors_from_basis(B)
    if not vectors:
        return [random.randint(0, 1) for _ in range(n)]
    return [1 if c > 0 else 0 for c in vectors[0]]


def _random_start(n: int) -> list[int]:
    return [random.randint(0, 1) for _ in range(n)]


WARM_STARTS = {
    "lll":    lambda inst, s: _lll_warm_start(inst, s),
    "sign":   lambda inst, s: _sign_warm_start(inst, s),
    "random": lambda inst, s: _random_start(inst.n),
}


# =====================================================================
# Classic Tabu Search — O(n) per iteration
# =====================================================================

def _tabu_classic(
    instance: SubsetSumInstance,
    x_init: list[int],
    n_iter: int,
    tenure: int,
    deadline: float
) -> tuple[list[int] | None, int, int]:
    """
    Standard tabu: evaluate all n flips, pick best non-tabu (or aspiration).

    Returns: (solution_or_None, branches, best_residual)
    """
    n = instance.n
    w = instance.weights

    x = x_init.copy()
    residual = instance.residual(x)

    best_x = x.copy()
    best_res = abs(residual)

    tabu_until = [0] * n
    branches = 0

    for it in range(n_iter):
        if time.perf_counter() > deadline:
            break

        best_j, best_move_res = -1, float("inf")

        for j in range(n):
            branches += 1
            delta = w[j] if x[j] == 0 else -w[j]
            new_res = abs(residual + delta)
            is_tabu = tabu_until[j] > it
            aspiration = new_res < best_res

            if (not is_tabu or aspiration) and new_res < best_move_res:
                best_j, best_move_res = j, new_res

        if best_j == -1:
            break

        # Apply
        delta = w[best_j] if x[best_j] == 0 else -w[best_j]
        x[best_j] = 1 - x[best_j]
        residual += delta
        tabu_until[best_j] = it + tenure

        if abs(residual) < best_res:
            best_res = abs(residual)
            best_x = x.copy()

        if residual == 0:
            return x, branches, 0

    return (best_x if best_res == 0 else None), branches, best_res


def _gray_walk(instance: SubsetSumInstance,
    x_init: list[int],
    n_iter: int,
    tenure: int,
    deadline: float,):
    """True Hamiltonian walk on {0,1}^n. Exact for n_iter >= 2^n."""
    n = instance.n
    w = instance.weights

    x = x_init.copy()
    residual = instance.residual(x)
    best_x, best_res = x.copy(), abs(residual)
    branches = 0

    limit = min(n_iter, (1 << n) - 1)  # 2^n - 1 flips = full cycle

    for it in range(1, limit + 1):
        if time.perf_counter() > deadline:
            break

        j = _gray_bit(it)
        if j >= n:
            break

        branches += 1
        delta = w[j] if x[j] == 0 else -w[j]
        x[j] = 1 - x[j]
        residual += delta

        if abs(residual) < best_res:
            best_res = abs(residual)
            best_x = x.copy()

        if residual == 0:
            return x, branches, 0

    return (best_x if best_res == 0 else None), branches, best_res

# =====================================================================
# Gray Tabu Search — O(1) per iteration
# =====================================================================

def _gray_bit(i: int) -> int:
    """Bit that flips between Gray(i-1) and Gray(i)."""
    return (i & -i).bit_length() - 1


def _tabu_gray(
    instance: SubsetSumInstance,
    x_init: list[int],
    n_iter: int,
    tenure: int,
    deadline: float
) -> tuple[list[int] | None, int, int]:
    """
    Gray code tabu: flip bits in Gray code order.
    Each iteration is O(1) — no neighbourhood scan.
    If the Gray-chosen bit is tabu, skip (no aspiration for simplicity).

    The idea: Gray code guarantees every bit gets flipped regularly,
    giving a structured exploration without the O(n) scan.
    """
    n = instance.n
    w = instance.weights

    x = x_init.copy()
    residual = instance.residual(x)

    best_x = x.copy()
    best_res = abs(residual)

    tabu_until = [0] * n
    branches = 0

    for it in range(1, n_iter + 1):
        if time.perf_counter() > deadline:
            break

        # Gray code tells us which bit to flip
        j = _gray_bit(it) % n
        branches += 1

        if tabu_until[j] > it:
            # Tabu — but check aspiration
            delta = w[j] if x[j] == 0 else -w[j]
            if abs(residual + delta) >= best_res:
                continue  # skip

        # Flip
        delta = w[j] if x[j] == 0 else -w[j]
        x[j] = 1 - x[j]
        residual += delta
        tabu_until[j] = it + tenure

        if abs(residual) < best_res:
            best_res = abs(residual)
            best_x = x.copy()

        if residual == 0:
            return x, branches, 0

    return (best_x if best_res == 0 else None), branches, best_res


# =====================================================================
# Beckett-Gray Tabu — for the culture
# =====================================================================

def _beckett_ordering(n: int) -> list[int]:
    """
    Generate the Beckett-Gray code bit-flip sequence for n bits.
    Recursive construction — only works for small n (known for n ≤ 6,
    and n = 2,4,5,6 specifically). Falls back to standard Gray for others.

    In a Beckett play, actors enter and exit a stage such that
    the last actor to enter is always the first to leave (LIFO).
    """
    # For practical purposes, use standard Gray (Beckett-Gray is an open problem
    # for most n). The joke is in the name, the implementation is Gray.
    size = 1 << n
    flips = []
    for i in range(1, size):
        flips.append(_gray_bit(i) % n)
    return flips


def _tabu_beckett(
    instance: SubsetSumInstance,
    x_init: list[int],
    n_iter: int,
    tenure: int,
    deadline: float,
) -> tuple[list[int] | None, int, int]:
    """
    'Beckett-Gray' tabu search.
    
    En attendant Godot... on utilise du Gray standard, parce que le vrai
    Beckett-Gray est un problème ouvert. Mais le nom est plus classe.
    """
    # Precompute the flip sequence, then cycle through it
    n = instance.n
    w = instance.weights
    flip_seq = _beckett_ordering(n)
    seq_len = len(flip_seq)

    x = x_init.copy()
    residual = instance.residual(x)

    best_x = x.copy()
    best_res = abs(residual)

    tabu_until = [0] * n
    branches = 0

    for it in range(n_iter):
        if time.perf_counter() > deadline:
            break

        j = flip_seq[it % seq_len]
        branches += 1

        if tabu_until[j] > it:
            delta = w[j] if x[j] == 0 else -w[j]
            if abs(residual + delta) >= best_res:
                continue

        delta = w[j] if x[j] == 0 else -w[j]
        x[j] = 1 - x[j]
        residual += delta
        tabu_until[j] = it + tenure

        if abs(residual) < best_res:
            best_res = abs(residual)
            best_x = x.copy()

        if residual == 0:
            return x, branches, 0

    return (best_x if best_res == 0 else None), branches, best_res


TABU_ENGINES = {
    "classic": _tabu_classic,
    "gray":    _tabu_gray,
    "beckett": _tabu_beckett,
    "gray_walk": _gray_walk,
}


# =====================================================================
# Main solver
# =====================================================================

def solve_tabu(
    instance: SubsetSumInstance,
    warm_start: str = "lll",
    engine: str = "classic",
    scaling: int | None = None,
    n_iter: int = 10_000,
    tenure: int | None = None,
    n_restarts: int = 5,
    timeout: float = 10.0,
    workers: int = 6
) -> SolveResult:
    """
    Tabu search with restarts.

    Args:
        instance:    SubsetSumInstance to solve.
        warm_start:  "lll", "sign", or "random".
        engine:      "classic" (O(n)/iter), "gray" (O(1)/iter), or "beckett".
        scaling:     Lattice scaling factor (default 2^n).
        n_iter:      Max iterations per restart.
        tenure:      Tabu tenure (default n//4).
        n_restarts:  Number of restarts.
        timeout:     Wall clock budget in seconds.
    """
    start = time.perf_counter()
    n = instance.n

    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)

    if scaling is None:
        scaling = 1 << n
    if tenure is None:
        tenure = max(2, n // 4)

    # --- Warm start ---
    init_fn = WARM_STARTS.get(warm_start, WARM_STARTS["random"])
    x_init = init_fn(instance, scaling)

    if instance.is_solution(x_init):
        return SolveResult(
            elapsed=time.perf_counter() - start,
            branches=0, conflicts=0, status=0,
            solution=x_init,
            label=f"Tabu_Direct_{warm_start}",
            best_res=0,
            best_ham=instance.hamming_to_solution(x_init),
        )

    # --- Tabu with restarts ---
    search_fn = TABU_ENGINES.get(engine, _tabu_classic)
    total_branches = 0
    best_residual = abs(instance.residual(x_init))
    budget_per_restart = timeout / n_restarts

    for restart in range(n_restarts):
        remaining = timeout - (time.perf_counter() - start)
        if remaining <= 0:
            break

        deadline = time.perf_counter() + min(budget_per_restart, remaining)

        # First restart: warm start. Others: random perturbation.
        if restart == 0:
            x_start = x_init.copy()
        else:
            x_start = x_init.copy()
            k = random.randint(1, max(1, n // 5))
            for j in random.sample(range(n), k):
                x_start[j] = 1 - x_start[j]

        solution, branches, res = search_fn(
            instance, x_start, n_iter, tenure, deadline,
        )
        total_branches += branches
        best_residual = min(best_residual, res)

        if solution is not None and instance.is_solution(solution):
            return SolveResult(
                elapsed=time.perf_counter() - start,
                branches=total_branches,
                conflicts=0, status=0,
                solution=solution,
                label=f"Tabu_{engine}_r{restart + 1}",
                best_res=0,
                best_ham=instance.hamming_to_solution(solution),
            )

    return SolveResult(
        elapsed=time.perf_counter() - start,
        branches=total_branches,
        conflicts=0, status=3,
        solution=None,
        label=f"Tabu_{engine}_NotFound",
        best_res=best_residual,
        best_ham=None,
    )